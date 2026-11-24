"""Collaborative (neighbour) filtering across gauges and basins: can predicted gauge metrics co-validate losses? (stage 62)

Idea: the gauge x day matrix of flow percentiles and the basin x day matrix of claims share one latent factor (storm intensity at
that place). A held-out gauge / basin is "completed" from (i) its own rain, (ii) what other gauges 10-60 km away observed that day
(inverse-distance weights, nested/overlapping basins excluded by the 10 km floor), (iii) their claims.
  T1 predict a held-out gauge's daily max-Q percentile: rain-only model vs neighbours-only vs both (spatial 5-fold block CV by
     gauge cluster, train <= 2016 on the other folds, test 2017-2023 on the held-out fold)
  T2 use that out-of-fold prediction (no own flow) as a claim feature: does a PREDICTED gauge product add to policies + rain,
     compared with the OBSERVED own flow?
  T3 neighbours' observed claims as a feature (loss co-validation)
  T4 verification: among days the rain-based score ranks in its top decile, does the observed flow / neighbour flow separate
     true from false alarms?
Wet days (3-day basin rain >= 10 mm) are used so the sample is not selected on the target.
Inputs: as script 116.   Run: python scripts/117_gauge_collaborative.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

spec = importlib.util.spec_from_file_location("g116", Path(__file__).with_name("116_gauge_outcomes_vs_claims.py"))
g116 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g116)
D, DAYS, auc, km = g116.D, g116.DAYS, g116.auc, g116.km
rng = np.random.default_rng(0)


def idw(M, dist, s_idx, lo=10, hi=60, k=5):
    """Inverse-distance mean of other columns of M (day x gauge) for column s_idx, neighbours between lo and hi km."""
    d = dist[s_idx]
    ok = np.flatnonzero((d >= lo) & (d <= hi))
    ok = ok[np.argsort(d[ok])][:k]
    if len(ok) == 0:
        return np.full(M.shape[0], np.nan), 0
    x = M[:, ok]
    w = 1.0 / d[ok]
    num = np.nansum(x * w, 1)
    den = ((~np.isnan(x)) * w).sum(1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan), len(ok)


def main():
    from sklearn.cluster import KMeans
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    g, S, R1, R3, mem, ring, bg, pos, ex, ncl, paid = g116.build()
    rain = pd.read_parquet(D / "tx_basin_rain.parquet", columns=["site", "day", "p7", "p30", "pmax"])
    rain["day"] = pd.to_datetime(rain.day)
    ext = {c: rain.pivot(index="day", columns="site", values=c).reindex(DAYS) for c in ("p7", "p30", "pmax")}
    fr = []
    for s in g.index:
        f = g116.gauge_frame(s, S, R1, R3, mem, bg, pos, ex, ncl, paid)
        if f is None:
            continue
        for c in ext:
            f[c] = ext[c][s].to_numpy() if s in ext[c].columns else np.nan
        f["larea"] = np.log(g.loc[s, "drainage_area"])
        fr.append(f)
    F = pd.concat(fr, ignore_index=True)
    F = F[F.pol.groupby(F.site).transform("mean") >= 20]
    F = F[F.qmax_pct.notna() | True]
    sites = list(F.site.unique())
    gg = g.loc[sites]
    lat, lon = gg.lat.to_numpy(), gg.lon.to_numpy()
    dist = km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    sidx = {s: i for i, s in enumerate(sites)}
    QP = F.pivot(index="day", columns="site", values="qmax_pct").reindex(index=DAYS, columns=sites).to_numpy()
    CR = (1000 * F.assign(r=F.claims / F.pol.replace(0, np.nan)).pivot(index="day", columns="site", values="r")
          .reindex(index=DAYS, columns=sites)).to_numpy()
    di = {d: i for i, d in enumerate(DAYS)}
    nb_q, nb_c, nn = np.full(len(F), np.nan), np.full(len(F), np.nan), np.zeros(len(F))
    for s in sites:
        m = (F.site == s).to_numpy()
        q, k = idw(QP, dist, sidx[s])
        c, _ = idw(np.log1p(CR), dist, sidx[s])
        ii = F.day[m].map(di).to_numpy()
        nb_q[m], nb_c[m], nn[m] = q[ii], c[ii], k
    F["nb_q"], F["nb_c"], F["n_nb"] = nb_q, nb_c, nn
    F["y"] = (F.claims > 0).astype(int)
    F["lp"], F["lr"], F["year"] = np.log1p(F.pol), np.log1p(F.p3), F.day.dt.year
    F["rain_pct"] = F.groupby("site").p3.rank(pct=True)
    W = F[(F.p3 >= 10) & F.qmax_pct.notna() & F.nb_q.notna() & F.pol.gt(0)].copy()
    fold = KMeans(5, n_init=5, random_state=0).fit_predict(np.c_[lat, lon])
    W["fold"] = W.site.map({s: fold[sidx[s]] for s in sites})
    print(f"\nwet gauge-days with own flow and >= 1 neighbour (10-60 km): {len(W):,}; gauges {W.site.nunique()}; "
          f"median neighbours {W.groupby('site').n_nb.first().median():.0f}; claim days {int(W.y.sum())}")

    # ---- T1: predict the gauge metric, spatial block CV ----
    feats_r = ["p1", "p3", "p7", "p30", "pmax", "larea"]
    feats_n = ["nb_q"]
    res = {k: np.full(len(W), np.nan) for k in ("rain", "nb", "both")}
    for f in range(5):
        trm = ((W.fold != f) & (W.year <= 2016)).to_numpy()
        tem = ((W.fold == f) & (W.year >= 2017)).to_numpy()
        for k, cols in (("rain", feats_r), ("nb", feats_n), ("both", feats_r + feats_n)):
            m = HGB(max_iter=200, learning_rate=0.08, max_depth=4, random_state=0).fit(W.loc[trm, cols], W.loc[trm, "qmax_pct"])
            res[k][tem] = m.predict(W.loc[tem, cols])
    te = W[W.year >= 2017].copy()
    tm = (W.year >= 2017).to_numpy()
    for k in res:
        te["pred_" + k] = res[k][tm]
    hi = (te.qmax_pct >= 0.99).astype(int)
    print("\n=== T1. Predicting a held-out gauge's daily max-Q percentile (held-out gauges, test 2017-2023) ===")
    rows = []
    for k in ("rain", "nb", "both"):
        p = te["pred_" + k]
        rows.append((k, np.corrcoef(p, te.qmax_pct)[0, 1] ** 2, te[["qmax_pct"]].assign(p=p).corr("spearman").iloc[0, 1],
                     auc(hi, p), auc((te.qmax_pct >= 0.9).astype(int), p)))
    print(pd.DataFrame(rows, columns=["inputs", "R2", "spearman", "AUC q>=0.99", "AUC q>=0.90"]).round(3).to_string(index=False))
    print(f"(n = {len(te):,}, {int(hi.sum())} days with q >= 0.99)")

    # ---- T2 / T3: as claim features ----
    print("\n=== T2/T3. Claim AUC (any claim in the basin, held-out gauges, test 2017-2023), gauge-clustered bootstrap vs policies + rain ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    trc = W[W.year <= 2016].copy()
    # training rows need an out-of-fold predicted flow too: use in-sample-free fold predictions on train years
    for k, cols in (("rain", feats_r), ("both", feats_r + feats_n)):
        trc["pred_" + k] = np.nan
        for f in range(5):
            a = ((W.fold != f) & (W.year <= 2016)).to_numpy()
            b = (trc.fold == f).to_numpy()
            m = HGB(max_iter=200, learning_rate=0.08, max_depth=4, random_state=0).fit(W.loc[a, cols], W.loc[a, "qmax_pct"])
            trc.loc[b, "pred_" + k] = m.predict(trc.loc[b, cols])
    sets = {"pol+rain": ["lp", "rain_pct"],
            "+ OWN observed flow": ["lp", "rain_pct", "qmax_pct"],
            "+ predicted flow (rain only)": ["lp", "rain_pct", "pred_rain"],
            "+ predicted flow (rain+neighbours)": ["lp", "rain_pct", "pred_both"],
            "+ neighbour observed flow": ["lp", "rain_pct", "nb_q"],
            "+ neighbour claims (10-60 km)": ["lp", "rain_pct", "nb_c"],
            "+ neighbour flow + claims": ["lp", "rain_pct", "nb_q", "nb_c"],
            "+ own flow + neighbour claims": ["lp", "rain_pct", "qmax_pct", "nb_c"]}
    sc_ = {}
    for nm, cols in sets.items():
        sc = StandardScaler().fit(trc[cols])
        m = LogisticRegression(max_iter=500).fit(sc.transform(trc[cols]), trc.y)
        sc_[nm] = m.decision_function(sc.transform(te[cols]))
    y, sit = te.y.to_numpy(), te.site.to_numpy()
    us = np.unique(sit)
    gi = {u: np.flatnonzero(sit == u) for u in us}
    base = sc_["pol+rain"]
    draws = [np.concatenate([gi[u] for u in rng.choice(us, len(us))]) for _ in range(200)]
    for nm, s_ in sc_.items():
        d = [auc(y[ix], s_[ix]) - auc(y[ix], base[ix]) for ix in draws] if nm != "pol+rain" else [0, 0]
        print(f"  {nm:<36} AUC {auc(y, s_):.3f}   gain {auc(y, s_) - auc(y, base):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

    # ---- T4: verification of rain-based alarms ----
    print("\n=== T4. Verification: among the 10% of test wet days with the highest rain-model claim score ===")
    top = base >= np.quantile(base, 0.9)
    tt = te[top]
    print(f"  alarms {top.sum():,}, observed claim rate {tt.y.mean():.3f}")
    for nm, col in (("own observed flow pct", "qmax_pct"), ("neighbour flow pct", "nb_q"), ("neighbour claims", "nb_c"),
                    ("predicted flow (rain+nb)", "pred_both")):
        v = tt[col]
        t = tt.assign(h=v >= v.median())
        print(f"  {nm:<26} AUC within alarms {auc(tt.y, v):.3f}; claim rate with {nm} above / below its median "
              f"{t[t.h].y.mean():.3f} / {t[~t.h].y.mean():.3f}")
    # storm-level agreement: per week, correlation across gauges of observed flow pct and claim rate
    te["wk"] = te.day.dt.to_period("W")
    wk = te.groupby("wk").filter(lambda d: len(d) >= 15 and d.y.sum() >= 3)
    r = [(d.qmax_pct.corr(d.claims / d.pol, method="spearman"), d.pred_both.corr(d.claims / d.pol, method="spearman"),
          d.p3.corr(d.claims / d.pol, method="spearman")) for _, d in wk.groupby("wk")]
    r = pd.DataFrame(r, columns=["flow", "pred_flow", "rain"])
    print(f"\n  within-week (storm) rank correlation across gauges with claim rate, {len(r)} weeks: median "
          f"own flow {r.flow.median():.2f}, predicted flow {r.pred_flow.median():.2f}, rain {r.rain.median():.2f}")


if __name__ == "__main__":
    main()
