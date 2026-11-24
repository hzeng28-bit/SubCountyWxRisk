"""Forecast-mode hydrologic verification at small unregulated gauges: GEFSv12 rain -> gauge flow -> claims (stage 64).

Issue time: 00Z of day S-1 (the reforecast of script 47). Valid days S and S+1 (leads 36-84 h). Units: gauge x day S, 2010-2019
(train 2010-2016, test 2017-2019), kept when EITHER the ensemble-mean forecast or the observed 0.25-degree cell rain reaches 10 mm
over the two days, so misses and false alarms stay in the sample (selection does not use the target).
Inputs known at issue: GEFS member rain at the cell of the basin centroid (days S, S+1), observed antecedent rain (days <= S-2).
Response models (HGB, train 2010-2016 on OBSERVED cell rain so the input resolution matches the forecast):
  reg  -> within-gauge percentile of the daily-max discharge, peak of days S, S+1
  clf  -> P(that percentile >= 0.99)
applied member by member, ensemble mean = forecast flow product; observed-rain versions give the ceiling.
Verification: V1 rain forecast vs observed cell rain; V2 forecast flow vs observed flow (AUC, reliability, Brier skill);
V3 claims (any claim in the basin on S..S+1) with forecast-derived features; V4 contingency: alert x observed-flow confirmation x claims.
Inputs : usgs_tx files (scripts 108-115), gefs/apcp_*.npy (script 47), gridMET, clean claims / policies.
Run    : python scripts/119_forecast_hydro_verification.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.ensemble import HistGradientBoostingClassifier as HGBC  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import zstio  # noqa: E402

spec = importlib.util.spec_from_file_location("g116", Path(__file__).with_name("116_gauge_outcomes_vs_claims.py"))
g116 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g116)
D, DAYS, auc = g116.D, g116.DAYS, g116.auc
GEFS = fp.PILOT.parent / "gefs"
rng = np.random.default_rng(0)


def build_forecast_arrays(sites, cent):
    """fc[day, member, 2, gauge], oc[day, 2, gauge]: GEFS member rain and observed gridMET cell-mean rain for days S, S+1."""
    meta = zstio.load(GEFS / "gefs_meta.npz", allow_pickle=True)
    glat, glon = meta["lat"], meta["lon"]
    cy = np.abs(glat[None, :] - cent[:, 1][:, None]).argmin(1)
    cx = np.abs(glon[None, :] - cent[:, 0][:, None]).argmin(1)
    days, lat, lon, P = fp.load_precip()
    py = np.abs(glat[None, :] - lat[:, None]).argmin(1)
    px = np.abs(glon[None, :] - lon[:, None]).argmin(1)
    cop = (py[:, None] * len(glon) + px[None, :]).ravel()
    ncell = len(glat) * len(glon)
    cnt = np.bincount(cop, minlength=ncell).astype(np.float32)
    pidx = {d: i for i, d in enumerate(days)}
    cache = {}

    def obs_cell(k):
        if k not in cache:
            f = np.nan_to_num(np.asarray(P[k]), nan=0.0).ravel()
            cache[k] = (np.bincount(cop, weights=f, minlength=ncell) / np.maximum(cnt, 1)).reshape(len(glat), len(glon))[cy, cx]
        return cache[k]

    sd = pd.date_range("2010-01-01", "2019-12-31")
    fc = np.full((len(sd), 5, 2, len(sites)), np.nan, np.float32)
    oc = np.full((len(sd), 2, len(sites)), np.nan, np.float32)
    for i, s in enumerate(sd):
        f = GEFS / f"apcp_{s - pd.Timedelta(1, 'D'):%Y%m%d}.npy"
        if not f.exists():
            continue
        a = zstio.load(f).astype(np.float32)
        fc[i, :, 0] = a[:, :4].sum(1)[:, cy, cx]
        fc[i, :, 1] = a[:, 4:].sum(1)[:, cy, cx]
        k = pidx[s]
        oc[i, 0], oc[i, 1] = obs_cell(k), obs_cell(k + 1)
        if i % 500 == 0:
            print("  gefs/obs cell", s.date(), flush=True)
    return sd, fc, oc


def main():
    import shapely  # noqa: F401
    from shapely.geometry import shape

    g, S, R1, R3, mem, ring, bg, pos, ex, ncl, paid = g116.build()
    fr = [f for s in g.index if (f := g116.gauge_frame(s, S, R1, R3, mem, bg, pos, ex, ncl, paid)) is not None]
    F = pd.concat(fr, ignore_index=True)
    F = F[F.pol.groupby(F.site).transform("mean") >= 20]
    sites = list(F.site.unique())
    fc_ = {f["properties"]["site"]: shape(f["geometry"]).centroid
           for f in json.loads((D / "tx_basins.geojson").read_text())["features"]}
    cent = np.array([[fc_[s].x, fc_[s].y] for s in sites])
    sd, fc, oc = build_forecast_arrays(sites, cent)
    rain = pd.read_parquet(D / "tx_basin_rain.parquet", columns=["site", "day", "p7", "p30"])
    rain["day"] = pd.to_datetime(rain.day)
    # antecedent rain known at issue: p7 / p30 ending on day S-2 (gridMET day S-1 is half after the 00Z issue)
    A7 = rain.pivot(index="day", columns="site", values="p7").reindex(columns=sites)
    A30 = rain.pivot(index="day", columns="site", values="p30").reindex(columns=sites)
    Q = F.pivot(index="day", columns="site", values="qmax_pct").reindex(index=g116.DAYS, columns=sites)
    C = F.pivot(index="day", columns="site", values="claims").reindex(index=g116.DAYS, columns=sites)
    POL = F.pivot(index="day", columns="site", values="pol").reindex(index=g116.DAYS, columns=sites)
    OB = F.pivot(index="day", columns="site", values="p1").reindex(index=g116.DAYS, columns=sites)
    la = np.log(g.loc[sites, "drainage_area"].to_numpy())
    n, ng = len(sd), len(sites)
    di = np.arange(n)
    qpk = np.fmax(Q.to_numpy()[di], Q.to_numpy()[di + 1])
    cl2 = C.to_numpy()[di] + C.to_numpy()[di + 1]
    pol = POL.to_numpy()[di]
    a7 = A7.reindex(sd - pd.Timedelta(2, "D")).to_numpy()
    a30 = A30.reindex(sd - pd.Timedelta(2, "D")).to_numpy()
    obb = np.fmax(OB.to_numpy()[di], OB.to_numpy()[di + 1])  # observed basin-mean daily max (finer than the cell)
    obs2 = oc.sum(1)                       # [day, gauge] observed cell sum over S, S+1
    obsm = oc.max(1)
    fcs = fc.sum(2)                        # [day, member, gauge]
    fcm = fc.max(2)
    ens_s = fcs.mean(1)
    ens_m = fcm.mean(1)
    valid = ~np.isnan(ens_s) & ~np.isnan(obs2) & ~np.isnan(qpk) & ~np.isnan(a7) & ~np.isnan(a30) & (pol > 0)
    wet = (ens_s >= 10) | (obs2 >= 10)
    ok = valid & wet
    dd, gg = np.nonzero(ok)
    U = pd.DataFrame({"site": np.array(sites)[gg], "gi": gg, "di": dd, "day": sd[dd], "year": sd[dd].year, "q": qpk[dd, gg],
                      "claims": cl2[dd, gg], "pol": pol[dd, gg], "a7": a7[dd, gg], "a30": a30[dd, gg], "larea": la[gg],
                      "o_s": obs2[dd, gg], "o_m": obsm[dd, gg], "ob_b": obb[dd, gg], "f_s": ens_s[dd, gg], "f_m": ens_m[dd, gg]})
    U["y"] = (U.claims > 0).astype(int)
    U["hi"] = (U.q >= 0.99).astype(int)
    U["lp"] = np.log1p(U.pol)
    tr, te = U.year <= 2016, U.year >= 2017
    print(f"\ngauges {len(sites)}; units (gauge x day, either source >= 10 mm): {len(U):,} (train {tr.sum():,}, test {te.sum():,}); "
          f"test: flow >= 0.99 on {int(U[te].hi.sum())}, claim units {int(U[te].y.sum())}")

    # ---- response models on OBSERVED cell rain ----
    cols = ["o_m", "o_s", "a7", "a30", "larea"]
    reg = HGB(max_iter=250, learning_rate=0.06, max_depth=4, random_state=0).fit(U.loc[tr, cols], U.loc[tr, "q"])
    clf = HGBC(max_iter=250, learning_rate=0.06, max_depth=4, random_state=0).fit(U.loc[tr, cols], U.loc[tr, "hi"])
    regb = HGB(max_iter=250, learning_rate=0.06, max_depth=4, random_state=0).fit(
        U.loc[tr, ["ob_b", "o_s", "a7", "a30", "larea"]], U.loc[tr, "q"])

    def member_pred(model, kind):
        out = np.zeros((len(U), 5))
        for m in range(5):
            X = U[["a7", "a30", "larea"]].copy()
            X.insert(0, "o_s", fcs[U.di, m, U.gi])
            X.insert(0, "o_m", fcm[U.di, m, U.gi])
            X = X[cols]
            out[:, m] = model.predict(X) if kind == "reg" else model.predict_proba(X)[:, 1]
        return out
    Pm_reg, Pm_clf = member_pred(reg, "reg"), member_pred(clf, "clf")
    U["fq_reg"], U["fq_clf"] = Pm_reg.mean(1), Pm_clf.mean(1)
    U["fq_reg_spread"] = Pm_reg.std(1)
    U["oq_reg"] = reg.predict(U[cols])
    U["oq_clf"] = clf.predict_proba(U[cols])[:, 1]
    U["oq_basin"] = regb.predict(U[["ob_b", "o_s", "a7", "a30", "larea"]])
    T = U[te]

    # ---- V1 ----
    print("\n=== V1. Rain: GEFS ensemble mean vs observed gridMET, same 0.25-degree cell (test 2017-2019) ===")
    from scipy.stats import spearmanr
    print(f"  2-day sum: Spearman {spearmanr(T.f_s, T.o_s).statistic:.2f}; mean forecast {T.f_s.mean():.1f} mm vs observed {T.o_s.mean():.1f} mm; "
          f"on observed >= 25 mm: forecast mean {T[T.o_s >= 25].f_s.mean():.1f} vs {T[T.o_s >= 25].o_s.mean():.1f}")
    print(f"  AUC of the forecast 2-day sum for observed cell rain >= 25 mm: {auc((T.o_s >= 25).astype(int), T.f_s):.3f}; >= 50 mm: "
          f"{auc((T.o_s >= 50).astype(int), T.f_s):.3f}")

    # ---- V2 ----
    print("\n=== V2. Forecast flow vs observed flow (observed peak percentile of daily-max Q on S, S+1; test) ===")
    rows = []
    for nm, s_ in (("forecast rain, ensemble-mean 2-day sum", T.f_s), ("forecast flow percentile (reg, member mean)", T.fq_reg),
                   ("forecast P(flow >= 0.99) (clf, member mean)", T.fq_clf), ("OBSERVED cell rain 2-day sum", T.o_s),
                   ("flow from observed cell rain (reg)", T.oq_reg), ("flow from observed basin rain (finer)", T.oq_basin)):
        rows.append((nm, auc(T.hi, s_), auc((T.q >= 0.9).astype(int), s_), spearmanr(s_, T.q).statistic))
    print(pd.DataFrame(rows, columns=["input", "AUC q>=0.99", "AUC q>=0.90", "Spearman with q"]).round(3).to_string(index=False))
    base_rate = U[tr].hi.mean()
    p = T.fq_clf.to_numpy()
    bs = ((p - T.hi) ** 2).mean()
    bs0 = ((base_rate - T.hi) ** 2).mean()
    po = T.oq_clf.to_numpy()
    print(f"  Brier skill vs the train base rate {base_rate:.3f}: forecast clf {1 - bs / bs0:+.3f}; observed-rain clf {1 - ((po - T.hi) ** 2).mean() / bs0:+.3f}")
    rel = T.assign(b=pd.cut(T.fq_clf, [0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0], include_lowest=True)).groupby("b", observed=True).agg(
        n=("hi", "size"), forecast_p=("fq_clf", "mean"), observed_freq=("hi", "mean"))
    print("  reliability of the forecast exceedance probability:\n" + rel.round(3).to_string())

    # ---- V3 ----
    print("\n=== V3. Claims in the basin (any claim on S..S+1): forecast-arm AUC, train 2010-2016, test 2017-2019, gauge bootstrap ===")
    U["fr"] = np.log1p(U.f_s)
    U["orr"] = np.log1p(U.o_s)
    sets = {"pol + forecast rain": ["lp", "fr"], "pol + forecast rain + forecast flow (reg)": ["lp", "fr", "fq_reg"],
            "pol + forecast rain + forecast P(flow>=0.99)": ["lp", "fr", "fq_clf"],
            "pol + forecast rain + both + spread": ["lp", "fr", "fq_reg", "fq_clf", "fq_reg_spread"],
            "pol + OBSERVED rain": ["lp", "orr"], "pol + forecast rain + OBSERVED flow (post-event)": ["lp", "fr", "q"],
            "pol + OBSERVED rain + OBSERVED flow": ["lp", "orr", "q"]}
    Tr, T = U[tr], U[te]
    sc = {}
    for nm, c in sets.items():
        s_ = StandardScaler().fit(Tr[c])
        sc[nm] = LogisticRegression(max_iter=500).fit(s_.transform(Tr[c]), Tr.y).decision_function(s_.transform(T[c]))
    y, sit = T.y.to_numpy(), T.site.to_numpy()
    us = np.unique(sit)
    gi = {u: np.flatnonzero(sit == u) for u in us}
    draws = [np.concatenate([gi[u] for u in rng.choice(us, len(us))]) for _ in range(200)]
    b0 = sc["pol + forecast rain"]
    for nm, s_ in sc.items():
        d = [auc(y[i], s_[i]) - auc(y[i], b0[i]) for i in draws] if nm != "pol + forecast rain" else [0, 0]
        print(f"  {nm:<52} AUC {auc(y, s_):.3f}  vs forecast-rain base {auc(y, s_) - auc(y, b0):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")

    # ---- V4 ----
    print("\n=== V4. Alert x observed-flow confirmation x claims (test; alert = forecast P(flow >= 0.99) >= 0.15) ===")
    T = T.assign(alert=T.fq_clf >= 0.15, conf=T.q >= 0.99)
    t4 = T.groupby(["alert", "conf"]).agg(units=("y", "size"), claim_rate=("y", "mean"), claims=("claims", "sum"),
                                          pol=("pol", "sum"))
    t4["claims_per_1000_pol"] = 1000 * t4.claims / t4.pol
    print(t4.round(3).to_string())
    a = T[T.alert]
    print(f"  among {len(a):,} alerts: hit (flow confirms) {a.conf.mean():.3f}; claim rate if confirmed {a[a.conf].y.mean():.3f} vs not confirmed "
          f"{a[~a.conf].y.mean():.3f}; AUC of observed flow for claims within alerts {auc(a.y, a.q):.3f}")
    m = T[~T.alert & T.conf]
    print(f"  misses (no alert, flow >= 0.99): {len(m)} units, claim rate {m.y.mean():.3f}; share of all claim units {m.y.sum() / max(T.y.sum(), 1):.3f}")


if __name__ == "__main__":
    main()
