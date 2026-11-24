"""Daily-max discharge, stage and mean discharge vs NFIP claims: space, time and cause (stage 61).

Uses the 148 small unregulated gauges of script 114 and the daily statistics of script 115.
  A outcomes   same-day AUC of each gauge outcome (within-gauge percentile) for 'any claim in the basin', on top of policies + rain
  B time       AUC of the outcome at day d+k for claims on day d, k = -3..+3; lag between loss date and the day of peak
  C space      claims per 1000 policy-days on high-flow days by distance of block-group centroid to the gauge, inside / outside the basin
  D cause      mediation (rain -> flow -> claims), storm fixed effects, placebo (flow of a far gauge, flow after the claim)
Inputs : usgs_tx/{tx_small_gauges,tx_small_daily_stats,tx_basin_rain,tx_basins.geojson}, clean claims / policies
Run    : python scripts/116_gauge_outcomes_vs_claims.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import flood_pilot as fp  # noqa: E402

spec = importlib.util.spec_from_file_location("g114", Path(__file__).with_name("114_small_gauge_loss.py"))
g114 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g114)
D, DAYS = g114.D, g114.DAYS
rng = np.random.default_rng(0)
auc = g114.auc


def km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def pct(a):
    return pd.Series(a).rank(pct=True).to_numpy()


def build():
    import json

    import shapely
    from shapely.geometry import shape

    g = pd.read_parquet(D / "tx_small_gauges.parquet")
    st = pd.read_parquet(D / "tx_small_daily_stats.parquet")
    st = st[st.monitoring_location_id.isin(g.index)]
    mean_q = pd.read_parquet(D / "tx_daily_q.parquet").pivot_table(index="time", columns="monitoring_location_id", values="value")
    tab = {"qmax": ("00060", "00001"), "hmax": ("00065", "00001"), "hmean": ("00065", "00003")}
    S = {k: st[(st.param == p) & (st.stat == s)].pivot_table(index="time", columns="monitoring_location_id", values="value")
         .reindex(DAYS) for k, (p, s) in tab.items()}
    S["qmean"] = mean_q.reindex(DAYS)
    fc = {f["properties"]["site"]: shape(f["geometry"]).buffer(0.002)
          for f in json.loads((D / "tx_basins.geojson").read_text())["features"]}
    bg = fp.bg_geometry()
    pts = shapely.points(bg.lon.to_numpy(), bg.lat.to_numpy())
    rain = pd.read_parquet(D / "tx_basin_rain.parquet", columns=["site", "day", "p1", "p3"])
    rain = rain[rain.site.isin(g.index)]
    rain["day"] = pd.to_datetime(rain.day)
    R1 = rain.pivot(index="day", columns="site", values="p1").reindex(DAYS)
    R3 = rain.pivot(index="day", columns="site", values="p3").reindex(DAYS)
    # block groups: inside basin, or outside within 50 km of the gauge
    mem, ring = {}, {}
    for s in g.index:
        inside = shapely.contains(fc[s], pts)
        d = km(g.loc[s, "lat"], g.loc[s, "lon"], bg.lat.to_numpy(), bg.lon.to_numpy())
        mem[s] = (np.flatnonzero(inside), d[inside])
        o = np.flatnonzero(~inside & (d <= 50))
        ring[s] = (o, d[o])
    used = sorted({i for s in g.index for i in np.r_[mem[s][0], ring[s][0]]})
    bgs = bg.bg10.iloc[used].tolist()
    pos = {b: i for i, b in enumerate(bgs)}
    print(f"gauges {len(g)}; block groups inside basins or within 50 km: {len(bgs):,}")
    ex = fp.exposure_matrices(bgs)["n_pol"]
    ncl, paid, nc = g114.loss_arrays(bgs)
    print(f"claims in these block groups: {nc:,.0f}")
    return g, S, R1, R3, mem, ring, bg, pos, ex, ncl, paid


def gauge_frame(s, S, R1, R3, mem, bg, pos, ex, ncl, paid):
    m = np.array([pos[bg.bg10.iloc[i]] for i in mem[s][0]], dtype=int)
    if len(m) == 0:
        return None
    f = pd.DataFrame({"day": DAYS, "pol": ex[m].sum(0), "claims": ncl[m].sum(0), "paid": paid[m].sum(0),
                      "p1": R1[s].to_numpy(), "p3": R3[s].to_numpy()})
    for k in S:
        a = S[k][s].to_numpy() if s in S[k].columns else np.full(len(DAYS), np.nan)
        f[k] = a
        f[k + "_pct"] = np.where(np.isnan(a), np.nan, pct(np.where(np.isnan(a), -np.inf, a)))
        if np.isnan(a).all() or np.isnan(a).mean() > 0.8:
            f[k + "_pct"] = np.nan
    f["site"] = s
    return f


def logit_auc(tr, te, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(tr[cols])
    m = LogisticRegression(max_iter=500).fit(sc.transform(tr[cols]), tr.y)
    return m.decision_function(sc.transform(te[cols])), m.coef_[0]


def boot(te, score_a, score_b, n=200):
    sites, y = te.site.to_numpy(), te.y.to_numpy()
    us = np.unique(sites)
    gi = {u: np.flatnonzero(sites == u) for u in us}
    d = []
    for _ in range(n):
        ix = np.concatenate([gi[u] for u in rng.choice(us, len(us))])
        d.append(auc(y[ix], score_b[ix]) - auc(y[ix], score_a[ix]))
    return np.percentile(d, [2.5, 97.5])


def main():
    g, S, R1, R3, mem, ring, bg, pos, ex, ncl, paid = build()
    F = pd.concat([f for s in g.index if (f := gauge_frame(s, S, R1, R3, mem, bg, pos, ex, ncl, paid)) is not None])
    F = F[F.pol.groupby(F.site).transform("mean") >= 20]
    F["y"] = (F.claims > 0).astype(int)
    F["lp"], F["lr"], F["lr1"] = np.log1p(F.pol), np.log1p(F.p3), np.log1p(F.p1)
    F["year"] = F.day.dt.year
    have_h = F.groupby("site").hmax_pct.transform(lambda x: x.notna().mean()) > 0.5
    print(f"\ngauges with stage (gage height) record: {F[have_h].site.nunique()} of {F.site.nunique()}")
    cand = ((F.qmax_pct >= 0.9) | (F.p3 >= 25)) & F.pol.gt(0) & F.p3.notna()

    # ---- A: outcomes ----
    print("\n=== A. Same-day outcomes vs 'any claim in the basin' (candidate gauge-days; gauges with stage record) ===")
    P = F[cand & have_h].dropna(subset=["qmax_pct", "qmean_pct", "hmax_pct", "hmean_pct"])
    tr, te = P[P.year <= 2016], P[P.year >= 2017]
    print(f"candidate gauge-days {len(P):,} (test {len(te):,}, claims {int(te.y.sum())}), gauges {P.site.nunique()}")
    base, _ = logit_auc(tr, te, ["lp", "lr"])
    print(f"policies + rain: AUC {auc(te.y, base):.3f}")
    rows = []
    for k in ("qmean_pct", "qmax_pct", "hmean_pct", "hmax_pct"):
        a, _ = logit_auc(tr, te, [k])
        b, _ = logit_auc(tr, te, ["lp", "lr", k])
        lo, hi = boot(te, base, b)
        c = P[P.claims > 0]
        from scipy.stats import spearmanr
        rows.append((k, auc(te.y, a), auc(te.y, b), auc(te.y, b) - auc(te.y, base), lo, hi,
                     spearmanr(c[k], c.claims).statistic, spearmanr(c[k], c.paid).statistic))
    print(pd.DataFrame(rows, columns=["outcome", "alone", "+pol+rain", "gain", "lo", "hi", "rho_claims|claimdays", "rho_paid"]).round(3).to_string(index=False))
    a, _ = logit_auc(tr, te, ["lp", "lr", "qmax_pct", "hmax_pct"])
    print(f"pol+rain+qmax+hmax: {auc(te.y, a):.3f}")
    print("rain only (p3 pct) alone:", round(auc(te.y, te.p3), 3))

    # ---- B: time ----
    print("\n=== B. Timing: AUC of outcome at day d+k for claims on day d (candidate days by max flow or rain at d) ===")
    out = []
    Fk = F.copy()
    for k in range(-3, 4):
        r = {"k": k}
        for col in ("qmax_pct", "hmax_pct", "p3"):
            sh = Fk.groupby("site")[col].shift(-k)
            m = cand & sh.notna() & (have_h if col == "hmax_pct" else True)
            r[col] = auc(Fk.y[m], sh[m])
        out.append(r)
    print(pd.DataFrame(out).round(3).to_string(index=False))
    print("(k>0: outcome AFTER the loss date; k<0: BEFORE)")
    # event-level lag: claim-weighted mean loss date vs day of peak within +-6 days
    ev = []
    for s, f in F.groupby("site"):
        f = f.reset_index(drop=True)
        big = f.index[f.claims >= 5]
        done = -99
        for i in big:
            if i - done < 7:
                continue
            w = f.iloc[max(0, i - 6): i + 7]
            if w.claims.sum() < 10 or w.qmax.isna().all():
                continue
            cd = (w.claims * np.arange(len(w))).sum() / w.claims.sum()
            row = {"site": s, "claims": w.claims.sum(), "d_claim": cd, "d_qmax": float(np.nanargmax(w.qmax.to_numpy())),
                   "d_rain": float(np.nanargmax(w.p1.to_numpy())), "first": float(np.argmax(w.claims.to_numpy() > 0))}
            if w.hmax.notna().sum() > 3:
                row["d_hmax"] = float(np.nanargmax(w.hmax.to_numpy()))
            ev.append(row)
            done = i
    E = pd.DataFrame(ev)
    print(f"events (>= 10 claims within +-6 d, {len(E)} gauge-events, {E.site.nunique()} gauges). Days relative to claim-weighted mean loss date:")
    print(f"  peak max-Q day - mean loss day: median {np.median(E.d_qmax - E.d_claim):+.1f} d  IQR {np.percentile(E.d_qmax - E.d_claim, 25):+.1f}..{np.percentile(E.d_qmax - E.d_claim, 75):+.1f}")
    print(f"  peak rain day - mean loss day: median {np.median(E.d_rain - E.d_claim):+.1f} d  IQR {np.percentile(E.d_rain - E.d_claim, 25):+.1f}..{np.percentile(E.d_rain - E.d_claim, 75):+.1f}")
    if "d_hmax" in E:
        e = E.dropna(subset=["d_hmax"])
        print(f"  peak stage day - mean loss day: median {np.median(e.d_hmax - e.d_claim):+.1f} d (n={len(e)})")
    print(f"  peak max-Q day - peak rain day: median {np.median(E.d_qmax - E.d_rain):+.1f} d")

    # ---- C: space ----
    print("\n=== C. Space: claims per 1000 policy-days on high-flow days (qmax pct >= 0.99) by block-group centroid distance to the gauge ===")
    rows = []
    for s in g.index:
        f = F[F.site == s]
        hi = f.qmax_pct.ge(0.99).to_numpy()
        if hi.sum() == 0:
            continue
        di = np.flatnonzero(hi)
        for where, (ix, dist) in (("inside basin", mem[s]), ("outside basin", ring[s])):
            ids = np.array([pos[bg.bg10.iloc[i]] for i in ix], dtype=int)
            for lo, up in ((0, 5), (5, 10), (10, 25), (25, 50)):
                sel = ids[(dist >= lo) & (dist < up)]
                if len(sel) == 0:
                    continue
                rows.append((where, f"{lo}-{up} km", ncl[sel][:, di].sum(), ex[sel][:, di].sum(),
                             ncl[sel].sum(), ex[sel].sum(), s))
    C = pd.DataFrame(rows, columns=["where", "dist", "c_hi", "p_hi", "c_all", "p_all", "site"])
    t = C.groupby(["where", "dist"]).sum(numeric_only=True)
    t["per1000_hi"] = 1000 * t.c_hi / t.p_hi
    t["per1000_all"] = 1000 * t.c_all / t.p_all
    t["ratio"] = t.per1000_hi / t.per1000_all
    print(t[["c_hi", "per1000_hi", "per1000_all", "ratio"]].round(3).to_string())
    # per-gauge spatial: share of gauge-days with claims in basin when flow high, by basin area
    F["abin"] = pd.cut(F.site.map(g.drainage_area), [0, 25, 100, 250])
    hh = F[F.qmax_pct >= 0.99].groupby("abin", observed=True).agg(days=("y", "size"), any_claim=("y", "mean"))
    print("any claim on high-flow days by drainage area:\n", hh.round(3).to_string())

    # ---- D: cause ----
    print("\n=== D. Cause: rain -> flow -> claims (candidate gauge-days, train <= 2016 / test 2017+) ===")
    P = F[cand].dropna(subset=["qmax_pct"]).copy()
    P["rain_pct"] = P.groupby("site").p3.rank(pct=True)
    tr, te = P[P.year <= 2016], P[P.year >= 2017]
    _, c0 = logit_auc(tr, te, ["lp", "rain_pct"])
    _, c1 = logit_auc(tr, te, ["lp", "rain_pct", "qmax_pct"])
    print(f"standardised logit coefficient of rain: {c0[1]:.3f} alone -> {c1[1]:.3f} with flow (flow {c1[2]:.3f}); "
          f"share of the rain effect carried through flow ~ {(1 - c1[1] / c0[1]) * 100:.0f}%")
    # storm fixed effects: demean by (week, all gauges) and by gauge
    P["storm"] = P.day.dt.to_period("W")
    Z = P[["y", "rain_pct", "qmax_pct", "hmax_pct", "lp"]].copy()
    for _ in range(3):
        for key in (P.storm, P.site):
            Z = Z - Z.groupby(key.to_numpy()).transform("mean")
    X = Z[["rain_pct", "qmax_pct"]].to_numpy()
    beta = np.linalg.lstsq(X, Z.y.to_numpy(), rcond=None)[0]
    X0 = Z[["rain_pct"]].to_numpy()
    r2 = lambda Xm: 1 - np.var(Z.y.to_numpy() - Xm @ np.linalg.lstsq(Xm, Z.y.to_numpy(), rcond=None)[0]) / np.var(Z.y.to_numpy())
    print(f"with week-of-storm and gauge fixed effects (LPM): beta rain {beta[0]:.3f}, flow {beta[1]:.3f}; within R2 rain {r2(X0):.3f} -> rain+flow {r2(X):.3f}")
    # placebo: flow of the nearest other gauge > 40 km away (same storm, different basin)
    pos_xy = g[["lat", "lon"]].to_numpy()
    far = {}
    for i, s in enumerate(g.index):
        d = km(pos_xy[i, 0], pos_xy[i, 1], pos_xy[:, 0], pos_xy[:, 1])
        d[i] = 1e9
        ok = np.flatnonzero(d > 40)
        far[s] = g.index[ok[np.argmin(d[ok])]] if len(ok) else None
    QP = F.pivot(index="day", columns="site", values="qmax_pct")
    P["far_q"] = [QP.at[d, far[s]] if far[s] in QP.columns else np.nan for d, s in zip(P.day, P.site)]
    P2 = P.dropna(subset=["far_q"])
    tr2, te2 = P2[P2.year <= 2016], P2[P2.year >= 2017]
    for nm, cols in (("pol+rain", ["lp", "rain_pct"]), ("+ own flow", ["lp", "rain_pct", "qmax_pct"]),
                     ("+ far gauge flow (>40 km)", ["lp", "rain_pct", "far_q"]), ("+ both", ["lp", "rain_pct", "qmax_pct", "far_q"])):
        s_, _ = logit_auc(tr2, te2, cols)
        print(f"  {nm:<28} AUC {auc(te2.y, s_):.3f}")
    # lead placebo: flow at d+3 and d-3 on top of rain
    for k in (-3, -1, 1, 3):
        P[f"q{k}"] = P.groupby("site").qmax_pct.shift(-k)
    P3 = P.dropna(subset=["q-3", "q-1", "q1", "q3"])
    tr3, te3 = P3[P3.year <= 2016], P3[P3.year >= 2017]
    b, _ = logit_auc(tr3, te3, ["lp", "rain_pct"])
    print(f"  pol+rain {auc(te3.y, b):.3f}; adding flow at d+k: ", end="")
    for k in (-3, -1, 0, 1, 3):
        col = "qmax_pct" if k == 0 else f"q{k}"
        s_, _ = logit_auc(tr3, te3, ["lp", "rain_pct", col])
        print(f"k={k:+d}: {auc(te3.y, s_) - auc(te3.y, b):+.4f}  ", end="")
    print()


if __name__ == "__main__":
    main()
