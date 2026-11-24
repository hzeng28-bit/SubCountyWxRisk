"""Small, unregulated Texas streamgages vs NFIP flood loss inside their basins (stage 60).

Question (user): keep only small stations, drop dams and other fixed hydraulic works, and ask whether the gauge's flow is
related to the loss level (claim count, paid building loss) of the insured buildings in its basin.

Selection (each step printed with its count):
  size       drainage area <= AMAX mi2 (default 250; sensitivity 50)
  record     >= 3 years of daily Q with >= 80% coverage in 2010-2023
  name       drop names that mark a structure: dam, lake/reservoir outflow or "bl/blw <lake>", weir, spillway, canal, ditch,
             diversion, pipeline, pump, effluent, floodway, tunnel (the USGS site type was not fetched)
  upstream   drop a gauge whose NLDI basin contains a name-flagged gauge (a reservoir or canal sits upstream)
  flashy     drop a gauge with a Richards-Baker flashiness index below RB_MIN (flow is damped like a regulated reach)
  basin      NLDI basin polygon exists and holds >= 20 policies on average (block-group centroids inside the basin)
Panel: gauge x day on candidate days (peak percentile of daily Q on d..d+1 >= 0.9, or 3-day basin rain >= 25 mm).
Outcome: NFIP claims (cause 2/3/4, non-tropical-named) with loss date d..d+1 in the basin's block groups; paid building loss.
Tests: claim rate per 1000 policies by flow-percentile bin; any-claim AUC on test years 2017-2023 (train <= 2016) for
  log policies + basin rain vs + flow percentile, gauge-clustered bootstrap; severity (paid loss) vs flow percentile.

Inputs : usgs_tx/{tx_gauges_q,tx_daily_q,tx_basins.geojson,tx_basin_rain} (scripts 50, 108, 109), clean claims / policies
Outputs: stdout (tee to $WXRISK_DATA/audit/small_gauge_loss.txt), usgs_tx/tx_small_gauges.parquet
Run    : python scripts/114_small_gauge_loss.py [--amax 250] [--rb-min 0.03]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

D = wb2.cache_path("labels_extra") / "flood" / "usgs_tx"
DAYS = pd.date_range("2010-01-01", "2023-12-31")
NAME_PAT = (r"\bdam\b|spillway|outflow|\bweir\b|canal|ditch|diversion|pipeline|pump|effluent|floodway|tunnel|"
            r"\b(?:bl|blw|below|bel)\b.*\b(?:lk|lake|res|reservoir)\b|\b(?:res|reservoir)\b(?! nr)|\bat (?:lk|lake)\b|"
            r"\b(?:lk|lake)\b.*\bdam\b|\bcnl\b|\bflume\b|\bpower\b")
rng = np.random.default_rng(0)


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def select(amax, rb_min):
    import shapely
    from shapely.geometry import shape

    g = pd.read_parquet(D / "tx_gauges_q.parquet").set_index("id")
    q = pd.read_parquet(D / "tx_daily_q.parquet").pivot_table(index="time", columns="monitoring_location_id", values="value")
    q = q.reindex(DAYS)
    g = g[g.index.isin(q.columns)].copy()
    steps = [("all gauges with daily Q", len(g))]
    g = g[g.drainage_area.le(amax) & g.drainage_area.gt(0)]
    steps.append((f"drainage area <= {amax} mi2", len(g)))
    cov = q[g.index].notna().groupby(q.index.year).mean()
    g = g[((cov >= 0.8).sum() >= 3).reindex(g.index).fillna(False)]
    steps.append(("record >= 3 yr at 80% coverage", len(g)))
    allg = pd.read_parquet(D / "tx_gauges_q.parquet").set_index("id")
    flagged = allg.index[allg.monitoring_location_name.str.lower().str.contains(NAME_PAT)]
    g = g[~g.index.isin(flagged)]
    steps.append(("name is not a structure", len(g)))
    # upstream structure: a flagged gauge lies inside the basin
    fc = {f["properties"]["site"]: shape(f["geometry"]).buffer(0.002)
          for f in json.loads((D / "tx_basins.geojson").read_text())["features"]}
    fp_pts = shapely.points(allg.loc[flagged].lon.to_numpy(), allg.loc[flagged].lat.to_numpy())
    bad = [s for s in g.index if s in fc and shapely.contains(fc[s], fp_pts).any()]
    g = g[~g.index.isin(bad)]
    steps.append(("no flagged structure gauge upstream", len(g)))
    qq = q[g.index]
    rb = qq.diff().abs().sum() / qq.sum()
    g["rb"] = rb
    g = g[g.rb >= rb_min]
    steps.append((f"Richards-Baker flashiness >= {rb_min}", len(g)))
    g = g[g.index.isin(fc)]
    steps.append(("NLDI basin available", len(g)))
    for n, k in steps:
        print(f"  {n:<40}{k:>6}")
    print(f"  flagged as structure by name: {len(flagged)}; flagged-upstream drops: {len(bad)}; RB quantiles of the kept set "
          f"{np.round(g.rb.quantile([.05, .25, .5, .75]).to_numpy(), 3)}")
    print("  examples dropped by name:", "; ".join(allg.loc[flagged].monitoring_location_name.head(8)))
    return g, q, fc


def loss_arrays(bgs):
    import polars as pl

    from wxrisk.prep import build as prep

    idx = {b: i for i, b in enumerate(bgs)}
    s = (prep.scan("claims_nfip")
         .filter((pl.col("state") == "TX") & pl.col("cause").is_in(["2", "3", "4"]) & pl.col("geoid").is_not_null()
                 & ~pl.col("flood_event").fill_null("").str.contains(fp.TC_PATTERN))
         .select(pl.col("geoid").alias("bg"), pl.col("loss_date").alias("date"),
                 pl.col("paid_building").fill_null(0.0).alias("paid"))
         .collect().to_pandas())
    s["date"] = pd.to_datetime(s["date"])
    cw = fp.bg_crosswalk.build().set_index("code")
    s["bg10"] = s.bg.map(cw.bg10)
    s = s[s.bg10.isin(idx) & s.date.between(DAYS[0], DAYS[-1])]
    bi, di = s.bg10.map(idx).to_numpy(), (s.date - DAYS[0]).dt.days.to_numpy()
    n = np.zeros((len(bgs), len(DAYS)), np.float32)
    p = np.zeros_like(n)
    np.add.at(n, (bi, di), 1)
    np.add.at(p, (bi, di), s.paid.to_numpy(np.float32))
    return n, p, len(s)


def panel(g, q, fc):
    import shapely

    bg = fp.bg_geometry()
    pts = shapely.points(bg.lon.to_numpy(), bg.lat.to_numpy())
    members = {s: np.flatnonzero(shapely.contains(fc[s], pts)) for s in g.index}
    used = sorted({i for v in members.values() for i in v})
    bgs = bg.bg10.iloc[used].tolist()
    pos = {b: i for i, b in enumerate(bgs)}
    print(f"  block groups inside the selected basins: {len(bgs):,}")
    ex = fp.exposure_matrices(bgs)["n_pol"]
    nclm, paid, nc = loss_arrays(bgs)
    print(f"  claims in those block groups, 2010-2023: {nc:,.0f}")
    rain = pd.read_parquet(D / "tx_basin_rain.parquet", columns=["site", "day", "p3"])
    rain = rain[rain.site.isin(g.index)].pivot(index="day", columns="site", values="p3")
    rain.index = pd.to_datetime(rain.index)
    rain = rain.reindex(DAYS)
    qp = q.rank(pct=True)
    rows = []
    for s in g.index:
        m = np.array([pos[bg.bg10.iloc[i]] for i in members[s]], dtype=int)
        if len(m) == 0:
            continue
        pol = ex[m].sum(0)
        if pol.mean() < 20:
            continue
        qa = qp[s].to_numpy()
        qpk = np.fmax(qa, np.r_[qa[1:], np.nan])  # peak percentile on d, d+1
        c = nclm[m].sum(0)
        pd_ = paid[m].sum(0)
        c2 = c + np.r_[c[1:], 0]
        p2 = pd_ + np.r_[pd_[1:], 0]
        r3 = np.fmax(rain[s].to_numpy(), np.r_[rain[s].to_numpy()[1:], np.nan])
        keep = ((qpk >= 0.9) | (r3 >= 25)) & ~np.isnan(qpk) & ~np.isnan(r3) & (pol > 0)
        di = np.flatnonzero(keep)
        rows.append(pd.DataFrame({"site": s, "day": DAYS[di], "qpct": qpk[di], "rain3": r3[di], "pol": pol[di],
                                  "claims": c2[di], "paid": p2[di]}))
    return pd.concat(rows, ignore_index=True)


def report(P, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    P = P.assign(y=(P.claims > 0).astype(int), lp=np.log1p(P.pol), lr=np.log1p(P.rain3), year=P.day.dt.year)
    print(f"\n=== {label}: {P.site.nunique()} gauges, {len(P):,} candidate gauge-days, any-claim rate {P.y.mean():.4f} ===")
    P["bin"] = pd.cut(P.qpct, [0.9, 0.95, 0.98, 0.99, 0.999, 1.0], include_lowest=True)
    t = P.groupby("bin", observed=True).agg(days=("y", "size"), any_claim=("y", "mean"), claims=("claims", "sum"),
                                              pol_days=("pol", "sum"), paid=("paid", "sum"))
    t["claims_per_1000_pol_days"] = 1000 * t.claims / t.pol_days
    t["paid_per_claim"] = t.paid / t.claims.replace(0, np.nan)
    print(t.round(3).to_string())
    P["rbin"] = pd.cut(P.rain3, [-1, 12.7, 25, 50, 100, 1e4])
    t2 = P.groupby("rbin", observed=True).agg(days=("y", "size"), any_claim=("y", "mean"), claims=("claims", "sum"),
                                                pol_days=("pol", "sum"))
    t2["claims_per_1000_pol_days"] = 1000 * t2.claims / t2.pol_days
    print("by basin rain:\n" + t2.round(3).to_string())
    tr, te = P[P.year <= 2016], P[P.year >= 2017]
    if te.y.sum() < 20 or tr.y.sum() < 20:
        print("too few claim days for the model test")
        return
    out = {}
    for name, cols in (("pol", ["lp"]), ("pol+rain", ["lp", "lr"]), ("pol+flow", ["lp", "qpct"]),
                       ("pol+rain+flow", ["lp", "lr", "qpct"]), ("flow only", ["qpct"])):
        sc = StandardScaler().fit(tr[cols])
        m = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(tr[cols]), tr.y)
        out[name] = m.decision_function(sc.transform(te[cols]))
    sites = te.site.to_numpy()
    us = np.unique(sites)
    gi = {u: np.flatnonzero(sites == u) for u in us}
    y = te.y.to_numpy()
    base = {k: auc(y, v) for k, v in out.items()}
    d_rf, d_f = [], []
    for _ in range(200):
        ix = np.concatenate([gi[u] for u in rng.choice(us, len(us))])
        a = {k: auc(y[ix], v[ix]) for k, v in out.items() if k in ("pol+rain", "pol+rain+flow", "pol")}
        d_rf.append(a["pol+rain+flow"] - a["pol+rain"])
        d_f.append(a["pol+rain"] - a["pol"])
    print(f"test 2017-2023: {len(te):,} gauge-days, {y.sum()} with a claim")
    print("AUC:", {k: round(v, 3) for k, v in base.items()})
    print(f"flow added to pol+rain: {base['pol+rain+flow'] - base['pol+rain']:+.4f} "
          f"[{np.percentile(d_rf, 2.5):+.4f}, {np.percentile(d_rf, 97.5):+.4f}] (gauge bootstrap)")
    # severity: among claim days
    c = P[P.claims > 0]
    if len(c) > 30:
        from scipy.stats import spearmanr

        for col, nm in (("paid", "paid building loss"), ("claims", "claim count")):
            r = spearmanr(c.qpct, c[col])
            r2 = spearmanr(c.rain3, c[col])
            print(f"claim days (n={len(c)}): Spearman flow pct vs {nm} {r.statistic:+.3f}; basin rain vs {nm} {r2.statistic:+.3f}")
        c = c.assign(ppc=c.paid / c.claims)
        print(f"  paid per claim vs flow pct {spearmanr(c.qpct, c.ppc).statistic:+.3f}; vs rain {spearmanr(c.rain3, c.ppc).statistic:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amax", type=float, default=250)
    ap.add_argument("--rb-min", type=float, default=0.03)
    a = ap.parse_args()
    print("Selection")
    g, q, fc = select(a.amax, a.rb_min)
    g.to_parquet(D / "tx_small_gauges.parquet")
    P = panel(g, q, fc)
    P.to_parquet(D / "tx_small_gauge_panel.parquet")
    report(P, f"small unregulated (<= {a.amax:.0f} mi2)")
    s50 = set(g.index[g.drainage_area <= 50])
    report(P[P.site.isin(s50)], "very small (<= 50 mi2)")


if __name__ == "__main__":
    main()
