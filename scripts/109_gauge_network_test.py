"""Flow-network conservation and rain -> discharge learnability at Texas streamgages (stage 59).

Steps (each prints a table; tee to $WXRISK_DATA/audit/gauge_network_test.txt):
  net    immediate-upstream gauge pairs (gauge inside another gauge's NLDI basin), drainage-area coverage, best travel lag, mass balance
         r = Q_down(t) - sum Q_up(t - lag). Loss share on flood days (Q_down well below the upstream sum while the
         gauged upstream area covers most of the basin) is the candidate overbank / diversion / storage signal.
  rain   basin-mean gridMET rain (today, 2 d, 3 d, 7 d, max cell) per gauge basin -> usgs_tx/tx_basin_rain.parquet
  learn  does basin rain predict a discharge surge (Q_t >= 3 Q_{t-1} and >= the gauge's own 95th percentile)?
         Train <= 2016, test 2017-2023; AUC with gauge-clustered bootstrap, with / without Q_{t-1} and area.
  claims county-episode flood claims vs the county's gauge peak percentile and vs county rain (pilot table).

Inputs : usgs_tx/tx_gauges_q, tx_daily_q, tx_basins.geojson (script 108), gridMET (flood_pilot.load_precip)
Run    : python scripts/109_gauge_network_test.py net|rain|learn|claims
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

D = wb2.cache_path("labels_extra") / "flood" / "usgs_tx"
rng = np.random.default_rng(0)


def load():
    g = pd.read_parquet(D / "tx_gauges_q.parquet").set_index("id")
    q = pd.read_parquet(D / "tx_daily_q.parquet")
    q = q.pivot_table(index="time", columns="monitoring_location_id", values="value")
    q = q.reindex(pd.date_range("2010-01-01", "2023-12-31"))
    return g[g.index.isin(q.columns)], q


def upstream_sets(g):
    """u is upstream of s when gauge u lies inside the NLDI basin of s (basin polygons come from script 108)."""
    import shapely
    from shapely.geometry import shape

    fc = json.loads((D / "tx_basins.geojson").read_text())["features"]
    pts = shapely.points(g.lon.to_numpy(), g.lat.to_numpy())
    ups = {}
    for f in fc:
        s = f["properties"]["site"]
        if s not in g.index:
            continue
        # a gauge sits on the basin outlet line, so allow a small tolerance for the boundary
        m = shapely.contains(shape(f["geometry"]).buffer(0.002), pts)
        ups[s] = {u for u in g.index[m] if u != s}
    return ups


def immediate_pairs(g):
    ups = upstream_sets(g)
    pairs = []
    for s, U in ups.items():
        # u is immediate if no other upstream gauge of s lies downstream of u
        imm = [u for u in U if not any(u in ups.get(v, ()) for v in U if v != u)]
        pairs.append((s, imm))
    return pairs


def net():
    g, q = load()
    rows = []
    for s, imm in immediate_pairs(g):
        da = g.drainage_area
        if not imm or not (da[s] > 0):
            continue
        cov = da[imm].sum() / da[s]
        both = q[[s, *imm]].dropna(how="any")
        if len(both) < 365 or not 0.6 <= cov <= 1.05:
            continue
        best = None
        for lag in (0, 1, 2, 3):
            su = q[imm].shift(lag).sum(axis=1, min_count=len(imm))
            ok = pd.concat([q[s], su], axis=1).dropna()
            if len(ok) < 365:
                continue
            c = np.corrcoef(np.log1p(ok.iloc[:, 0]), np.log1p(ok.iloc[:, 1]))[0, 1]
            if best is None or c > best[1]:
                best = (lag, c, su)
        if best is None:
            continue
        lag, c, su = best
        d = pd.concat([q[s].rename("dn"), su.rename("up")], axis=1).dropna()
        flood = d[d.up >= d.up.quantile(0.99)]
        ratio = (flood.dn / flood.up.where(flood.up > 0)).dropna()
        rows.append(dict(site=s, n_up=len(imm), cov=cov, lag=lag, r=c, n=len(d), n_flood=len(flood),
                         ratio_med_all=(d.dn / d.up.where(d.up > 0)).median(), ratio_med_flood=ratio.median(),
                         loss_share=(ratio < 0.7).mean(), gain_share=(ratio > 1.3).mean()))
    t = pd.DataFrame(rows)
    t.to_parquet(D / "tx_mass_balance.parquet")
    print(f"pairs with coverage 0.6-1.05 and >= 1 yr of joint data: {len(t)} (single-upstream: {(t.n_up == 1).sum()})")
    print(t[["cov", "lag", "r", "ratio_med_all", "ratio_med_flood", "loss_share", "gain_share"]].describe().round(3))
    print("lag counts:", t.lag.value_counts().sort_index().to_dict())
    print(t.sort_values("loss_share", ascending=False).head(10).round(3).to_string())


def basin_rain():
    from shapely.geometry import shape
    import shapely

    days, lat, lon, P = fp.load_precip()
    days = pd.DatetimeIndex(days)
    fc = json.loads((D / "tx_basins.geojson").read_text())["features"]
    LON, LAT = np.meshgrid(lon, lat)
    out = {}
    for k, f in enumerate(fc):
        poly = shape(f["geometry"])
        x0, y0, x1, y1 = poly.bounds
        ii = np.where((lat >= y0 - 0.05) & (lat <= y1 + 0.05))[0]
        jj = np.where((lon >= x0 - 0.05) & (lon <= x1 + 0.05))[0]
        if not len(ii) or not len(jj):
            continue
        sl = (slice(ii[0], ii[-1] + 1), slice(jj[0], jj[-1] + 1))
        m = shapely.contains_xy(poly, LON[sl], LAT[sl])
        if not m.any():  # basin smaller than a grid cell: nearest cell
            ci = np.abs(lat - poly.centroid.y).argmin()
            cj = np.abs(lon - poly.centroid.x).argmin()
            x = np.asarray(P[:, ci, cj], dtype=np.float32)[:, None]
        else:
            x = np.asarray(P[:, sl[0], sl[1]], dtype=np.float32)[:, m]
        s = f["properties"]["site"]
        mean = np.nanmean(x, axis=1)
        out[s] = pd.DataFrame({"p1": mean, "pmax": np.nanmax(x, axis=1)}, index=days)
        if k % 100 == 0:
            print(f"  basin {k}/{len(fc)}", flush=True)
    r = pd.concat(out, names=["site", "day"]).reset_index()
    r["p2"] = r.groupby("site").p1.transform(lambda v: v.rolling(2, min_periods=1).sum())
    r["p3"] = r.groupby("site").p1.transform(lambda v: v.rolling(3, min_periods=1).sum())
    r["p7"] = r.groupby("site").p1.transform(lambda v: v.rolling(7, min_periods=1).sum())
    r["p30"] = r.groupby("site").p1.transform(lambda v: v.rolling(30, min_periods=1).sum())
    r.to_parquet(D / "tx_basin_rain.parquet")
    print(f"{len(r):,} basin-days at {r.site.nunique()} basins")


def auc(y, s):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(y, s)


def learn():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB

    g, q = load()
    r = pd.read_parquet(D / "tx_basin_rain.parquet")
    ql = q.stack().rename("q").reset_index()
    ql.columns = ["day", "site", "q"]
    ql = ql.sort_values(["site", "day"])
    ql["q1"] = ql.groupby("site").q.shift(1)
    ql["p95"] = ql.groupby("site").q.transform(lambda v: v[v.index.isin(v.index)].quantile(0.95))
    ql["surge"] = ((ql.q >= 3 * ql.q1) & (ql.q >= ql.p95) & (ql.q1 >= 0)).astype(int)
    d = ql.merge(r, on=["site", "day"], how="inner").dropna(subset=["q1", "p1"])
    d = d.join(np.log(g.drainage_area).rename("larea"), on="site")
    d["lq1"] = np.log1p(d.q1)
    d["rel_q1"] = d.lq1 - np.log1p(d.p95)
    d["doy"] = d.day.dt.dayofyear
    tr, te = d[d.day.dt.year <= 2016], d[d.day.dt.year >= 2017]
    print(f"gauge-days train {len(tr):,} test {len(te):,}; surge rate train {tr.surge.mean():.4f} test {te.surge.mean():.4f}")
    sets = {"rain only (p1,p2,p3,pmax)": ["p1", "p2", "p3", "pmax"],
            "+ antecedent rain (p7,p30)": ["p1", "p2", "p3", "pmax", "p7", "p30"],
            "+ area": ["p1", "p2", "p3", "pmax", "p7", "p30", "larea"],
            "+ area + Q(t-1) relative": ["p1", "p2", "p3", "pmax", "p7", "p30", "larea", "rel_q1"],
            "Q(t-1) relative only": ["rel_q1", "larea"],
            "p1 alone (score)": None}
    sites = te.site.unique()
    for name, cols in sets.items():
        if cols is None:
            s = te.p1.to_numpy()
        else:
            m = HGB(max_iter=200, learning_rate=0.06, max_depth=5, random_state=0).fit(tr[cols], tr.surge)
            s = m.predict_proba(te[cols])[:, 1]
        te = te.assign(_s=s)
        a = auc(te.surge, s)
        # arrays per gauge once; a bootstrap draw concatenates them instead of re-filtering 1.2 M rows
        by = {k: (x.surge.to_numpy(), x._s.to_numpy()) for k, x in te.groupby("site")}
        keys = list(by)
        bs = []
        for _ in range(100):
            pick = rng.choice(len(keys), len(keys))
            yy = np.concatenate([by[keys[i]][0] for i in pick])
            ss = np.concatenate([by[keys[i]][1] for i in pick])
            if yy.min() != yy.max():
                bs.append(auc(yy, ss))
        print(f"{name:34s} AUC {a:.3f} [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]")


def claims():
    g, q = load()
    r = pd.read_parquet(D / "tx_basin_rain.parquet")
    t = pd.read_parquet(wb2.cache_path("labels_extra") / "flood" / "pilot" / "flood_pilot_table.parquet",
                        columns=["ep", "cty", "start", "end", "y", "n_claims", "year"])
    ec = t.groupby(["ep", "cty"]).agg(start=("start", "first"), end=("end", "first"), claims=("n_claims", "sum"),
                                      year=("year", "first")).reset_index()
    ec["start"], ec["end"] = pd.to_datetime(ec.start), pd.to_datetime(ec.end)
    g = g.assign(cty=g.county_code.astype(str).str.zfill(3))
    print(ec.head(3).to_string(), "\ncounties in pilot sample:", ec.cty.nunique(), "gauge county code example:",
          g.cty.iloc[0])
    ec["cty3"] = ec.cty.astype(str).str[-3:]
    qp = q.rank(pct=True)
    out = []
    for cty, gg in g.groupby("cty"):
        ids = [i for i in gg.index if i in q.columns]
        if not ids:
            continue
        e = ec[ec.cty3 == cty]
        for _, row in e.iterrows():
            w = slice(row.start - pd.Timedelta(days=1), row.end + pd.Timedelta(days=2))
            pk = qp.loc[w, ids].max().max()
            out.append((row.ep, row.cty, row.claims, pk, q.loc[w, ids].max().max()))
    o = pd.DataFrame(out, columns=["ep", "cty", "claims", "qpct", "qmax"]).dropna()
    o["y"] = (o.claims > 0).astype(int)
    print(f"county-episodes with a gauge: {len(o):,}; with claims {o.y.mean():.3f}")
    print("AUC of gauge peak percentile for 'any claim':", round(auc(o.y, o.qpct), 3))
    print("Spearman peak percentile vs claim count:", round(o[["qpct", "claims"]].corr("spearman").iloc[0, 1], 3))


if __name__ == "__main__":
    {"net": net, "rain": basin_rain, "learn": learn, "claims": claims}[sys.argv[1]]()
