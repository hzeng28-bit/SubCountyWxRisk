"""River network overlay and survivorship check: is the gauge/claim data good where the claims are, not where the rivers are? (stage 68)

Overlays NHDPlus flowlines (script 123) on claims and gauges, then measures
  T1  share of stream length within 10 / 25 km of a gauge, by stream order and by local policy density
  T2  claims vs stream length: share of claims vs share of stream length near a gauge
  T3  unclaimed floods: high-flow gauge-days (flow percentile >= 0.99) with no claim in the basin, by policies in the basin
  T4  gauge record by local claim density (record length, active over 2010-2023)
Inputs : gis/tx_flowlines.npz (123), gis/tx_counties_claims_gauges.geojson, gis/tx_gauges.geojson (122), usgs_tx/tx_small_gauge_panel.parquet
Outputs: docs/figures/gis/river_network_survivorship.png, audit/river_survivorship.txt
Run    : uv run python scripts/124_river_network_survivorship.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = wb2.cache_path(".")
GIS = DATA / "gis"
U = wb2.cache_path("labels_extra") / "flood" / "usgs_tx"
FIG = ROOT / "docs" / "figures" / "gis"
LAT0 = 31.0
KX, KY = 111.32 * np.cos(np.radians(LAT0)), 110.57


def proj(lon, lat):
    return np.c_[np.asarray(lon) * KX, np.asarray(lat) * KY]


def grid_sum(xy, w, cell=5.0, box=3, ext=None):
    x0, y0 = xy.min(0) - cell
    nx = int((xy[:, 0].max() - x0) / cell) + 2
    ny = int((xy[:, 1].max() - y0) / cell) + 2
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[nx, ny], range=[[x0, x0 + nx * cell], [y0, y0 + ny * cell]], weights=w)
    S = uniform_filter(H, size=box, mode="constant") * box * box
    return S, x0, y0, cell


def sample(S, x0, y0, cell, xy):
    i = np.clip(((xy[:, 0] - x0) / cell).astype(int), 0, S.shape[0] - 1)
    j = np.clip(((xy[:, 1] - y0) / cell).astype(int), 0, S.shape[1] - 1)
    return S[i, j]


def wq(v, w, qs):
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return np.interp(qs, c, v[o])


def main():
    fl = np.load(GIS / "tx_flowlines.npz")
    xy_ll, off, order, da = fl["xy"], fl["off"], fl["order"], fl["da"]
    line_id = np.repeat(np.arange(len(order)), np.diff(off))
    P = proj(xy_ll[:, 0], xy_ll[:, 1])
    seg = np.r_[np.hypot(*(P[1:] - P[:-1]).T), 0.0]
    last = np.r_[off[1:] - 1]
    seg[last] = 0
    prev = np.r_[0.0, seg[:-1]]
    first = off[:-1]
    prev[first] = 0
    wlen = (seg + prev) / 2  # km of river represented by each vertex
    keep = wlen > 0
    P, wlen, line_id, lon, lat = P[keep], wlen[keep], line_id[keep], xy_ll[keep, 0], xy_ll[keep, 1]
    ordv = order[line_id]
    dav = da[line_id]
    print(f"{len(order):,} lines, {len(P):,} vertices, {wlen.sum():,.0f} km of flowline (order >= 2)")

    bg = fp.bg_geometry()
    cl = fp.claims_tx()
    cl = cl[cl.date.between("2010-01-01", "2023-12-31")]
    bg = bg.join(cl.groupby("bg10").size().rename("claims"), on="bg10")
    bg["claims"] = bg.claims.fillna(0)
    ex = fp.exposure_matrices(bg.bg10.tolist())
    bg["pol_years"] = ex["n_pol"].sum(1) / 365.25
    B = proj(bg.lon, bg.lat)
    Sp, x0, y0, cell = grid_sum(B, bg.pol_years.to_numpy())
    Sc, *_ = grid_sum(B, bg.claims.to_numpy())
    pol15 = sample(Sp, x0, y0, cell, P)  # policy-years within ~15 km box
    gq = pd.DataFrame(json.loads((GIS / "tx_gauges.geojson").read_text())["features"])
    gq = pd.DataFrame([{**f["properties"], "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
                       for f in json.loads((GIS / "tx_gauges.geojson").read_text())["features"]])
    G = proj(gq.lon, gq.lat)
    small = gq["class"].str.startswith("small").to_numpy()
    t_all, t_small = cKDTree(G), cKDTree(G[small])
    d_all = t_all.query(P)[0]
    d_small = t_small.query(P)[0]

    L = []
    p = L.append
    p(f"=== T1. Share of stream length near a gauge ({wlen.sum():,.0f} km, NHDPlus order >= 2) ===")
    grp = np.select([ordv == 2, ordv == 3, ordv == 4, ordv >= 5], ["order 2", "order 3", "order 4", "order 5+"], default="other")
    rows = []
    for g in ["order 2", "order 3", "order 4", "order 5+", "all"]:
        m = np.ones(len(P), bool) if g == "all" else grp == g
        w = wlen[m]
        rows.append((g, round(w.sum()), *[round(float((w * (d[m] <= r)).sum() / w.sum()), 3)
                                         for d in (d_all, d_small) for r in (10, 25)]))
    p(pd.DataFrame(rows, columns=["stream class", "km", "all gauges <=10km", "all <=25km", "small gauges <=10km", "small <=25km"]).to_string(index=False))
    o = np.argsort(pol15, kind="stable")  # ties (many reaches with 0 policies) are split by length, not dropped
    cf = np.empty(len(o))
    cf[o] = (np.cumsum(wlen[o]) - wlen[o] / 2) / wlen.sum()
    qi = np.minimum((cf * 5).astype(int), 4)
    names = ["Q1 lowest exposure", "Q2", "Q3", "Q4", "Q5 highest exposure"]
    rows = []
    for k in range(5):
        m = qi == k
        w = wlen[m]
        rows.append((names[k], round(w.sum()), round(float(np.median(pol15[m]))),
                     round(float((w * (d_all[m] <= 10)).sum() / w.sum()), 3), round(float((w * (d_small[m] <= 10)).sum() / w.sum()), 3),
                     round(float(sample(Sc, x0, y0, cell, P[m]).sum() * 0 + np.average(sample(Sc, x0, y0, cell, P[m]), weights=w)), 1)))
    p("\n  by exposure of the river reach (policy-years within ~15 km), quintiles of stream length:")
    rowsQ = rows
    p(pd.DataFrame(rows, columns=["quintile", "km", "median pol-yr nearby", "share <=10km of any gauge", "share <=10km of small gauge", "mean claims nearby"]).to_string(index=False))

    p("\n=== T2. Claims vs stream length near a gauge ===")
    Bq = cKDTree(P)
    dstr = Bq.query(B)[0]
    dga = t_all.query(B)[0]
    dgs = t_small.query(B)[0]
    c = bg.claims.to_numpy()
    rows = [("share of stream length", float((wlen * (d_all <= 10)).sum() / wlen.sum()), float((wlen * (d_small <= 10)).sum() / wlen.sum())),
            ("share of claims", float(c[dga <= 10].sum() / c.sum()), float(c[dgs <= 10].sum() / c.sum())),
            ("share of policy-years", float(bg.pol_years[dga <= 10].sum() / bg.pol_years.sum()), float(bg.pol_years[dgs <= 10].sum() / bg.pol_years.sum()))]
    p(pd.DataFrame(rows, columns=["within 10 km of", "any gauge", "small gauge"]).round(3).to_string(index=False))
    p(f"  claims within 2 km / 5 km of a mapped stream: {float(c[dstr <= 2].sum()/c.sum()):.3f} / {float(c[dstr <= 5].sum()/c.sum()):.3f}; "
      f"policy-years: {float(bg.pol_years[dstr<=2].sum()/bg.pol_years.sum()):.3f} / {float(bg.pol_years[dstr<=5].sum()/bg.pol_years.sum()):.3f}")

    p("\n=== T3. Unclaimed floods: gauge-days with flow percentile >= 0.99, by policies in the basin ===")
    pn = pd.read_parquet(U / "tx_small_gauge_panel.parquet")
    f = pn[pn.qpct >= 0.99].copy()
    pb = f.groupby("site").pol.median()
    f["polbin"] = pd.cut(f.pol, [-1, 50, 200, 1000, 1e9], labels=["<50", "50-200", "200-1000", ">=1000"])
    r = f.groupby("polbin", observed=True).agg(gauge_days=("site", "size"), any_claim=("claims", lambda s: float((s > 0).mean())),
                                               mean_claims=("claims", "mean"))
    p(r.round(3).to_string())
    p(f"  flood-flow gauge-days with no claim: {float((f.claims == 0).mean()):.3f}; in basins with < 200 policies: "
      f"{float((f.pol < 200).mean()):.3f} of flood-flow gauge-days (these floods never appear in the loss record)")
    p(f"  gauges with median policies < 200 in basin: {(pb < 200).mean():.3f} of {len(pb)}")

    p("\n=== T4. Gauge record by local claim density (claims within ~15 km) ===")
    gc = sample(Sc, x0, y0, cell, G)
    gq["claims15"] = gc
    meta = pd.read_parquet(U / "tx_gauges_q.parquet").reset_index()
    meta["id"] = meta["id"] if "id" in meta.columns else meta["index"]
    meta["begin"] = pd.to_datetime(meta.begin_utc, utc=True).dt.tz_localize(None)
    meta["end"] = pd.to_datetime(meta.end_utc, utc=True).dt.tz_localize(None)
    gq = gq.merge(meta[["id", "begin", "end"]], on="id", how="left")
    gq["rec_y"] = (gq.end.clip(upper=pd.Timestamp("2023-12-31")) - gq.begin.clip(lower=pd.Timestamp("2010-01-01"))).dt.days / 365.25
    gq["full"] = (gq.begin <= "2010-12-31") & (gq.end >= "2023-01-01")
    gq["claim_q"] = pd.qcut(gq.claims15.rank(method="first"), 4, labels=["Q1 few claims", "Q2", "Q3", "Q4 many claims"])
    rr = gq.groupby("claim_q", observed=True).agg(gauges=("id", "size"), median_claims15=("claims15", "median"),
                                                   record_years=("rec_y", "median"), share_full_record=("full", "mean"))
    p(rr.round(2).to_string())
    p(f"  Spearman(record years in 2010-2023, claims nearby) = {float(gq.rec_y.rank().corr(gq.claims15.rank())):.3f}")
    txt = "\n".join(L)
    print(txt)
    (DATA / "audit").mkdir(exist_ok=True)
    (DATA / "audit" / "river_survivorship.txt").write_text(txt)
    near_line = np.zeros(len(order), bool)
    np.logical_or.at(near_line, line_id, d_small <= 10)
    plot(fl, near_line, gq, bg, rows, rowsQ, r)


def plot(fl, near_line, gq, bg, rows2, rowsQ, t3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon as MPoly

    xy, off, order = fl["xy"], fl["off"], fl["order"]
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1, 1], wspace=0.3, hspace=0.38)
    ax = fig.add_subplot(gs[:, 0])
    polys, vals = [], []
    for ft in json.loads((GIS / "tx_counties_claims_gauges.geojson").read_text())["features"]:
        gm = ft["geometry"]
        parts = gm["coordinates"] if gm["type"] == "MultiPolygon" else [gm["coordinates"]]
        for part in parts:
            polys.append(MPoly(np.asarray(part[0]), closed=True))
            vals.append(ft["properties"]["claims_per_1000km2"] or 0)
    ax.add_collection(PatchCollection(polys, array=np.clip(vals, 0.5, None), cmap="Greys", norm=LogNorm(0.5, 4000), alpha=0.6,
                                      edgecolor="#cccccc", linewidth=0.2))
    segs = [xy[off[k]:off[k + 1]] for k in range(len(order))]
    lw = 0.15 + 0.22 * np.clip(order - 2, 0, 6)
    far = np.where(~near_line)[0]
    nr = np.where(near_line)[0]
    ax.add_collection(LineCollection([segs[k] for k in far], linewidths=lw[far], colors="#e08a1e", alpha=0.75, rasterized=True))
    ax.add_collection(LineCollection([segs[k] for k in nr], linewidths=lw[nr] + 0.7, colors="#1565c0", alpha=1.0, rasterized=True, zorder=3))
    sm = gq["class"].str.startswith("small")
    ax.scatter(gq.lon[sm], gq.lat[sm], s=16, c="#d62728", edgecolor="white", linewidth=0.4, zorder=5, label=f"small gauge (n={sm.sum()})")
    top = bg[bg.claims >= 20]
    ax.scatter(top.lon, top.lat, s=np.sqrt(top.claims) * 3, facecolor="none", edgecolor="#6a1b9a", linewidth=0.6, alpha=0.9, zorder=4,
               label="block group >= 20 claims")
    ax.plot([], [], c="#1565c0", lw=2.2, label="reach within 10 km of a small gauge")
    ax.plot([], [], c="#e08a1e", lw=1.5, label="reach farther than 10 km (ungauged)")
    ax.set_xlim(-106.8, -93.3)
    ax.set_ylim(25.6, 36.8)
    ax.set_aspect(1 / np.cos(np.radians(31)))
    ax.set_title("NHDPlus rivers (order >= 2) vs small gauges and claims (grey fill: claims per km2)", fontsize=10)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

    a2 = fig.add_subplot(gs[0, 1])
    lab = ["river length", "claims", "policy-years"]
    a2.bar(np.arange(3) - 0.2, [r[1] for r in rows2], 0.4, label="any gauge", color="#64b5f6")
    a2.bar(np.arange(3) + 0.2, [r[2] for r in rows2], 0.4, label="small gauge", color="#0d47a1")
    a2.set_xticks(range(3), lab)
    a2.set_ylabel("share within 10 km of a gauge")
    a2.legend(fontsize=8)
    a2.set_title("the loss record sits near gauges; the rivers do not", fontsize=9)
    a3 = fig.add_subplot(gs[0, 2])
    x = np.arange(5)
    a3.bar(x - 0.2, [r[3] for r in rowsQ], 0.4, label="any gauge", color="#64b5f6")
    a3.bar(x + 0.2, [r[4] for r in rowsQ], 0.4, label="small gauge", color="#0d47a1")
    a3.set_xticks(x, ["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    a3.set_xlabel("exposure of the river reach (policy-years nearby)")
    a3.set_ylabel("share of river length within 10 km")
    a3.set_title("gauged river length by exposure quintile", fontsize=9)
    a4 = fig.add_subplot(gs[1, 1:])
    xb = np.arange(len(t3))
    a4.bar(xb, t3.any_claim, color="#8e24aa")
    for i, (v, n) in enumerate(zip(t3.any_claim, t3.gauge_days, strict=True)):
        a4.text(i, v + 0.008, f"{v:.0%}\nn={n}", ha="center", fontsize=8)
    a4.set_xticks(xb, [f"{i} policies" for i in t3.index])
    a4.set_ylabel("share with any claim in the basin")
    a4.set_ylim(0, 0.38)
    a4.set_xlabel("policies in the gauge basin")
    a4.set_title("flood-scale flow days (flow percentile >= 0.99): most never reach the loss record", fontsize=9)
    fig.savefig(FIG / "river_network_survivorship.png", dpi=150, bbox_inches="tight")
    print("figure ->", FIG / "river_network_survivorship.png")


if __name__ == "__main__":
    main()
