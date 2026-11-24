"""GIS map: NFIP claims, policies and USGS gauges on one Texas map, and how gauge density relates to the claim area (stage 67).

Layers (also written as GeoJSON for QGIS / the GEE viewer): counties (dissolved 2010 block groups) with claims, policies,
claims per 1000 policy-years, gauge counts; block-group centroids with claims; gauge points with class; small-gauge basins.
Tests (county level, 254 counties): does gauge density follow claims or just policies / population? Spearman, partial
(controlling log policy density), and distance claim -> nearest gauge vs the exposure-weighted block groups.
Inputs : pilot/bg2010_tx.zip, clean claims / policies, usgs_tx/{tx_gauges_q,tx_small_gauges,tx_basins.geojson}
Outputs: $WXRISK_DATA/gis/*.geojson, docs/figures/gis/claims_gauges_map.png, audit/gis_claims_gauges.txt
Run    : uv run python scripts/122_gis_claims_gauges_map.py
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = wb2.DATA if hasattr(wb2, "DATA") else Path.home() / "wxrisk_data"
U = fp.PILOT.parent / "usgs_tx"
OUT = DATA / "gis"
FIG = ROOT / "docs" / "figures" / "gis"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def bg_polygons():
    import shapefile
    from shapely.geometry import shape

    z = zipfile.ZipFile(fp.PILOT / "bg2010_tx.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                          dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    out = {}
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        out[rec["STATE"] + rec["COUNTY"] + rec["TRACT"] + rec["BLKGRP"]] = shape(sr.shape.__geo_interface__)
    return out


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def partial_spearman(a, b, c):
    ra, rb, rc = (pd.Series(x).rank().to_numpy() for x in (a, b, c))
    res = []
    for y in (ra, rb):
        A = np.c_[np.ones_like(rc), rc]
        res.append(y - A @ np.linalg.lstsq(A, y, rcond=None)[0])
    return float(np.corrcoef(res[0], res[1])[0, 1])


def boot(fn, n_units, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n_units, n_units)) for _ in range(reps)]
    return np.nanpercentile(v, [2.5, 97.5])


def main():
    from shapely.geometry import mapping, Point
    from shapely.ops import unary_union

    polys = bg_polygons()
    bg = fp.bg_geometry()
    print(f"block groups {len(bg):,}")
    cl = fp.claims_tx()
    cl = cl[cl.date.between("2010-01-01", "2023-12-31")]
    bgc = cl.groupby("bg10").size().rename("claims")
    ex = fp.exposure_matrices(bg.bg10.tolist())
    bg["pol_years"] = ex["n_pol"].sum(1) / 365.25
    bg = bg.join(bgc, on="bg10")
    bg["claims"] = bg.claims.fillna(0)
    print(f"claims 2010-2023 {int(bg.claims.sum()):,}; policy-years {bg.pol_years.sum():,.0f}")

    gq = pd.read_parquet(U / "tx_gauges_q.parquet")
    sm = pd.read_parquet(U / "tx_small_gauges.parquet")
    gq = gq.reset_index() if "id" not in gq.columns else gq
    sm = sm.reset_index() if "id" not in sm.columns else sm
    gq["begin"] = pd.to_datetime(gq.begin_utc, utc=True).dt.tz_localize(None)
    gq["end"] = pd.to_datetime(gq.end_utc, utc=True).dt.tz_localize(None)
    gq = gq[(gq.begin <= "2023-12-31") & (gq.end >= "2010-01-01")].copy()
    gq["cls"] = np.where(gq.id.isin(sm.id), "small screened",
                         np.where(gq.drainage_area <= 250, "small unscreened",
                                  np.where(gq.drainage_area > 250, "large river", "unknown area")))
    print(gq.cls.value_counts().to_string())

    # --- county layer
    county_geom = {c: unary_union([polys[b] for b in g.bg10]) for c, g in bg.groupby("cty")}
    cty = bg.groupby("cty").agg(claims=("claims", "sum"), pol_years=("pol_years", "sum"), area_km2=("area_km2", "sum"),
                               lon=("lon", "mean"), lat=("lat", "mean"))
    for name, mask in [("gauges_all", gq.cls != ""), ("gauges_small", gq.cls.str.startswith("small")),
                       ("gauges_screened", gq.cls == "small screened")]:
        pts = gq[mask]
        cnt = {c: sum(county_geom[c].contains(Point(x, y)) for x, y in zip(pts.lon, pts.lat, strict=True))
               for c in cty.index}
        cty[name] = pd.Series(cnt)
    cty["claims_per_1000_pol_yr"] = 1000 * cty.claims / cty.pol_years.clip(lower=1)
    cty["claims_per_1000km2"] = 1000 * cty.claims / cty.area_km2
    cty["pol_per_km2"] = cty.pol_years / cty.area_km2
    for k in ("gauges_all", "gauges_small", "gauges_screened"):
        cty[k.replace("gauges", "gdens")] = 1000 * cty[k] / cty.area_km2  # gauges per 1000 km2

    # --- GeoJSON for QGIS
    feats = [{"type": "Feature", "geometry": mapping(county_geom[c].simplify(0.005)),
              "properties": {"county": c, **{k: (None if pd.isna(v) else round(float(v), 4)) for k, v in r.items()}}}
             for c, r in cty.iterrows()]
    (OUT / "tx_counties_claims_gauges.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    gfe = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
            "properties": {"id": r.id, "name": r.monitoring_location_name, "area_mi2": r.drainage_area, "class": r.cls}}
           for r in gq.itertuples()]
    (OUT / "tx_gauges.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": gfe}))
    b2 = bg[bg.claims > 0]
    bfe = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r.lon, r.lat]},
            "properties": {"bg10": r.bg10, "claims": int(r.claims), "pol_years": round(r.pol_years, 1)}}
           for r in b2.itertuples()]
    (OUT / "tx_bg_claims.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": bfe}))

    # --- tests, county level
    L = []
    P = L.append
    P("=== County level (254 counties): does gauge density follow claims or exposure? ===")
    pcty = np.log10(cty.pol_per_km2.clip(lower=1e-4))
    keep = cty.pol_years > 0
    rows = []
    for gk in ("gdens_all", "gdens_small", "gdens_screened"):
        x = cty.loc[keep, gk].to_numpy()
        for yk in ("claims_per_1000km2", "claims_per_1000_pol_yr"):
            y = cty.loc[keep, yk].to_numpy()
            pc = pcty[keep].to_numpy()
            n = len(x)
            r = spearman(x, y)
            ci = boot(lambda i: spearman(x[i], y[i]), n)
            rp = partial_spearman(x, y, pc)
            cip = boot(lambda i: partial_spearman(x[i], y[i], pc[i]), n)
            rows.append((gk, yk, round(r, 3), f"[{ci[0]:+.2f},{ci[1]:+.2f}]", round(rp, 3), f"[{cip[0]:+.2f},{cip[1]:+.2f}]"))
    P(pd.DataFrame(rows, columns=["gauge density", "claim measure", "Spearman", "95% CI", "partial | policy density", "95% CI"]).to_string(index=False))
    P(f"  policy density vs gauge density (all): Spearman {spearman(cty.loc[keep,'gdens_all'], cty.loc[keep,'pol_per_km2']):.3f}")
    P(f"  counties with >=1 claim: {(cty.claims>0).sum()}; with a gauge (all): {(cty.gauges_all>0).sum()}; with a screened small gauge: {(cty.gauges_screened>0).sum()}")
    t = cty[keep].copy()
    t["q"] = pd.qcut(t.claims_per_1000km2.rank(method="first"), 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high claims"])
    P("\n  by quintile of claims per 1000 km2:")
    P(t.groupby("q", observed=True).agg(counties=("claims", "size"), claims=("claims", "sum"), gauges_all=("gauges_all", "sum"),
                                         gauges_small=("gauges_small", "sum"), screened=("gauges_screened", "sum"),
                                         pol_per_km2=("pol_per_km2", "median"), gdens_all=("gdens_all", "median")).round(3).to_string())

    # --- distance claim -> nearest gauge vs policy-weighted block groups
    P("\n=== Distance to the nearest gauge (km): claims vs exposure-weighted block groups ===")
    out = []
    for name, mask in [("all gauges", gq.cls != ""), ("small gauges", gq.cls.str.startswith("small")),
                       ("screened small", gq.cls == "small screened")]:
        gg = gq[mask]
        d = np.array([km(la, lo, gg.lat.to_numpy(), gg.lon.to_numpy()).min() for la, lo in zip(bg.lat, bg.lon, strict=True)])
        bg["d_" + name] = d
        wc = np.average(d, weights=bg.claims.clip(lower=0) + 1e-9)
        wp = np.average(d, weights=bg.pol_years + 1e-9)
        cs = np.sort(d)
        qs = lambda w: [np.interp(0.5, np.cumsum(w[np.argsort(d)]) / w.sum(), cs)]  # noqa: E731
        out.append((name, round(wc, 1), round(wp, 1), round(qs(bg.claims.to_numpy() + 1e-9)[0], 1),
                    round(qs(bg.pol_years.to_numpy() + 1e-9)[0], 1),
                    round(float(bg.claims[d <= 25].sum() / bg.claims.sum()), 3), round(float(bg.pol_years[d <= 25].sum() / bg.pol_years.sum()), 3)))
    P(pd.DataFrame(out, columns=["gauge set", "mean d claims", "mean d policies", "median d claims", "median d policies",
                                 "share claims <=25 km", "share policies <=25 km"]).to_string(index=False))
    P("(claims sit about as far from gauges as the policies do if the shares and distances are equal: gauges follow where people insure, not where they flood)")

    # --- claim share by distance band, rate per policy-year
    bands = [0, 10, 25, 50, 100, 1e9]
    dd = bg["d_all gauges"]
    b = pd.cut(dd, bands, labels=["<10", "10-25", "25-50", "50-100", ">100"], right=False)
    r = bg.groupby(b, observed=True).agg(bgs=("bg10", "size"), claims=("claims", "sum"), pol_years=("pol_years", "sum"))
    r["claims_per_1000_pol_yr"] = (1000 * r.claims / r.pol_years).round(2)
    P("\n  claim rate by distance band to the nearest gauge (all gauges):")
    P(r.round(0).to_string())
    (DATA / "audit").mkdir(exist_ok=True)
    txt = "\n".join(L)
    print(txt)
    (DATA / "audit" / "gis_claims_gauges.txt").write_text(txt)

    plot(bg, cty, county_geom, gq)


def plot(bg, cty, county_geom, gq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon as MPoly
    from matplotlib.collections import PatchCollection

    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1, 1], height_ratios=[1, 1], wspace=0.28, hspace=0.33)
    ax = fig.add_subplot(gs[:, 0])

    def patches(g):
        geoms = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        return [MPoly(np.asarray(p.exterior.coords), closed=True) for p in geoms]

    cmap = plt.get_cmap("YlOrRd")
    vals = (cty.claims_per_1000km2.clip(lower=0.5)).to_dict()
    pc = []
    cols = []
    for c, g in county_geom.items():
        for p in patches(g.simplify(0.004)):
            pc.append(p)
            cols.append(vals[c])
    coll = PatchCollection(pc, array=np.array(cols), cmap=cmap, norm=LogNorm(0.5, max(cols)), edgecolor="#9a9a9a", linewidth=0.25)
    ax.add_collection(coll)
    top = bg[bg.claims >= 20].sort_values("claims")
    ax.scatter(top.lon, top.lat, s=np.sqrt(top.claims) * 3, facecolor="none", edgecolor="#3b0a6b", linewidth=0.6, alpha=0.8,
               label="block group, >= 20 claims")
    sty = {"large river": ("#9aa7b4", 9, "^"), "small unscreened": ("#4aa3df", 14, "o"), "small screened": ("#0b3d91", 26, "o")}
    for k, (c, s, m) in sty.items():
        q = gq[gq.cls == k]
        ax.scatter(q.lon, q.lat, s=s, c=c, marker=m, edgecolor="white", linewidth=0.3, label=f"{k} gauge (n={len(q)})", zorder=4)
    ax.set_xlim(-106.8, -93.3)
    ax.set_ylim(25.6, 36.8)
    ax.set_aspect(1 / np.cos(np.radians(31)))
    ax.set_title("NFIP inland claims 2010-2023 (county fill: claims per 1000 km2) and USGS discharge gauges", fontsize=11)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    cb = fig.colorbar(coll, ax=ax, fraction=0.03, pad=0.01)
    cb.set_label("claims per 1000 km2 (log)")
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    keep = cty.pol_years > 0
    t = cty[keep]
    a1 = fig.add_subplot(gs[0, 1])
    a1.scatter(t.claims_per_1000km2.clip(lower=0.05), t.gdens_all.clip(lower=0.05), s=12, c="#0b3d91", alpha=0.6)
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_xlabel("claims per 1000 km2")
    a1.set_ylabel("gauges per 1000 km2")
    a1.set_title(f"county: gauge density vs claim density\nSpearman {spearman(t.gdens_all, t.claims_per_1000km2):.2f}", fontsize=9)
    a2 = fig.add_subplot(gs[0, 2])
    a2.scatter(t.pol_per_km2.clip(lower=1e-3), t.gdens_all.clip(lower=0.05), s=12, c="#c0392b", alpha=0.6)
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("policy-years per km2")
    a2.set_ylabel("gauges per 1000 km2")
    a2.set_title(f"county: gauge density vs policy density\nSpearman {spearman(t.gdens_all, t.pol_per_km2):.2f}", fontsize=9)
    a3 = fig.add_subplot(gs[1, 1:])
    bands = [0, 10, 25, 50, 100, 1e9]
    lab = ["<10", "10-25", "25-50", "50-100", ">100"]
    dd = bg["d_all gauges"]
    b = pd.cut(dd, bands, labels=lab, right=False)
    r = bg.groupby(b, observed=True).agg(claims=("claims", "sum"), pol=("pol_years", "sum"))
    x = np.arange(len(r))
    a3.bar(x - 0.2, r.claims / r.claims.sum(), 0.4, color="#e67e22", label="share of claims")
    a3.bar(x + 0.2, r.pol / r.pol.sum(), 0.4, color="#7f8c8d", label="share of policy-years")
    a3.set_xticks(x, [f"{v} km" for v in r.index])
    a3.set_xlabel("distance from block-group centroid to the nearest gauge (all gauges active 2010-2023)")
    a3.set_ylabel("share")
    a3.legend(fontsize=8)
    a3.set_title("where claims and policies sit relative to the gauge network", fontsize=9)
    fig.savefig(FIG / "claims_gauges_map.png", dpi=150, bbox_inches="tight")
    print("figure ->", FIG / "claims_gauges_map.png")


if __name__ == "__main__":
    main()
