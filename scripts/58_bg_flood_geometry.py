"""Block-group flood-zone and HAND features (FEMA NFHL area shares, HAND zonal statistics).

Per 2010 block group:
  nfhl_sfha      share of land area inside the Special Flood Hazard Area (NFHL zone polygons with SFHA_TF = 'T')
  nfhl_x02       share inside shaded X (0.2 % annual chance / levee-protected X)
  nfhl_mapped    1 if the county has any NFHL polygon outside minimal-hazard X (0 can mean unmapped, not safe)
  h1_med, h1_p10, h1_le2, h1_le5   HAND (streams >= 1 km2) median, 10th percentile, share of cells <= 2 m, <= 5 m
  h10_med, h10_le5                 same for streams >= 10 km2
Unlike the policy-based `sfha_share` (insured policies in the SFHA, selected by mandatory purchase), the NFHL share is the area
share of the mapped hazard and does not depend on who bought insurance.

Inputs : labels_extra/flood/nfhl/nfhl_<county>.geojson (scripts/fetch/fetch_nfhl_tx.py), pilot/hand_z11_t{1,10}.npy (script 57),
         bg2010_tx.zip
Output : $WXRISK_DATA/labels_extra/flood/pilot/bg_flood_geometry.parquet
Run    : python scripts/58_bg_flood_geometry.py [--nfhl-only]      # full run about 2.5 h on a Mac, mostly the HAND zonal part
"""

from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import shapefile
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

NFHL = wb2.cache_path("labels_extra") / "flood" / "nfhl"
Z, X0, Y0 = 11, 416, 800


def to_pixel(lon, lat):
    n = 2.0**Z
    x = (lon + 180) / 360 * n
    y = (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / math.pi) / 2 * n
    return (x - X0) * 256, (y - Y0) * 256


def block_groups() -> dict[str, shapely.Geometry]:
    z = zipfile.ZipFile(fp.PILOT / "bg2010_tx.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                          dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    out = {}
    for sr in rd.iterShapeRecords():
        r = dict(zip(names, sr.record, strict=True))
        out[r["STATE"] + r["COUNTY"] + r["TRACT"] + r["BLKGRP"]] = shape(sr.shape.__geo_interface__)
    return out


def nfhl_shares(bgs: dict[str, shapely.Geometry]) -> pd.DataFrame:
    rows = {}
    for cty in sorted({b[:5] for b in bgs}):
        p = NFHL / f"nfhl_{cty}.geojson"
        feats = json.loads(p.read_text())["features"] if p.exists() else []
        sfha = [shape(f["geometry"]) for f in feats if f["geometry"] and f["properties"].get("SFHA_TF") == "T"]
        # shaded X only: plain minimal-hazard X carries the subtype "AREA OF MINIMAL FLOOD HAZARD" and is the complement
        x02 = [shape(f["geometry"]) for f in feats if f["geometry"] and f["properties"].get("SFHA_TF") != "T"
               and any(k in (f["properties"].get("ZONE_SUBTY") or "").upper() for k in ("0.2 PCT", "LEVEE"))]
        U = {k: shapely.make_valid(unary_union([shapely.make_valid(g) for g in v])) if v else None
             for k, v in (("sfha", sfha), ("x02", x02))}
        if U["sfha"] is not None and U["x02"] is not None:
            U["x02"] = U["x02"].difference(U["sfha"])  # dual-zone overlaps counted once, SFHA first
        for b in (b for b in bgs if b[:5] == cty):
            g = bgs[b]
            a = g.area
            rows[b] = {"nfhl_sfha": (g.intersection(U["sfha"]).area / a) if U["sfha"] is not None and a > 0 else 0.0,
                       "nfhl_x02": (g.intersection(U["x02"]).area / a) if U["x02"] is not None and a > 0 else 0.0,
                       "nfhl_mapped": float(len(feats) > 0)}
        print(cty, len(feats), "zones", flush=True)
    return pd.DataFrame.from_dict(rows, orient="index")


def hand_stats(bgs: dict[str, shapely.Geometry]) -> pd.DataFrame:
    H1 = np.load(fp.PILOT / "hand_z11_t1.npy", mmap_mode="r")
    H10 = np.load(fp.PILOT / "hand_z11_t10.npy", mmap_mode="r")
    rows = {}
    for b, g in bgs.items():
        pg = shapely.transform(g, lambda c: np.column_stack(to_pixel(c[:, 0], c[:, 1])))
        x0, y0, x1, y1 = pg.bounds
        xs = np.arange(max(int(x0), 0), min(int(x1) + 1, H1.shape[1]))
        ys = np.arange(max(int(y0), 0), min(int(y1) + 1, H1.shape[0]))
        if len(xs) == 0 or len(ys) == 0:
            continue
        gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
        m = shapely.contains_xy(pg, gx, gy)
        if not m.any():  # smaller than a cell: take the cell under the centroid
            c = pg.centroid
            m = np.zeros_like(m)
            m[min(max(int(c.y) - ys[0], 0), len(ys) - 1), min(max(int(c.x) - xs[0], 0), len(xs) - 1)] = True
        sl = (slice(ys[0], ys[-1] + 1), slice(xs[0], xs[-1] + 1))
        h1 = np.asarray(H1[sl], np.float32)[m]
        h10 = np.asarray(H10[sl], np.float32)[m]
        f1, f10 = h1[np.isfinite(h1)], h10[np.isfinite(h10)]
        rows[b] = {"h1_med": np.median(f1) if len(f1) else np.nan, "h1_p10": np.percentile(f1, 10) if len(f1) else np.nan,
                   "h1_le2": (f1 <= 2).mean() if len(f1) else np.nan, "h1_le5": (f1 <= 5).mean() if len(f1) else np.nan,
                   "h10_med": np.median(f10) if len(f10) else np.nan, "h10_le5": (f10 <= 5).mean() if len(f10) else np.nan}
    return pd.DataFrame.from_dict(rows, orient="index")


def main() -> None:
    bgs = block_groups()
    print(len(bgs), "block groups", flush=True)
    old = fp.PILOT / "bg_flood_geometry.parquet"
    if "--nfhl-only" in sys.argv and old.exists():  # HAND zonal statistics take about two hours; reuse them
        hand = pd.read_parquet(old).set_index("bg10")[["h1_med", "h1_p10", "h1_le2", "h1_le5", "h10_med", "h10_le5"]]
    else:
        hand = hand_stats(bgs)
    out = nfhl_shares(bgs).join(hand)
    out.index.name = "bg10"
    out = out.reset_index()
    out.to_parquet(fp.PILOT / "bg_flood_geometry.parquet")
    print(out.describe().round(3).T[["count", "mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
