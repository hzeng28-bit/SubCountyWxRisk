"""Block-group terrain features for the Texas pilot, from AWS Terrarium elevation tiles (zoom 10, ~150 m).

Per block-group centroid: elevation, height above the lowest point within 1 / 3 / 10 km (a coarse
stand-in for height above nearest drainage), slope, and relative position within the 3 km window.
This is static susceptibility information that a rainfall-only model cannot see.

    python scripts/27_terrain_features.py
"""

from __future__ import annotations

import io
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402

Z = 10
X0, X1, Y0, Y1 = 208, 246, 400, 436  # tile ranges covering the Texas box
BUCKET, KEY = "elevation-tiles-prod", "terrarium/{z}/{x}/{y}.png"  # anonymous S3
TILES = fp.PILOT / "terrain_tiles"


def tile(x: int, y: int) -> np.ndarray:
    p = TILES / f"{Z}_{x}_{y}.npy"
    if p.exists():
        return np.load(p)
    try:  # s3_bytes retries transient errors itself
        im = np.array(Image.open(io.BytesIO(fetch.s3_bytes(BUCKET, KEY.format(z=Z, x=x, y=y)))).convert("RGB"))
    except Exception:  # noqa: BLE001 - a missing tile becomes NaN below
        return np.full((256, 256), np.nan, np.float32)
    e = im.astype(np.float32)
    e = e[..., 0] * 256 + e[..., 1] + e[..., 2] / 256 - 32768
    np.save(p, e)
    return e


def lonlat_to_pixel(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = 2.0**Z
    x = (lon + 180) / 360 * n
    la = np.radians(lat)
    y = (1 - np.log(np.tan(la) + 1 / np.cos(la)) / math.pi) / 2 * n
    return (x - X0) * 256, (y - Y0) * 256


def main() -> None:
    TILES.mkdir(parents=True, exist_ok=True)
    xs, ys = range(X0, X1 + 1), range(Y0, Y1 + 1)
    jobs = [(x, y) for y in ys for x in xs]
    with ThreadPoolExecutor(16) as ex:
        tiles = dict(zip(jobs, ex.map(lambda j: tile(*j), jobs), strict=True))
    print(f"{len(jobs)} tiles", flush=True)
    M = np.full(((Y1 - Y0 + 1) * 256, (X1 - X0 + 1) * 256), np.nan, np.float32)
    for (x, y), t in tiles.items():
        M[(y - Y0) * 256:(y - Y0 + 1) * 256, (x - X0) * 256:(x - X0 + 1) * 256] = t
    print("mosaic", M.shape, f"NaN share {np.isnan(M).mean():.3f}", flush=True)

    g = fp.bg_geometry()
    px, py = lonlat_to_pixel(g.lon.to_numpy(), g.lat.to_numpy())
    cx, cy = np.clip(px.astype(int), 0, M.shape[1] - 1), np.clip(py.astype(int), 0, M.shape[0] - 1)

    f = np.nan_to_num(M, nan=-9999.0)
    out = {"elev": M[cy, cx]}
    for km, half in ((1, 7), (3, 20), (10, 65)):
        lo = minimum_filter(np.where(np.isnan(M), 1e9, M), size=2 * half + 1)
        out[f"rel{km}"] = M[cy, cx] - lo[cy, cx]
        if km == 3:
            hi = maximum_filter(f, size=2 * half + 1)
            out["pos3"] = (M[cy, cx] - lo[cy, cx]) / (hi[cy, cx] - lo[cy, cx] + 1.0)
        del lo
    # slope in degrees; metres per pixel shrink with cos(latitude) in Web Mercator
    row_lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * ((np.arange(M.shape[0]) / 256 + Y0) / 2.0**Z)))))
    res = 156543.03 * np.cos(np.radians(row_lat)) / 2.0**Z
    gy, gx = np.gradient(np.nan_to_num(M, nan=0.0))
    out["slope"] = np.degrees(np.arctan(np.hypot(gx, gy) / res[:, None]))[cy, cx]
    t = pd.DataFrame(out)
    t.insert(0, "bg10", g.bg10.to_numpy())
    t.to_parquet(fp.PILOT / "bg_terrain.parquet")
    print(t.describe().round(1).T[["mean", "std", "min", "max"]].to_string())
    print("saved bg_terrain.parquet")


if __name__ == "__main__":
    main()
