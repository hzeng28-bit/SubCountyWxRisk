"""Height above nearest drainage (HAND) for Texas from AWS Terrarium elevation tiles (zoom 11, ~65-75 m).

The official 3 m Texas HAND (ORNL/TACC) is hundreds of GB and its server refuses connections, so HAND is computed here:
D8 flow directions from the filled DEM (pyflwdir, priority-flood from the mosaic edge), streams = cells draining at least
T km2, HAND = elevation above the stream cell the water reaches. Two thresholds: 1 km2 (small urban creeks) and 10 km2 (channels).
Coarse by design: 70 m cannot resolve storm drains, levees or road embankments; it is a screening indicator of position in the valley.

Inputs : AWS Terrarium tiles (anonymous S3 `elevation-tiles-prod`, via wxrisk.fetch), cached as .npy per tile
Outputs: $WXRISK_DATA/labels_extra/flood/pilot/hand_z11_t1.npy, hand_z11_t10.npy  (float16 metres, NaN = no stream downstream)
Run    : python scripts/57_hand_raster.py        # ~5.8k tiles (about 1.5 GB cached), then ~10-20 min and ~10 GB RAM
"""

from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402

Z = 11
X0, X1, Y0, Y1 = 416, 493, 800, 873  # the z10 box of script 27, doubled
BUCKET, KEY = "elevation-tiles-prod", "terrarium/{z}/{x}/{y}.png"
TILES = fp.PILOT / "terrain_tiles_z11"
CELL_KM2 = 0.0043  # a 256-px Terrarium tile at z11 and 31 deg N: (156543.03 * cos(31 deg) / 2**11) ** 2 / 1e6
THRESH_KM2 = (1, 10)


def tile(x: int, y: int) -> np.ndarray:
    p = TILES / f"{Z}_{x}_{y}.npy"
    if p.exists():
        return np.load(p)
    try:
        im = np.array(Image.open(io.BytesIO(fetch.s3_bytes(BUCKET, KEY.format(z=Z, x=x, y=y)))).convert("RGB"))
    except Exception:  # noqa: BLE001 - a missing tile becomes NaN
        return np.full((256, 256), np.nan, np.float32)
    e = im.astype(np.float32)
    e = e[..., 0] * 256 + e[..., 1] + e[..., 2] / 256 - 32768
    np.save(p, e)
    return e


def mosaic() -> np.ndarray:
    TILES.mkdir(parents=True, exist_ok=True)
    jobs = [(x, y) for y in range(Y0, Y1 + 1) for x in range(X0, X1 + 1)]
    M = np.full(((Y1 - Y0 + 1) * 256, (X1 - X0 + 1) * 256), np.nan, np.float32)
    with ThreadPoolExecutor(16) as ex:
        for (x, y), t in zip(jobs, ex.map(lambda j: tile(*j), jobs), strict=True):
            M[(y - Y0) * 256:(y - Y0 + 1) * 256, (x - X0) * 256:(x - X0 + 1) * 256] = t
    return M


def main() -> None:
    import pyflwdir

    t0 = time.time()
    M = mosaic()
    print(f"mosaic {M.shape}, NaN share {np.isnan(M).mean():.3f}, {time.time() - t0:.0f} s", flush=True)
    M = np.where(np.isnan(M), -9999.0, np.maximum(M, 0.0)).astype(np.float32)  # sea and bathymetry -> sea level
    flw = pyflwdir.from_dem(M, nodata=-9999.0, outlets="edge")
    print(f"flow directions {time.time() - t0:.0f} s", flush=True)
    acc = flw.upstream_area(unit="cell")
    for km2 in THRESH_KM2:
        drain = acc >= km2 / CELL_KM2
        h = flw.hand(drain, M).astype(np.float32)
        h[(M <= -9999.0) | (h < 0)] = np.nan
        np.save(fp.PILOT / f"hand_z11_t{km2}.npy", h.astype(np.float16))
        ok = h[np.isfinite(h)]
        print(f"HAND stream >= {km2} km2: median {np.median(ok[::50]):.1f} m, share < 5 m {(ok < 5).mean():.2f}, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
