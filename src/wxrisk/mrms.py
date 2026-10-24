"""MRMS 24 h products on NOAA's AWS bucket, reduced to county-day maxima.

A file stamped D+1 07:00 UTC holds the 24 h maximum / accumulation ending then, i.e. the local
standard-time day D (UTC-7), matching Storm Events. The archive starts in October 2020.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import contains_xy

from . import fetch as dl
from . import wb2
from .labels_flood import county_polys  # noqa: F401  (re-exported)

BUCKET = "noaa-mrms-pds"  # anonymous S3, us-east-1


def fetch(product: str, day: pd.Timestamp, out_dir: Path) -> Path | None:
    end = day + pd.Timedelta(days=1)
    name = f"MRMS_{product}_{end:%Y%m%d}-070000.grib2.gz"
    p = out_dir / f"{day:%Y%m%d}.grib2"
    if p.exists():
        return p
    gz = p.with_suffix(".grib2.gz")
    try:
        dl.s3(BUCKET, f"CONUS/{product}/{end:%Y%m%d}/{name}", gz)
        with gzip.open(gz, "rb") as g, open(p.with_suffix(".part"), "wb") as f:
            shutil.copyfileobj(g, f)
        p.with_suffix(".part").replace(p)
        gz.unlink()
        return p
    except dl.InsufficientSpace:
        raise
    except Exception:  # noqa: BLE001 - a missing day is reported by the caller
        return None


def county_masks(lat: np.ndarray, lon: np.ndarray, polys: dict) -> dict:
    """Per county: (row slice, col slice, boolean mask) on the MRMS grid (lon given as 0-360)."""
    lat1, lon1 = lat[:, 0], lon[0] - 360.0
    masks = {}
    for k, poly in polys.items():
        x0, y0, x1, y1 = poly.bounds
        r = np.flatnonzero((lat1 >= y0) & (lat1 <= y1))
        c = np.flatnonzero((lon1 >= x0) & (lon1 <= x1))
        gx, gy = np.meshgrid(lon1[c], lat1[r])
        masks[k] = (slice(r.min(), r.max() + 1), slice(c.min(), c.max() + 1), contains_xy(poly, gx, gy))
    return masks


def county_day_max(product: str, days: list[pd.Timestamp], out_dir: Path, workers: int = 8,
                   fracs: tuple[float, ...] = ()) -> pd.DataFrame:
    """Max over each county's cells per day; `fracs` adds area shares above those values."""
    from concurrent.futures import ThreadPoolExecutor

    import pygrib

    out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        files = list(ex.map(lambda d: fetch(product, d, out_dir), days))
    print(f"{sum(f is not None for f in files)}/{len(days)} {product} days fetched", flush=True)
    polys, masks, rows = county_polys(), None, []
    for d, f in zip(days, files, strict=True):
        if f is None:
            continue
        g = pygrib.open(str(f)).message(1)
        v = np.ma.filled(g.values, 0.0)
        if masks is None:
            lat, lon = g.latlons()
            masks = county_masks(lat, lon, polys)
        for k, (rs, cs, m) in masks.items():
            cell = v[rs, cs][m]
            rows.append((k, d, float(cell.max()) if cell.size else 0.0,
                         *[float((cell >= t).mean()) if cell.size else 0.0 for t in fracs]))
    return pd.DataFrame(rows, columns=["cty", "date", "max_mm", *[f"frac_ge_{t:g}" for t in fracs]])


def cache(name: str) -> Path:
    return wb2.cache_path(name)
