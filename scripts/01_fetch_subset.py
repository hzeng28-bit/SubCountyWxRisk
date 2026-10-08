"""Fetch a cached subset: ENS members at one lead, ERA5 truth, ERA5 climatology thresholds.

Everything is pulled anonymously from the WeatherBench-2 public bucket and cached as .npz
under $WXRISK_DATA (default ~/wxrisk_data). Re-running skips files that already exist.

    python scripts/01_fetch_subset.py --years 2018 2021 --lead-hours 72
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

VAR = "total_precipitation_24hr"
CLIM_START, CLIM_END = "1990-01-01", "2017-12-31"  # disjoint from the 2018-2022 eval period


def fetch_climatology() -> Path:
    out = wb2.cache_path("era5_tp24_clim_1990_2017_00z.npz")
    if out.exists():
        print(f"[skip] {out.name}")
        return out
    era = wb2.open_era5()
    da = era[VAR].sel(time=slice(CLIM_START, CLIM_END))
    da = da.isel(time=(da.time.dt.hour == 0))
    print(f"[clim] {da.sizes} -> {out.name}")
    t0 = time.time()
    arr = da.transpose("time", "longitude", "latitude").values
    months = da.time.dt.month.values
    np.savez_compressed(out, field=arr.astype(np.float32), months=months.astype(np.int16))
    print(f"[clim] done in {time.time() - t0:.0f}s, {arr.shape}")
    return out


def fetch_year(year: int, lead_hours: int) -> Path:
    out = wb2.cache_path(f"ens_tp24_{year}_lead{lead_hours}h_00z.npz")
    if out.exists():
        print(f"[skip] {out.name}")
        return out
    ens, era = wb2.open_ens(), wb2.open_era5()
    sub = ens[VAR].sel(time=str(year)).sel(prediction_timedelta=lead_hours)
    sub = sub.isel(time=(sub.time.dt.hour == 0))
    init = sub.time.values
    valid = init + np.timedelta64(lead_hours, "h")
    print(f"[{year}] members {dict(sub.sizes)} -> {out.name}")
    t0 = time.time()
    members = sub.transpose("time", "number", "longitude", "latitude").values
    print(f"[{year}] members fetched in {time.time() - t0:.0f}s {members.shape}")
    truth = (
        era[VAR].sel(time=valid).transpose("time", "longitude", "latitude").values
    )
    static = {
        k: era[k].transpose("longitude", "latitude").values
        for k in ("land_sea_mask", "geopotential_at_surface")
        if k in era
    }
    np.savez_compressed(
        out,
        members=members.astype(np.float32),
        truth=truth.astype(np.float32),
        init_time=init.astype("datetime64[h]").astype(np.int64),
        valid_month=valid.astype("datetime64[M]").astype(np.int64) % 12 + 1,
        **{f"static_{k}": v.astype(np.float32) for k, v in static.items()},
    )
    print(f"[{year}] done in {time.time() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2018, 2021])
    ap.add_argument("--lead-hours", type=int, default=72)
    ap.add_argument("--skip-clim", action="store_true")
    a = ap.parse_args()
    if not a.skip_clim:
        fetch_climatology()
    for y in a.years:
        fetch_year(y, a.lead_hours)


if __name__ == "__main__":
    main()
