"""Fetch lead-time SEQUENCES for motion estimation (not single-lead snapshots).

Two subsets:
  A) 240x121 ENS-MEAN: Z500 / u500 / v500 / mslp over leads 0..LEAD_MAX, a handful of inits.
     Fine enough that a synoptic displacement spans several cells, and it carries the wind
     field needed as an INDEPENDENT check on any estimated motion vector.
  B) 64x32 full 50 members: mslp over all leads, same inits. This is the grid the risk index
     lives on, so we need to know whether motion is recoverable there at all.

    python scripts/03_fetch_sequences.py --n-init 12 --lead-max 120
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

MEAN_240 = (
    "gs://weatherbench2/datasets/ifs_ens/"
    "2018-2022-240x121_equiangular_with_poles_conservative_mean.zarr"
)


def pick_inits(times: np.ndarray, n: int, year: int) -> np.ndarray:
    """Evenly spaced 00Z inits across one year, so every season is represented."""
    t = times[(times.astype("datetime64[Y]").astype(int) + 1970 == year)]
    t = t[t.astype("datetime64[h]").astype(np.int64) % 24 == 0]
    return t[np.linspace(0, t.size - 1, n).astype(int)]


def fetch_mean_240(n_init: int, lead_max: int, year: int) -> Path:
    out = wb2.cache_path(f"seq240_mean_{year}_n{n_init}_lead{lead_max}.npz")
    if out.exists():
        print(f"[skip] {out.name}")
        return out
    ds = wb2.open_store(MEAN_240)
    inits = pick_inits(ds.time.values, n_init, year)
    leads = ds.prediction_timedelta.values
    leads = leads[leads <= lead_max]
    sub = ds.sel(time=inits, prediction_timedelta=leads)
    t0 = time.time()
    arrs = {}
    for name, var, kw in [
        ("z500", "geopotential", {"level": 500}),
        ("u500", "u_component_of_wind", {"level": 500}),
        ("v500", "v_component_of_wind", {"level": 500}),
        ("mslp", "mean_sea_level_pressure", {}),
    ]:
        da = sub[var].sel(**kw) if kw else sub[var]
        arrs[name] = da.transpose("time", "prediction_timedelta", "longitude", "latitude").values
        print(f"  [240] {name} {arrs[name].shape} {time.time() - t0:.0f}s")
    np.savez_compressed(
        out,
        latitude=ds.latitude.values,
        longitude=ds.longitude.values,
        lead_hours=leads.astype(np.int64),
        init_time=inits.astype("datetime64[h]").astype(np.int64),
        **{k: v.astype(np.float32) for k, v in arrs.items()},
    )
    print(f"[240] done {time.time() - t0:.0f}s -> {out.name}")
    return out


def fetch_members_64(n_init: int, lead_max: int, year: int) -> Path:
    out = wb2.cache_path(f"seq64_members_{year}_n{n_init}_lead{lead_max}.npz")
    if out.exists():
        print(f"[skip] {out.name}")
        return out
    ds = wb2.open_ens()
    inits = pick_inits(ds.time.values, n_init, year)
    leads = ds.prediction_timedelta.values
    leads = leads[leads <= lead_max]
    sub = ds["mean_sea_level_pressure"].sel(time=inits, prediction_timedelta=leads)
    t0 = time.time()
    arr = sub.transpose(
        "time", "number", "prediction_timedelta", "longitude", "latitude"
    ).values
    print(f"  [64] mslp members {arr.shape} {time.time() - t0:.0f}s")
    np.savez_compressed(
        out,
        mslp=arr.astype(np.float32),
        latitude=ds.latitude.values,
        longitude=ds.longitude.values,
        lead_hours=leads.astype(np.int64),
        init_time=inits.astype("datetime64[h]").astype(np.int64),
    )
    print(f"[64] done {time.time() - t0:.0f}s -> {out.name}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-init", type=int, default=12)
    ap.add_argument("--lead-max", type=int, default=120)
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--only", choices=["240", "64"], default=None)
    a = ap.parse_args()
    if a.only != "64":
        fetch_mean_240(a.n_init, a.lead_max, a.year)
    if a.only != "240":
        fetch_members_64(a.n_init, a.lead_max, a.year)


if __name__ == "__main__":
    main()
