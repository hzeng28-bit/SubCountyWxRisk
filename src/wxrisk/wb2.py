"""WeatherBench-2 public-bucket access (anonymous, no credentials)."""

from __future__ import annotations

import os
from pathlib import Path

import xarray as xr

BUCKET = "gs://weatherbench2"
ENS_STORE = f"{BUCKET}/datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr"
ERA5_STORE = f"{BUCKET}/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr"

# Higher-resolution ladder, for after the 64x32 conclusions hold.
ENS_STORE_240 = (
    f"{BUCKET}/datasets/ifs_ens/2018-2022-240x121_equiangular_with_poles_conservative.zarr"
)
ERA5_STORE_240 = (
    f"{BUCKET}/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)

DATA_DIR = Path(os.environ.get("WXRISK_DATA", Path.home() / "wxrisk_data"))


def open_store(url: str) -> xr.Dataset:
    return xr.open_zarr(url, storage_options={"token": "anon"}, consolidated=True, chunks={})


def open_ens(res: str = "64x32") -> xr.Dataset:
    return open_store(ENS_STORE if res == "64x32" else ENS_STORE_240)


def open_era5(res: str = "64x32") -> xr.Dataset:
    return open_store(ERA5_STORE if res == "64x32" else ERA5_STORE_240)


def cache_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name
