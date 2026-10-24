"""County-day convective-environment features from the CDS ERA5 downloads (scripts/09).

Each county takes its nearest 0.25 deg grid cell. Features are built per day for two snapshots:
12Z ("morning") and 18Z ("last analysis before the 20-03Z convective peak").
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from . import wb2

ENV = ["cape", "cin", "dd", "pw", "q850", "lapse75", "shear", "w700", "msl"]


def _at_counties(ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    ix = {"latitude": xr.DataArray(lat, dims="cty"), "longitude": xr.DataArray(lon, dims="cty")}
    return ds.sel(**ix, method="nearest")


def _year(year: int, cty: pd.DataFrame) -> pd.DataFrame:
    d = wb2.DATA_DIR / "cds"
    lat, lon = cty.lat.to_numpy(), cty.lon.to_numpy()
    sg = _at_counties(xr.open_dataset(d / f"single_{year}.nc"), lat, lon).load()
    pl = _at_counties(xr.open_dataset(d / f"plev_{year}.nc"), lat, lon).sel(pressure_level=[850, 700, 500]).load()
    pr = _at_counties(xr.open_dataset(d / f"precip_{year}.nc"), lat, lon).load()

    t = pd.DatetimeIndex(sg.valid_time.values)
    out = {}
    for hour in (12, 18):
        i = np.flatnonzero(t.hour == hour)
        s = sg.isel(valid_time=i)
        p = pl.sel(valid_time=s.valid_time)
        wind = lambda u, v: np.hypot(u, v)  # noqa: E731
        f = {
            "cape": s.cape.values,
            "cin": np.abs(np.nan_to_num(s.cin.values, nan=0.0)),  # undefined where CAPE == 0
            "dd": s.t2m.values - s.d2m.values,
            "t2m": s.t2m.values,
            "pw": s.tcwv.values,
            "q850": p.q.sel(pressure_level=850).values,
            "lapse75": p.t.sel(pressure_level=700).values - p.t.sel(pressure_level=500).values,
            "shear": wind(p.u.sel(pressure_level=500).values - s.u10.values,
                          p.v.sel(pressure_level=500).values - s.v10.values),
            "w700": wind(p.u.sel(pressure_level=700).values, p.v.sel(pressure_level=700).values),
            "msl": s.msl.values,
            "d2m": s.d2m.values,
        }
        out[hour] = (t[i].normalize(), f)
    days, f12 = out[12]
    d18, f18 = out[18]
    assert (days == d18).all()
    # hourly tp is in metres per hour; windows end at 12Z and 18Z
    ptime = pd.DatetimeIndex(pr.valid_time.values)
    tp = pr.tp.values * 1000.0
    pr_day = ptime.normalize()
    p_am = np.stack([tp[(pr_day == d) & (ptime.hour <= 12)].sum(0) for d in days])
    p_pm = np.stack([tp[(pr_day == d) & (ptime.hour > 12)].sum(0) for d in days])

    n_day, n_cty = f12["cape"].shape
    cols = {}
    for k, v in f12.items():
        cols[f"{k}_12"] = v.ravel()
    for k, v in f18.items():
        cols[f"{k}_18"] = v.ravel()
    for k in ("cape", "t2m", "d2m", "pw"):
        cols[f"d_{k}"] = (f18[k] - f12[k]).ravel()
    cols["tp_am"], cols["tp_pm"] = p_am.ravel(), p_pm.ravel()
    df = pd.DataFrame(cols)
    df["date"] = np.repeat(days.to_numpy(), n_cty)
    df["cty"] = np.tile(cty.cty.to_numpy(), n_day)
    return df


def load(years: range, cty: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_year(y, cty) for y in years], ignore_index=True)
