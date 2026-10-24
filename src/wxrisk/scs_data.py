"""Joined county-day table (labels + point env + scale features + D-3 lag) and tier column sets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import county_env, wb2
from .scale_features import CHANNELS, MOTION_CHANNELS, WAVELENGTHS

HEADS = ["y0", "y10k", "y100k"]  # y1m has 16 test positives: not scoreable
SCALES = [int(w) for w in WAVELENGTHS]
STATIC = ["doy_s", "doy_c", "lat", "lon"]
ENV = county_env.ENV


def load(years: list[int]) -> pd.DataFrame:
    lab = pd.read_parquet(wb2.cache_path("labels_scs3_conm_az.parquet"))
    cty = lab.drop_duplicates("cty")[["cty", "lat", "lon"]].reset_index(drop=True)
    env = county_env.load(years, cty)
    sf = pd.concat([pd.read_parquet(wb2.cache_path(f"scalefeat_{y}.parquet")) for y in years])
    df = lab.merge(env, on=["cty", "date"]).merge(sf, on=["cty", "date"])
    for c in df.columns:
        if c.startswith(("cape_", "cin_", "tp_")) and "_b" not in c and "_e" not in c:
            df[c] = np.log1p(df[c].clip(lower=0))
    num = df.select_dtypes("float64").columns.difference(["lat", "lon"])
    df[num] = df[num].astype(np.float32)
    # D-3 frame: 12Z columns of the same county three days earlier
    lag_cols = ([f"{k}_12" for k in ENV]
                + [c for c in df.columns if c.split("_")[0] in CHANNELS and c.split("_")[1] == "12"])
    lag = df[["cty", "date", *dict.fromkeys(lag_cols)]].copy()  # point cols match both rules
    lag["date"] = lag["date"] + pd.Timedelta(3, "D")
    lag.columns = [c if c in ("cty", "date") else c + "_L3" for c in lag.columns]
    df = df.merge(lag, on=["cty", "date"])
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    doy = df.date.dt.dayofyear.to_numpy()
    df["doy_s"], df["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df["state"] = df.cty.str[:2].map({"08": "CO", "35": "NM", "04": "AZ"})
    return df.dropna().reset_index(drop=True)


def tier_spec(tier: str) -> dict:
    """Column sets: 'point' (baseline) and per-scale expert groups ('lp' = context expert)."""
    if tier == "A3":
        hours, sfx = (12,), "_L3"
    elif tier == "B":
        hours, sfx = (12,), ""
    else:
        hours, sfx = (12, 18), ""
    point = [f"{k}_{h}{sfx}" for h in hours for k in ENV]
    groups = {}
    for s in SCALES:
        g = [f"{c}_{h}_{t}{s}{sfx}" for h in hours for c in CHANNELS for t in ("b", "e")]
        if tier == "C":  # "Cnm" = tier C without the motion channels (ablation)
            g += [f"mv_{c}_{s}_{d}" for c in MOTION_CHANNELS for d in ("dx", "dy")]
        groups[f"s{s}"] = g
    groups["lp"] = [f"{c}_{h}_lp{sfx}" for h in hours for c in CHANNELS]
    if tier in ("C", "Cnm"):
        extra = ["d_cape", "d_t2m", "d_d2m", "d_pw", "tp_am", "tp_pm"]
        point += extra
        groups["lp"] += extra
    return {"point": point, "groups": groups}
