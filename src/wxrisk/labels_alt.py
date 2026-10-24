"""Alternative county-day SCS3 labels built from Storm Events magnitudes (and warnings).

Same universe as scripts/08 (112 CO/NM/AZ counties x every May-Sep day, 2000-2022, local date).
All labels are 0/1 per county-day:
  a_any     any SCS3 county report (identical to y0)
  a_sev     NWS-severe: hail >= 1.00 in, wind >= 50 kt or a wind-damage report with no gust, tornado
  a_sig     significant severe: hail >= 2.00 in, wind >= 65 kt, tornado EF2+/F2+
  a_hail    hail >= 1.00 in
  a_wind    wind >= 50 kt, or a wind-damage report with no gust
  a_tor     tornado
  a_sub     reports that are sub-severe only (hail < 1 in, or wind < 50 kt with a gust given)
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import wb2

STATES = {"08": "CO", "35": "NM", "04": "AZ"}
TYPES = ["Hail", "Thunderstorm Wind", "Tornado"]
COLS = ["BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME", "STATE_FIPS", "CZ_FIPS", "CZ_TYPE",
        "EVENT_TYPE", "MAGNITUDE", "MAGNITUDE_TYPE", "TOR_F_SCALE", "DAMAGE_PROPERTY",
        "CZ_TIMEZONE"]


def parse_damage(s: str) -> float:
    s = str(s).strip().upper()
    if not s or s == "NAN":
        return 0.0
    try:
        return float(s[:-1]) * {"K": 1e3, "M": 1e6, "B": 1e9}[s[-1]] if s[-1] in "KMB" else float(s)
    except ValueError:
        return 0.0


def storm_events(years=range(2000, 2023)) -> pd.DataFrame:
    """SCS3 county records for the three states, May-Sep, with parsed magnitude and damage."""
    fr = []
    for y in years:
        f = glob.glob(str(wb2.cache_path("storm_events")) + f"/StormEvents_details*_d{y}_*.csv.gz")[0]
        fr.append(pd.read_csv(f, usecols=COLS, dtype=str, low_memory=False))
    ev = pd.concat(fr, ignore_index=True)
    ev = ev[ev.EVENT_TYPE.isin(TYPES) & (ev.CZ_TYPE == "C")
            & ev.STATE_FIPS.str.zfill(2).isin(STATES)
            & ev.BEGIN_YEARMONTH.str[4:6].isin(["05", "06", "07", "08", "09"])].copy()
    ev["cty"] = ev.STATE_FIPS.str.zfill(2) + ev.CZ_FIPS.str.zfill(3)
    ev["date"] = pd.to_datetime(ev.BEGIN_YEARMONTH + ev.BEGIN_DAY.str.zfill(2), format="%Y%m%d")
    ev["mag"] = pd.to_numeric(ev.MAGNITUDE, errors="coerce")
    ev["prop"] = ev.DAMAGE_PROPERTY.map(parse_damage)
    return ev


def severity_flags(ev: pd.DataFrame) -> pd.DataFrame:
    """Per-record booleans; wind with no gust given counts as a damage-based severe report."""
    hail, wind, tor = (ev.EVENT_TYPE == "Hail", ev.EVENT_TYPE == "Thunderstorm Wind",
                       ev.EVENT_TYPE == "Tornado")
    mag = ev.mag
    efs = ev.TOR_F_SCALE.fillna("").str.extract(r"(\d)")[0].astype(float)
    f = pd.DataFrame(index=ev.index)
    f["a_hail"] = hail & (mag >= 1.0)
    f["a_wind"] = wind & ((mag >= 50) | mag.isna() | (mag == 0))
    f["a_tor"] = tor
    f["a_sev"] = f.a_hail | f.a_wind | f.a_tor
    f["a_sig"] = (hail & (mag >= 2.0)) | (wind & (mag >= 65)) | (tor & (efs >= 2))
    f["a_any"] = True
    f["a_sub"] = ~f.a_sev
    return f


def county_day_labels(counties: pd.DataFrame, years=range(2000, 2023)) -> pd.DataFrame:
    """Full county x day universe with every alternative label (0/1)."""
    ev = storm_events(years)
    f = severity_flags(ev)
    flags = list(f.columns)
    g = pd.concat([ev[["cty", "date"]], f], axis=1).groupby(["cty", "date"], as_index=False)[flags].max()
    days = pd.DatetimeIndex([d for d in pd.date_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
                             if d.month in (5, 6, 7, 8, 9)])
    full = pd.MultiIndex.from_product([counties, days], names=["cty", "date"]).to_frame(index=False)
    full = full.merge(g, how="left", on=["cty", "date"])
    full[flags] = full[flags].fillna(False).astype(np.int8)
    return full
