"""Build the CO/NM/AZ county-day SCS3 label table from NOAA Storm Events.

SCS3 = Hail + Thunderstorm Wind + Tornado, county-type records only (CZ_TYPE == "C").
Per county-day the property damage is summed, then discretised into four nested targets:
occurrence (any report), >=$10k, >=$100k, >=$1M (nominal $).

Universe is every county x every day in May-Sep (days without a report are explicit zeros).
Output: $WXRISK_DATA/labels_scs3_conm_az.parquet  (cty, date, lat, lon, y0, y10k, y100k, y1m)

    python scripts/08_build_labels.py --se-dir <dir with StormEvents_details-*.csv.gz>
"""

from __future__ import annotations

import argparse
import glob
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

STATES = {"08": "CO", "35": "NM", "04": "AZ"}
TYPES = ["Hail", "Thunderstorm Wind", "Tornado"]
GAZ = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_counties_national.zip"
YEARS = range(2000, 2023)  # ERA5 in WB2 ends 2023-01-10, so 2023 has no features
MONTHS = (5, 6, 7, 8, 9)
USECOLS = ["BEGIN_YEARMONTH", "BEGIN_DAY", "STATE_FIPS", "CZ_FIPS", "CZ_TYPE", "EVENT_TYPE",
           "DAMAGE_PROPERTY"]


def parse_damage(s: str) -> float:
    m = re.match(r"^([0-9.]+)([KMBkmb]?)$", str(s).strip())
    if not m:
        return 0.0
    return float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2).upper()]


def counties() -> pd.DataFrame:
    raw = fetch.get_bytes(GAZ)
    import zipfile

    z = zipfile.ZipFile(io.BytesIO(raw))
    df = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype={"GEOID": str})
    df.columns = [c.strip() for c in df.columns]
    df = df[df.GEOID.str[:2].isin(STATES)]
    return df.rename(columns={"GEOID": "cty", "INTPTLAT": "lat", "INTPTLONG": "lon"})[
        ["cty", "NAME", "lat", "lon"]
    ].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--se-dir", required=True)
    args = ap.parse_args()

    fr = [pd.read_csv(f, usecols=USECOLS, dtype=str, low_memory=False)
          for f in sorted(glob.glob(f"{args.se_dir}/StormEvents_details*.csv.gz"))]
    ev = pd.concat(fr, ignore_index=True)
    ev = ev[ev.EVENT_TYPE.isin(TYPES) & (ev.CZ_TYPE == "C")
            & ev.STATE_FIPS.str.zfill(2).isin(STATES)].copy()
    ev["cty"] = ev.STATE_FIPS.str.zfill(2) + ev.CZ_FIPS.str.zfill(3)
    ev["date"] = pd.to_datetime(ev.BEGIN_YEARMONTH + ev.BEGIN_DAY.str.zfill(2), format="%Y%m%d")
    ev["prop"] = ev.DAMAGE_PROPERTY.map(parse_damage)
    cd = ev.groupby(["cty", "date"], as_index=False).prop.sum()

    cty = counties()
    days = pd.DatetimeIndex([d for d in pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31")
                             if d.month in MONTHS])
    full = pd.MultiIndex.from_product([cty.cty, days], names=["cty", "date"]).to_frame(index=False)
    full = full.merge(cd, how="left", on=["cty", "date"]).merge(cty[["cty", "lat", "lon"]], on="cty")
    # a damage-less report is still an occurrence, so occurrence comes from presence, not prop
    full["y0"] = full.prop.notna().astype(np.int8)
    p = full.prop.fillna(0.0)
    for name, thr in (("y10k", 1e4), ("y100k", 1e5), ("y1m", 1e6)):
        full[name] = (p >= thr).astype(np.int8)
    out = wb2.cache_path("labels_scs3_conm_az.parquet")
    full.drop(columns="prop").to_parquet(out)
    print(f"{out}: {len(full)} county-days, {cty.cty.nunique()} counties")
    print(full.groupby(full.date.dt.year >= 2017)[["y0", "y10k", "y100k", "y1m"]].agg(["sum", "mean"]))


if __name__ == "__main__":
    main()
