"""Fetch the inputs of the Texas sub-county flood pilot.

  gridMET   daily precipitation (4 km) over a Texas box, 2000-2023, one NetCDF per year, through
            the THREDDS NetCDF Subset Service (anonymous)
  HURDAT2   NHC best tracks, used to drop tropical-cyclone-influenced days
  TIGER     Census cartographic block groups for Texas, 2010 and 2020 vintages (NFIP does not say
            which vintage its 12-digit codes follow, so both are kept and the match rate is measured)

    python scripts/22_fetch_pilot_inputs.py --workers 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

PILOT = wb2.cache_path("labels_extra") / "flood" / "pilot"
NCSS = ("http://thredds.northwestknowledge.net:8080/thredds/ncss/agg_met_pr_1979_CurrentYear_CONUS.nc"
        "?var=precipitation_amount&north=36.6&west=-106.7&east=-93.4&south=25.8&horizStride=1"
        "&time_start={y}-01-01T00:00:00Z&time_end={y}-12-31T00:00:00Z&accept=netcdf")
OTHER = {
    "hurdat2.txt": "https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt",
    "bg2020_tx.zip": "https://www2.census.gov/geo/tiger/GENZ2020/shp/cb_2020_48_bg_500k.zip",
    "bg2010_tx.zip": "https://www2.census.gov/geo/tiger/GENZ2010/gz_2010_48_150_00_500k.zip",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2000, 2023])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    (PILOT / "gridmet").mkdir(parents=True, exist_ok=True)
    jobs = [fetch.Job(PILOT / n, url=u) for n, u in OTHER.items()]
    # NCSS builds each subset on request: no HEAD probe (it would do the work twice), size is only a hint
    jobs += [fetch.Job(PILOT / "gridmet" / f"pr_tx_{y}.nc", url=NCSS.format(y=y), probe=False, size_hint=100 << 20)
             for y in range(a.years[0], a.years[1] + 1)]
    res = fetch.fetch_many(jobs, workers=a.workers)
    if bad := {k: v for k, v in res.items() if v.startswith("FAIL")}:
        sys.exit(f"{len(bad)} downloads failed: {bad}")


if __name__ == "__main__":
    main()
