"""Economic covariates for the take-up model (scripts/37), keyless public files, through wxrisk.fetch.

  Freddie Mac PMMS history   30-year mortgage rate, weekly, 1971+ (the source of FRED's MORTGAGE30US; the FRED
                             CSV endpoint times out from this network, so the original publisher is used)
  Zillow ZHVI (county)       home value index, single family + condo, smoothed, seasonally adjusted, monthly
  Redfin county market tracker  monthly county home sales (homes_sold), prices, inventory, 2012+, 241 MB gz
  Census building permits    county annual files co<YEAR>a.txt (single-family permits and value), 2010-2023

Output: $WXRISK_DATA/labels_extra/econ/. Zillow and FRED files are rolling snapshots (they change as new months
arrive and as Zillow revises its history), so they are `mutable` in data.lock.json.

    python scripts/fetch/fetch_econ.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

OUT = wb2.cache_path("labels_extra") / "econ"
ZHVI = "https://files.zillowstatic.com/research/public_csvs/zhvi/County_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
PMMS = "https://www.freddiemac.com/pmms/docs/PMMS_history.csv"
BPS = "https://www2.census.gov/econ/bps/County/co{y}a.txt"
REDFIN = "https://redfin-public-data.s3.us-west-2.amazonaws.com/redfin_market_tracker/county_market_tracker.tsv000.gz"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [fetch.Job(OUT / "pmms_history.csv", url=PMMS, headers=UA), fetch.Job(OUT / "zillow_zhvi_county.csv", url=ZHVI),
            fetch.Job(OUT / "redfin_county_market_tracker.tsv.gz", url=REDFIN, headers=UA)]
    jobs += [fetch.Job(OUT / f"bps_county_{y}.txt", url=BPS.format(y=y)) for y in range(2010, 2024)]
    for k, v in fetch.fetch_many(jobs, workers=4).items():
        print(k, v)


if __name__ == "__main__":
    main()
