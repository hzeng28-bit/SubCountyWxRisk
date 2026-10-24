"""Download NFIP claims (OpenFEMA v3 NfipClaims) for 2000 onward, one parquet per year.

Fields kept: where (state, county, 12-digit census block group, ZIP), when (date of loss), what
(cause code, water depth, duration, flood characteristics, named event), how much (paid and
damage amounts), and the building (flood zone, occupancy, coverage, foundation, community).
Latitude and longitude are published to one decimal place only, so they are not kept.

    python scripts/20_fetch_nfip_claims.py --years 2000 2023 --workers 4
"""

from __future__ import annotations

import argparse
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

URL = "https://www.fema.gov/api/open/v3/NfipClaims"
FIELDS = ("id,state,countyCode,censusGeoid,reportedZipCode,dateOfLoss,yearOfLoss,floodEvent,"
          "causeOfDamage,waterDepth,exteriorWaterDepth,interiorWaterDepth,floodWaterDuration,"
          "floodCharacteristicsIndicator,amountPaidOnBuildingClaim,amountPaidOnContentsClaim,"
          "netBuildingPaymentAmount,netContentsPaymentAmount,buildingDamageAmount,contentsDamageAmount,"
          "ratedFloodZone,occupancyType,totalBuildingInsuranceCoverage,nfipRatedCommunityNumber,"
          "foundationType,policyCount,elevatedBuildingIndicator,postFIRMConstructionIndicator")
STEP = 10000
OUT = wb2.cache_path("labels_extra") / "flood" / "nfip_national"


def one_year(year: int) -> str:
    out = OUT / f"nfip_{year}.parquet"
    if out.exists():
        return f"[skip] {year}"
    expected = fetch.openfema_count("NfipClaims", "v3", f"yearOfLoss eq {year}")  # the server's own row count
    rows, skip = [], 0
    while True:
        q = urllib.parse.urlencode({"$filter": f"yearOfLoss eq {year}", "$top": STEP, "$skip": skip,
                                    "$select": FIELDS, "$orderby": "id"})
        r = fetch.get_json(f"{URL}?{q}", timeout=180)["NfipClaims"]
        rows += r
        skip += STEP
        if len(r) < STEP:
            break
    if len(rows) != expected:  # paging gap, or the dataset was updated while we were reading it
        raise RuntimeError(f"{year}: got {len(rows)} claims, server reports {expected}; re-run")
    pd.DataFrame(rows).to_parquet(out.with_suffix(".part"))  # atomic: a crash must not leave a 'done' file
    out.with_suffix(".part").replace(out)
    return f"[{year}] {len(rows)} claims"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2000, 2023])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(one_year, range(a.years[0], a.years[1] + 1)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
