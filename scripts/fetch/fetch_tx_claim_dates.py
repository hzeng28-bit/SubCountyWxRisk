"""Texas NFIP claims with the claim open date and first payment dates (scenario report: how late the insurer's own claim stream starts).

`scripts/20` keeps the loss fields but not `openDate`; the claim open date is the first moment an insurer holds a claim record. Same OpenFEMA v3 NfipClaims entity, paged
with the server row count as the check; Texas, loss years 2017-2023, all causes (filter by `causeOfDamage` downstream).
Output : $WXRISK_DATA/labels_extra/flood/nfip_tx_claim_dates.parquet  (id, dateOfLoss, openDate, mostRecentPaymentDate, causeOfDamage, floodEvent, countyCode, censusGeoid, amountPaidOnBuildingClaim)
Run    : python scripts/fetch/fetch_tx_claim_dates.py
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

URL = "https://www.fema.gov/api/open/v3/NfipClaims"
FIELDS = "id,dateOfLoss,openDate,mostRecentPaymentDate,causeOfDamage,floodEvent,countyCode,censusGeoid,amountPaidOnBuildingClaim,asOfDate"
STEP = 10000
OUT = wb2.cache_path("labels_extra") / "flood" / "nfip_tx_claim_dates.parquet"


def main() -> None:
    if OUT.exists():
        print(f"[skip] {OUT}")
        return
    flt = "state eq 'TX' and yearOfLoss ge 2017 and yearOfLoss le 2023"
    expected = fetch.openfema_count("NfipClaims", "v3", flt)
    rows, skip = [], 0
    while True:
        q = urllib.parse.urlencode({"$filter": flt, "$top": STEP, "$skip": skip, "$select": FIELDS, "$orderby": "id"})
        r = fetch.get_json(f"{URL}?{q}", timeout=180)["NfipClaims"]
        rows += r
        skip += STEP
        print(f"  {len(rows)}/{expected}", flush=True)
        if len(r) < STEP:
            break
    if len(rows) != expected:
        raise RuntimeError(f"got {len(rows)} claims, server reports {expected}; re-run")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(OUT.with_suffix(".part"))
    OUT.with_suffix(".part").replace(OUT)
    print(f"wrote {OUT} ({len(df)} claims)")


if __name__ == "__main__":
    main()
