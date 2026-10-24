"""Extract Texas rows of the OpenFEMA NFIP policy bulk file (74.7M rows) into a small parquet.

Kept: block-group code, ZIP, policy term dates, count, flood zone and a few building and coverage
fields. Output: $WXRISK_DATA/labels_extra/flood/pilot/policies_tx.parquet

    python scripts/24_extract_tx_policies.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

SRC = wb2.cache_path("labels_extra") / "flood" / "nfip_policies_v3.parquet"
OUT = wb2.cache_path("labels_extra") / "flood" / "pilot" / "policies_tx.parquet"
COLS = ["propertyState", "censusGeoid", "reportedZipCode", "policyEffectiveDate", "policyTerminationDate",
        "cancellationDateOfFloodPolicy", "policyCount", "ratedFloodZone", "floodZoneCurrent", "occupancyType",
        "totalBuildingInsuranceCoverage", "buildingReplacementCost", "elevationDifference",
        "postFIRMConstructionIndicator", "primaryResidenceIndicator", "originalConstructionDate",
        "mandatoryPurchaseFlag", "policyTermIndicator", "foundationType", "numberOfFloorsInInsuredBuilding"]


def check_bulk_file() -> None:
    """The bulk file should hold as many rows as the OpenFEMA catalog advertises (truncated download otherwise)."""
    rows = pq.ParquetFile(SRC).metadata.num_rows
    try:
        want = fetch.openfema_record_count("NfipPolicies", 3)
    except Exception as e:  # noqa: BLE001 - offline is fine, this is only a sanity check
        print(f"[skip] could not read the OpenFEMA record count: {e}")
        return
    if rows == want:
        print(f"bulk file rows {rows:,} == OpenFEMA recordCount")
    elif rows < want:
        print(f"WARNING bulk file has {rows:,} rows, the catalog now says {want:,}: file is older or truncated")
    else:
        print(f"WARNING bulk file has {rows:,} rows, MORE than the catalog's {want:,}")


def main() -> None:
    check_bulk_file()
    dset = ds.dataset(SRC, format="parquet")
    sc = dset.scanner(columns=COLS, filter=ds.field("propertyState") == "TX", batch_size=500_000)
    tmp = OUT.with_suffix(".part")  # atomic: a crash must not leave a partial 'done' file
    w, n = None, 0
    for b in sc.to_batches():
        t = pa.Table.from_batches([b])
        w = w or pq.ParquetWriter(tmp, t.schema)
        w.write_table(t)
        n += t.num_rows
        print(f"{n:,} Texas rows", flush=True) if n % 2_000_000 < 500_000 else None
    w.close()
    tmp.replace(OUT)
    print(f"done: {n:,} Texas policy records -> {OUT}")


if __name__ == "__main__":
    main()
