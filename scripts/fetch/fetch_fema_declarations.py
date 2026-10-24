"""FEMA disaster declarations (OpenFEMA DisasterDeclarationsSummaries v2) for one state, all incident types, 2000+.

One row per declaration x county. Used as the "known flood event" definition (scripts 35-36), because the NFIP
`floodEvent` field covers only ~81% of claims and ~72 events. Output:
$WXRISK_DATA/labels_extra/flood/fema_declarations_<state>.parquet (skipped if it exists)

    python scripts/fetch/fetch_fema_declarations.py            # TX
    python scripts/fetch/fetch_fema_declarations.py --state LA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_fema_conmaz import OUT, page_all, save  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="TX")
    a = ap.parse_args()
    name = f"fema_declarations_{a.state.lower()}.parquet"
    if (OUT / name).exists():
        print("exists:", OUT / name)
        return
    dd = page_all("DisasterDeclarationsSummaries", "v2",
                  f"state eq '{a.state}' and declarationDate ge '2000-01-01T00:00:00.000z'",
                  "DisasterDeclarationsSummaries")
    print(f"{len(dd):,} declaration-county rows;", dd.incidentType.value_counts().head(8).to_dict())
    save(dd, name)


if __name__ == "__main__":
    main()
