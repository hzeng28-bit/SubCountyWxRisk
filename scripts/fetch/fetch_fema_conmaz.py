"""CO/NM/AZ NFIP claims (OpenFEMA v3) and Flood disaster declarations (v2), all fields, one parquet each.

Superseded for claims by scripts/20_fetch_nfip_claims.py (national, 2000+, selected fields); kept because
scripts/15 and 18 read nfip_claims_conmaz.parquet. Output goes to $WXRISK_DATA/labels_extra/flood/.

    python scripts/fetch/fetch_fema_conmaz.py
"""

from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

OUT = wb2.cache_path("labels_extra") / "flood"


def page_all(entity: str, ver: str, flt: str, key: str, step: int = 1000) -> pd.DataFrame:
    rows, skip = [], 0
    while True:
        q = urllib.parse.urlencode({"$filter": flt, "$top": step, "$skip": skip})
        r = fetch.get_json(f"https://www.fema.gov/api/open/{ver}/{entity}?{q}", timeout=120)[key]
        rows += r
        skip += step
        if len(r) < step:
            return pd.DataFrame(rows)


def save(df: pd.DataFrame, name: str) -> None:
    tmp = OUT / f"{name}.part"
    df.to_parquet(tmp)
    tmp.replace(OUT / name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / "nfip_claims_conmaz.parquet").exists():
        parts = []
        for s in ("CO", "NM", "AZ"):
            d = page_all("NfipClaims", "v3", f"state eq '{s}'", "NfipClaims")
            print(s, len(d), flush=True)
            parts.append(d)
        save(pd.concat(parts), "nfip_claims_conmaz.parquet")
    if not (OUT / "fema_flood_declarations.parquet").exists():
        dd = page_all("DisasterDeclarationsSummaries", "v2",
                      "(state eq 'CO' or state eq 'NM' or state eq 'AZ') and incidentType eq 'Flood'",
                      "DisasterDeclarationsSummaries")
        print("declarations", len(dd))
        save(dd, "fema_flood_declarations.parquet")


if __name__ == "__main__":
    main()
