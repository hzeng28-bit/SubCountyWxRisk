"""OpenFEMA field dictionary and dataset description for NfipClaims and NfipPolicies (v3).

The provider's own statement of what each field means, which codes are documented and which values are placeholders.
It is the evidence for the "provider-specified" rules in the cleaning register (docs/CLEANING_REGISTER.md).

    python scripts/fetch/fetch_openfema_dictionary.py
Output: $WXRISK_DATA/labels_extra/flood/openfema_dictionary/{NfipClaims,NfipPolicies}.fields.json and .dataset.json
"""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wxrisk import fetch, wb2  # noqa: E402

OUT = wb2.cache_path("labels_extra") / "flood" / "openfema_dictionary"
BASE = "https://www.fema.gov/api/open/v1/"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("NfipClaims", "NfipPolicies"):
        q = urllib.parse.urlencode({"$filter": f"openFemaDataSet eq '{name}'", "$top": 1000})
        fields = fetch.get_json(BASE + "OpenFemaDataSetFields?" + q)["OpenFemaDataSetFields"]
        q = urllib.parse.urlencode({"$filter": f"name eq '{name}'"})
        ds = fetch.get_json(BASE + "OpenFemaDataSets?" + q)["OpenFemaDataSets"]
        # lastRefresh / hash change with every upstream refresh; they stay in the file as the retrieval record
        (OUT / f"{name}.fields.json").write_text(json.dumps(fields, indent=1, sort_keys=True) + "\n")
        (OUT / f"{name}.dataset.json").write_text(json.dumps(ds, indent=1, sort_keys=True) + "\n")
        print(f"{name}: {len(fields)} fields, {len(ds)} dataset record(s)")


if __name__ == "__main__":
    main()
