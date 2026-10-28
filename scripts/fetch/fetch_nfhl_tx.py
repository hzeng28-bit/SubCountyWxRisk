"""FEMA NFHL flood hazard zones for Texas, one GeoJSON file per county, from FEMA's public ArcGIS service.

The state bulk package on msc.fema.gov needs a browser session, so the zones are paged out of
hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28 (Flood Hazard Zones, max 2000 records per request).
Kept: every zone except plain minimal-hazard X, i.e. the SFHA (A, AE, AO, AH, AR, A99, V, VE) and shaded X (0.2 % annual chance,
levee-protected X); minimal-hazard X is the complement and is not needed for area shares.
Geometry is generalised on the server (maxAllowableOffset, in degrees) to about the 38 m of the DEM used for HAND.

Output : $WXRISK_DATA/labels_extra/flood/nfhl/nfhl_<county fips>.geojson  (idempotent: existing non-empty files are skipped)
Run    : python scripts/fetch/fetch_nfhl_tx.py [--county 48201] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from wxrisk import fetch, wb2  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402

URL = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query"
OUT = wb2.cache_path("labels_extra") / "flood" / "nfhl"
PAGE = 1000
OFFSET_DEG = 0.0003  # ~30 m
FIELDS = "FLD_ZONE,ZONE_SUBTY,SFHA_TF,STATIC_BFE,DFIRM_ID"


def query(county: str, offset: int) -> dict:
    p = {"where": f"DFIRM_ID LIKE '48{county[2:]}%' AND NOT (FLD_ZONE = 'X' AND ZONE_SUBTY IS NULL)",
         "outFields": FIELDS, "outSR": "4326", "f": "geojson", "returnGeometry": "true",
         "maxAllowableOffset": str(OFFSET_DEG), "resultOffset": str(offset), "resultRecordCount": str(PAGE),
         "orderByFields": "OBJECTID"}
    return fetch.get_json(URL + "?" + urllib.parse.urlencode(p), timeout=300)


def one(county: str) -> tuple[str, int]:
    dest = OUT / f"nfhl_{county}.geojson"
    if dest.exists() and dest.stat().st_size > 0:
        return county, -1
    feats, off = [], 0
    while True:
        j = query(county, off)
        if "error" in j:
            raise RuntimeError(f"{county}: {j['error']}")
        got = j.get("features", [])
        feats += got
        if len(got) < PAGE and not j.get("exceededTransferLimit"):
            break
        off += len(got)
    tmp = dest.with_suffix(".part")
    tmp.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    tmp.replace(dest)
    return county, len(feats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--county", nargs="*")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    counties = a.county or sorted(fp.bg_geometry().cty.unique())
    with ThreadPoolExecutor(a.workers) as ex:
        for c, n in ex.map(one, counties):
            print(c, "skipped" if n < 0 else f"{n} zones", flush=True)


if __name__ == "__main__":
    main()
