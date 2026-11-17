"""Upload a GeoJSON / property table from gee_export to an Earth Engine table asset without Cloud Storage (docs/GEE_PLAN.md, step 2).

Builds the FeatureCollection client-side from a GeoJSON file, in parts of at most about 4 MB (the client request limit is about 10 MB), exports each
with Export.table.toAsset, merges the parts server side and deletes them; waits for the tasks and checks the row count.
Properties keep their JSON types, so GEOID stays a string (leading zeros survive).

Inputs : $WXRISK_DATA/gee_export/<name>.geojson
Output : projects/<EE_PROJECT>/assets/wxrisk/<name>  (EE_PROJECT env var, default impact-foreacsting-flooding)
Run    : python scripts/111_gee_upload_table.py county_geometry [--overwrite]
Needs  : uv sync --extra gee ; earthengine authenticate
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

CHUNK_BYTES = 4_000_000
PROJECT = os.environ.get("EE_PROJECT", "impact-foreacsting-flooding")


def main() -> None:
    import ee

    name = sys.argv[1]
    overwrite = "--overwrite" in sys.argv
    ee.Initialize(project=PROJECT)
    asset = f"projects/{PROJECT}/assets/wxrisk/{name}"
    try:
        ee.data.getAsset(asset)
        if not overwrite:
            sys.exit(f"{asset} exists; pass --overwrite to replace it")
        ee.data.deleteAsset(asset)
    except ee.EEException:
        pass

    gj = json.loads((wb2.cache_path("gee_export") / f"{name}.geojson").read_text())
    feats = gj["features"]
    n = len(feats)
    size = len(json.dumps(feats, separators=(",", ":")))
    k = max(1, -(-size // CHUNK_BYTES))  # parts, so that every client request stays well below the ~10 MB limit
    per = -(-n // k)
    parts = [feats[i:i + per] for i in range(0, n, per)]
    if len(parts) == 1:
        run(ee, to_fc(ee, parts[0]), asset, name)
    else:
        ids = [f"{asset}__part{i}" for i in range(len(parts))]
        for pid, part in zip(ids, parts, strict=True):
            run(ee, to_fc(ee, part), pid, f"{name}_part")
        merged = ee.FeatureCollection([ee.FeatureCollection(i) for i in ids]).flatten()  # server side, no payload
        run(ee, merged, asset, name)
        for pid in ids:
            ee.data.deleteAsset(pid)
    got = ee.FeatureCollection(asset).size().getInfo()
    assert got == n, f"asset has {got} rows, expected {n}"
    print(f"ok: {asset} has {got} rows ({len(parts)} upload part(s))")


def to_fc(ee, feats: list[dict]):
    """FeatureCollection from GeoJSON features; a null geometry (attribute-only table) becomes a geometry-less ee.Feature."""
    return ee.FeatureCollection([ee.Feature(ee.Geometry(f["geometry"]) if f["geometry"] else None, f["properties"]) for f in feats])


def run(ee, fc, asset: str, desc: str) -> None:
    task = ee.batch.Export.table.toAsset(fc, description=f"wxrisk_{desc}"[:100], assetId=asset)
    task.start()
    print(f"started {task.id} -> {asset}", flush=True)
    while task.status()["state"] in ("READY", "RUNNING"):
        time.sleep(5)
    st = task.status()
    if st["state"] != "COMPLETED":
        sys.exit(f"task ended {st['state']}: {st.get('error_message')}")


if __name__ == "__main__":
    main()
