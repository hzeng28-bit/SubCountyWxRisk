"""County geometry for the Earth Engine display layer (docs/GEE_PLAN.md, step 1a).

Why counties: NFIP block-group codes mix 2010 and 2020 vintages (AGENTS.md trap 5), so the mentor asked for display at county level. The
county FIPS is the first 5 digits of any block-group GEOID and is stable across vintages in these states, so no crosswalk is needed to
aggregate (county score = sum of block-group scores, as scripts/94 does).

Reads the 2020 Census 1:20m cartographic county file already in the data dir (also used by scripts 16, 94, 99); no download.
Keeps TX (flood line) and LA, FL (flood pipelines, stages 44-45) and CO, NM, AZ (SCS3 line). Writes one GeoJSON in lon/lat (NAD83,
treated as WGS84: the offset is below the 1:20m generalisation) with properties GEOID, STATEFP, COUNTYFP, NAME, STUSPS, ALAND.
Self-checks: unique GEOIDs, valid geometries, expected county counts (TX 254, LA 64, FL 67, CO 64, NM 33, AZ 15).

Inputs : labels_extra/census/cb_2020_us_county_20m.zip
Output : $WXRISK_DATA/gee_export/county_geometry.geojson
Run    : python scripts/110_gee_county_geometry.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shapely.geometry import mapping, shape  # noqa: E402
from shapely.validation import explain_validity  # noqa: E402

from wxrisk import wb2  # noqa: E402

# state FIPS -> expected number of counties (20m file keeps all of them)
STATES = {"48": 254, "22": 64, "12": 67, "08": 64, "35": 33, "04": 15}
KEEP = ("GEOID", "STATEFP", "COUNTYFP", "NAME", "STUSPS", "ALAND")


def main() -> None:
    import shapefile

    z = zipfile.ZipFile(wb2.cache_path("labels_extra") / "census" / "cb_2020_us_county_20m.zip")
    stem = next(n for n in z.namelist() if n.endswith(".shp"))[:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                          dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]

    feats, counts = [], dict.fromkeys(STATES, 0)
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        if rec["STATEFP"] not in STATES:
            continue
        geom = shape(sr.shape.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)  # fixes self-touching rings; checked below
            assert geom.is_valid, f"{rec['GEOID']}: {explain_validity(geom)}"
        counts[rec["STATEFP"]] += 1
        feats.append({"type": "Feature", "geometry": mapping(geom),
                      "properties": {k: rec[k] for k in KEEP}})

    bad = {s: (counts[s], n) for s, n in STATES.items() if counts[s] != n}
    assert not bad, f"county counts differ from expected (got, expected): {bad}"
    ids = [f["properties"]["GEOID"] for f in feats]
    assert len(ids) == len(set(ids)), "duplicate GEOID"

    out = wb2.cache_path("gee_export") / "county_geometry.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":")))
    print(f"{len(feats)} counties {counts} -> {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
