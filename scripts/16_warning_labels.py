"""County-day NWS warning labels from IEM storm-based warning polygons (2002 onward).

A warning counts for a county-day if its polygon intersects the county polygon; the day is the
issue time in local standard time (UTC-7), the same convention as Storm Events. Columns:
n_sv severe thunderstorm, n_to tornado, n_ff flash flood warnings; Marine is ignored.

Output: $WXRISK_DATA/warnings_county_day.parquet  (cty, date, n_sv, n_to, n_ff)

    python scripts/16_warning_labels.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import shapefile
from shapely.geometry import shape
from shapely.strtree import STRtree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

EXTRA = wb2.cache_path("labels_extra")
BOX = (-117.0, 30.5, -101.0, 42.5)  # lon_min, lat_min, lon_max, lat_max around CO/NM/AZ
PHEN = {"SV": "n_sv", "TO": "n_to", "FF": "n_ff"}


def counties():
    z = zipfile.ZipFile(EXTRA / "census" / "cb_2020_us_county_20m.zip")
    rd = shapefile.Reader(shp=io.BytesIO(z.read("cb_2020_us_county_20m.shp")),
                          shx=io.BytesIO(z.read("cb_2020_us_county_20m.shx")),
                          dbf=io.BytesIO(z.read("cb_2020_us_county_20m.dbf")))
    names = [f[0] for f in rd.fields[1:]]
    geoms, ids = [], []
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        if rec["STATEFP"] in ("04", "08", "35"):
            geoms.append(shape(sr.shape.__geo_interface__))
            ids.append(rec["GEOID"])
    return geoms, ids


def year_rows(year: int, geoms, ids, tree) -> list[tuple[str, pd.Timestamp, str]]:
    z = zipfile.ZipFile(EXTRA / "iem" / f"{year}_tsmf_sbw.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                          dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    out = []
    for sr in rd.iterShapeRecords():
        x0, y0, x1, y1 = sr.shape.bbox
        if x1 < BOX[0] or x0 > BOX[2] or y1 < BOX[1] or y0 > BOX[3]:
            continue
        rec = dict(zip(names, sr.record, strict=True))
        if rec["PHENOM"] not in PHEN:
            continue
        issued = pd.Timestamp(rec["ISSUED"]) if not isinstance(rec["ISSUED"], str) else pd.Timestamp(
            rec["ISSUED"])
        day = (issued - pd.Timedelta(hours=7)).normalize()
        poly = shape(sr.shape.__geo_interface__)
        for i in tree.query(poly, predicate="intersects"):
            out.append((ids[i], day, PHEN[rec["PHENOM"]]))
    return out


def main() -> None:
    geoms, ids = counties()
    tree = STRtree(geoms)
    print(f"{len(ids)} counties", flush=True)
    rows = []
    for y in range(2002, 2023):
        r = year_rows(y, geoms, ids, tree)
        rows += r
        print(f"[{y}] {len(r)} county-warning hits", flush=True)
    d = pd.DataFrame(rows, columns=["cty", "date", "kind"])
    d = d[d.date.dt.month.isin([5, 6, 7, 8, 9])]
    out = d.groupby(["cty", "date", "kind"]).size().unstack(fill_value=0).reset_index()
    for c in PHEN.values():
        if c not in out:
            out[c] = 0
    out.to_parquet(wb2.cache_path("warnings_county_day.parquet"))
    print(f"{len(out)} county-days with a warning; totals", out[list(PHEN.values())].sum().to_dict())


if __name__ == "__main__":
    main()
