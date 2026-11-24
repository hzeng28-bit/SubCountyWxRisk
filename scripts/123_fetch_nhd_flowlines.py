"""Texas NHDPlus flowlines (stream order >= 2) from the USGS National Map service, for the river-network overlay (stage 68).

Service : hydro.nationalmap.gov nhd/MapServer layer 4 (Flowline - Small Scale, NHDPlus attributes: StreamOrde, TotDASqKM).
Pages of 2000 features are cached one JSON file each (resume-safe); geometry generalised to ~200 m (maxAllowableOffset).
Outputs: $WXRISK_DATA/gis/nhd_pages/page_*.json, $WXRISK_DATA/gis/tx_flowlines.npz (arrays: line vertices + order + drainage area)
Run    : uv run python scripts/123_fetch_nhd_flowlines.py
"""
from __future__ import annotations

import json
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import fetch, wb2  # noqa: E402

BASE = "https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/4/query"
ENV = "-106.7,25.8,-93.4,36.6"
PAGE = 2000
OUT = wb2.cache_path(".") / "gis"
PAGES = OUT / "nhd_pages"


def q(**kw):
    p = dict(where="StreamOrde>=2", geometry=ENV, geometryType="esriGeometryEnvelope", inSR=4326,
             spatialRel="esriSpatialRelIntersects", f="json", outSR=4326)
    p.update(kw)
    return BASE + "?" + urllib.parse.urlencode(p)


def page(k):
    f = PAGES / f"page_{k:03d}.json"
    if f.exists() and f.stat().st_size > 0:
        return k
    url = q(outFields="OBJECTID,StreamOrde,TotDASqKM", returnGeometry="true", maxAllowableOffset=0.002,
            orderByFields="OBJECTID", resultOffset=k * PAGE, resultRecordCount=PAGE)
    b = fetch.get_bytes(url, timeout=120)
    j = json.loads(b)
    if "features" not in j:
        raise RuntimeError(f"page {k}: {str(j)[:200]}")
    tmp = f.with_suffix(".part")
    tmp.write_bytes(b)
    tmp.rename(f)
    return k


def main():
    PAGES.mkdir(parents=True, exist_ok=True)
    n = fetch.get_json(q(returnCountOnly="true"))["count"]
    npg = -(-n // PAGE)
    print(f"{n:,} flowlines, {npg} pages")
    with ThreadPoolExecutor(4) as ex:
        for k in ex.map(page, range(npg)):
            if k % 10 == 0:
                print("page", k, flush=True)
    pts, off, order, da = [], [0], [], []
    for k in range(npg):
        for ft in json.loads((PAGES / f"page_{k:03d}.json").read_text())["features"]:
            a = ft["attributes"]
            for path in ft["geometry"]["paths"]:
                arr = np.asarray(path, np.float32)[:, :2]
                pts.append(arr)
                off.append(off[-1] + len(arr))
                order.append(a["StreamOrde"] or 0)
                da.append(a["TotDASqKM"] or np.nan)
    np.savez(OUT / "tx_flowlines.npz", xy=np.concatenate(pts), off=np.array(off, np.int64),
             order=np.array(order, np.int16), da=np.array(da, np.float32))
    print(f"saved {len(order):,} lines, {off[-1]:,} vertices")


if __name__ == "__main__":
    main()
