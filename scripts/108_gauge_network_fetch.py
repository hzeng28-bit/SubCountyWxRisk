"""Texas streamgage discharge and river network for a flow-conservation test (stage 59, fetch step).

Mentor suggestion: use river discharge at gauges as a fine-grained outcome. Here the gauges are linked by the river network:
water that passes an upstream gauge must reach the next gauge downstream unless it is lost (overbank spill, levee break,
diversion, reservoir storage) or gained (local inflow).

Fetches (all anonymous or with the dataretrieval key, see script 50):
  daily mean discharge 2010-2023 for the 896 gauges of stage 18        -> usgs_tx/tx_daily_q.parquet
  NLDI basin polygons (simplified), cached per gauge                    -> usgs_tx/nldi_cache/, tx_basins.geojson

Inputs : usgs_tx/tx_gauges_q.parquet (script 50)
Outputs: $WXRISK_DATA/labels_extra/flood/usgs_tx/
Run    : python scripts/108_gauge_network_fetch.py [daily|nldi|all]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch as dl  # noqa: E402
from wxrisk import wb2  # noqa: E402

D = wb2.cache_path("labels_extra") / "flood" / "usgs_tx"
NLDI = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
HDR = {"User-Agent": "Mozilla/5.0"}


def daily() -> None:
    from dataretrieval import waterdata

    out = D / "tx_daily_q.parquet"
    ids = list(pd.read_parquet(D / "tx_gauges_q.parquet").id)
    rows = []
    for y in range(2010, 2024):
        for k in range(0, len(ids), 50):
            for attempt in range(4):
                try:
                    d, _ = waterdata.get_daily(monitoring_location_id=ids[k:k + 50], parameter_code="00060",
                                               statistic_id="00003", time=f"{y}-01-01/{y}-12-31", skip_geometry=True)
                    break
                except Exception as e:  # rate limit or a transient 5xx: back off and retry
                    print(f"  {y} {k}: {type(e).__name__}, retry", flush=True)
                    time.sleep(30 * (attempt + 1))
            else:
                raise RuntimeError(f"daily fetch failed for {y} chunk {k}")
            rows.append(d[["monitoring_location_id", "time", "value", "approval_status", "qualifier"]])
        print(f"{y} done, {sum(len(r) for r in rows):,} rows", flush=True)
    q = pd.concat(rows, ignore_index=True)
    q["value"] = pd.to_numeric(q["value"], errors="coerce")
    q["time"] = pd.to_datetime(q["time"]).dt.tz_localize(None).dt.normalize()
    q["qualifier"] = q["qualifier"].astype(str)
    q.to_parquet(out)
    print(f"{len(q):,} gauge-days at {q.monitoring_location_id.nunique()} gauges -> {out}")


def _get(url: str, cache: Path):
    """One cached JSON request. NLDI allows 800 requests an hour anonymously; a 429 carries Retry-After, wait it out."""
    import urllib.error

    if cache.exists():
        return json.loads(cache.read_text())
    for _ in range(40):
        try:
            j = json.loads(dl.get_bytes(url, headers=HDR, retries=1, timeout=60))
            cache.write_text(json.dumps(j))
            return j
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = int(e.headers.get("Retry-After", "300")) + 5
                print(f"  rate limited, waiting {wait} s", flush=True)
                time.sleep(wait)
            elif e.code in (404, 400):
                cache.write_text("null")  # site has no NLDI feature
                return None
            else:
                time.sleep(5)
        except Exception:
            time.sleep(5)
    return None


def nldi() -> None:
    """Basin polygon per gauge (one request each); upstream gauges are derived in script 109 by point-in-polygon, which
    halves the request count against the hourly cap."""
    ids = list(pd.read_parquet(D / "tx_gauges_q.parquet").id)
    C = D / "nldi_cache"
    C.mkdir(exist_ok=True)
    feats = []
    for k, s in enumerate(ids):
        j = _get(f"{NLDI}/{s}/basin?simplified=true", C / f"{s}_basin.json")
        if j and j.get("features"):
            f = j["features"][0]
            f["properties"] = {"site": s}
            feats.append(f)
        if k % 50 == 0:
            print(f"  {k}/{len(ids)} basins, {len(feats)} ok", flush=True)
    (D / "tx_basins.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    print(f"basins {len(feats)} of {len(ids)}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("daily", "all"):
        daily()
    if what in ("nldi", "all"):
        nldi()
