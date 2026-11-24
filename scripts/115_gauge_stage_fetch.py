"""Daily max discharge, daily max / mean gage height and USGS site type for the small Texas gauges (stage 61, fetch step).

Why: script 114 used daily MEAN discharge and screened regulation by name. Here we add the daily MAX of discharge (00060)
and of gage height (00065), and the official site type, so the outcomes can be compared against claims.
Gauges: all with drainage area <= 250 mi2 in tx_gauges_q.
Outputs: usgs_tx/tx_small_daily_stats.parquet (id, time, param, stat, value), usgs_tx/tx_site_types.parquet
Run    : python scripts/115_gauge_stage_fetch.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataretrieval import waterdata  # noqa: E402

from wxrisk import wb2  # noqa: E402

D = wb2.cache_path("labels_extra") / "flood" / "usgs_tx"


def retry(fn, **kw):
    for a in range(5):
        try:
            return fn(**kw)[0]
        except Exception as e:  # rate limit / transient
            print("  retry", type(e).__name__, flush=True)
            time.sleep(30 * (a + 1))
    raise RuntimeError("fetch failed")


def main():
    g = pd.read_parquet(D / "tx_gauges_q.parquet")
    ids = list(g[g.drainage_area.le(250)].id)
    st = retry(waterdata.get_monitoring_locations, monitoring_location_id=ids, skip_geometry=True,
               properties=["monitoring_location_id", "site_type_code", "site_type", "construction_date"])
    st.to_parquet(D / "tx_site_types.parquet")
    print(st.site_type.value_counts().to_string())
    rows = []
    for par, stat in (("00060", "00001"), ("00065", "00001"), ("00065", "00003")):
        for y in range(2010, 2024):
            for k in range(0, len(ids), 50):
                d = retry(waterdata.get_daily, monitoring_location_id=ids[k:k + 50], parameter_code=par,
                          statistic_id=stat, time=f"{y}-01-01/{y}-12-31", skip_geometry=True)
                if len(d):
                    d = d[["monitoring_location_id", "time", "value"]].assign(param=par, stat=stat)
                    rows.append(d)
            print(par, stat, y, sum(len(r) for r in rows), flush=True)
    o = pd.concat(rows, ignore_index=True)
    o["value"] = pd.to_numeric(o.value, errors="coerce")
    o["time"] = pd.to_datetime(o.time).dt.tz_localize(None).dt.normalize()
    o.to_parquet(D / "tx_small_daily_stats.parquet")
    print(o.groupby(["param", "stat"]).monitoring_location_id.nunique())


if __name__ == "__main__":
    main()
