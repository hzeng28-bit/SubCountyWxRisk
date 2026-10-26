"""Hourly MRMS MultiSensor QPE (Pass2) crops around heavy-rain Texas episodes, block-averaged to ~4 km.

Testbed for the sparse-to-dense / motion-compensated reconstruction test (script 40). gridMET is daily, which
is too coarse for storm motion; MRMS hourly starts 2020-10, so only 2021-2023 episodes are usable.

Inputs : flood_pilot_table.parquet (episode dates, county), MRMS on s3://noaa-mrms-pds (anonymous)
Outputs: $WXRISK_DATA/labels_extra/flood/mrms_hourly/ev_<ep>.npy (R[t, y, x] mm/h, memory-mappable)
         and ev_<ep>_meta.npz (lat, lon, times, cty)
Run    : python scripts/39_mrms_hourly_events.py --n-events 8
"""

from __future__ import annotations

import argparse
import gzip
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch as dl  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

BUCKET = "noaa-mrms-pds"
PROD = "MultiSensor_QPE_01H_Pass2_00.00"
OUT = wb2.cache_path("labels_extra") / "flood" / "mrms_hourly"
HALF_LAT, HALF_LON, BLOCK = 2.7, 3.2, 4  # ~600 km box; 0.01 deg cells averaged 4x4


def hour_key(t: pd.Timestamp) -> str:
    return f"CONUS/{PROD}/{t:%Y%m%d}/MRMS_{PROD}_{t:%Y%m%d-%H%M%S}.grib2.gz"


def read_crop(t: pd.Timestamp, box: tuple[float, float, float, float]):
    import tempfile

    import pygrib

    try:
        raw = dl.s3_bytes(BUCKET, hour_key(t))
    except Exception:  # noqa: BLE001 - a missing hour is left as NaN
        return None
    with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
        f.write(gzip.decompress(raw))
        f.flush()
        g = pygrib.open(f.name).message(1)
        la0, la1, lo0, lo1 = box
        v, lat, lon = g.data(lat1=la0, lat2=la1, lon1=lo0 + 360, lon2=lo1 + 360)
    v = np.ma.filled(v, np.nan).astype(np.float32)
    v[v < 0] = np.nan  # -3 / -999 are "no coverage" flags, not dry
    ny, nx = (v.shape[0] // BLOCK) * BLOCK, (v.shape[1] // BLOCK) * BLOCK
    v = v[:ny, :nx].reshape(ny // BLOCK, BLOCK, nx // BLOCK, BLOCK)
    return np.nanmean(v, axis=(1, 3)), lat[:ny:BLOCK, 0] - 0, lon[0, :nx:BLOCK] - 360


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-events", type=int, default=8)
    ap.add_argument("--max-days", type=int, default=3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "cty", "start", "end", "y"])
    e = t.groupby("ep").agg(cty=("cty", "first"), s=("start", "first"), e=("end", "first"), y=("y", "sum"))
    e = e[e.s >= "2020-10-15"].sort_values("y", ascending=False)
    cc = fp.county_table().set_index("cty")
    chosen, used = [], []
    for ep, r in e.iterrows():  # keep events apart in time/space so crops are not duplicates
        c = cc.loc[r.cty]
        if any(abs((r.s - s).days) < 3 and abs(c.lat - la) < 3 for s, la in used):
            continue
        chosen.append((ep, r.cty, r.s, r.e, c.lon, c.lat))
        used.append((r.s, c.lat))
        if len(chosen) == a.n_events:
            break

    for ep, cty, s, en, lon, lat in chosen:
        dest = OUT / f"ev_{ep}.npy"
        if dest.exists():
            continue
        end = min(en, s + pd.Timedelta(days=a.max_days - 1))
        hours = pd.date_range(s, end + pd.Timedelta(hours=24), freq="h")
        box = (lat - HALF_LAT, lat + HALF_LAT, lon - HALF_LON, lon + HALF_LON)
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda h: read_crop(h, box), hours))
        ok = [r for r in res if r is not None]
        if not ok:
            print(f"ep {ep}: no data", flush=True)
            continue
        shp = ok[0][0].shape
        R = np.stack([r[0] if r is not None and r[0].shape == shp else np.full(shp, np.nan, np.float32)
                      for r in res])
        np.savez(dest.with_name(f"ev_{ep}_meta.npz"), lat=ok[0][1], lon=ok[0][2], times=hours.values, cty=cty)
        np.save(dest, R)  # last, so a present .npy means the event is complete
        print(f"ep {ep} cty {cty} {s:%Y-%m-%d}: {len(ok)}/{len(hours)} h, grid {shp}, "
              f"max {np.nanmax(R):.1f} mm/h", flush=True)


if __name__ == "__main__":
    main()
