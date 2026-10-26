"""GEFSv12 reforecast precipitation over Texas for the flood forecast-mode pilot (stage 16).

One 00Z initialisation per needed day (the day before an episode starts), all members the bucket holds for that day
(5; the 11 of Wednesdays are cut to 5), 6-hour APCP accumulations for leads 36-84 h (the two gridMET days S and S+1), cropped to a Texas box at 0.25 deg. Only the needed GRIB
messages are read (byte ranges from the .idx file), so a day costs ~3 MB per member instead of 24 MB.

Inputs : flood_pilot_table.parquet (episode start days), s3://noaa-gefs-retrospective (anonymous)
Outputs: $WXRISK_DATA/labels_extra/flood/gefs/apcp_<YYYYMMDD>.npy   float16 [5, 8, ny, nx] mm per 6 h;
         axis 1 = 6-h windows ending at lead 42, 48, ..., 84 h; gefs_meta.npz (lat, lon, lead_end, members per day)
Run    : python scripts/47_fetch_gefs_reforecast.py --dry-run
         python scripts/47_fetch_gefs_reforecast.py --limit 20      # prototype days
         python scripts/47_fetch_gefs_reforecast.py                 # everything, resumable (existing days are skipped)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch as dl  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

BUCKET = "noaa-gefs-retrospective"
OUT = wb2.cache_path("labels_extra") / "flood" / "gefs"
# gridMET day t = 12Z t .. 12Z t+1 (pooled correlation with the ensemble mean peaks at that window, 0.70 against 0.55-0.61 for
# the neighbouring shifts). From the 00Z run of the day before an episode starts (S-1), day S = leads 36-60 and day S+1 = 60-84.
LEAD_START = list(range(36, 79, 6))  # 8 six-hour windows [36,42] ... [78,84]
NMEM = 5  # control + p01..p04: the 11-member Wednesday runs are cut to the same 5 so every day has the same ensemble
BOX = (24.5, 37.5, -107.5, -92.5)  # lat0, lat1, lon0, lon1 around Texas
LAST_YEAR = 2019  # the reforecast ends there


def prefix(day: pd.Timestamp) -> str:
    return f"GEFSv12/reforecast/{day:%Y}/{day:%Y%m%d}00/"


def members(day: pd.Timestamp) -> list[str]:
    r = dl.s3_client().list_objects_v2(Bucket=BUCKET, Prefix=prefix(day), Delimiter="/")
    return sorted(x["Prefix"].split("/")[-2] for x in r.get("CommonPrefixes", []))[:NMEM]


def message_ranges(day: pd.Timestamp, mem: str) -> tuple[str, list[tuple[int, int]]]:
    key = f"{prefix(day)}{mem}/Days:1-10/apcp_sfc_{day:%Y%m%d}00_{mem}.grib2"
    rows = dl.s3_bytes(BUCKET, key + ".idx").decode().splitlines()
    off = [int(r.split(":")[1]) for r in rows]
    size = dl.probe_s3(BUCKET, key).size
    off.append(size)
    want, got = set(LEAD_START), {}
    for i, r in enumerate(rows):
        m = re.search(r":(\d+)-(\d+) hour acc fcst", r)
        a, b = int(m.group(1)), int(m.group(2))
        if b - a == 6 and a in want:
            got[a] = (off[i], off[i + 1] - 1)
    if set(got) != want:
        raise ValueError(f"{key}: windows {sorted(want - set(got))} missing")
    return key, [got[a] for a in LEAD_START]


def read_member(day: pd.Timestamp, mem: str):
    import pygrib

    key, rng = message_ranges(day, mem)
    la0, la1, lo0, lo1 = BOX
    out, lat, lon = [], None, None
    for a, b in rng:
        raw = dl.s3_range(BUCKET, key, a, b)
        with tempfile.NamedTemporaryFile(suffix=".grib2") as f:
            f.write(raw)
            f.flush()
            g = pygrib.open(f.name).message(1)
            v, la, lo = g.data(lat1=la0, lat2=la1, lon1=lo0 + 360, lon2=lo1 + 360)
        out.append(np.ma.filled(v, np.nan).astype(np.float32))
        if lat is None:
            lat, lon = la[:, 0], lo[0] - 360
    return np.stack(out), lat, lon


def fetch_day(day: pd.Timestamp):
    dest = OUT / f"apcp_{day:%Y%m%d}.npy"
    if dest.exists():
        return day, None, "skip"
    mems = members(day)
    if not mems:
        return day, None, "no members"
    arrs = []
    for m in mems:
        a, lat, lon = read_member(day, m)
        arrs.append(a)
    ny, nx = arrs[0].shape[1:]
    full = np.full((NMEM, len(LEAD_START), ny, nx), np.nan, np.float16)
    full[: len(arrs)] = np.stack(arrs).astype(np.float16)
    part = dest.with_name(dest.name + ".part.npy")
    np.save(part, full)
    part.replace(dest)
    return day, (lat, lon, len(mems)), "ok"


def init_days(table: Path) -> list[pd.Timestamp]:
    s = pd.read_parquet(table, columns=["start"])["start"].drop_duplicates()
    d = (s - pd.Timedelta(1, "D")).dt.normalize().drop_duplicates().sort_values()
    return [t for t in d if 2010 <= t.year <= LAST_YEAR]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(fp.PILOT / "flood_pilot_table.parquet"))
    ap.add_argument("--limit", type=int, default=0, help="first N days (prototype)")
    ap.add_argument("--stride", type=int, default=1, help="every k-th day (spread a prototype over the years)")
    ap.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"), help="only initialisation days in these years (Slurm array shards)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 6)))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    days = init_days(Path(a.table))[:: a.stride]
    if a.years:
        days = [d for d in days if a.years[0] <= d.year <= a.years[1]]
    if a.limit:
        days = days[: a.limit]
    print(f"{len(days)} initialisation days, {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.dry_run:
        d = days[len(days) // 2]
        mem = members(d)
        key, rng = message_ranges(d, mem[0])
        per = sum(b - a_ + 1 for a_, b in rng)
        print(f"example {d:%Y-%m-%d}: {len(mem)} members, {per / 1e6:.1f} MB per member; "
              f"~{per * 5 * len(days) / 1e9:.0f} GB transferred for the whole list (5 members per day)")
        return
    meta, done, fail = {}, 0, []
    # processes, not threads: pygrib holds the GIL while it unpacks a global 0.25 deg message
    with ProcessPoolExecutor(a.workers) as ex:
        for day, info, status in ex.map(_safe, days, chunksize=4):
            done += 1
            if status not in ("ok", "skip"):
                fail.append((f"{day:%Y%m%d}", status))
            if info:
                meta[f"{day:%Y%m%d}"] = info[2]
                meta["_lat"], meta["_lon"] = info[0], info[1]
            if done % 50 == 0:
                print(f"  {done}/{len(days)}  failures {len(fail)}", flush=True)
    mp = OUT / "gefs_meta.npz"
    old = dict(np.load(mp, allow_pickle=True)) if mp.exists() else {}
    nm = {k: v for k, v in meta.items() if not k.startswith("_")}
    old_n = old.get("members", np.array({}, dtype=object)).item() if "members" in old else {}
    old_n.update(nm)
    lat = meta.get("_lat", old.get("lat"))
    lon = meta.get("_lon", old.get("lon"))
    tmp = mp.with_name(f"gefs_meta.{os.getpid()}.tmp.npz")  # array shards finish at different times: replace atomically
    np.savez(tmp, lat=lat, lon=lon, lead_end=np.array(LEAD_START) + 6, members=np.array(old_n, dtype=object))
    tmp.replace(mp)
    print(f"done; {len(fail)} failures", flush=True)
    for d, s in fail[:20]:
        print("  ", d, s)


def _safe(day):
    try:
        return fetch_day(day)
    except Exception as e:  # noqa: BLE001 - one bad day must not stop a long batch
        return day, None, f"{type(e).__name__}: {str(e)[:80]}"


if __name__ == "__main__":
    main()
