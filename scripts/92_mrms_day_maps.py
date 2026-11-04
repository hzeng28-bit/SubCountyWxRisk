"""Hourly MRMS rain reduced to day products on the 4 km E1 grid (RESULTS stage 52): the radar-side channels for the gridMET -> MRMS fine-tuning.

Source: dynamical.org Icechunk repo `noaa-mrms-conus-analysis-hourly` (CC-BY-4.0, anonymous S3, us-west-2; `precipitation_surface` = gauge-corrected QPE before 2020-10).
It is NaN before 2015-03 and has holes of a few percent of hours afterwards, so the maps start on 2015-03-01 (E1 day index of that date in `mrms4k_days.npy`).
For every E1 day S (script 80: the 4 km gridMET peak / sum over S and S+1) the 48 hourly 1 km rates from S 00 UTC + `offset` h are reduced onto the E1 4 km cells
(mean and maximum of the 1 km pixels whose centres lie in the cell):
  0 sum48        mm over the 48 h (missing hours count as 0)
  1 hmax         largest hourly rate of the cell mean, mm/h
  2 hmax_1km     largest hourly rate of any 1 km pixel of the cell, mm/h
  3 hours_ge1    share of the 48 h with cell-mean rate >= 1 mm/h
  4 hours_ge5    share with >= 5 mm/h
  5 valid        share of the 48 h with data
`check` picks the UTC offset (hours) whose 48 h sum best matches the gridMET 2-day sum on the heaviest days of a sub-box; `build` writes the maps (resumable by 27-day group).
This script does not import wxrisk (the dynamical.org stack is not a project dependency): run it in an isolated environment, e.g.
  uv run --no-project --with "icechunk>=2" --with "zarr>=3.2" --with xarray --with numpy --with pandas python -I scripts/92_mrms_day_maps.py check
Inputs : $WXRISK_DATA/cnn/static.npz, obs4k_day_maps.npy (script 80)
Output : $WXRISK_DATA/cnn/mrms4k_day_maps.npy float16 [day, 6, 260, 320], mrms4k_days.npy, mrms4k_offset.txt
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import icechunk
import numpy as np
import pandas as pd
import xarray as xr

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import zstio  # noqa: E402

D = Path(os.environ.get("WXRISK_DATA", Path.home() / "wxrisk_data")) / "cnn"
T0 = pd.Timestamp("2014-11-01")
START = pd.Timestamp("2015-03-01")
GROUP = 27  # days per read group = one time chunk of the store
NBAND = 5


def open_ds():
    st = icechunk.s3_storage(bucket="dynamical-noaa-mrms", prefix="noaa-mrms-conus-analysis-hourly/v0.3.0.icechunk", region="us-west-2", anonymous=True)
    return xr.open_zarr(icechunk.Repository.open(st).readonly_session(branch="main").store, consolidated=False, chunks=None)


class Grid:
    """Mapping of the 1 km MRMS pixels to the 4 km E1 cells (contiguous groups, so reduceat works)."""

    def __init__(self, ds):
        S = np.load(D / "static.npz", allow_pickle=True)
        self.lat4, self.lon4 = S["lat4"], S["lon4"]
        self.days = pd.DatetimeIndex(S["days"])
        self.valid4 = S["obs4_valid"]
        step = abs(self.lat4[1] - self.lat4[0])
        lat_m, lon_m = ds.latitude.values, ds.longitude.values
        assert lat_m[0] > lat_m[-1], "latitude axis is expected to be descending"
        crow = np.floor((self.lat4[0] + step / 2 - lat_m) / step).astype(int)
        ccol = np.floor((lon_m - (self.lon4[0] - step / 2)) / step).astype(int)
        self.crow, self.ccol = crow, ccol
        self.ny, self.nx = len(self.lat4), len(self.lon4)

    def band(self, r0: int, r1: int, c0: int = 0, c1: int | None = None):
        """MRMS index slices and reduceat starts for the 4 km cells rows r0:r1, columns c0:c1."""
        c1 = self.nx if c1 is None else c1
        ri = np.flatnonzero((self.crow >= r0) & (self.crow < r1))
        ci = np.flatnonzero((self.ccol >= c0) & (self.ccol < c1))
        rs = np.flatnonzero(np.diff(self.crow[ri], prepend=-1) != 0)
        cs = np.flatnonzero(np.diff(self.ccol[ci], prepend=-1) != 0)
        assert len(rs) == r1 - r0 and len(cs) == c1 - c0, "a 4 km cell has no 1 km pixel"
        return slice(ri[0], ri[-1] + 1), slice(ci[0], ci[-1] + 1), rs, cs


def reduce_hourly(a: np.ndarray, rs, cs):
    """[T, nr, nc] rate (kg m-2 s-1) -> hourly cell mean and 1 km maximum in mm/h, NaN where the cell has no valid pixel."""
    ok = ~np.isnan(a)
    a0 = np.where(ok, a, 0).astype(np.float32) * 3600.0
    s = np.add.reduceat(np.add.reduceat(a0, rs, axis=1), cs, axis=2)
    n = np.add.reduceat(np.add.reduceat(ok.astype(np.float32), rs, axis=1), cs, axis=2)
    mx = np.maximum.reduceat(np.maximum.reduceat(a0, rs, axis=1), cs, axis=2)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan), np.where(n > 0, mx, np.nan)


def day_products(mean: np.ndarray, mx: np.ndarray) -> np.ndarray:
    """[48, R, C] hourly cell mean / 1 km maximum -> the six channels [6, R, C]."""
    ok = ~np.isnan(mean)
    m0 = np.where(ok, mean, 0)
    return np.stack([m0.sum(0), m0.max(0), np.where(ok, mx, 0).max(0), (m0 >= 1).mean(0), (m0 >= 5).mean(0), ok.mean(0)]).astype(np.float32)


def hour_index(day: pd.Timestamp, off: int) -> int:
    return int((day - T0) / pd.Timedelta(hours=1)) + off


def check() -> None:
    ds = open_ds()
    g = Grid(ds)
    O = zstio.load(D / "obs4k_day_maps.npy", mmap_mode="r")
    r0, r1, c0, c1 = 60, 120, 100, 190  # sub-box over the upper Texas coast / east Texas (about 2.5 x 3.7 degrees)
    cand = np.flatnonzero((g.days >= "2016-01-01") & (g.days <= "2018-12-31"))
    tot = np.array([float(np.asarray(O[n, 1, r0:r1, c0:c1], np.float32).sum()) for n in cand])
    days = cand[np.argsort(-tot)[:12]]
    ri, ci, rs, cs = g.band(r0, r1, c0, c1)
    print(f"sub-box rows {r0}:{r1}, cols {c0}:{c1}; heaviest gridMET days: {[str(g.days[n].date()) for n in days]}", flush=True)
    offsets = list(range(10, 18))
    res = {o: [] for o in offsets}
    bias = {o: [] for o in offsets}
    for n in days:
        ref = np.asarray(O[n, 1, r0:r1, c0:c1], np.float32)
        a = ds["precipitation_surface"].isel(time=slice(hour_index(g.days[n], 0), hour_index(g.days[n], 0) + 48 + 24), latitude=ri, longitude=ci).values
        mean, _ = reduce_hourly(a, rs, cs)
        for o in offsets:
            s = np.nan_to_num(mean[o:o + 48]).sum(0)
            res[o].append(np.corrcoef(np.log1p(s).ravel(), np.log1p(ref).ravel())[0, 1])
            bias[o].append(s.sum() / ref.sum())
    print("offset(h)  median corr(log1p)  median MRMS/gridMET total", flush=True)
    for o in offsets:
        print(f"{o:8d}   {np.median(res[o]):.3f}              {np.median(bias[o]):.2f}", flush=True)


def build(off: int) -> None:
    ds = open_ds()
    g = Grid(ds)
    first = int(np.flatnonzero(g.days >= START)[0])
    days = g.days[first:]
    nd = len(days)
    path = D / "mrms4k_day_maps.npy"
    if zstio.zst_path(path).exists():
        zstio.restore(path)  # resume writes in place; an archived copy would otherwise be silently recreated empty
    mode = "r+" if path.exists() else "w+"
    out = np.lib.format.open_memmap(path, mode=mode, dtype=np.float16, shape=(nd, 6, g.ny, g.nx))
    np.save(D / "mrms4k_days.npy", days.values.astype("datetime64[D]"))
    (D / "mrms4k_offset.txt").write_text(str(off))
    prog = D / "mrms4k_progress.txt"
    done = set(prog.read_text().split()) if prog.exists() else set()
    edges = np.linspace(0, g.ny, NBAND + 1).astype(int)
    t_all = time.time()
    for k0 in range(0, nd, GROUP):
        if str(k0) in done:
            continue
        k1 = min(k0 + GROUP, nd)
        h0 = hour_index(days[k0], off)
        h1 = hour_index(days[k1 - 1], off) + 48
        t0 = time.time()
        for b in range(NBAND):
            ri, ci, rs, cs = g.band(edges[b], edges[b + 1])
            a = ds["precipitation_surface"].isel(time=slice(h0, h1), latitude=ri, longitude=ci).values
            mean, mx = reduce_hourly(a, rs, cs)
            del a
            for k in range(k0, k1):
                s = (k - k0) * 24
                out[k, :, edges[b]:edges[b + 1]] = day_products(mean[s:s + 48], mx[s:s + 48])
        out.flush()
        done.add(str(k0))
        prog.write_text("\n".join(sorted(done, key=int)))
        print(f"group {k0 // GROUP + 1}/{-(-nd // GROUP)} ({days[k0].date()}): {time.time() - t0:.0f} s, elapsed {(time.time() - t_all) / 60:.1f} min", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["check", "build"])
    ap.add_argument("--offset", type=int, default=0, help="UTC hours added to 00 UTC of day S (from `check`)")
    a = ap.parse_args()
    check() if a.mode == "check" else build(a.offset)
