"""Scale-resolved (Morlet) features of each episode's peak-rain field, sampled at block-group cells.

Same encoder as scale_features (stage 3), applied to log1p of the episode's 1-day peak rainfall over
the Texas gridMET grid: for wavelengths of 4, 8, 16, 32 and 64 cells (about 16-256 km) the band-pass
value and log local amplitude at the block group's cell, plus a 16-cell Gaussian low-pass.

    python scripts/28_wavelet_rain_features.py --workers 6
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk.wavelet_motion import morlet_bank  # noqa: E402

WL = (4, 8, 16, 32, 64)
PAD = 40
_T = None


def filters(shape):
    return np.stack([np.stack([t for t, _ in morlet_bank(*shape, float(w), 4, 0.6)]) for w in WL])


def work(args):
    global _T
    eps, = args
    days, lat, lon, P = fp.load_precip()
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "bg", "start", "end"])
    t = t[t.ep.isin(eps)]
    g = fp.bg_geometry()
    iy, ix = fp.grid_index(lat, lon, g.lat.to_numpy(), g.lon.to_numpy())
    if _T is None:
        _T = filters((P.shape[1] + 2 * PAD, P.shape[2] + 2 * PAD))
    out = []
    for ep, grp in t.groupby("ep"):
        s = (grp.start.iloc[0] - days[0]).days
        e = (grp.end.iloc[0] - days[0]).days
        F = np.log1p(np.nan_to_num(np.asarray(P[s:e + 1]), nan=0.0).max(axis=0))
        Fp = np.pad(F, PAD, mode="edge")
        fh = np.fft.fft2(Fp - Fp.mean())
        R = np.fft.ifft2(fh[None, None] * _T)  # (scale, orientation, y, x)
        band = R.real.mean(1)
        amp = np.log(np.sqrt((np.abs(R) ** 2).mean(1)) + 1e-6)
        lp = gaussian_filter(Fp, 16, mode="nearest")
        yy, xx = iy[grp.bg.to_numpy()] + PAD, ix[grp.bg.to_numpy()] + PAD
        cols = {"ep": np.full(len(grp), ep), "bg": grp.bg.to_numpy(), "w_lp": lp[yy, xx].astype(np.float32)}
        for k, w in enumerate(WL):
            cols[f"w_b{w}"] = band[k][yy, xx].astype(np.float32)
            cols[f"w_e{w}"] = amp[k][yy, xx].astype(np.float32)
        out.append(pd.DataFrame(cols))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep"])
    eps = np.sort(t.ep.unique())
    chunks = [(c,) for c in np.array_split(eps, a.workers * 6)]
    with ProcessPoolExecutor(a.workers) as ex:
        parts = []
        for i, r in enumerate(ex.map(work, chunks)):
            parts.append(r)
            print(f"  chunk {i + 1}/{len(chunks)}", flush=True)
    w = pd.concat(parts, ignore_index=True)
    w.to_parquet(fp.PILOT / "flood_pilot_wavelet.parquet")
    print(f"{len(w):,} rows, {w.shape[1] - 2} features saved")


if __name__ == "__main__":
    main()
