"""Scale-resolved county-day features from the CDS ERA5 box (Morlet bank + phase flow).

For each day and each of the 12Z / 18Z frames, every environment channel is decomposed with the
Morlet bank of `wavelet_motion` into WAVELENGTHS (in 0.25 deg cells, ~125-800 km):
  b{s}  band-pass value at the county: signed anomaly at that scale (sum of oriented real parts)
  e{s}  log local amplitude at that scale (how active that scale is around the county)
  lp    Gaussian low-pass (sigma = 16 cells): the part larger than the biggest wavelength
Motion channels also get an independent phase flow per scale from the 12Z -> 18Z frame pair.

Arrays are laid out (lon east-increasing, lat north-increasing), so flow dx is cells eastward and
dy cells northward over the 6 h. The box is not periodic, so fields are edge-padded by PAD cells
before the FFTs; counties sit >= 20 cells inside the box.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter

from . import wb2
from .wavelet_motion import morlet_bank, scale_resolved_flow

WAVELENGTHS = (32.0, 16.0, 8.0, 5.0)
PAD = 32
SIGMA_LP = 16.0
CHANNELS = ["cape", "cin", "dd", "pw", "q850", "lapse75", "shear", "w700", "msl"]
MOTION_CHANNELS = ["pw", "msl"]
HOURS = (12, 18)


def _g(da: xr.DataArray) -> np.ndarray:
    da = da.sortby("latitude").sortby("longitude")
    return da.transpose("valid_time", "longitude", "latitude").values.astype(np.float32)


def load_grid(year: int):
    """Derived channels on the box: times, lon, lat, fields[time, lon, lat, channel]."""
    d = wb2.DATA_DIR / "cds"
    sg = xr.open_dataset(d / f"single_{year}.nc")
    pl = xr.open_dataset(d / f"plev_{year}.nc")
    lvl = lambda v, p: _g(pl[v].sel(pressure_level=p))  # noqa: E731
    cape = np.log1p(np.clip(_g(sg.cape), 0, None))
    cin = np.log1p(np.abs(np.nan_to_num(_g(sg.cin), nan=0.0)))
    f = {
        "cape": cape,
        "cin": cin,
        "dd": _g(sg.t2m) - _g(sg.d2m),
        "pw": _g(sg.tcwv),
        "q850": lvl("q", 850),
        "lapse75": lvl("t", 700) - lvl("t", 500),
        "shear": np.hypot(lvl("u", 500) - _g(sg.u10), lvl("v", 500) - _g(sg.v10)),
        "w700": np.hypot(lvl("u", 700), lvl("v", 700)),
        "msl": _g(sg.msl),
    }
    t = pd.DatetimeIndex(sg.valid_time.values)
    lon = np.sort(sg.longitude.values)
    lat = np.sort(sg.latitude.values)
    return t, lon, lat, np.stack([f[c] for c in CHANNELS], axis=-1)


def _filters(shape: tuple[int, int]) -> np.ndarray:
    """Transfer functions, (scale, orientation, nx, ny)."""
    return np.stack([np.stack([t for t, _ in morlet_bank(*shape, lam, 4, 0.6)])
                     for lam in WAVELENGTHS])


def _pad(a: np.ndarray) -> np.ndarray:
    """Edge-pad the last two axes."""
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(PAD, PAD)] * 2, mode="edge")


def year_features(year: int, cty: pd.DataFrame) -> pd.DataFrame:
    t, lon, lat, fields = load_grid(year)
    ix = np.abs(lon[None] - cty.lon.to_numpy()[:, None]).argmin(1) + PAD
    iy = np.abs(lat[None] - cty.lat.to_numpy()[:, None]).argmin(1) + PAD
    days = pd.DatetimeIndex(sorted(set(t.normalize())))
    sel = {h: np.array([np.flatnonzero((t.normalize() == d) & (t.hour == h))[0] for d in days])
           for h in HOURS}
    shape = (fields.shape[1] + 2 * PAD, fields.shape[2] + 2 * PAD)
    T = _filters(shape)
    nch, ns, nc = len(CHANNELS), len(WAVELENGTHS), len(cty)
    band = {h: np.empty((len(days), nch, ns, nc), np.float32) for h in HOURS}
    amp = {h: np.empty_like(band[h]) for h in HOURS}
    lp = {h: np.empty((len(days), nch, nc), np.float32) for h in HOURS}
    mv = np.empty((len(days), len(MOTION_CHANNELS), ns, 2, nc), np.float32)
    mi = [CHANNELS.index(c) for c in MOTION_CHANNELS]

    for di in range(len(days)):
        pads = {}
        for h in HOURS:
            F = _pad(np.moveaxis(fields[sel[h][di]], -1, 0))  # (nch, nx, ny)
            pads[h] = F
            fh = np.fft.fft2(F - F.mean(axis=(1, 2), keepdims=True))
            R = np.fft.ifft2(fh[:, None, None] * T[None])  # (nch, ns, nk, nx, ny)
            band[h][di] = R.real.mean(2)[:, :, ix, iy]
            amp[h][di] = np.log(np.sqrt((np.abs(R) ** 2).mean(2))[:, :, ix, iy] + 1e-6)
            lp[h][di] = gaussian_filter(F, (0, SIGMA_LP, SIGMA_LP), mode="nearest")[:, ix, iy]
        for m, c in enumerate(mi):
            for s, fl in enumerate(scale_resolved_flow(pads[12][c], pads[18][c], WAVELENGTHS)):
                mv[di, m, s, 0], mv[di, m, s, 1] = fl["dx"][ix, iy], fl["dy"][ix, iy]

    cols: dict[str, np.ndarray] = {}
    for h in HOURS:
        for ci, c in enumerate(CHANNELS):
            for si, lam in enumerate(WAVELENGTHS):
                s = int(lam)
                cols[f"{c}_{h}_b{s}"] = band[h][:, ci, si].ravel()
                cols[f"{c}_{h}_e{s}"] = amp[h][:, ci, si].ravel()
            cols[f"{c}_{h}_lp"] = lp[h][:, ci].ravel()
    for m, c in enumerate(MOTION_CHANNELS):
        for si, lam in enumerate(WAVELENGTHS):
            cols[f"mv_{c}_{int(lam)}_dx"] = mv[:, m, si, 0].ravel()
            cols[f"mv_{c}_{int(lam)}_dy"] = mv[:, m, si, 1].ravel()
    df = pd.DataFrame(cols)
    df["date"] = np.repeat(days.to_numpy(), nc)
    df["cty"] = np.tile(cty.cty.to_numpy(), len(days))
    return df
