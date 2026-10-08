"""Does the motion-vector trajectory along lead time have FREQUENCY structure?

The motion field is strongly persistent in lead time (script 06: r ~ 0.87), so of course a few
low-frequency coefficients capture most of its energy -- that is what "smooth" means and it is
not evidence of anything. The non-trivial question is whether the trajectory concentrates MORE
energy than its own lag-1 autocorrelation already implies.

Control: for every pixel, an AR(1) surrogate with the SAME variance and the SAME measured
lag-1 autocorrelation, same series length, 8 realizations. If the real MV trajectory does not
beat its AR(1) surrogate, then "bring the frequency head back" buys nothing beyond a smoother,
and the honest conclusion is to model the motion field as AR(1) instead.

Run: python scripts/07_mv_spectrum.py --n-init 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.wavelet_motion import phase_flow  # noqa: E402


def detrend(field: np.ndarray, hp_sigma: float = 12.0) -> np.ndarray:
    f = field - field.mean(axis=0, keepdims=True)
    return f - gaussian_filter(f, hp_sigma, mode=("wrap", "nearest"))


def energy_fraction(series: np.ndarray, kmax: int = 6) -> np.ndarray:
    """Cumulative FFT energy fraction in the first k non-DC coefficients. series: [..., T]."""
    x = series - series.mean(axis=-1, keepdims=True)
    f = np.fft.rfft(x, axis=-1)
    p = np.abs(f) ** 2
    p = p[..., 1:]  # drop DC, already removed by the mean subtraction
    tot = p.sum(axis=-1, keepdims=True)
    cum = np.cumsum(p, axis=-1) / np.maximum(tot, 1e-30)
    return cum[..., : min(kmax, cum.shape[-1])]


def ar1_surrogate(series: np.ndarray, n_real: int, rng: np.random.Generator) -> np.ndarray:
    """AR(1) surrogates matching each pixel's variance and lag-1 autocorrelation."""
    x = series - series.mean(axis=-1, keepdims=True)
    t = x.shape[-1]
    var = x.var(axis=-1)
    num = np.sum(x[..., 1:] * x[..., :-1], axis=-1)
    den = np.sum(x[..., :-1] ** 2, axis=-1)
    r1 = np.clip(num / np.maximum(den, 1e-30), -0.99, 0.99)
    out = np.zeros((n_real,) + x.shape, dtype=np.float32)
    sd = np.sqrt(np.maximum(var, 0) * (1 - r1**2))
    for i in range(n_real):
        y = np.zeros_like(x, dtype=np.float32)
        y[..., 0] = rng.normal(size=x.shape[:-1]) * np.sqrt(np.maximum(var, 0))
        for j in range(1, t):
            y[..., j] = r1 * y[..., j - 1] + rng.normal(size=x.shape[:-1]) * sd
        out[i] = y
    return out, r1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="seq240_mean_2018_n12_lead120.npz")
    ap.add_argument("--n-init", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=0.25)
    ap.add_argument("--wavelengths", type=float, nargs="+", default=[32, 16, 8, 5])
    ap.add_argument("--subsample", type=int, default=4, help="pixel stride for the statistics")
    a = ap.parse_args()

    d = np.load(wb2.cache_path(a.file))
    z, lat = d["z500"], d["latitude"]
    leads = d["lead_hours"]
    n_init = min(a.n_init, z.shape[0])
    n_lead = z.shape[1]
    mid = (np.abs(lat) >= 25) & (np.abs(lat) <= 70)
    wl = tuple(a.wavelengths)
    print(f"{z.shape}, {n_lead} leads ({leads[0]}..{leads[-1]} h), {n_init} inits")

    flows = np.zeros((n_init, n_lead - 1, z.shape[2], z.shape[3], 2), dtype=np.float32)
    t0 = time.time()
    for ti in range(n_init):
        fields = [detrend(z[ti, li]) for li in range(n_lead)]
        for li in range(n_lead - 1):
            pf = phase_flow(fields[li], fields[li + 1], wavelengths=wl, smooth=a.smooth)
            flows[ti, li, ..., 0] = pf["dx"]
            flows[ti, li, ..., 1] = pf["dy"]
        print(f"  init {ti + 1}/{n_init} done, {time.time() - t0:.0f}s", flush=True)
    out = wb2.cache_path(f"mvflow_{a.file.replace('.npz', '')}_n{n_init}.npy")
    np.save(out, flows)
    print(f"saved {out.name} {flows.shape}")

    s = a.subsample
    sub = flows[:, :, ::s, :, :][:, :, :, mid, :]  # [init, lead-1, x, y, 2]
    series = np.moveaxis(sub, 1, -1)  # [init, x, y, 2, T]
    series = series.reshape(-1, series.shape[-1])
    print(f"\ntrajectories: {series.shape[0]} pixel-components x {series.shape[1]} intervals")

    rng = np.random.default_rng(0)
    sur, r1 = ar1_surrogate(series, 8, rng)
    ef_real = energy_fraction(series)
    ef_sur = energy_fraction(sur.reshape(-1, sur.shape[-1]))
    print(f"measured lag-1 autocorrelation of the trajectories: "
          f"median {np.median(r1):.3f}, IQR [{np.percentile(r1, 25):.3f}, "
          f"{np.percentile(r1, 75):.3f}]")
    print("\ncumulative energy in the first k non-DC frequency coefficients (median over "
          "trajectories)")
    print(f"{'k':>3s} {'real':>8s} {'AR(1) surrogate':>16s} {'difference':>11s}")
    print("-" * 42)
    for k in range(ef_real.shape[-1]):
        r = float(np.median(ef_real[:, k]))
        s_ = float(np.median(ef_sur[:, k]))
        print(f"{k + 1:3d} {r:8.3f} {s_:16.3f} {r - s_:+11.3f}")
    print("\nIf the difference column is ~0, the motion trajectory is just a smooth AR(1)")
    print("process and a frequency representation adds nothing beyond that.")


if __name__ == "__main__":
    main()
