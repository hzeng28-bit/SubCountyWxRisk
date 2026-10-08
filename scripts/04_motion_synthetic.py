"""Fair synthetic comparison: windowed-FFT phase correlation vs complex-wavelet phase flow.

Same fields, same metric (per-pixel endpoint error in grid cells), only the estimator differs.
Four regimes, chosen so that the two methods are expected to separate:

  S1 uniform translation                       -- both should nail it
  S2 uniform translation + amplitude decay + noise -- phase methods should survive
  S3 spatially varying flow (shear/rotation)   -- one vector per block must average it away
  S4 two scales moving in DIFFERENT directions -- a single analysis window can only return one
                                                  vector; this is the decisive case

Run: python scripts/04_motion_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk.motion import block_motion  # noqa: E402
from wxrisk.wavelet_motion import phase_flow, scale_resolved_flow, warp_flow  # noqa: E402

NX, NY = 96, 96
RNG = np.random.default_rng(7)
EDGE = 16  # ignore a border ring: latitude is clamped, not periodic


def band_field(lo: int, hi: int, n_modes: int = 14, nx: int = NX, ny: int = NY) -> np.ndarray:
    """NARROWBAND field: a handful of modes inside [lo, hi].

    Deliberately kept as its own case: with only a few modes the local structure is close to a
    single plane wave, so the component of motion along the crests is unobservable (the optical
    flow aperture problem). Both estimators are then limited by the DATA, not by the method,
    and comparing them on such a field says nothing.
    """
    f = np.zeros((nx, ny), dtype=complex)
    for _ in range(n_modes):
        kx = RNG.integers(lo, hi + 1) * RNG.choice([-1, 1])
        ky = RNG.integers(lo, hi + 1) * RNG.choice([-1, 1])
        f[kx % nx, ky % ny] = RNG.normal() + 1j * RNG.normal()
    out = np.real(np.fft.ifft2(f))
    return out / out.std()


def powerlaw_field(
    slope: float = 2.0, k_min: int = 2, k_max: int = 24, nx: int = NX, ny: int = NY
) -> np.ndarray:
    """BROADBAND field with amplitude ~ k^-slope and random phases.

    Atmospheric fields have a red, broadband spectrum (Z500 roughly k^-3 in energy), which
    means structure at many orientations and scales at once -- the regime where a flow
    estimator is actually testable.
    """
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.fftfreq(ny) * ny
    kk = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    amp = np.where((kk >= k_min) & (kk <= k_max), np.power(np.maximum(kk, 1e-9), -slope), 0.0)
    phase = RNG.uniform(0, 2 * np.pi, size=(nx, ny))
    f = amp * np.exp(1j * phase)
    out = np.real(np.fft.ifft2(f))
    return out / out.std()


def block_flow_to_pixels(a: np.ndarray, b: np.ndarray, block: int, stride: int) -> tuple:
    """Run block phase correlation and nearest-interpolate the block vectors to pixels."""
    from scipy.ndimage import zoom

    m = block_motion(a, b, block=block, stride=stride)
    sx, sy = m["sx"], m["sy"]
    zx = NX / sx.shape[0]
    zy = NY / sx.shape[1]
    px = zoom(sx, (zx, zy), order=1, mode="nearest")[:NX, :NY]
    py = zoom(sy, (zx, zy), order=1, mode="nearest")[:NX, :NY]
    return px, py


def epe(dx: np.ndarray, dy: np.ndarray, tx: np.ndarray, ty: np.ndarray) -> float:
    """Median endpoint error over the interior."""
    s = (slice(None), slice(EDGE, NY - EDGE))
    e = np.sqrt((dx[s] - tx[s]) ** 2 + (dy[s] - ty[s]) ** 2)
    return float(np.median(e))


def report(case: str, truth: tuple, ests: dict) -> None:
    tx, ty = truth
    print(f"\n[{case}]  true |d| median = {np.median(np.hypot(tx, ty)):.2f} cells")
    for name, (dx, dy) in ests.items():
        print(f"  {name:38s} median EPE = {epe(dx, dy, tx, ty):6.3f} cells")


def main() -> None:
    wl = (32.0, 16.0, 8.0, 5.0)

    # ---------------- S1: uniform translation, broadband field ----------------
    a = powerlaw_field()
    sx, sy = 2.5, -1.5
    b = warp_flow(a, np.full_like(a, sx), np.full_like(a, sy))
    tx, ty = np.full_like(a, sx), np.full_like(a, sy)
    pf = phase_flow(a, b, wavelengths=wl)
    report(
        "S1 uniform translation (+2.50, -1.50)",
        (tx, ty),
        {
            "FFT whole field (one vector)": block_flow_to_pixels(a, b, NX, NX),
            "FFT block=48 phase corr": block_flow_to_pixels(a, b, 48, 24),
            "FFT block=32 phase corr": block_flow_to_pixels(a, b, 32, 16),
            "FFT block=16 phase corr": block_flow_to_pixels(a, b, 16, 8),
            "wavelet phase flow (coarse->fine)": (pf["dx"], pf["dy"]),
        },
    )

    # ---------------- S1b: same translation, NARROWBAND field (aperture-limited) ---------
    a1b = band_field(2, 6)
    b1b = warp_flow(a1b, np.full_like(a1b, sx), np.full_like(a1b, sy))
    pf1b = phase_flow(a1b, b1b, wavelengths=wl)
    report(
        "S1b same translation, NARROWBAND field (aperture problem)",
        (tx, ty),
        {
            "FFT whole field (one vector)": block_flow_to_pixels(a1b, b1b, NX, NX),
            "FFT block=32 phase corr": block_flow_to_pixels(a1b, b1b, 32, 16),
            "wavelet phase flow (coarse->fine)": (pf1b["dx"], pf1b["dy"]),
        },
    )

    # ---------------- S2: translation + decay + noise ----------------
    b2 = 0.7 * b + RNG.normal(size=b.shape) * 0.2 * b.std()
    pf2 = phase_flow(a, b2, wavelengths=wl)
    report(
        "S2 same translation, x0.7 amplitude, SNR 5",
        (tx, ty),
        {
            "FFT block=32 phase corr": block_flow_to_pixels(a, b2, 32, 16),
            "FFT block=16 phase corr": block_flow_to_pixels(a, b2, 16, 8),
            "wavelet phase flow (coarse->fine)": (pf2["dx"], pf2["dy"]),
        },
    )

    # ---------------- S3: spatially varying flow ----------------
    gx, gy = np.meshgrid(np.arange(NX), np.arange(NY), indexing="ij")
    amp = 3.0
    tx3 = amp * np.sin(2 * np.pi * gy / NY)
    ty3 = amp * np.sin(2 * np.pi * gx / NX)
    a3 = powerlaw_field()
    b3 = warp_flow(a3, tx3, ty3)
    pf3 = phase_flow(a3, b3, wavelengths=wl)
    report(
        "S3 spatially varying flow (shear, amp 3)",
        (tx3, ty3),
        {
            "FFT block=32 phase corr": block_flow_to_pixels(a3, b3, 32, 16),
            "FFT block=16 phase corr": block_flow_to_pixels(a3, b3, 16, 8),
            "FFT block=12 phase corr": block_flow_to_pixels(a3, b3, 12, 6),
            "wavelet phase flow (coarse->fine)": (pf3["dx"], pf3["dy"]),
        },
    )

    # ---------------- S4: two scales, different directions ----------------
    coarse = powerlaw_field(slope=2.0, k_min=2, k_max=4)
    fine = powerlaw_field(slope=2.0, k_min=10, k_max=16)
    c_sx, c_sy = 3.0, 0.0
    f_sx, f_sy = 0.0, 2.5
    a4 = coarse + 0.8 * fine
    b4 = warp_flow(coarse, np.full_like(coarse, c_sx), np.full_like(coarse, c_sy)) + 0.8 * (
        warp_flow(fine, np.full_like(fine, f_sx), np.full_like(fine, f_sy))
    )
    print(
        f"\n[S4 two scales: coarse(+{c_sx:.1f},{c_sy:+.1f}) + 0.8*fine({f_sx:+.1f},{f_sy:+.1f})]"
    )
    for name, (dx, dy) in {
        "FFT block=32 phase corr": block_flow_to_pixels(a4, b4, 32, 16),
        "FFT block=16 phase corr": block_flow_to_pixels(a4, b4, 16, 8),
    }.items():
        s = (slice(None), slice(EDGE, NY - EDGE))
        print(
            f"  {name:38s} median est = ({np.median(dx[s]):+.2f}, {np.median(dy[s]):+.2f})"
            f"   EPE vs coarse {epe(dx, dy, np.full_like(a4, c_sx), np.full_like(a4, c_sy)):.3f}"
            f"   vs fine {epe(dx, dy, np.full_like(a4, f_sx), np.full_like(a4, f_sy)):.3f}"
        )
    print("  wavelet, INDEPENDENT per-scale flow:")
    for s_out in scale_resolved_flow(a4, b4, wavelengths=(32.0, 24.0, 16.0, 8.0, 6.0, 4.0)):
        sl = (slice(None), slice(EDGE, NY - EDGE))
        dx, dy = s_out["dx"], s_out["dy"]
        print(
            f"    lambda={s_out['wavelength']:5.1f} cells  median est = "
            f"({np.median(dx[sl]):+.2f}, {np.median(dy[sl]):+.2f})"
            f"   EPE vs coarse {epe(dx, dy, np.full_like(a4, c_sx), np.full_like(a4, c_sy)):6.3f}"
            f"   vs fine {epe(dx, dy, np.full_like(a4, f_sx), np.full_like(a4, f_sy)):6.3f}"
            f"   cond {np.median(s_out['condition'][sl]):.3f}"
        )


if __name__ == "__main__":
    main()
