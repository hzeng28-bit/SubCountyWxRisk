"""Synthetic checks: sign convention, sub-pixel accuracy, warp/estimate round-trip."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk.motion import estimate_shift, warp  # noqa: E402

NX, NY = 64, 64
RNG = np.random.default_rng(0)


def smooth_field(nx: int = NX, ny: int = NY, k: int = 4) -> np.ndarray:
    """Band-limited random field: a few low Fourier modes, like a synoptic pattern."""
    f = np.zeros((nx, ny), dtype=complex)
    for _ in range(12):
        kx, ky = RNG.integers(1, k + 1), RNG.integers(1, k + 1)
        f[kx, ky] = RNG.normal() + 1j * RNG.normal()
    return np.real(np.fft.ifft2(f, s=(nx, ny))) * nx * ny


def test_sign_and_subpixel() -> None:
    field = smooth_field()
    errs = []
    for sx, sy in [(3.0, 0.0), (0.0, 2.0), (2.5, -1.5), (-4.0, 3.25), (0.4, 0.6)]:
        shifted = warp(field, sx, sy)
        ex, ey, co = estimate_shift(field, shifted)
        errs.append((sx, sy, ex, ey, co))
        assert abs(ex - sx) < 0.25, f"x: wanted {sx}, got {ex}"
        assert abs(ey - sy) < 0.25, f"y: wanted {sy}, got {ey}"
        assert co > 1.0
    for sx, sy, ex, ey, co in errs:
        print(f"  true=({sx:+.2f},{sy:+.2f})  est=({ex:+.2f},{ey:+.2f})  coh={co:.1f}")


def test_noise_robustness() -> None:
    field = smooth_field()
    shifted = warp(field, 2.5, -1.5)
    for snr in (10.0, 3.0, 1.0):
        noise = RNG.normal(size=field.shape) * field.std() / snr
        ex, ey, co = estimate_shift(field, shifted + noise)
        print(f"  snr={snr:4.1f}  est=({ex:+.2f},{ey:+.2f})  coh={co:.1f}")
        if snr >= 3.0:
            assert abs(ex - 2.5) < 0.5 and abs(ey + 1.5) < 0.5


def test_pure_noise_has_low_coherence() -> None:
    a = RNG.normal(size=(NX, NY))
    b = RNG.normal(size=(NX, NY))
    _, _, co_noise = estimate_shift(a, b)
    f = smooth_field()
    _, _, co_sig = estimate_shift(f, warp(f, 2.0, 1.0))
    print(f"  coherence: noise={co_noise:.2f}  translation={co_sig:.2f}")
    assert co_sig > co_noise


if __name__ == "__main__":
    for fn in (test_sign_and_subpixel, test_noise_robustness, test_pure_noise_has_low_coherence):
        print(fn.__name__)
        fn()
    print("all synthetic motion checks passed")
