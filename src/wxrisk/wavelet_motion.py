"""Scale-resolved motion from complex-wavelet (Morlet/Gabor) local phase.

Why wavelets instead of a windowed FFT: phase correlation needs ONE analysis window for the
whole block, so it returns ONE vector per block and its accuracy is capped by that window's
fixed time/space-frequency tradeoff. A translation is a linear phase ramp in the Fourier domain
only if the whole window moves rigidly. Real atmospheric flow is (a) spatially varying and
(b) multi-scale with DIFFERENT motion at different scales (a synoptic wave moving east with
embedded features moving north). A complex-wavelet bank attaches an analysis window to each
scale, so every scale gets its own displacement estimate, per pixel.

Method (phase-based optical flow, Fleet & Jepson 1990, in 2D):
for a complex Morlet filter with centre wavevector k, a local translation d produces a phase
difference  dphi = k . d.  With >= 2 orientations per scale this is a 2x2 weighted
least-squares problem per pixel, amplitude-weighted. Coarse-to-fine: each scale estimates a
RESIDUAL displacement on top of the already-warped field, which both unwraps the phase
(|k . d| < pi) and lets coarse scales capture large motion that fine scales cannot resolve.

Convention matches wxrisk.motion: the returned flow (dx, dy) satisfies b(x) ~= a(x - d(x)).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "morlet_bank",
    "gabor_response",
    "phase_flow",
    "warp_flow",
    "scale_resolved_flow",
]


def _freq_grid(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    wx = 2 * np.pi * np.fft.fftfreq(nx)[:, None]
    wy = 2 * np.pi * np.fft.fftfreq(ny)[None, :]
    return np.broadcast_to(wx, (nx, ny)), np.broadcast_to(wy, (nx, ny))


def morlet_bank(
    nx: int, ny: int, wavelength: float, n_orient: int = 4, bandwidth: float = 0.6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fourier-domain analytic Morlet filters at one scale, several orientations.

    Returns a list of (transfer_function[nx, ny], centre_wavevector[2]). `bandwidth` is the
    relative width of the Gaussian envelope in frequency: larger = better localized in space,
    worse in frequency. The spatial support scales with `wavelength`, which is exactly the
    adaptive-window property a fixed FFT window lacks.
    """
    wxg, wyg = _freq_grid(nx, ny)
    k0 = 2 * np.pi / wavelength
    sigma = bandwidth * k0
    out = []
    for theta in np.linspace(0.0, np.pi, n_orient, endpoint=False):
        kx, ky = k0 * np.cos(theta), k0 * np.sin(theta)
        # analytic (one-sided) filter: only the lobe around +k, so the response has a
        # meaningful phase instead of a real-valued standing wave
        g = np.exp(-((wxg - kx) ** 2 + (wyg - ky) ** 2) / (2 * sigma**2))
        out.append((g, np.array([kx, ky])))
    return out


def gabor_response(field: np.ndarray, transfer: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(field - field.mean())
    return np.fft.ifft2(f * transfer)


def warp_flow(field: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Warp by a per-pixel flow: result[x, y] ~= field[x - dx, y - dy]."""
    from scipy.ndimage import map_coordinates

    nx, ny = field.shape
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    sx = (gx - dx) % nx
    sy = np.clip(gy - dy, 0, ny - 1)
    return map_coordinates(
        field, [sx.ravel(), sy.ravel()], order=1, mode="grid-wrap"
    ).reshape(field.shape)


def _gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    # longitude wraps, latitude does not
    return gaussian_filter(arr, sigma=sigma, mode=("wrap", "nearest"))


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _solve_scale(
    a: np.ndarray,
    b: np.ndarray,
    wavelength: float,
    n_orient: int,
    bandwidth: float,
    smooth: float,
    ridge: float,
) -> dict[str, np.ndarray]:
    """One scale: amplitude-weighted 2x2 phase least squares per pixel."""
    nx, ny = a.shape
    bank = morlet_bank(nx, ny, wavelength, n_orient, bandwidth)
    axx = np.zeros((nx, ny))
    axy = np.zeros((nx, ny))
    ayy = np.zeros((nx, ny))
    rx = np.zeros((nx, ny))
    ry = np.zeros((nx, ny))
    amp_tot = np.zeros((nx, ny))
    for transfer, k in bank:
        ra, rb = gabor_response(a, transfer), gabor_response(b, transfer)
        dphi = _wrap_pi(np.angle(rb) - np.angle(ra))
        w = np.abs(ra) * np.abs(rb)
        # pool the normal equations over a window that scales with the wavelet, not a
        # fixed block: this is the aperture problem solved at the scale being measured
        s = max(smooth * wavelength, 1.0)
        w_s = _gaussian(w, s)
        wd = _gaussian(w * dphi, s)
        axx += w_s * k[0] * k[0]
        axy += w_s * k[0] * k[1]
        ayy += w_s * k[1] * k[1]
        # Sign: with b(x) = a(x - d) the filter response obeys R_b(x) ~= R_a(x - d), so
        # angle(R_b) - angle(R_a) = -k.d. The normal equations therefore take -dphi on the
        # right-hand side. Getting this wrong flips the whole flow field while leaving its
        # magnitude correct, which is exactly the kind of silent direction bug that still
        # prints plausible numbers (measured: -3.04 instead of +3.00 before the fix).
        rx -= k[0] * wd
        ry -= k[1] * wd
        amp_tot += w_s
    lam = ridge * (axx + ayy + 1e-30)
    axx_r, ayy_r = axx + lam, ayy + lam
    det = axx_r * ayy_r - axy**2
    det = np.where(np.abs(det) < 1e-30, np.nan, det)
    dx = (ayy_r * rx - axy * ry) / det
    dy = (axx_r * ry - axy * rx) / det
    # conditioning of the 2x2 system = how well-determined the vector is (aperture problem)
    tr = axx + ayy
    disc = np.sqrt(np.maximum((axx - ayy) ** 2 + 4 * axy**2, 0.0))
    eig_min = 0.5 * (tr - disc)
    return {
        "dx": np.nan_to_num(dx),
        "dy": np.nan_to_num(dy),
        "amplitude": amp_tot,
        "eig_min": eig_min,
        "condition": eig_min / (0.5 * (tr + disc) + 1e-30),
    }


def phase_flow(
    a: np.ndarray,
    b: np.ndarray,
    wavelengths: tuple[float, ...] = (32.0, 16.0, 8.0, 5.0),
    n_orient: int = 4,
    bandwidth: float = 0.6,
    smooth: float = 0.5,
    ridge: float = 1e-3,
    max_step: float | None = None,
) -> dict[str, np.ndarray]:
    """Coarse-to-fine per-pixel flow. `wavelengths` must be ordered coarse -> fine.

    Returns the accumulated flow plus the per-scale increments and confidences, so that
    "which scale carried the motion" is an output, not a hidden detail.
    """
    dx = np.zeros_like(a, dtype=np.float64)
    dy = np.zeros_like(a, dtype=np.float64)
    per_scale = []
    for lam in wavelengths:
        a_w = warp_flow(a, dx, dy)
        s = _solve_scale(a_w, b, lam, n_orient, bandwidth, smooth, ridge)
        cap = max_step if max_step is not None else lam / 2.0
        inc_x = np.clip(s["dx"], -cap, cap)
        inc_y = np.clip(s["dy"], -cap, cap)
        dx = dx + inc_x
        dy = dy + inc_y
        per_scale.append(
            {
                "wavelength": lam,
                "inc_x": inc_x,
                "inc_y": inc_y,
                "amplitude": s["amplitude"],
                "condition": s["condition"],
            }
        )
    return {"dx": dx, "dy": dy, "per_scale": per_scale}


def scale_resolved_flow(
    a: np.ndarray,
    b: np.ndarray,
    wavelengths: tuple[float, ...] = (32.0, 16.0, 8.0, 5.0),
    n_orient: int = 4,
    bandwidth: float = 0.6,
    smooth: float = 0.5,
    ridge: float = 1e-3,
) -> list[dict]:
    """INDEPENDENT flow per scale (no coarse-to-fine coupling).

    Use this to ask whether different scales move differently -- the coarse-to-fine version
    deliberately folds all scales into one vector and would hide exactly that signal.
    """
    out = []
    for lam in wavelengths:
        s = _solve_scale(a, b, lam, n_orient, bandwidth, smooth, ridge)
        cap = lam / 2.0
        out.append(
            {
                "wavelength": lam,
                "dx": np.clip(s["dx"], -cap, cap),
                "dy": np.clip(s["dy"], -cap, cap),
                "amplitude": s["amplitude"],
                "condition": s["condition"],
            }
        )
    return out
