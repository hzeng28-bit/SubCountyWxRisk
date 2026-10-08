"""Motion estimation by FFT phase correlation (sub-pixel), plus motion-compensated warping.

Why phase correlation and not integer block matching: on a 5.625 deg grid one cell is ~625 km,
while a 20 m/s system moves ~430 km in 6 h -- less than one cell. Integer-pixel block matching
would return a zero vector almost everywhere. A translation is an exact linear phase ramp in the
Fourier domain, so the shift can be recovered at sub-pixel precision from the cross-power
spectrum. This is also the direct link back to the frequency-domain line of work: the motion
vector IS a phase measurement.

Convention (verified against synthetic shifts in tests/test_motion.py):
    estimate_shift(a, b) -> (sx, sy) such that  b(x, y) ~= a(x - sx, y - sy)
i.e. the returned vector points from the feature's position in `a` to its position in `b`,
in grid cells, along (axis 0 = x = longitude, axis 1 = y = latitude).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hann2d",
    "estimate_shift",
    "block_motion",
    "warp",
    "motion_compensation_gain",
]


def hann2d(nx: int, ny: int) -> np.ndarray:
    return np.outer(np.hanning(nx), np.hanning(ny))


def _parabolic_subpixel(c: np.ndarray, i: int, n: int) -> float:
    """Parabolic interpolation of a correlation peak at index i along a periodic axis."""
    im, ip = (i - 1) % n, (i + 1) % n
    y0, y1, y2 = c[im], c[i], c[ip]
    denom = y0 - 2 * y1 + y2
    if denom == 0:
        return 0.0
    d = 0.5 * (y0 - y2) / denom
    return float(np.clip(d, -1.0, 1.0))


def estimate_shift(
    a: np.ndarray,
    b: np.ndarray,
    window: bool = False,
    eps: float = 1e-12,
    amp_frac: float = 1e-3,
) -> tuple[float, float, float]:
    """Sub-pixel shift taking `a` to `b` by phase correlation.

    Returns (sx, sy, coherence) with sx/sy in grid cells and coherence = normalized peak
    height of the phase-correlation surface.

    `window` defaults to False on purpose. A Hann taper is the textbook default for phase
    correlation, but on band-limited synoptic fields it DESTROYS the estimate: the window is
    fixed in space while the content moves under it, so for a field whose energy sits in a few
    low modes the window's own spectrum dominates the cross-power spectrum and the peak
    collapses to zero lag. Measured on a synthetic band-limited field with a true shift of
    +3.00 cells: window=True gives 0.23, window=False gives 3.00. This is the concrete cost
    of a FIXED analysis window, and the reason the wavelet-phase estimator in
    `wxrisk.wavelet_motion` (window width tied to scale) is the primary method here.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    if window:
        w = hann2d(*a.shape)
        a, b = a * w, b * w
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    cross = fb * np.conj(fa)
    # Fully normalized ("pure") phase correlation is amplitude-blind, which on a band-limited
    # synoptic field amplifies numerically empty frequency bins to unit weight and can put the
    # correlation peak anywhere: measured on a synthetic field with true shift (0, +2), pure
    # phase correlation returned (+6.16, ...). Keep only bins with meaningful joint amplitude.
    mag = np.abs(fa) * np.abs(fb)
    keep = mag >= amp_frac * mag.max()
    cross = np.where(keep, cross / (np.abs(cross) + eps), 0.0)
    corr = np.real(np.fft.ifft2(cross))
    nx, ny = corr.shape
    ix, iy = np.unravel_index(np.argmax(corr), corr.shape)
    peak = float(corr[ix, iy])
    # integer part, mapped to (-n/2, n/2]
    sx_int = ((ix + nx / 2) % nx) - nx / 2
    sy_int = ((iy + ny / 2) % ny) - ny / 2
    # Sub-pixel refinement by the PHASE SLOPE of the cross spectrum, not by parabolic
    # interpolation of the correlation peak. With b(x) = a(x - s) we have
    # cross = fb * conj(fa) = |fa|^2 exp(-i w.s), so the residual phase after removing the
    # integer shift is a linear ramp in w whose slope is the fractional shift. Parabolic peak
    # interpolation only reached ~0.4 cell accuracy on a synthetic (2.5, -1.5) shift; the
    # phase-slope fit is exact for a pure translation.
    wx = 2 * np.pi * np.fft.fftfreq(nx)[:, None]
    wy = 2 * np.pi * np.fft.fftfreq(ny)[None, :]
    phase_res = np.angle(cross * np.exp(1j * (wx * sx_int + wy * sy_int)))
    w = mag * keep
    wxb = np.broadcast_to(wx, w.shape)
    wyb = np.broadcast_to(wy, w.shape)
    gxx = float(np.sum(w * wxb**2))
    gyy = float(np.sum(w * wyb**2))
    gxy = float(np.sum(w * wxb * wyb))
    rx = float(-np.sum(w * wxb * phase_res))
    ry = float(-np.sum(w * wyb * phase_res))
    det = gxx * gyy - gxy * gxy
    if abs(det) > 1e-20:
        fx = (gyy * rx - gxy * ry) / det
        fy = (gxx * ry - gxy * rx) / det
    else:
        fx = fy = 0.0
    fx = float(np.clip(fx, -0.5, 0.5))
    fy = float(np.clip(fy, -0.5, 0.5))
    coherence = peak / (np.abs(corr).mean() + eps)
    return float(sx_int + fx), float(sy_int + fy), float(coherence)


def _take_block(
    field: np.ndarray, x0: int, y0: int, bx: int, by: int
) -> np.ndarray:
    """Block with longitude wrap-around and latitude clamping."""
    xi = np.arange(x0, x0 + bx) % field.shape[0]
    yi = np.clip(np.arange(y0, y0 + by), 0, field.shape[1] - 1)
    return field[np.ix_(xi, yi)]


def block_motion(
    a: np.ndarray, b: np.ndarray, block: int = 16, stride: int = 8
) -> dict[str, np.ndarray]:
    """Per-block motion field. Returns sx, sy, coherence, and block centre indices."""
    nx, ny = a.shape
    xs = np.arange(0, nx, stride)
    ys = np.arange(0, max(ny - block, 0) + 1, stride)
    sx = np.zeros((xs.size, ys.size))
    sy = np.zeros_like(sx)
    co = np.zeros_like(sx)
    for i, x0 in enumerate(xs):
        for j, y0 in enumerate(ys):
            pa = _take_block(a, x0, y0, block, block)
            pb = _take_block(b, x0, y0, block, block)
            sx[i, j], sy[i, j], co[i, j] = estimate_shift(pa, pb)
    cx = (xs + block / 2) % nx
    cy = np.clip(ys + block / 2, 0, ny - 1)
    return {"sx": sx, "sy": sy, "coherence": co, "cx": cx, "cy": cy}


def warp(field: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Shift `field` by (sx, sy) cells with bilinear interpolation.

    Longitude (axis 0) wraps; latitude (axis 1) uses nearest-edge. warp(a, sx, sy)[x, y]
    approximates a[x - sx, y - sy], matching the estimate_shift convention.
    """
    from scipy.ndimage import map_coordinates

    nx, ny = field.shape
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    src_x = (gx - sx) % nx
    src_y = np.clip(gy - sy, 0, ny - 1)
    return map_coordinates(
        field, [src_x.ravel(), src_y.ravel()], order=1, mode="grid-wrap"
    ).reshape(field.shape)


def motion_compensation_gain(
    a: np.ndarray,
    b: np.ndarray,
    block: int = 16,
    stride: int = 16,
    min_coherence: float = 0.0,
) -> dict:
    """Codec-style residual comparison, per block.

    For every block: zero-MV residual |b - a| (the "no motion" / persistence mode) against
    motion-compensated residual |b - warp(a, mv)|, plus a single GLOBAL mv for the whole
    field as the "one mode for the entire frame" control.
    Returns per-block arrays so that gains can be stratified rather than only pooled.
    """
    gsx, gsy, gco = estimate_shift(a, b)
    a_glob = warp(a, gsx, gsy)
    nx, ny = a.shape
    xs = np.arange(0, nx, stride)
    ys = np.arange(0, max(ny - block, 0) + 1, stride)
    rows = []
    for x0 in xs:
        for y0 in ys:
            pa = _take_block(a, x0, y0, block, block)
            pb = _take_block(b, x0, y0, block, block)
            sx, sy, co = estimate_shift(pa, pb)
            e_zero = float(np.mean(np.abs(pb - pa)))
            e_glob = float(np.mean(np.abs(pb - _take_block(a_glob, x0, y0, block, block))))
            if co < min_coherence:
                e_mv = e_zero
                sx = sy = 0.0
            else:
                pa_w = warp(pa, sx, sy)
                # drop a border ring: warped edges are contaminated by wrap-around
                m = block // 8 or 1
                e_mv = float(np.mean(np.abs(pb[m:-m, m:-m] - pa_w[m:-m, m:-m])))
                e_zero_in = float(np.mean(np.abs(pb[m:-m, m:-m] - pa[m:-m, m:-m])))
                e_glob_in = float(
                    np.mean(
                        np.abs(
                            pb[m:-m, m:-m]
                            - _take_block(a_glob, x0, y0, block, block)[m:-m, m:-m]
                        )
                    )
                )
                e_zero, e_glob = e_zero_in, e_glob_in
            rows.append((x0, y0, sx, sy, co, e_zero, e_mv, e_glob))
    arr = np.array(rows, dtype=np.float64)
    cols = ("x0", "y0", "sx", "sy", "coherence", "e_zero", "e_mv", "e_global")
    out = {c: arr[:, k] for k, c in enumerate(cols)}
    out["global_mv"] = np.array([gsx, gsy, gco])
    return out
