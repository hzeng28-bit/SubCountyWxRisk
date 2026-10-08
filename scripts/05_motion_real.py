"""Stage 1 on real IFS ENS fields: is the estimated motion vector physically real?

Two questions, both answerable without any learning:

  Q1 (INDEPENDENT cross-check) Does the estimated displacement of the Z500 field between two
     lead times agree with the model's own 500 hPa WIND field? The wind never enters the phase
     estimation, so this is not self-referential -- unlike "my motion vector reduces my own
     residual", which it does by construction.
  Q2 (codec gain) Does motion compensation reduce the residual relative to the zero-motion
     ("intra"/persistence) mode, and does a per-pixel/per-scale vector beat one vector per
     frame?

Run: python scripts/05_motion_real.py --dt-steps 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.motion import block_motion  # noqa: E402
from wxrisk.wavelet_motion import phase_flow, warp_flow  # noqa: E402

DEG_M = 111_320.0


def load_seq(name: str) -> dict:
    f = np.load(wb2.cache_path(name))
    return {k: f[k] for k in f.files}


def cells_to_ms(
    dx: np.ndarray, dy: np.ndarray, lat: np.ndarray, dlon_deg: float, dlat_deg: float, dt_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Grid-cell displacement -> m/s, with the cos(lat) metric factor on longitude."""
    coslat = np.cos(np.deg2rad(lat))[None, :]
    u = dx * dlon_deg * DEG_M * coslat / dt_s
    v = dy * dlat_deg * DEG_M / dt_s
    return u, v


def block_to_pixels(a: np.ndarray, b: np.ndarray, block: int, stride: int) -> tuple:
    m = block_motion(a, b, block=block, stride=stride)
    nx, ny = a.shape
    zx, zy = nx / m["sx"].shape[0], ny / m["sx"].shape[1]
    px = zoom(m["sx"], (zx, zy), order=1, mode="nearest")[:nx, :ny]
    py = zoom(m["sy"], (zx, zy), order=1, mode="nearest")[:nx, :ny]
    return px, py


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def detrend(field: np.ndarray, mode: str, hp_sigma: float = 12.0) -> np.ndarray:
    """Remove the quasi-stationary background before estimating motion.

    Z500 is dominated in amplitude by the climatological pole-to-equator geopotential
    gradient, which does NOT advect. Any amplitude-weighted phase estimator is therefore
    pulled toward zero displacement by a feature that never moves. Weather features are
    anomalies on top of that background, so the background has to go first.
    """
    if mode == "none":
        return field
    if mode == "zonal":  # anomaly from the zonal mean at each latitude
        return field - field.mean(axis=0, keepdims=True)
    if mode == "highpass":  # keep scales smaller than hp_sigma cells
        return field - gaussian_filter(field, hp_sigma, mode=("wrap", "nearest"))
    if mode == "zonal+highpass":
        f = field - field.mean(axis=0, keepdims=True)
        return f - gaussian_filter(f, hp_sigma, mode=("wrap", "nearest"))
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="seq240_mean_2018_n12_lead120.npz")
    ap.add_argument("--dt-steps", type=int, default=1, help="lead steps between the two frames")
    ap.add_argument("--n-init", type=int, default=12)
    ap.add_argument("--max-pairs-per-init", type=int, default=4)
    ap.add_argument("--wavelengths", type=float, nargs="+", default=[32, 16, 8, 5])
    ap.add_argument(
        "--detrend",
        default="none",
        choices=["none", "zonal", "highpass", "zonal+highpass"],
    )
    ap.add_argument("--hp-sigma", type=float, default=12.0)
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--ridge", type=float, default=1e-3)
    a = ap.parse_args()

    d = load_seq(a.file)
    z, u5, v5 = d["z500"], d["u500"], d["v500"]
    lat, lon = d["latitude"], d["longitude"]
    leads = d["lead_hours"]
    dlat_deg = float(abs(lat[1] - lat[0]))
    dlon_deg = float(abs(lon[1] - lon[0]))
    dt_s = float(leads[a.dt_steps] - leads[0]) * 3600.0
    print(
        f"{a.file}: {z.shape} grid {dlon_deg}x{dlat_deg} deg, dt = {dt_s / 3600:.0f} h, "
        f"one cell = {dlon_deg * DEG_M / 1000:.0f} km at the equator"
    )
    print(f"a 20 m/s system moves {20 * dt_s / 1000:.0f} km in dt "
          f"= {20 * dt_s / (dlon_deg * DEG_M):.2f} cells")

    # latitude masks: geostrophic advection of Z500 is a midlatitude story
    mid = (np.abs(lat) >= 25) & (np.abs(lat) <= 70)
    trop = np.abs(lat) < 25
    acc: dict[str, list] = {k: [] for k in ("wav", "fft32", "fft16")}
    rows_u: dict[str, list] = {k: [] for k in ("wav", "fft32", "fft16")}
    rows_v: dict[str, list] = {k: [] for k in ("wav", "fft32", "fft16")}
    truth_u: list = []
    truth_v: list = []
    wind_cells: list = []
    est_cells: list = []
    resid: dict[str, list] = {
        k: [] for k in ("zero", "wav", "fft32", "fft16", "global", "wind", "wind_half")
    }

    n_init = min(a.n_init, z.shape[0])
    lead_idx = list(range(0, z.shape[1] - a.dt_steps))[: a.max_pairs_per_init]
    print(f"pairs: {n_init} inits x {len(lead_idx)} lead pairs = {n_init * len(lead_idx)}")
    print(f"detrend = {a.detrend} (hp_sigma = {a.hp_sigma} cells)")

    for ti in range(n_init):
        for li in lead_idx:
            fa = detrend(z[ti, li], a.detrend, a.hp_sigma)
            fb = detrend(z[ti, li + a.dt_steps], a.detrend, a.hp_sigma)
            # mid-interval wind, the physically correct advecting flow for the interval
            uw = 0.5 * (u5[ti, li] + u5[ti, li + a.dt_steps])
            vw = 0.5 * (v5[ti, li] + v5[ti, li + a.dt_steps])

            pf = phase_flow(
                fa, fb, wavelengths=tuple(a.wavelengths), smooth=a.smooth, ridge=a.ridge
            )
            ests = {
                "wav": (pf["dx"], pf["dy"]),
                "fft32": block_to_pixels(fa, fb, 32, 16),
                "fft16": block_to_pixels(fa, fb, 16, 8),
            }
            resid["zero"].append(mae(fb - fa))
            for name, (dx, dy) in ests.items():
                ue, ve = cells_to_ms(dx, dy, lat, dlon_deg, dlat_deg, dt_s)
                # compare at the analysis scale: smooth the wind to the coarsest wavelet window
                sig = 0.5 * float(min(a.wavelengths))
                uw_s = gaussian_filter(uw, sig, mode=("wrap", "nearest"))
                vw_s = gaussian_filter(vw, sig, mode=("wrap", "nearest"))
                rows_u[name].append(ue[:, mid].ravel())
                rows_v[name].append(ve[:, mid].ravel())
                if name == "wav":
                    truth_u.append(uw_s[:, mid].ravel())
                    truth_v.append(vw_s[:, mid].ravel())
                resid[name].append(mae(fb - warp_flow(fa, dx, dy)))
            # one vector for the whole frame = "one mode per frame" control
            gx, gy = block_to_pixels(fa, fb, min(fa.shape), min(fa.shape))
            resid["global"].append(mae(fb - warp_flow(fa, gx, gy)))
            # PHYSICS control: warp by the model's own wind (pure advection). If this is worse
            # than the estimated flow, then a low correlation with the wind is expected rather
            # than a failure -- feature phase speed is not the wind speed (Rossby waves
            # propagate westward relative to the flow).
            coslat = np.cos(np.deg2rad(lat))[None, :]
            wdx = uw * dt_s / (dlon_deg * DEG_M * coslat)
            wdy = vw * dt_s / (dlat_deg * DEG_M)
            resid["wind"].append(mae(fb - warp_flow(fa, wdx, wdy)))
            resid["wind_half"].append(mae(fb - warp_flow(fa, 0.5 * wdx, 0.5 * wdy)))
            wind_cells.append((np.median(np.abs(wdx[:, mid])), np.median(np.abs(wdy[:, mid]))))
            est_cells.append(
                (np.median(np.abs(pf["dx"][:, mid])), np.median(np.abs(pf["dy"][:, mid])))
            )
            acc["wav"].append((np.abs(pf["dx"]).mean(), np.abs(pf["dy"]).mean()))

    tu = np.concatenate(truth_u)
    tv = np.concatenate(truth_v)
    print("\nQ1  estimated displacement vs the model's own 500 hPa wind (midlat 25-70 deg)")
    print(f"{'method':10s} {'r(u)':>7s} {'rho(u)':>7s} {'slope_u':>8s} "
          f"{'r(v)':>7s} {'rho(v)':>7s} {'slope_v':>8s}")
    print("-" * 60)
    for name in ("wav", "fft32", "fft16"):
        eu = np.concatenate(rows_u[name])
        ev = np.concatenate(rows_v[name])
        ru = pearsonr(eu, tu)[0]
        rv = pearsonr(ev, tv)[0]
        su = np.polyfit(tu, eu, 1)[0]
        sv = np.polyfit(tv, ev, 1)[0]
        print(f"{name:10s} {ru:7.3f} {spearmanr(eu, tu)[0]:7.3f} {su:8.3f} "
              f"{rv:7.3f} {spearmanr(ev, tv)[0]:7.3f} {sv:8.3f}")
    print(f"  wind magnitude: median |u| {np.median(np.abs(tu)):.1f} m/s, "
          f"|v| {np.median(np.abs(tv)):.1f} m/s")
    print(f"  tropics excluded: {int(trop.sum())} of {lat.size} latitudes")

    print("\nQ2  motion-compensated residual (Z500, m^2/s^2), mean over all pairs")
    z0 = np.mean(resid["zero"])
    print(f"{'mode':26s} {'MAE':>10s} {'vs zero-MV':>11s}")
    print("-" * 50)
    for name, label in [
        ("zero", "zero-MV (persistence)"),
        ("wind", "warp by true 500hPa wind"),
        ("wind_half", "warp by 0.5 x true wind"),
        ("global", "one vector per frame"),
        ("fft32", "FFT block=32"),
        ("fft16", "FFT block=16"),
        ("wav", "wavelet phase flow"),
    ]:
        m = np.mean(resid[name])
        print(f"{label:26s} {m:10.2f} {m / z0:11.3f}")
    wc = np.array(wind_cells)
    ec = np.array(est_cells)
    print(
        f"  displacement magnitudes (cells, midlat median): wind |dx| {wc[:, 0].mean():.2f} "
        f"|dy| {wc[:, 1].mean():.2f}  |  wavelet |dx| {ec[:, 0].mean():.2f} "
        f"|dy| {ec[:, 1].mean():.2f}"
    )
    w = np.array(resid["wav"])
    zz = np.array(resid["zero"])
    print(f"  wavelet beats zero-MV on {100 * np.mean(w < zz):.0f}% of pairs "
          f"({len(w)} pairs); median ratio {np.median(w / zz):.3f}")


if __name__ == "__main__":
    main()
