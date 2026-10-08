"""Q3: does the motion vector carry TEMPORAL evolution information?

The residual tests in script 05 fit the flow on the very pair they score, so a more flexible
flow trivially lowers the residual (more degrees of freedom). This script removes that
loophole: the flow is estimated on the pair (l-1, l) and then used to predict lead l+1, which
it has never seen. If the motion vector encodes real evolution, warping forward must beat
persistence, and the flow estimated at one interval must correlate with the next one.

Modes compared at lead l+1:
    persistence           : f(l+1) ~= f(l)                      (zero motion)
    extrapolated flow     : f(l+1) ~= warp(f(l), d(l-1 -> l))   (held out)
    oracle flow           : f(l+1) ~= warp(f(l), d(l -> l+1))   (upper bound, not held out)
    linear field extrap   : f(l+1) ~= 2 f(l) - f(l-1)           (amplitude extrapolation, the
                                                                 control that uses no motion)

Run: python scripts/06_motion_extrapolation.py --detrend zonal+highpass
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.wavelet_motion import phase_flow, warp_flow  # noqa: E402


def detrend(field: np.ndarray, mode: str, hp_sigma: float = 12.0) -> np.ndarray:
    if mode == "none":
        return field
    f = field - field.mean(axis=0, keepdims=True) if "zonal" in mode else field
    if "highpass" in mode:
        f = f - gaussian_filter(f, hp_sigma, mode=("wrap", "nearest"))
    return f


def mae(x: np.ndarray, mask: np.ndarray) -> float:
    return float(np.mean(np.abs(x[:, mask])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="seq240_mean_2018_n12_lead120.npz")
    ap.add_argument("--n-init", type=int, default=12)
    ap.add_argument("--dt-steps", type=int, default=1)
    ap.add_argument("--max-triples", type=int, default=6)
    ap.add_argument("--detrend", default="zonal+highpass")
    ap.add_argument("--smooth", type=float, default=0.25)
    ap.add_argument("--wavelengths", type=float, nargs="+", default=[32, 16, 8, 5])
    a = ap.parse_args()

    d = np.load(wb2.cache_path(a.file))
    z, lat = d["z500"], d["latitude"]
    leads = d["lead_hours"]
    mid = (np.abs(lat) >= 25) & (np.abs(lat) <= 70)
    k = a.dt_steps
    wl = tuple(a.wavelengths)
    print(f"{a.file} {z.shape}, dt = {leads[k] - leads[0]:.0f} h, detrend = {a.detrend}, "
          f"smooth = {a.smooth}")

    res = {m: [] for m in ("pers", "extrap", "oracle", "linear", "extrap_half")}
    flow_corr = {"u": [], "v": []}
    n_init = min(a.n_init, z.shape[0])
    triples = [(li, li + k, li + 2 * k) for li in range(0, z.shape[1] - 2 * k)][: a.max_triples]
    print(f"{n_init} inits x {len(triples)} triples = {n_init * len(triples)} cases")

    for ti in range(n_init):
        fields = [detrend(z[ti, li], a.detrend) for li in range(z.shape[1])]
        for i0, i1, i2 in triples:
            f0, f1, f2 = fields[i0], fields[i1], fields[i2]
            p01 = phase_flow(f0, f1, wavelengths=wl, smooth=a.smooth)
            p12 = phase_flow(f1, f2, wavelengths=wl, smooth=a.smooth)
            res["pers"].append(mae(f2 - f1, mid))
            res["extrap"].append(mae(f2 - warp_flow(f1, p01["dx"], p01["dy"]), mid))
            res["extrap_half"].append(
                mae(f2 - warp_flow(f1, 0.5 * p01["dx"], 0.5 * p01["dy"]), mid)
            )
            res["oracle"].append(mae(f2 - warp_flow(f1, p12["dx"], p12["dy"]), mid))
            res["linear"].append(mae(f2 - (2 * f1 - f0), mid))
            flow_corr["u"].append(
                pearsonr(p01["dx"][:, mid].ravel(), p12["dx"][:, mid].ravel())[0]
            )
            flow_corr["v"].append(
                pearsonr(p01["dy"][:, mid].ravel(), p12["dy"][:, mid].ravel())[0]
            )

    p = np.array(res["pers"])
    print("\nResidual at lead l+1 (midlat 25-70), mean over cases")
    print(f"{'mode':34s} {'MAE':>9s} {'vs persistence':>15s} {'win rate':>9s}")
    print("-" * 72)
    for name, label in [
        ("pers", "persistence (zero motion)"),
        ("linear", "linear field extrapolation"),
        ("extrap_half", "0.5 x held-out flow"),
        ("extrap", "HELD-OUT flow from (l-1, l)"),
        ("oracle", "oracle flow from (l, l+1)"),
    ]:
        v = np.array(res[name])
        print(f"{label:34s} {v.mean():9.2f} {v.mean() / p.mean():15.3f} "
              f"{100 * np.mean(v < p):8.0f}%")

    cu, cv = np.array(flow_corr["u"]), np.array(flow_corr["v"])
    print("\nTemporal persistence of the motion field itself (consecutive intervals):")
    print(f"  r(dx_t, dx_t+1) = {np.nanmean(cu):.3f} +- {np.nanstd(cu):.3f}")
    print(f"  r(dy_t, dy_t+1) = {np.nanmean(cv):.3f} +- {np.nanstd(cv):.3f}")
    print("  (a motion field that is pure noise would give ~0 here; this is the direct test")
    print("   of whether the motion vector encodes temporal evolution at all)")


if __name__ == "__main__":
    main()
