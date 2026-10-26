"""Does a motion-compensated prior turn sparse rain observations into a dense field? (sparse-to-dense test on hourly MRMS)

Hypothesis (stage 6 idea): a dense but OLD field, advected by the motion estimated from the two frames before it, carries information about where
the rain is NOW, so a few observations plus the advected field beat the same observations alone. MRMS is the stand-in for "truth"; the sparse
observations are a random subset of its ~4 km cells (perfect point observations: an upper bound on what a gauge network gives).

For every rainy hour t of an event (frames t-L-1, t-L, t all valid):
  S    sparse only            inverse-distance interpolation (8 nearest observed cells) of the observations at t
  P    persistence            F[t-L], no observations
  M    advected prior         F[t-L] shifted by L x the motion between F[t-L-1] and F[t-L] (phase correlation), no observations
  PO   persistence + obs      P plus the interpolated residual (obs - P) at the observed cells
  MO   advected prior + obs   M plus the interpolated residual
Scores on the NON-observed cells: RMSE (mm/h), CSI at >= 5 mm/h. Skill = 1 - RMSE(method) / RMSE(S). Intervals: bootstrap over EVENTS
(hours inside an event are strongly dependent, so the event is the unit).

Inputs : $WXRISK_DATA/labels_extra/flood/mrms_hourly/ev_*.npy (+ _meta.npz) from scripts/39_mrms_hourly_events.py
Output : stdout (tee to $WXRISK_DATA/audit/mrms_sparse.txt) and audit/mrms_sparse.json
Run    : python scripts/46_mrms_sparse_reconstruction.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import shift as nd_shift
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import (
    compute,  # noqa: E402
    wb2,  # noqa: E402
)

DIR = wb2.cache_path("labels_extra") / "flood" / "mrms_hourly"
FRACTIONS = (0.005, 0.02, 0.08)
LAGS = (1, 3)
CSI_AT = 5.0
RAINY = 0.02  # an hour is evaluated when >= 2% of the cells report >= 1 mm/h
MIN_COH = 3.0  # phase-correlation peak over the mean correlation; below it the motion is not trusted and set to zero
REPS = 3  # independent random observation sets per hour and fraction
RNG = np.random.default_rng(0)


def idw(shape: tuple[int, int], obs_idx: np.ndarray, values: np.ndarray, k: int = 8) -> np.ndarray:
    ys, xs = np.divmod(obs_idx, shape[1])
    tree = cKDTree(np.c_[ys, xs])
    gy, gx = np.mgrid[: shape[0], : shape[1]]
    d, i = tree.query(np.c_[gy.ravel(), gx.ravel()], k=k)
    w = 1.0 / np.maximum(d, 0.5) ** 2
    return ((w * values[i]).sum(1) / w.sum(1)).reshape(shape)


def motion(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Shift s with b(x) ~ a(x - s), array-axis units, on log1p fields; zero when the peak is weak."""
    s = compute.estimate_shift_batch(np.log1p(a)[None], np.log1p(b)[None])[0]
    return (s[:2], s[2]) if s[2] >= MIN_COH else (np.zeros(2), s[2])


def scores(truth: np.ndarray, est: np.ndarray, free: np.ndarray) -> tuple[float, float]:
    t, e = truth[free], np.maximum(est[free], 0)
    rmse = float(np.sqrt(np.mean((t - e) ** 2)))
    hit = np.sum((t >= CSI_AT) & (e >= CSI_AT))
    union = np.sum((t >= CSI_AT) | (e >= CSI_AT))
    return rmse, (hit / union if union else np.nan)


def event_rows(path: Path) -> list[dict]:
    R = np.array(np.load(path, mmap_mode="r"))
    valid = ~np.isnan(R).all(axis=(1, 2))
    R0 = np.nan_to_num(R, nan=0.0)
    rows = []
    for lag in LAGS:
        for t in range(lag + 1, len(R)):
            if not (valid[t] and valid[t - lag] and valid[t - lag - 1]):
                continue
            if (R0[t] >= 1.0).mean() < RAINY:
                continue
            s, coh = motion(R0[t - lag - 1], R0[t - lag])
            prior_p = R0[t - lag]
            prior_m = nd_shift(prior_p, s * lag, order=1, mode="constant", cval=0.0)
            for f in FRACTIONS:
                n = max(int(f * R0[t].size), 16)
                for _ in range(REPS):
                    idx = RNG.choice(R0[t].size, n, replace=False)
                    free = np.ones(R0[t].size, bool)
                    free[idx] = False
                    free = free.reshape(R0[t].shape)
                    obs = R0[t].ravel()[idx]
                    est = {
                        "S": idw(R0[t].shape, idx, obs),
                        "P": prior_p,
                        "M": prior_m,
                        "PO": prior_p + idw(R0[t].shape, idx, obs - prior_p.ravel()[idx]),
                        "MO": prior_m + idw(R0[t].shape, idx, obs - prior_m.ravel()[idx]),
                    }
                    for name, e in est.items():
                        rmse, csi = scores(R0[t], e, free)
                        rows.append({"event": path.stem, "lag": lag, "frac": f, "method": name, "rmse": rmse, "csi": csi,
                                     "shift_pix": float(np.hypot(*s)), "coh": coh})
    return rows


def boot(per_event: pd.DataFrame, a: str, b: str, col: str, n: int = 2000) -> tuple[float, float, float]:
    """Mean over events of (a - b) with a percentile bootstrap over events."""
    w = per_event.pivot(index="event", columns="method", values=col)
    d = (w[a] - w[b]).dropna().to_numpy()
    if len(d) < 3:
        return float(d.mean()) if len(d) else np.nan, np.nan, np.nan
    m = RNG.choice(d, (n, len(d))).mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    files = sorted(DIR.glob("ev_*.npy"))
    print(f"{len(files)} events; compute device {compute.device()}", flush=True)
    rows = []
    for f in files:
        r = event_rows(f)
        print(f"  {f.stem}: {len(r) // (5 * len(FRACTIONS) * REPS * len(LAGS)) or len(r)} rows-groups", flush=True)
        rows += r
    df = pd.DataFrame(rows)
    if df.empty:
        print("no evaluable hours")
        return
    out: dict = {"events": len(files), "hours_evaluated": int(df[df.method == "S"].groupby(["event", "lag"]).size().sum() / (len(FRACTIONS) * REPS))}
    pe = df.groupby(["lag", "frac", "event", "method"], as_index=False)[["rmse", "csi", "shift_pix", "coh"]].mean()
    print("\nmean over events of RMSE (mm/h) and CSI(>=5 mm/h), non-observed cells")
    print(pe.groupby(["lag", "frac", "method"])[["rmse", "csi"]].mean().unstack("method").round(3).to_string())
    print("\nRMSE skill against S = 1 - RMSE/RMSE(S), event-bootstrap 95% interval; and the motion effect (MO vs PO, M vs P)")
    res = {}
    for (lag, frac), g in pe.groupby(["lag", "frac"]):
        w = g.pivot(index="event", columns="method", values="rmse")
        key = f"lag{lag}_frac{frac}"
        res[key] = {"n_events": int(w.shape[0])}
        for m in ("P", "M", "PO", "MO"):
            sk = (1 - w[m] / w["S"]).dropna().to_numpy()
            bs = RNG.choice(sk, (2000, len(sk))).mean(1) if len(sk) >= 3 else np.array([np.nan, np.nan])
            res[key][f"skill_{m}"] = [round(float(sk.mean()), 3), round(float(np.nanpercentile(bs, 2.5)), 3), round(float(np.nanpercentile(bs, 97.5)), 3)]
        for a, b in (("MO", "PO"), ("M", "P"), ("MO", "S")):
            d, lo, hi = boot(g, a, b, "rmse")
            res[key][f"rmse_{a}_minus_{b}"] = [round(d, 3), round(lo, 3), round(hi, 3)]
        print(f"  lag {lag} h, {frac:.1%} of cells: " + "; ".join(f"{k} {v}" for k, v in res[key].items() if k != "n_events"))
    sh = pe[pe.method == "M"].groupby("lag")[["shift_pix", "coh"]].mean().round(2)
    print("\nmean estimated shift (pixels, ~4 km) and coherence of the motion:\n", sh.to_string())
    out["results"] = res
    out["note"] = "perfect point observations; MRMS is both the truth and the source of the prior; 8 events is the unit of the intervals"
    p = wb2.DATA_DIR / "audit" / "mrms_sparse.json"
    p.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("wrote", p)


if __name__ == "__main__":
    main()
