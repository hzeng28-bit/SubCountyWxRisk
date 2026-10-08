"""Stage 0: how strong is the plain equal-weight ensemble vote?

Same label, same metric, same split for every method -- only the method differs.
Train year(s) fit the climatology rate and the calibration curves; the test year is scored.

    python scripts/02_baseline_vote.py --train 2018 --test 2021 --quantile 0.99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import baselines as bl  # noqa: E402
from wxrisk import hazard, metrics, wb2  # noqa: E402

MM = 1e-3  # ERA5/IFS precipitation is in metres


def load_year(year: int, lead_hours: int) -> dict:
    f = np.load(wb2.cache_path(f"ens_tp24_{year}_lead{lead_hours}h_00z.npz"))
    return {k: f[k] for k in f.files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, nargs="+", default=[2018])
    ap.add_argument("--test", type=int, nargs="+", default=[2021])
    ap.add_argument("--lead-hours", type=int, default=72)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--floor-mm", type=float, default=10.0)
    a = ap.parse_args()

    # ---- climatological thresholds from 1990-2017 (disjoint from every split) -----------
    clim = np.load(wb2.cache_path("era5_tp24_clim_1990_2017_00z.npz"))
    thr_table = hazard.seasonal_thresholds(clim["field"], clim["months"], a.quantile)
    floor = a.floor_mm * MM
    print(
        f"threshold table q={a.quantile} floor={a.floor_mm}mm  "
        f"median thr over cells/seasons = {np.nanmedian(thr_table) / MM:.1f} mm"
    )

    def prep(years: list[int]) -> dict:
        parts = [load_year(y, a.lead_hours) for y in years]
        members = np.concatenate([p["members"] for p in parts])
        truth = np.concatenate([p["truth"] for p in parts])
        months = np.concatenate([p["valid_month"] for p in parts]).astype(np.int64)
        thr = hazard.threshold_for(months, thr_table, floor)
        y = (truth > thr).astype(np.int8)
        return {"members": members, "truth": truth, "months": months, "thr": thr, "y": y}

    tr, te = prep(a.train), prep(a.test)
    print(
        f"train {a.train}: {tr['y'].shape} base rate {tr['y'].mean():.4f} | "
        f"test {a.test}: {te['y'].shape} base rate {te['y'].mean():.4f}"
    )

    # ---- B0 climatology ------------------------------------------------------------------
    p0 = bl.climatology_prob(tr["y"], tr["months"], te["months"])

    # ---- B1 equal-weight vote ------------------------------------------------------------
    v_tr = bl.vote_fraction(tr["members"], tr["thr"])
    v_te = bl.vote_fraction(te["members"], te["thr"])

    # ---- B2/B3 calibrated vote -----------------------------------------------------------
    p2 = bl.calibrate_global(v_tr, tr["y"], v_te)
    nlat = tr["y"].shape[2]
    band = np.broadcast_to(np.arange(nlat) // 4, tr["y"].shape[1:])  # 8 latitude bands
    p3 = bl.calibrate_grouped(v_tr, tr["y"], v_te, np.ascontiguousarray(band))

    # ---- B4 vote + spread features -------------------------------------------------------
    f_tr = bl.spread_features(tr["members"], tr["thr"])
    f_te = bl.spread_features(te["members"], te["thr"])
    p4 = bl.logistic_on_features(f_tr, tr["y"], f_te)

    rows = [
        metrics.summarize("B0 climatology", p0, te["y"], p0),
        metrics.summarize("B1 equal-weight vote", v_te, te["y"], p0),
        metrics.summarize("B2 vote + global isotonic", p2, te["y"], p0),
        metrics.summarize("B3 vote + per-band isotonic", p3, te["y"], p0),
        metrics.summarize("B4 logistic on ens features", p4, te["y"], p0),
    ]
    hdr = (
        f"{'method':30s} {'BS':>9s} {'BSS/B0':>8s} {'rel':>9s} "
        f"{'res':>9s} {'AUC-PR':>7s} {'AUC':>6s}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['method']:30s} {r['BS']:9.3e} {r['BSS_vs_ref']:8.4f} "
            f"{r['reliability']:9.3e} {r['resolution']:9.3e} {r['AUC_PR']:7.4f} {r['AUC_ROC']:6.4f}"
        )
    print(f"\ntest base rate = {rows[0]['base_rate']:.4f}")

    # ---- paired bootstrap over init days: are the small gaps real? ----------------------
    print("\nPaired bootstrap of BS difference vs B1 (resampling init days, 2000 reps):")
    print("  negative = better than B1; CI spanning 0 = indistinguishable")
    for name, p in [("B2", p2), ("B3", p3), ("B4", p4), ("B0", p0)]:
        d, lo, hi = metrics.paired_bootstrap_bs_diff(p, v_te, te["y"])
        verdict = "better" if hi < 0 else ("worse" if lo > 0 else "tie")
        print(f"  {name} - B1: {d:+.3e}  95% CI [{lo:+.3e}, {hi:+.3e}]  -> {verdict}")

    # ---- stratified: a single pooled global number hides where the skill is --------------
    print("\nBSS of B1 (vs B0) by latitude band:")
    for g in range(8):
        m = band == g
        p_ref = p0[:, m]
        print(
            f"  band {g} (lat idx {g * 4}-{g * 4 + 3}): "
            f"base {te['y'][:, m].mean():.4f} "
            f"BSS {metrics.brier_skill_score(v_te[:, m], te['y'][:, m], p_ref):7.4f}"
        )


if __name__ == "__main__":
    main()
