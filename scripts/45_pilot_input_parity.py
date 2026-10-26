"""Parity of the flood-pilot inputs built from the CLEAN tables against the ones the pilot table was built from.

Compares, on the 2010 block-group geography of the pilot:
  exposure   n_pol / n_sfha / n_single / cov_sum per block group and day: clean policies (coverage_end) vs the raw extract
             (effective + 366 days, minus mid-term cancellations) and vs the stored expo_2010_2023.npz
  claims     cumulative non-tropical inland claim counts: clean claims table vs the stored claims_cum.npy
Prints totals, the share of block-group-days that differ and the largest relative difference; `--write` stores the clean-built
arrays as expo_2010_2023_clean.npz / claims_cum_clean.npy next to the originals (scripts/25 reads the originals unless told otherwise).

    python scripts/45_pilot_input_parity.py [--write] | tee $WXRISK_DATA/audit/pilot_input_parity.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402


def compare(name: str, a: np.ndarray, b: np.ndarray) -> None:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    diff = ~np.isclose(a, b, rtol=1e-6, atol=1e-6)
    denom = np.maximum(np.abs(b), 1.0)
    print(f"  {name:9s} total new {a.sum():.6g} old {b.sum():.6g} ({a.sum() / b.sum() - 1:+.3%}); "
          f"cells differing {diff.mean():.3%}; max |diff|/max(old,1) {np.max(np.abs(a - b) / denom):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ids = fp.bg_geometry().bg10.tolist()
    stored = np.load(fp.PILOT / "expo_2010_2023.npz")
    assert list(stored["bg"]) == ids
    new, raw = fp.exposure_matrices(ids), fp.exposure_matrices_raw(ids)
    print("exposure, clean vs stored npz")
    for k in new:
        compare(k, new[k], stored[k])
    print("exposure, raw function re-run vs stored npz (does the stored file come from this code?)")
    for k in raw:
        compare(k, raw[k], stored[k])
    cl_new = fp.claim_cumulative(fp.claims_tx(), ids)
    cl_old = np.load(fp.PILOT / "claims_cum.npy", mmap_mode="r")
    print("claims, clean vs stored claims_cum.npy")
    compare("claims", cl_new, cl_old)
    print(f"  claims in window: new {int(cl_new[:, -1].sum()):,} old {int(cl_old[:, -1].sum()):,}")
    if a.write:
        np.savez(fp.PILOT / "expo_2010_2023_clean.npz", bg=np.array(ids), **new)
        np.save(fp.PILOT / "claims_cum_clean.npy", cl_new)
        print("written expo_2010_2023_clean.npz, claims_cum_clean.npy")


if __name__ == "__main__":
    main()
