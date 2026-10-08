"""Hazard definitions: climatological thresholds and binary event labels.

A hazard event at grid cell (h, w) and valid time t is

    1[ accumulation/intensity over the 24h window ending at t  >  thr(cell, season) ]

The threshold is a per-cell, per-season quantile of a CLIMATOLOGY PERIOD that is disjoint
from the evaluation period, so the label definition cannot leak evaluation-period information.
"""

from __future__ import annotations

import numpy as np

SEASONS = ("DJF", "MAM", "JJA", "SON")
_MONTH_TO_SEASON = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def season_index(months: np.ndarray) -> np.ndarray:
    """Map calendar month numbers (1-12) to season index 0..3."""
    return np.vectorize(_MONTH_TO_SEASON.get)(np.asarray(months)).astype(np.int64)


def seasonal_thresholds(field: np.ndarray, months: np.ndarray, q: float) -> np.ndarray:
    """Per-cell, per-season quantile.

    field: [T, H, W] climatology-period samples (one per non-overlapping window)
    months: [T] calendar month of each sample
    returns: [4, H, W]
    """
    seas = season_index(months)
    out = np.full((4,) + field.shape[1:], np.nan, dtype=np.float64)
    for s in range(4):
        sel = field[seas == s]
        if sel.size:
            out[s] = np.nanquantile(sel, q, axis=0)
    return out


def exceedance_labels(
    field: np.ndarray,
    months: np.ndarray,
    thresholds: np.ndarray,
    floor: float = 0.0,
) -> np.ndarray:
    """Binary labels [T, H, W] from a seasonal threshold table [4, H, W].

    `floor` is an absolute minimum intensity, so that arid cells with a tiny p99 do not
    generate meaningless "extreme" events.
    """
    thr = np.maximum(thresholds[season_index(months)], floor)
    return (field > thr).astype(np.int8)


def threshold_for(months: np.ndarray, thresholds: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Broadcast the seasonal threshold table to [T, H, W] (shared by labels and member votes)."""
    return np.maximum(thresholds[season_index(months)], floor)
