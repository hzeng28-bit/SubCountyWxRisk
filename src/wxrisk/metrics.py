"""Probabilistic verification metrics for rare binary hazard events."""

from __future__ import annotations

import numpy as np


def brier(p: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[m] - y[m]) ** 2))


def brier_skill_score(p: np.ndarray, y: np.ndarray, p_ref: np.ndarray) -> float:
    """BSS against an explicit reference forecast (not against the pooled base rate)."""
    bs, bs_ref = brier(p, y), brier(p_ref, y)
    return float(1.0 - bs / bs_ref) if bs_ref > 0 else float("nan")


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 11) -> dict:
    """Murphy decomposition: BS = reliability - resolution + uncertainty."""
    m = np.isfinite(p) & np.isfinite(y)
    p, y = p[m], y[m]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ybar = y.mean()
    rel = res = 0.0
    rows = []
    for b in range(bins):
        s = idx == b
        n = int(s.sum())
        if n == 0:
            rows.append((b, 0, np.nan, np.nan))
            continue
        pk, ok = float(p[s].mean()), float(y[s].mean())
        rel += n * (pk - ok) ** 2
        res += n * (ok - ybar) ** 2
        rows.append((b, n, pk, ok))
    n_tot = y.size
    return {
        "rows": rows,
        "reliability": rel / n_tot,
        "resolution": res / n_tot,
        "uncertainty": float(ybar * (1 - ybar)),
        "base_rate": float(ybar),
    }


def auc_pr(p: np.ndarray, y: np.ndarray) -> float:
    """Average precision; the right summary when the base rate is ~1%."""
    from sklearn.metrics import average_precision_score

    m = np.isfinite(p) & np.isfinite(y)
    if y[m].sum() == 0:
        return float("nan")
    return float(average_precision_score(y[m], p[m]))


def auc_roc(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    m = np.isfinite(p) & np.isfinite(y)
    if y[m].sum() == 0 or y[m].sum() == y[m].size:
        return float("nan")
    return float(roc_auc_score(y[m], p[m]))


def paired_bootstrap_bs_diff(
    p_a: np.ndarray,
    p_b: np.ndarray,
    y: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for BS(a) - BS(b), resampling whole init times.

    Resampling init times (not individual grid cells) is required: all ~2000 cells of one
    forecast share the same synoptic situation and are nowhere near independent.
    Returns (mean diff, 2.5%, 97.5%). Negative means method a has the lower Brier score.
    """
    rng = np.random.default_rng(seed)
    da = (p_a - y) ** 2
    db = (p_b - y) ** 2
    per_init = np.nanmean(da.reshape(da.shape[0], -1), axis=1) - np.nanmean(
        db.reshape(db.shape[0], -1), axis=1
    )
    n = per_init.shape[0]
    boots = per_init[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    return float(per_init.mean()), float(np.quantile(boots, 0.025)), float(
        np.quantile(boots, 0.975)
    )


def summarize(name: str, p: np.ndarray, y: np.ndarray, p_ref: np.ndarray) -> dict:
    rt = reliability_table(p, y)
    return {
        "method": name,
        "BS": brier(p, y),
        "BSS_vs_ref": brier_skill_score(p, y, p_ref),
        "reliability": rt["reliability"],
        "resolution": rt["resolution"],
        "AUC_PR": auc_pr(p, y),
        "AUC_ROC": auc_roc(p, y),
        "base_rate": rt["base_rate"],
    }
