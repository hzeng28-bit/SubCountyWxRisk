"""Shared pieces for the SCS3 county-day experiments (K-means baseline, scale encoder)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .prep.scaling import FeaturePlan, FeatureRule

HEADS = ["y0", "y10k", "y100k", "y1m"]
SHRINK = 20.0  # pseudo-counts pulling a cluster toward its prior
STATIC = {"doy_s", "doy_c", "lat", "lon"}


def anomalise_plain(df: pd.DataFrame, cols: list[str], fit: pd.Series) -> np.ndarray:
    """The original normalisation (kept for the comparison in RESULTS stage 14): raw county x month mean, NaN for a county-month unseen in the fit rows."""
    out = np.empty((len(df), len(cols)), np.float32)
    key = df.cty + "_" + df.month.astype(str)
    fit = fit.to_numpy()
    for j, c in enumerate(cols):
        x = df[c]
        if c in STATIC:
            out[:, j] = (x - x[fit].mean()) / x[fit].std()
            continue
        mu = x[fit].groupby(key[fit]).mean()
        z = x - key.map(mu).to_numpy()
        out[:, j] = z / z[fit].std()
    return out


def scs_plan(cols: list[str]) -> FeaturePlan:
    """County x month anomaly (fitted on the training rows, shrunk toward the global mean for sparse cells, global fallback for an unseen cell), unit variance on the fit rows."""
    return FeaturePlan(tuple(
        FeatureRule(c, basis="calendar / location: standardise only") if c in STATIC
        else FeatureRule(c, group="_ck", group_stat="center_z", basis="county x month anomaly (docs/SCALING_PLAN.md, SCS K-means)")
        for c in cols))


def anomalise(df: pd.DataFrame, cols: list[str], fit: pd.Series) -> np.ndarray:
    """Subtract the train county x month mean (shrunk), scale by the train anomaly std; calendar/location are only standardised. Fit rows only."""
    d = df[cols].assign(_ck=(df.cty + "_" + df.month.astype(str)).to_numpy())
    plan = scs_plan(cols)
    prm = plan.fit(d, fit.to_numpy(), "fit rows of the experiment")
    return plan.apply(d, prm).to_numpy()


def climatology(df: pd.DataFrame, fit: pd.Series) -> dict[str, np.ndarray]:
    """p0 = county rate x month rate / overall rate, fit on `fit` rows only."""
    out = {}
    for h in HEADS:
        base = df.loc[fit, h].mean()
        r_c = df.loc[fit].groupby("cty")[h].mean()
        r_m = df.loc[fit].groupby("month")[h].mean()
        p0 = df.cty.map(r_c).to_numpy() * df.month.map(r_m).to_numpy() / base
        out[h] = np.clip(p0, 1e-5, 1.0)
    return out


def cluster_stats(lab: np.ndarray, fit: np.ndarray, p0: np.ndarray, k: int, y: np.ndarray):
    """Per-cluster (rate, lift) from the fit rows."""
    n = np.bincount(lab[fit], minlength=k)
    pos = np.bincount(lab[fit], weights=y[fit], minlength=k)
    exp = np.bincount(lab[fit], weights=p0[fit], minlength=k)
    prior = exp / np.maximum(n, 1)
    return (pos + SHRINK * prior) / (n + SHRINK), (pos + SHRINK) / (exp + SHRINK)


def readouts(lab: np.ndarray, fit: np.ndarray, p0: np.ndarray, k: int, y: np.ndarray):
    rate, lift = cluster_stats(lab, fit, p0, k, y)
    return rate[lab], np.clip(p0 * lift[lab], 0, 1)


def fit_predict(X: np.ndarray, fit: np.ndarray, k: int, seed: int = 0, n_init: int = 3,
                max_fit: int | None = None) -> np.ndarray:
    idx = np.flatnonzero(fit)
    if max_fit and len(idx) > max_fit:
        idx = np.random.default_rng(seed).choice(idx, max_fit, replace=False)
    return KMeans(k, n_init=n_init, random_state=seed).fit(X[idx]).predict(X)


def brier(p, y) -> float:
    return float(np.mean((p - y) ** 2))


def bss_ci(p, p_ref, y, years, rng, n_boot=1000):
    """BSS of p against p_ref with year-grouped bootstrap; returns (point, lo, hi)."""
    ys = np.unique(years)
    se = np.array([((p - y)[years == v] ** 2).sum() for v in ys])
    se0 = np.array([((p_ref - y)[years == v] ** 2).sum() for v in ys])
    pt = 1 - se.sum() / se0.sum()
    idx = rng.integers(0, len(ys), (n_boot, len(ys)))
    bs = 1 - se[idx].sum(1) / se0[idx].sum(1)
    return pt, *np.percentile(bs, [2.5, 97.5])
