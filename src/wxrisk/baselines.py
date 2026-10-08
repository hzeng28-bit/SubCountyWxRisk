"""Baselines B0-B4. Every one of them is fit on TRAIN only and applied to TEST.

Shapes: fields are [T, H, W]; ensemble fields are [T, M, H, W].
All calibrators are fit on the training split only; nothing here ever sees test labels.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .hazard import season_index


# ---------------------------------------------------------------- B0 climatology
def climatology_prob(
    y_train: np.ndarray, months_train: np.ndarray, months_test: np.ndarray, prior: float = 1.0
) -> np.ndarray:
    """Per-cell, per-season event rate from the training split, Laplace-smoothed.

    `prior` is a pseudo-count toward the global base rate (shrinkage for cells that never
    saw an event in training).
    """
    s_tr, s_te = season_index(months_train), season_index(months_test)
    glob = float(y_train.mean())
    rate = np.zeros((4,) + y_train.shape[1:], dtype=np.float64)
    for s in range(4):
        sel = y_train[s_tr == s]
        n = sel.shape[0]
        rate[s] = (sel.sum(axis=0) + prior * glob) / (n + prior) if n else glob
    return rate[s_te]


# ---------------------------------------------------------------- B1 democratic vote
def vote_fraction(members: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Equal-weight member vote: fraction of members exceeding the threshold.

    members: [T, M, H, W], thr: [T, H, W] broadcast over members.
    This is what operational ensemble probability does.
    """
    return (members > thr[:, None]).mean(axis=1)


# ---------------------------------------------------------------- calibration
def _fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p.ravel(), y.ravel())
    return iso


def calibrate_global(
    p_train: np.ndarray, y_train: np.ndarray, p_test: np.ndarray
) -> np.ndarray:
    """B2: one isotonic curve shared by every grid cell."""
    iso = _fit_isotonic(p_train, y_train)
    return iso.predict(p_test.ravel()).reshape(p_test.shape)


def calibrate_grouped(
    p_train: np.ndarray,
    y_train: np.ndarray,
    p_test: np.ndarray,
    group: np.ndarray,
    min_events: int = 30,
) -> np.ndarray:
    """B3: one isotonic curve per spatial group (e.g. latitude band), global fallback.

    group: [H, W] integer group id. Groups with fewer than `min_events` training events fall
    back to the global curve -- fitting a calibration curve on 3 events is noise, not skill.
    """
    out = np.empty_like(p_test, dtype=np.float64)
    iso_glob = _fit_isotonic(p_train, y_train)
    out[:] = iso_glob.predict(p_test.ravel()).reshape(p_test.shape)
    for g in np.unique(group):
        mask = group == g
        ytr = y_train[:, mask]
        if ytr.sum() < min_events:
            continue
        iso = _fit_isotonic(p_train[:, mask], ytr)
        out[:, mask] = iso.predict(p_test[:, mask].ravel()).reshape(p_test[:, mask].shape)
    return out


# ---------------------------------------------------------------- B4 vote + spread features
def spread_features(members: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Scalar ensemble summaries per cell, all normalized by the local threshold.

    Returns [T, H, W, F]. Dividing by the threshold is the per-cell robust-scale
    normalization that lets cells of wildly different climatological intensity be pooled.
    """
    t = np.maximum(thr, 1e-12)[:, None]
    x = members / t
    feats = [
        (x > 1.0).mean(axis=1),            # vote fraction
        x.mean(axis=1),                    # ensemble mean, in threshold units
        x.std(axis=1),                     # spread
        np.max(x, axis=1),                 # ensemble max
        np.median(x, axis=1),
        np.quantile(x, 0.9, axis=1),
    ]
    return np.stack(feats, axis=-1)


def logistic_on_features(
    f_train: np.ndarray, y_train: np.ndarray, f_test: np.ndarray, c: float = 1.0
) -> np.ndarray:
    """Pooled logistic regression on the ensemble summary features.

    Deliberately a single GLOBAL model: the point is to measure whether a pooled linear
    combiner over heterogeneous cells beats the plain vote (prior work says it often does
    not), not to win.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ftr = f_train.reshape(-1, f_train.shape[-1])
    fte = f_test.reshape(-1, f_test.shape[-1])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
    model.fit(ftr, y_train.ravel())
    return model.predict_proba(fte)[:, 1].reshape(f_test.shape[:-1])
