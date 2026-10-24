"""Poisson regression with group fixed effects concentrated out (no statsmodels dependency).

For counts y with exposure `expo` and covariates x, E[y] = expo * exp(alpha_g + x @ beta). For a given beta the
group effect has a closed form, alpha_g = log(sum_g y / sum_g expo * exp(x @ beta)), so only beta is searched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def poisson_fe(y, x, expo, grp) -> np.ndarray:
    """beta for covariate matrix x (n, p)."""
    codes, uniq = pd.factorize(grp)
    y, expo = np.asarray(y, float), np.asarray(expo, float)
    s_y = np.bincount(codes, weights=y, minlength=len(uniq))

    def nll(b):
        eta = x @ b
        s_mu = np.bincount(codes, weights=expo * np.exp(eta), minlength=len(uniq))
        keep = s_mu > 0
        return -(np.sum(y * eta) - np.sum(s_y[keep] * np.log(s_mu[keep])))

    return minimize(nll, np.zeros(x.shape[1]), method="BFGS").x


def group_effects(y, x, expo, grp, beta) -> pd.Series:
    """alpha_g given beta, indexed by group label."""
    codes, uniq = pd.factorize(grp)
    s_y = np.bincount(codes, weights=np.asarray(y, float), minlength=len(uniq))
    s_mu = np.bincount(codes, weights=np.asarray(expo, float) * np.exp(x @ beta), minlength=len(uniq))
    return pd.Series(np.log(np.clip(s_y, 0.5, None) / np.clip(s_mu, 1e-12, None)), index=uniq)
