"""Scaling parameters: fitted on the training period only, stored as JSON, applied as a pure function.

Why separate fit and apply: the parameters are an artifact with an identity (hash), so a model, a test set and a
teammate's machine all use the same numbers, and the fit rows are recorded. `fit` demands an explicit training
mask, because letting test years influence mean/scale is the leak that AGENTS.md's year-split rule exists to stop.

Reproducibility: parameters are rounded to 12 significant digits before they are stored or used, and `apply` uses
only + - * / (correctly rounded in IEEE-754) plus log1p, whose last bit can differ between C libraries. So the
same params give bit-identical output on one platform, and agreement to ~1e-12 relative across platforms
(`verify_close`).
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl

METHODS = ("standard", "robust", "log1p_standard")
SIG = 12


def _round(x: float) -> float:
    return float(f"{x:.{SIG - 1}e}")


def _stats(v: np.ndarray, method: str) -> tuple[float, float]:
    if method == "log1p_standard":
        v = np.log1p(v)
    if method == "robust":
        q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75], method="linear")
        return _round(med), _round(q3 - q1)
    # sort first: numpy's pairwise sum then sees the same order on every machine
    v = np.sort(v)
    return _round(float(v.mean(dtype=np.float64))), _round(float(v.std(dtype=np.float64)))


def fit(df: pl.DataFrame, methods: dict[str, str], train: pl.Expr, train_desc: str) -> dict:
    """`train` selects the fit rows (e.g. pl.col('eff').dt.year() <= 2016); `train_desc` is stored with the params."""
    sub = df.filter(train)
    params = {"fit_rows": sub.height, "fit_where": train_desc, "cols": {}}
    for col, m in methods.items():
        assert m in METHODS, m
        v = sub[col].drop_nulls().to_numpy().astype(np.float64)
        assert v.size > 0, f"{col}: no non-null training values"
        center, scale = _stats(v, m)
        assert scale > 0, f"{col}: zero scale in the training period"
        params["cols"][col] = {"method": m, "center": center, "scale": scale, "n": int(v.size),
                               "n_null": int(sub.height - v.size)}
    params["sha256"] = digest(params)
    return params


def digest(params: dict) -> str:
    body = {k: v for k, v in params.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def apply(df: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Adds `<col>_z` (float32). Nulls stay null: missingness is carried by the `_missing` flag, never by a fill."""
    assert params["sha256"] == digest(params), "normalisation params were edited after fitting"
    out = []
    for col, p in params["cols"].items():
        x = pl.col(col).cast(pl.Float64)
        if p["method"] == "log1p_standard":
            x = x.log1p()
        out.append(((x - p["center"]) / p["scale"]).cast(pl.Float32).alias(f"{col}_z"))
    return df.with_columns(out)


def verify_close(a: pl.DataFrame, b: pl.DataFrame, rtol: float = 1e-6) -> bool:
    """Cross-platform comparison of normalised columns (float32 output, so rtol well above 1e-12)."""
    for c in (c for c in a.columns if c.endswith("_z")):
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if not np.allclose(x, y, rtol=rtol, atol=1e-7, equal_nan=True):
            return False
    return True
