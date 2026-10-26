"""Scaling stage of the pipeline: feature plans (transform, scale, group-aware, discretise) and target transforms with their inverse.

Two different jobs that must not be mixed up:

  features   never come back out of the model, so no inverse is needed; their plan only has to be fitted on the training rows and applied
             identically everywhere (`FeaturePlan`).
  targets    the model's output lives in the transformed space and MUST be mapped back before it is scored or reported (`TargetTransform`).
             `Scaled` tags such an output with the space it is in, and the metrics here refuse anything that is not in original units, so
             a forgotten inverse is an error, not a wrong number.

Group-aware statistics come in two kinds, chosen per column in the plan (docs/SCALING_PLAN.md says when):
  fitted    per-group centre / scale computed on the TRAINING rows, shrunk toward the global value for small groups, with a global fallback for a group
            never seen in training (e.g. the SCS county x month anomalies);
  sample    computed from the rows of the same group in the batch being transformed (within-episode z-score / percentile). It uses no label and
            no other group, but it needs the whole group to be present at prediction time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SIG = 12
SHRINK = 20.0  # pseudo-observations pulling a small group's mean toward the global mean


# ------------------------------------------------------------------ elementwise transforms and their inverses

TRANSFORMS = {
    "identity": (lambda x: x, lambda y: y),
    "log1p": (np.log1p, np.expm1),
    "log": (np.log, np.exp),
    "asinh": (np.arcsinh, np.sinh),
    "sqrt": (np.sqrt, np.square),
    "logit": (lambda p: np.log(p / (1 - p)), lambda z: 1 / (1 + np.exp(-z))),
}


def _r(x: float) -> float:
    return float(f"{x:.{SIG - 1}e}")


# ------------------------------------------------------------------ features


@dataclass(frozen=True)
class FeatureRule:
    col: str
    transform: str = "identity"  # one of TRANSFORMS, applied first
    scale: str = "standard"  # standard | robust | none
    cyclic: float | None = None  # period: the column becomes <col>_sin, <col>_cos (no scaling afterwards)
    group: str | None = None  # group-aware column (name of the grouping column)
    group_mode: str = "fitted"  # fitted | sample
    group_stat: str = "center"  # center | center_z | zscore | percentile   (percentile: sample mode only; center_z: fitted only)
    bins: tuple[float, ...] | None = None  # supervised cut points (fitted by `fit_bins`); output is the bin index one-hot
    basis: str = ""  # why: evidence key or provider statement, quoted in the plan document


@dataclass
class FeaturePlan:
    rules: tuple[FeatureRule, ...]
    min_group: int = 5

    def digest(self) -> str:
        blob = json.dumps([r.__dict__ for r in self.rules], sort_keys=True, default=list) + str(self.min_group)
        return hashlib.sha256(blob.encode()).hexdigest()

    # ------------------------------------------------ fit
    def fit(self, df: pd.DataFrame, train: np.ndarray, train_desc: str) -> dict:
        """Parameters from the training rows only. `train` is an explicit boolean mask (the leak guard); `train_desc` is stored with the result."""
        tr = df[train]
        params = {"plan_sha256": self.digest(), "fit_where": train_desc, "fit_rows": int(len(tr)), "cols": {}}
        for r in self.rules:
            if r.cyclic or r.bins is not None:
                params["cols"][r.col] = {"kind": "cyclic" if r.cyclic else "bins"}
                continue
            x = TRANSFORMS[r.transform][0](tr[r.col].astype(float).to_numpy())
            x = x[np.isfinite(x)]
            assert x.size, f"{r.col}: no finite training values"
            p = self._global(x, r.scale)
            if r.group and r.group_mode == "fitted":
                p["groups"] = self._fit_groups(tr, r, p)
            params["cols"][r.col] = p
        params["sha256"] = _digest(params)
        return params

    @staticmethod
    def _global(x: np.ndarray, scale: str) -> dict:
        if scale == "none":
            return {"center": 0.0, "scale": 1.0, "n": int(x.size)}
        if scale == "robust":
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
            c, s = med, q3 - q1
        else:
            xs = np.sort(x)  # sorted first: the same summation order on every machine
            c, s = float(xs.mean(dtype=np.float64)), float(xs.std(dtype=np.float64))
        assert s > 0, "zero scale in the training period"
        return {"center": _r(c), "scale": _r(s), "n": int(x.size)}

    def _fit_groups(self, tr: pd.DataFrame, r: FeatureRule, g: dict) -> dict:
        x = pd.Series(TRANSFORMS[r.transform][0](tr[r.col].astype(float).to_numpy()), index=tr.index)
        key = tr[r.group]
        n = x.groupby(key).count()
        mu = x.groupby(key).mean()
        sd = x.groupby(key).std(ddof=0)
        # empirical-Bayes style shrinkage: a group with few rows leans on the global centre and the global spread
        w = n / (n + SHRINK)
        mu_s = w * mu + (1 - w) * g["center"]
        sd_s = np.sqrt(w * sd.fillna(0) ** 2 + (1 - w) * g["scale"] ** 2)
        out = {str(k): [_r(float(mu_s[k])), _r(float(max(sd_s[k], 1e-12))), int(n[k])] for k in n.index}
        if r.group_stat == "center_z":  # spread of the training anomalies themselves, so the output has unit variance on the fit rows
            out["__anomaly_scale__"] = [_r(float((x - key.map(mu_s)).std(ddof=0)))]
        return out

    # ------------------------------------------------ apply
    def apply(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        assert params["sha256"] == _digest(params), "parameters were edited after fitting"
        assert params["plan_sha256"] == self.digest(), "plan changed since the parameters were fitted: refit"
        out = {}
        for r in self.rules:
            x = TRANSFORMS[r.transform][0](df[r.col].astype(float).to_numpy())
            if r.cyclic:
                a = 2 * np.pi * df[r.col].astype(float).to_numpy() / r.cyclic
                out[f"{r.col}_sin"], out[f"{r.col}_cos"] = np.sin(a), np.cos(a)
                continue
            if r.bins is not None:
                b = np.searchsorted(np.asarray(r.bins), df[r.col].astype(float).to_numpy(), side="right")
                for i in range(len(r.bins) + 1):
                    out[f"{r.col}_bin{i}"] = (b == i).astype(np.float32)
                continue
            p = params["cols"][r.col]
            if r.group and r.group_mode == "sample":
                out[f"{r.col}_{_tag(r)}"] = self._within_sample(df, r, x)
                continue
            if r.group:  # fitted groups
                mu, sd = self._lookup(df[r.group].astype(str).to_numpy(), p)
                if r.group_stat == "center_z":
                    z = (x - mu) / p["groups"]["__anomaly_scale__"][0]
                else:
                    z = (x - mu) / (sd if r.group_stat == "zscore" else 1.0)
                out[f"{r.col}_{_tag(r)}"] = z
                continue
            out[r.col] = (x - p["center"]) / p["scale"]
        return pd.DataFrame(out, index=df.index).astype(np.float32)

    @staticmethod
    def _lookup(keys: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray]:
        g = p["groups"]
        mu = np.array([g[k][0] if k in g and k != "__anomaly_scale__" else p["center"] for k in keys])
        sd = np.array([g[k][1] if k in g and k != "__anomaly_scale__" else p["scale"] for k in keys])
        return mu, sd  # a group unseen in training falls back to the global statistics

    def _within_sample(self, df: pd.DataFrame, r: FeatureRule, x: np.ndarray) -> np.ndarray:
        s = pd.Series(x, index=df.index)
        by = s.groupby(df[r.group].to_numpy(), sort=False)
        n = by.transform("count")
        if r.group_stat == "percentile":
            v = by.rank(pct=True)
        else:
            m = by.transform("mean")
            sd = by.transform("std").replace(0, np.nan) if r.group_stat == "zscore" else 1.0
            v = (s - m) / sd
        return v.where(n >= self.min_group).fillna(0.5 if r.group_stat == "percentile" else 0.0).to_numpy()


def _tag(r: FeatureRule) -> str:
    return {"center": "dev", "center_z": "anom", "zscore": "z", "percentile": "pct"}[r.group_stat] + ("_" + r.group)


def _digest(params: dict) -> str:
    body = {k: v for k, v in params.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def fit_bins(x: np.ndarray, y: np.ndarray, leaves: int = 8, min_share: float = 0.01) -> tuple[float, ...]:
    """Supervised cut points from a shallow tree on TRAINING rows (thresholds chosen by the target); at least `min_share` of the rows per bin."""
    from sklearn.tree import DecisionTreeClassifier

    ok = np.isfinite(x)
    t = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=max(int(ok.sum() * min_share), 50), random_state=0)
    t.fit(x[ok].reshape(-1, 1), y[ok])
    return tuple(_r(float(v)) for v in np.sort(t.tree_.threshold[t.tree_.feature >= 0]))


# ------------------------------------------------------------------ targets: the part that needs an inverse


@dataclass(frozen=True)
class Scaled:
    """A model output tagged with the space it lives in. Metrics refuse anything but original units."""

    values: np.ndarray
    space: str = "identity"  # "identity" = original units; otherwise the TRANSFORMS name the model worked in

    def require_original(self) -> np.ndarray:
        if self.space != "identity":
            raise ValueError(f"prediction is in {self.space!r} space: map it back with TargetTransform.inverse before scoring or reporting")
        return self.values


@dataclass
class TargetTransform:
    """The target's transform, its inverse and (for log-type targets) the retransformation bias correction.

    exp(E[log y]) is the geometric mean, not the mean of y: reading a log-scale model back on the original scale underestimates the mean unless the
    residual spread is accounted for. `smearing` is Duan's factor mean(exp(residual)), estimated on TRAINING residuals; `inverse` multiplies by it for
    the log-type transforms when asked for a conditional mean (`mean=True`), and returns the conditional median/geometric mean otherwise.
    """

    name: str = "identity"
    smearing: float = 1.0

    def forward(self, y: np.ndarray) -> np.ndarray:
        return TRANSFORMS[self.name][0](np.asarray(y, float))

    def fit_smearing(self, resid_train: np.ndarray) -> TargetTransform:
        assert self.name in ("log", "log1p"), "smearing only applies to log-type targets"
        self.smearing = float(np.mean(np.exp(resid_train)))
        return self

    def wrap(self, model_output: np.ndarray) -> Scaled:
        return Scaled(np.asarray(model_output, float), self.name)

    def inverse(self, s: Scaled, mean: bool = False) -> Scaled:
        if s.space == "identity":
            return s
        assert s.space == self.name, f"output is in {s.space!r} space, this transform inverts {self.name!r}"
        y = TRANSFORMS[self.name][1](s.values)
        if mean and self.name in ("log", "log1p"):
            y = y * self.smearing if self.name == "log" else (y + 1) * self.smearing - 1
        return Scaled(y, "identity")


def mse(y_true: np.ndarray, pred: Scaled) -> float:
    return float(np.mean((np.asarray(y_true, float) - pred.require_original()) ** 2))


def brier(y_true: np.ndarray, pred: Scaled) -> float:
    return mse(y_true, pred)


def mape_weighted(y_true: np.ndarray, pred: Scaled) -> float:
    p = pred.require_original()
    y = np.asarray(y_true, float)
    return float(np.abs(p - y).sum() / np.abs(y).sum())


@dataclass(frozen=True)
class TargetSpec:
    """Registry entry: what each project target is, how it is modelled, and where the inverse must be applied."""

    name: str
    where: str
    model_space: str
    transform: str
    inverse_needed: bool
    inverse_point: str
    basis: str = field(default="")
