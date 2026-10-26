"""Feature plans fit on training rows only, group-aware statistics behave, and a log-scale prediction cannot be scored without its inverse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk.prep import scaling as sc  # noqa: E402


def frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    g = np.repeat(["a", "b", "c"], n // 3)
    mu = pd.Series(g).map({"a": 0.0, "b": 10.0, "c": 20.0}).to_numpy()
    return pd.DataFrame({"g": g, "x": mu + r.normal(size=n), "skew": np.exp(r.normal(size=n)), "month": r.integers(1, 13, n),
                         "ep": np.repeat(np.arange(n // 6), 6), "year": np.repeat([2014, 2015, 2018], n // 3)})


def test_parameters_use_training_rows_only() -> None:
    df = frame()
    tr = (df.year <= 2016).to_numpy()
    plan = sc.FeaturePlan((sc.FeatureRule("x"), sc.FeatureRule("skew", transform="log1p")))
    p1 = plan.fit(df, tr, "year<=2016")
    df2 = df.copy()
    df2.loc[~tr, "x"] = 1e6  # the test years change completely
    assert plan.fit(df2, tr, "year<=2016")["cols"] == p1["cols"]
    z = plan.apply(df, p1)
    assert abs(z.x[tr].mean()) < 1e-5 and abs(z.x[tr].std(ddof=0) - 1) < 1e-4
    assert z.dtypes.eq(np.float32).all()


def test_fitted_groups_shrink_small_groups_and_fall_back_for_unseen() -> None:
    df = frame()
    df.loc[df.g == "c", "x"] += 0  # keep
    tr = np.ones(len(df), bool)
    tr[(df.g == "c").to_numpy()] = False
    tr[np.flatnonzero((df.g == "c").to_numpy())[:2]] = True  # group c has only two training rows
    plan = sc.FeaturePlan((sc.FeatureRule("x", group="g", group_stat="center"),))
    p = plan.fit(df, tr, "mask")
    mu_a, mu_b, mu_c = (p["cols"]["x"]["groups"][k][0] for k in "abc")
    glob = p["cols"]["x"]["center"]
    # 200 rows per group still lean 9% on the global mean (n / (n + 20)), so the estimates sit slightly inside the true 0 and 10
    assert abs(mu_a - 0.0) < 0.1 * abs(glob) + 0.2 and abs(mu_b - 10.0) < 0.1 * abs(glob - 10.0) + 0.3
    assert abs(mu_c - glob) < abs(20.0 - glob)  # two rows: pulled toward the global mean
    new = df.assign(g="never_seen")
    out = plan.apply(new, p)
    assert np.allclose(out.iloc[:, 0], (new.x - glob).astype(np.float32), atol=1e-4)  # unseen group -> global centre


def test_within_sample_statistics_use_only_the_group_itself() -> None:
    df = pd.DataFrame({"ep": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3], "v": [1, 2, 3, 4, 5, 100, 200, 300, 400, 500, 7, 9]})
    plan = sc.FeaturePlan((sc.FeatureRule("v", group="ep", group_mode="sample", group_stat="percentile"),
                           sc.FeatureRule("v", group="ep", group_mode="sample", group_stat="zscore")), min_group=5)
    out = plan.apply(df, plan.fit(df, np.ones(len(df), bool), "all"))
    pct, z = out.iloc[:, 0], out.iloc[:, 1]
    assert np.allclose(pct[:5], pct[5:10])  # same shape, different level: identical
    assert np.allclose(z[:5], z[5:10], atol=1e-5)
    assert (pct[10:] == 0.5).all() and (z[10:] == 0).all()  # a group below min_group gets the neutral value, not a noisy statistic


def test_cyclic_and_bins_and_tamper_guard() -> None:
    df = frame()
    tr = np.ones(len(df), bool)
    plan = sc.FeaturePlan((sc.FeatureRule("month", cyclic=12), sc.FeatureRule("x", bins=(5.0, 15.0))))
    p = plan.fit(df, tr, "all")
    out = plan.apply(df, p)
    jan, dec = out[df.month == 1].iloc[0], out[df.month == 12].iloc[0]
    assert np.hypot(jan.month_sin - dec.month_sin, jan.month_cos - dec.month_cos) < 0.6  # December is next to January
    assert out[["x_bin0", "x_bin1", "x_bin2"]].sum(axis=1).eq(1).all()
    p["fit_rows"] += 1
    with pytest.raises(AssertionError):
        plan.apply(df, p)
    other = sc.FeaturePlan((sc.FeatureRule("month", cyclic=12),))
    with pytest.raises(AssertionError):
        other.apply(df, plan.fit(df, tr, "all"))  # plan changed since fitting


def test_supervised_bins_follow_the_target_not_the_quantiles() -> None:
    r = np.random.default_rng(1)
    x = r.uniform(0, 100, 20000)
    y = (r.random(20000) < np.where(x > 80, 0.5, 0.02)).astype(float)
    cuts = sc.fit_bins(x, y, leaves=4)
    assert any(78 < c < 82 for c in cuts)


def test_target_inverse_and_the_retransformation_bias() -> None:
    r = np.random.default_rng(2)
    n = 200000
    eps = r.normal(size=n) * 0.8
    y = np.exp(1.0 + eps)  # lognormal: mean = exp(1 + 0.32), geometric mean = e
    tt = sc.TargetTransform("log")
    assert np.allclose(tt.inverse(tt.wrap(tt.forward(y[:5]))).values, y[:5])
    pred_log = np.full(n, np.log(y).mean())
    tt.fit_smearing(np.log(y) - pred_log)
    geo = tt.inverse(tt.wrap(pred_log)).values.mean()
    mean = tt.inverse(tt.wrap(pred_log), mean=True).values.mean()
    assert abs(geo - y.mean()) / y.mean() > 0.2  # reading exp(E[log y]) as a mean is off by ~30%
    assert abs(mean - y.mean()) / y.mean() < 0.02  # Duan smearing repairs it


def test_scoring_requires_original_units() -> None:
    y = np.array([0.0, 3.0, 10.0])
    tt = sc.TargetTransform("log1p")
    out = tt.wrap(tt.forward(np.array([0.5, 2.5, 9.0])))
    with pytest.raises(ValueError, match="log1p"):
        sc.mse(y, out)
    assert sc.mse(y, tt.inverse(out)) == pytest.approx(np.mean((y - np.array([0.5, 2.5, 9.0])) ** 2))
    assert sc.mse(y, sc.Scaled(np.array([0.5, 2.5, 9.0]))) > 0  # original-unit output passes
    with pytest.raises(AssertionError):
        sc.TargetTransform("log").inverse(out)  # inverting with the wrong transform


def test_pilot_plan_and_target_registry_are_consistent() -> None:
    from wxrisk.prep.scaling_plan import PILOT_PLAN, TARGETS

    cols = [r.col for r in PILOT_PLAN.rules]
    assert len(cols) == len(set(cols)), "a column appears twice in the plan"
    assert all(r.transform in sc.TRANSFORMS for r in PILOT_PLAN.rules)
    assert all(r.basis for r in PILOT_PLAN.rules), "every column needs its basis"
    assert not any(r.group for r in PILOT_PLAN.rules), "no group-aware column is adopted yet (docs/SCALING_PLAN.md, section 3)"
    assert next(r for r in PILOT_PLAN.rules if r.col == "month").cyclic == 12
    assert {"lat", "lon"} <= {r.col for r in PILOT_PLAN.rules if r.transform == "identity"}  # angles are only standardised
    for t in TARGETS:
        assert t.inverse_point, f"{t.name}: say where the inverse is applied (or why none is needed)"
    assert any(t.inverse_needed and "GEOMETRIC" in t.inverse_point for t in TARGETS)  # the stage 9 trap stays on record


def test_scs_anomalise_matches_plain_and_survives_unseen_cell():
    from wxrisk import scs_eval as se

    rng = np.random.default_rng(0)
    n = 6000
    df = pd.DataFrame({"cty": rng.choice(["08001", "35001"], n), "month": rng.integers(1, 13, n),
                       "lat": rng.normal(size=n), "cape_12": rng.gamma(2, size=n)})
    fit = pd.Series(np.arange(n) < 5000)
    a, b = se.anomalise(df, ["lat", "cape_12"], fit), se.anomalise_plain(df, ["lat", "cape_12"], fit)
    assert np.corrcoef(a[:, 1], b[:, 1])[0, 1] > 0.995  # shrinkage only moves sparse cells
    assert abs(a[fit.to_numpy(), 1].std() - 1) < 0.02
    df.loc[5500, ["cty", "month"]] = ["04001", 7]  # never seen in the fit rows
    assert np.isnan(se.anomalise_plain(df, ["cape_12"], fit)[5500]).all()
    assert np.isfinite(se.anomalise(df, ["cape_12"], fit)[5500]).all()
