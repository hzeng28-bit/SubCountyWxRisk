"""Scaling audit of the Texas flood pilot: which columns need rescaling, where group-aware statistics matter, what discretisation buys.

Learners that care about scale (logistic regression with L2, K-means, Poisson/BFGS) get a different treatment from trees, so the audit
measures both. Metric (AGENTS.md): within-event AUC, test 2017-2023, event-clustered paired bootstrap; a change is kept only if its gain
exceeds the interval. Everything is fitted on 2010-2016. Writes $WXRISK_DATA/audit/scaling_evidence.json.

  E1 feature diagnostics        skew, kurtosis, zeros, range -> suggested transform (rule in `suggest`)
  E2 scale sensitivity          logistic regression: raw / standardised / transformed / +cyclic month / +group-aware / +bins
  E3 tree invariance            boosting on raw vs transformed features (should not move)
  E4 group structure            share of each feature's variance that sits between episodes / counties / years
  E5 single-feature shape       is the response monotone? (within-event AUC of one feature is invariant to monotone transforms)
  E6 regularisation x scaling   C sweep on raw vs standardised vs transformed
  E7 retransformation           a log-scale fit read back on the original scale (script 30 style)
  E8 log-OLS vs log-link GLM    exp(coef) as ratio of geometric means vs ratio of means, same controls

    python scripts/42_scaling_audit.py | tee $WXRISK_DATA/audit/scaling_audit.txt
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, rankdata, skew
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

warnings.filterwarnings("ignore")
EV: dict = {}
RNG = np.random.default_rng(0)
MIN_BG = 10

COUNTY = ["cty_f1_max", "cty_f1_mean", "cty_s_mean", "cty_a14_mean", "dur", "month", "cty_nbg"]
EXPO = ["log_pol", "sfha_share", "single_share", "mean_cov"]
CELL = ["bg_f1", "bg_f3", "bg_s", "f1_vs_cty"]
NEIGH = ["nb3_mean", "nb7_mean", "nb15_mean", "nb7_max", "nb15_max", "nep7_25", "nep7_50", "nep15_25", "nep15_50"]
STATIC = ["lat", "lon", "log_area", "bg_a14"]
TERRAIN = ["elev", "rel1", "rel3", "pos3", "rel10", "slope"]
BASE = COUNTY + EXPO + CELL + NEIGH + STATIC + TERRAIN


def put(key: str, **kw) -> None:
    EV[key] = kw
    print(f"\n[{key}]")
    for k, v in kw.items():
        print(f"  {k}: {v}")


# ------------------------------------------------------------------ data


def load() -> pd.DataFrame:
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t["log_area"] = np.log(t.area_km2.clip(lower=0.01))
    t = t.merge(pd.read_parquet(fp.PILOT / "bg_terrain.parquet"), on="bg10", how="left")
    return t


# ------------------------------------------------------------------ metric


def episode_stats(ep: np.ndarray, y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Per episode: Mann-Whitney U numerator and number of pairs; pooled AUC = sum(U) / sum(pairs)."""
    df = pd.DataFrame({"ep": ep, "y": y, "s": s})
    rows = {}
    for e, g in df.groupby("ep", sort=False):
        n1 = int(g.y.sum())
        n0 = len(g) - n1
        if len(g) < MIN_BG or n1 == 0 or n0 == 0:
            continue
        r = rankdata(g.s.to_numpy())
        rows[e] = (r[g.y.to_numpy() == 1].sum() - n1 * (n1 + 1) / 2, n1 * n0)
    return pd.DataFrame(rows, index=["u", "pairs"]).T


def auc(st: pd.DataFrame) -> float:
    return float(st.u.sum() / st.pairs.sum())


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, n: int = 1000) -> tuple[float, float, float]:
    """Paired event-clustered bootstrap of AUC(a) - AUC(b)."""
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    pt = auc(a) - auc(b)
    k = RNG.integers(0, len(idx), (n, len(idx)))
    ua, pa, ub, pb = a.u.to_numpy(), a.pairs.to_numpy(), b.u.to_numpy(), b.pairs.to_numpy()
    d = ua[k].sum(1) / pa[k].sum(1) - ub[k].sum(1) / pb[k].sum(1)
    return pt, *np.percentile(d, [2.5, 97.5])


# ------------------------------------------------------------------ transforms used by the candidates


def suggest(x: pd.Series) -> str:
    """The rule written down in docs/SCALING_PLAN.md, applied to the training values of one column."""
    v = x.dropna().to_numpy(float)
    if np.isin(v, [0, 1]).all():
        return "none (binary)"
    lo, hi = v.min(), v.max()
    if lo >= 0 and hi <= 1:
        return "none (bounded share): standardise"
    sk = skew(v)
    if lo >= 0 and sk > 1.5:
        return "log1p, then standardise"
    if abs(sk) > 1.5:  # heavy tail with negative values: asinh is log-like for large |x| and linear near 0
        return "asinh, then standardise"
    return "standardise"


def apply_transform(t: pd.DataFrame, cols: list[str], plan: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=t.index)
    for c in cols:
        x = t[c].astype(float)
        p = plan.get(c, "standardise")
        if p.startswith("log1p"):
            out[c] = np.log1p(x.clip(lower=0))
        elif p.startswith(("asinh", "signed")):
            out[c] = np.arcsinh(x)
        else:
            out[c] = x
    return out


def with_cyclic_month(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    m = X.pop("month")
    X["month_sin"], X["month_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    return X


def group_features(t: pd.DataFrame) -> pd.DataFrame:
    """Within-episode and within-county statistics. They use only the features of the same episode / county (no labels, no other years)."""
    g = pd.DataFrame(index=t.index)
    by = t.groupby("ep", sort=False)
    for c in ["bg_f1", "bg_f3", "bg_s", "nb7_mean", "nb15_mean", "log_pol", "sfha_share"]:
        m, s = by[c].transform("mean"), by[c].transform("std").replace(0, np.nan)
        g[f"{c}_zep"] = ((t[c] - m) / s).fillna(0.0)
        g[f"{c}_pep"] = by[c].rank(pct=True)
    bc = t.groupby("cty", sort=False)
    for c in ["elev", "rel3", "slope"]:  # static geography relative to the county's block groups (label-free)
        g[f"{c}_rel_cty"] = t[c] - bc[c].transform("median")
    return g


def supervised_bins(x_tr: np.ndarray, y_tr: np.ndarray, leaves: int = 8) -> np.ndarray:
    """Cut points from a shallow tree on the training rows: thresholds chosen by the target, minimum 1% of the rows per bin."""
    tr = DecisionTreeClassifier(max_leaf_nodes=leaves, min_samples_leaf=max(5000, len(x_tr) // 100), random_state=0)
    tr.fit(x_tr.reshape(-1, 1), y_tr)
    th = np.sort(tr.tree_.threshold[tr.tree_.feature >= 0])
    return th


def onehot_bins(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    b = np.searchsorted(cuts, x, side="right")
    return np.eye(len(cuts) + 1, dtype=np.float32)[b]


# ------------------------------------------------------------------ experiments


def e1_diagnostics(t: pd.DataFrame, tr: np.ndarray) -> dict[str, str]:
    rows, plan = {}, {}
    for c in BASE:
        x = t.loc[tr, c].astype(float).replace([np.inf, -np.inf], np.nan)
        v = x.dropna()
        plan[c] = suggest(x)
        rows[c] = {"skew": round(float(skew(v)), 2), "excess_kurtosis": round(float(kurtosis(v)), 1), "zero_share": round(float((v == 0).mean()), 3),
                   "min": float(v.min()), "median": float(v.median()), "p99": float(v.quantile(0.99)), "max": float(v.max()),
                   "max_over_p99": round(float(v.max() / v.quantile(0.99)), 2) if v.quantile(0.99) > 0 else None, "suggested": plan[c]}
    put("E1.feature_diagnostics", rule="see suggest()", columns=rows)
    return plan


def fit_lr(Xtr, ytr, Xte, C: float) -> np.ndarray:
    m = LogisticRegression(C=C, max_iter=400, solver="lbfgs").fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def std_fit(Xtr: np.ndarray, Xte: np.ndarray):
    sc = StandardScaler().fit(np.nan_to_num(Xtr))
    return sc.transform(np.nan_to_num(Xtr)), sc.transform(np.nan_to_num(Xte))


def e2_sensitivity(t, tr, te, plan, cuts) -> dict[str, pd.DataFrame]:
    y = t.y.to_numpy(float)
    ep_te = t.ep.to_numpy()[te]
    res: dict[str, pd.DataFrame] = {}

    def run(name: str, X: pd.DataFrame, C: float = 0.1, scale: bool = True, learner: str = "lr") -> None:
        A = X.to_numpy(np.float32)
        if learner == "lr":
            if scale:
                a, b = std_fit(A[tr], A[te])
            else:
                a, b = np.nan_to_num(A[tr]), np.nan_to_num(A[te])
            p = fit_lr(a, y[tr], b, C)
        else:
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=5, min_samples_leaf=300, l2_regularization=1.0,
                                               random_state=0).fit(A[tr], y[tr])
            p = m.predict_proba(A[te])[:, 1]
        res[name] = episode_stats(ep_te, y[te], p)
        print(f"  {name}: within-event AUC {auc(res[name]):.4f}", flush=True)

    raw = t[BASE].astype(float)
    tf = apply_transform(t, BASE, plan)
    tfc = with_cyclic_month(tf)
    grp = group_features(t)
    grp_t = grp.copy()
    # percentile / z features are already bounded or standardised; keep as is. Add group features to the transformed + cyclic set.
    tfg = pd.concat([tfc, grp_t], axis=1)
    print("\nE2 logistic regression (C=0.1 unless noted), test 2017-2023", flush=True)
    run("LR raw, no scaling", raw, scale=False)
    run("LR raw + standardise (current)", raw)
    run("LR transformed + standardise", tf)
    run("LR transformed + cyclic month", tfc)
    run("LR transformed + cyclic + group-aware", tfg)
    bins = []
    for c, cu in cuts.items():
        bins.append(pd.DataFrame(onehot_bins(t[c].to_numpy(float), cu), index=t.index, columns=[f"{c}_bin{i}" for i in range(len(cu) + 1)]))
    tfgb = pd.concat([tfg, *bins], axis=1)
    run("LR transformed + cyclic + group-aware + supervised bins", tfgb)
    print("\nE3 boosting (invariant to monotone transforms in theory)", flush=True)
    run("GBM raw (current)", raw, learner="gbm")
    run("GBM transformed + cyclic", tfc, learner="gbm")
    run("GBM raw + group-aware", pd.concat([raw, grp], axis=1), learner="gbm")
    return res


def e2_report(res: dict[str, pd.DataFrame]) -> None:
    ref = res["LR raw + standardise (current)"]
    gref = res["GBM raw (current)"]
    out = {}
    for k, st in res.items():
        r = gref if k.startswith("GBM") else ref
        d = boot_diff(st, r) if st is not r else (0.0, 0.0, 0.0)
        out[k] = {"within_event_auc": round(auc(st), 4), "diff_vs_reference": [round(x, 4) for x in d],
                  "reference": "GBM raw (current)" if k.startswith("GBM") else "LR raw + standardise (current)"}
    put("E2_E3.within_event_auc", episodes=int(len(ref)), results=out,
        reading_rule="a change counts only if its paired event-clustered 95% interval excludes 0")


def e4_groups(t: pd.DataFrame, tr: np.ndarray) -> None:
    d = t.loc[tr]
    rows = {}
    for c in BASE + ["y"]:
        x = d[c].astype(float)
        tot = x.var()
        if tot == 0 or np.isnan(tot):
            continue
        rows[c] = {g: round(float(1 - (x - x.groupby(d[g]).transform("mean")).var() / tot), 3) for g in ("ep", "cty", "year")}
    put("E4.variance_share_between_groups", note="1 - within-group variance / total, on training rows (eta squared). ep = rain episode (one county, one event)", columns=rows)


def e5_shape(t, tr, te) -> dict[str, np.ndarray]:
    y = t.y.to_numpy(float)
    ep_te = t.ep.to_numpy()[te]
    feats = ["bg_f1", "cty_f1_max", "nb7_mean", "log_pol", "sfha_share", "mean_cov", "elev", "rel3", "slope", "cty_nbg", "log_area", "bg_a14"]
    cuts, rows = {}, {}
    for c in feats:
        x = t[c].to_numpy(float)
        ok = np.isfinite(x)
        cu = supervised_bins(x[tr & ok], y[tr & ok])
        cuts[c] = cu
        # train event rate across 10 quantile bins and its monotonicity
        q = pd.qcut(pd.Series(x[tr & ok]), 10, duplicates="drop")
        rate = pd.Series(y[tr & ok]).groupby(q.values, observed=True).mean().to_numpy()
        sp = pd.Series(rate).corr(pd.Series(np.arange(len(rate))), method="spearman")
        # single-feature scores, test, within event: raw (any monotone transform gives the same ranks) vs bins
        raw_auc = auc(episode_stats(ep_te, y[te], np.nan_to_num(x[te])))
        b_tr, b_te = onehot_bins(x[tr & ok], cu), onehot_bins(np.nan_to_num(x[te]), cu)
        p = LogisticRegression(C=1.0, max_iter=200).fit(b_tr, y[tr & ok]).predict_proba(b_te)[:, 1]
        bin_auc = auc(episode_stats(ep_te, y[te], p))
        rows[c] = {"cuts": [round(float(v), 4) for v in cu], "decile_rates": [round(float(v), 5) for v in rate], "spearman_decile_vs_rate": round(float(sp), 2),
                   "single_feature_within_auc_raw_or_any_monotone": round(raw_auc, 4), "single_feature_within_auc_bins": round(bin_auc, 4)}
    put("E5.single_feature_shape", note="a monotone transform cannot change one feature's within-event AUC; bins can, but only if the response is not monotone",
        columns=rows)
    return cuts


def e6_c_sweep(t, tr, te, plan) -> None:
    y = t.y.to_numpy(float)
    ep_te = t.ep.to_numpy()[te]
    raw = t[BASE].astype(float).to_numpy(np.float32)
    tfc = with_cyclic_month(apply_transform(t, BASE, plan)).to_numpy(np.float32)
    out = {}
    for name, A, scale in (("raw, unscaled", raw, False), ("raw, standardised", raw, True), ("transformed+cyclic, standardised", tfc, True)):
        for C in (0.001, 0.01, 0.1, 1.0, 10.0):
            if scale:
                a, b = std_fit(A[tr], A[te])
            else:
                a, b = np.nan_to_num(A[tr]), np.nan_to_num(A[te])
            m = LogisticRegression(C=C, max_iter=400).fit(a, y[tr])
            out[f"{name} C={C}"] = {"within_event_auc": round(auc(episode_stats(ep_te, y[te], m.predict_proba(b)[:, 1])), 4), "iterations": int(m.n_iter_[0])}
            print(f"  {name} C={C}: {out[f'{name} C={C}']}", flush=True)
    put("E6.regularisation_x_scaling", results=out)


def e7_retransform() -> None:
    """A model fitted on log(damage ratio) read back on the original scale: ratio of geometric means vs ratio of means."""
    import duckdb

    C = f"read_parquet('{wb2.DATA_DIR / 'clean' / 'claims_nfip.parquet'}')"
    d = duckdb.sql(f"""select case when foundation in ('1') then 'slab' when foundation in ('2') then 'basement' when foundation in ('4','5','6') then 'elevated'
        when foundation='3' then 'crawl' end g, damage_building/cov_building as dr, depth_ext
        from {C} where occupancy in (1,11) and cov_building>0 and damage_building is not null and depth_ext>0 and loss_year>=2016 and foundation is not null""").df()
    d = d[d.g.notna() & (d.dr > 0)].copy()
    d["dr"] = d.dr.clip(1e-4, 1.5)
    d["ln"] = np.log(d.dr)
    g = d.groupby("g").agg(n=("dr", "size"), mean=("dr", "mean"), median=("dr", "median"), geo_mean=("ln", lambda s: float(np.exp(s.mean()))), sd_log=("ln", "std"))
    g["mean_over_geo_mean"] = g["mean"] / g["geo_mean"]
    base = g.loc["slab"]
    rel = pd.DataFrame({"ratio_of_means": g["mean"] / base["mean"], "ratio_of_geometric_means": g["geo_mean"] / base["geo_mean"],
                        "ratio_of_medians": g["median"] / base["median"]})
    put("E7.retransformation", group_table=g.round(3).to_dict("index"), ratios_vs_slab=rel.round(3).to_dict("index"),
        duan_smearing_factor={k: round(float(np.exp(d.loc[d.g == k, "ln"] - d.loc[d.g == k, "ln"].mean()).mean()), 3) for k in g.index},
        note="exp(coefficient) from a log-damage regression is a ratio of geometric means; the ratio of means (what dollars add up to) differs")


def e5b_bins_ablation(t, tr, te, plan, cuts) -> None:
    """Which columns carry the gain of the supervised bins? Add the bins of ONE column (continuous column kept) to the transformed + cyclic LR."""
    y = t.y.to_numpy(float)
    ep_te = t.ep.to_numpy()[te]
    base = with_cyclic_month(apply_transform(t, BASE, plan))

    def fit(X: pd.DataFrame) -> pd.DataFrame:
        a, b = std_fit(X.to_numpy(np.float32)[tr], X.to_numpy(np.float32)[te])
        return episode_stats(ep_te, y[te], fit_lr(a, y[tr], b, 0.1))

    def bins_of(cols: list[str]) -> pd.DataFrame:
        return pd.concat([pd.DataFrame(onehot_bins(t[c].to_numpy(float), cuts[c]), index=t.index,
                                       columns=[f"{c}_bin{i}" for i in range(len(cuts[c]) + 1)]) for c in cols], axis=1)

    ref = fit(base)
    print(f"\nE5b reference LR transformed + cyclic: {auc(ref):.4f}", flush=True)
    out = {"reference_auc": round(auc(ref), 4)}
    groups = {"rain (bg_f1, cty_f1_max, nb7_mean)": ["bg_f1", "cty_f1_max", "nb7_mean"],
              "exposure (log_pol, sfha_share, mean_cov)": ["log_pol", "sfha_share", "mean_cov"],
              "geography (elev, rel3, slope, cty_nbg, log_area, bg_a14)": ["elev", "rel3", "slope", "cty_nbg", "log_area", "bg_a14"]}
    items = {c: [c] for c in cuts} | groups
    for name, cols in items.items():
        st = fit(pd.concat([base, bins_of(cols)], axis=1))
        d = boot_diff(st, ref)
        out[name] = {"auc": round(auc(st), 4), "diff_vs_reference": [round(float(v), 4) for v in d]}
        print(f"  + bins of {name}: {out[name]}", flush=True)
    put("E5b.bins_ablation", results=out)


def e8_log_ols_vs_glm() -> None:
    """Same controls, two estimators: OLS on log(damage ratio) (exp(coef) = ratio of geometric means) vs a log-link Poisson GLM on the ratio itself (exp(coef) = ratio of means)."""
    import duckdb
    from sklearn.linear_model import LinearRegression, PoissonRegressor

    C = f"read_parquet('{wb2.DATA_DIR / 'clean' / 'claims_nfip.parquet'}')"
    d = duckdb.sql(f"""select case foundation when '1' then 'slab' when '2' then 'bsmt' when '3' then 'crawl' else 'elev' end g,
        damage_building/cov_building dr, depth_ext, loss_year from {C} where occupancy in (1,11) and cov_building>0
        and damage_building is not null and depth_ext>0 and foundation in ('1','2','3','4','5','6') and loss_year>=2010""").df()
    d["dr"] = d.dr.clip(1e-4, 1.5)
    db = pd.cut(d.depth_ext, [0, 6, 12, 24, 48, 1e4], labels=["1-6", "7-12", "13-24", "25-48", ">48"])
    X = pd.get_dummies(pd.concat([d.g, db.rename("db"), d.loss_year.astype(str).rename("yr")], axis=1), drop_first=False)
    X = X.drop(columns=["g_slab", "db_1-6", "yr_" + str(d.loss_year.min())]).astype(float)
    ols = LinearRegression().fit(X, np.log(d.dr))
    glm = PoissonRegressor(alpha=0, max_iter=500).fit(X, d.dr)
    out = {}
    for g in ("bsmt", "crawl", "elev"):
        j = list(X.columns).index("g_" + g)
        out[g] = {"exp_coef_log_ols_ratio_of_geometric_means": round(float(np.exp(ols.coef_[j])), 3),
                  "exp_coef_loglink_glm_ratio_of_means": round(float(np.exp(glm.coef_[j])), 3)}
    put("E8.log_ols_vs_loglink_glm", n=int(len(d)), controls="depth bin (5) + loss year", vs="slab", results=out)


def e9_bins_blocked_cv(t, plan) -> None:
    """Confirmation of the exposure bins (E5b found the gain on one test period, 15 comparisons): blocked-by-year CV over year pairs.
    Cut points are refitted on the training years of every fold; test folds are disjoint, so the episodes are disjoint too."""
    y = t.y.to_numpy(float)
    base = with_cyclic_month(apply_transform(t, BASE, plan))
    cols = ["log_pol", "sfha_share", "single_share", "mean_cov"]
    years = list(range(2010, 2024))
    folds = [years[i:i + 2] for i in range(0, len(years), 2)]
    ref, new, per = [], [], []
    for f in folds:
        te = t.year.isin(f).to_numpy()
        tr = ~te
        cuts = {c: supervised_bins(t[c].to_numpy(float)[tr], y[tr]) for c in cols}
        bins = pd.concat([pd.DataFrame(onehot_bins(t[c].to_numpy(float), cuts[c]), index=t.index,
                                       columns=[f"{c}_bin{i}" for i in range(len(cuts[c]) + 1)]) for c in cols], axis=1)

        def fit(X, tr=tr, te=te):
            a, b = std_fit(X.to_numpy(np.float32)[tr], X.to_numpy(np.float32)[te])
            return episode_stats(t.ep.to_numpy()[te], y[te], fit_lr(a, y[tr], b, 0.1))

        r, n = fit(base), fit(pd.concat([base, bins], axis=1))
        ref.append(r)
        new.append(n)
        per.append({"test_years": f, "reference": round(auc(r), 4), "with_bins": round(auc(n), 4), "diff": round(auc(n) - auc(r), 4)})
        print(f"  fold {f}: {per[-1]}", flush=True)
    R, N = pd.concat(ref), pd.concat(new)
    d = boot_diff(N, R)
    print(f"E9 pooled within-event AUC: reference {auc(R):.4f}, with exposure bins {auc(N):.4f}, diff {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
    put("E9.exposure_bins_blocked_cv", folds=per, reference_auc=round(auc(R), 4), with_bins_auc=round(auc(N), 4),
        diff=[round(float(v), 4) for v in d], columns=cols,
        confirmed=bool(d[1] > 0 and sum(p["diff"] > 0 for p in per) >= len(per) - 1))


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    t = load()
    tr, te = (t.year <= 2016).to_numpy(), (t.year >= 2017).to_numpy()
    print(f"{len(t):,} rows; train {tr.sum():,} ({int(t.y[tr].sum()):,} pos), test {te.sum():,}; {t.ep.nunique():,} episodes", flush=True)
    if only == "plan":  # python scripts/42_scaling_audit.py plan : does PILOT_PLAN reproduce the best linear configuration of E2?
        from wxrisk.prep.scaling_plan import PILOT_PLAN

        prev = json.loads((wb2.DATA_DIR / "audit" / "scaling_evidence.json").read_text())
        y = t.y.to_numpy(float)
        params = PILOT_PLAN.fit(t, tr, "year <= 2016")
        Z = PILOT_PLAN.apply(t, params).fillna(0.0).to_numpy(np.float32)  # standardised, so 0 is the training mean
        m = LogisticRegression(C=0.1, max_iter=400).fit(Z[tr], y[tr])
        st = episode_stats(t.ep.to_numpy()[te], y[te], m.predict_proba(Z[te])[:, 1])
        ref = prev["E2_E3.within_event_auc"]["results"]["LR transformed + cyclic month"]["within_event_auc"]
        put("E10.plan_check", plan_sha256=params["sha256"][:16], columns_out=int(Z.shape[1]), iterations=int(m.n_iter_[0]),
            within_event_auc=round(auc(st), 4), audit_reference_transformed_cyclic=ref)
        prev["E10.plan_check"] = EV["E10.plan_check"]
        (wb2.DATA_DIR / "audit" / "scaling_evidence.json").write_text(json.dumps(prev, indent=1, sort_keys=True, default=str) + "\n")
        return
    if only == "retransform":  # python scripts/42_scaling_audit.py retransform : E7 + E8 only
        prev = json.loads((wb2.DATA_DIR / "audit" / "scaling_evidence.json").read_text())
        e7_retransform()
        e8_log_ols_vs_glm()
        prev["E7.retransformation"], prev["E8.log_ols_vs_loglink_glm"] = EV["E7.retransformation"], EV["E8.log_ols_vs_loglink_glm"]
        (wb2.DATA_DIR / "audit" / "scaling_evidence.json").write_text(json.dumps(prev, indent=1, sort_keys=True, default=str) + "\n")
        return
    if only == "bins-cv":  # python scripts/42_scaling_audit.py bins-cv : E9 only, merged into the evidence file
        prev = json.loads((wb2.DATA_DIR / "audit" / "scaling_evidence.json").read_text())
        plan = {k: v["suggested"] for k, v in prev["E1.feature_diagnostics"]["columns"].items()}
        e9_bins_blocked_cv(t, plan)
        prev["E9.exposure_bins_blocked_cv"] = EV["E9.exposure_bins_blocked_cv"]
        (wb2.DATA_DIR / "audit" / "scaling_evidence.json").write_text(json.dumps(prev, indent=1, sort_keys=True, default=str) + "\n")
        return
    if only == "bins":  # python scripts/42_scaling_audit.py bins : only the per-column bin ablation, merged into the evidence file
        prev = json.loads((wb2.DATA_DIR / "audit" / "scaling_evidence.json").read_text())
        plan = {k: v["suggested"] for k, v in prev["E1.feature_diagnostics"]["columns"].items()}
        cuts = e5_shape(t, tr, te)
        e5b_bins_ablation(t, tr, te, plan, cuts)
        prev["E5b.bins_ablation"] = EV["E5b.bins_ablation"]
        (wb2.DATA_DIR / "audit" / "scaling_evidence.json").write_text(json.dumps(prev, indent=1, sort_keys=True, default=str) + "\n")
        return
    plan = e1_diagnostics(t, tr)
    e4_groups(t, tr)
    cuts = e5_shape(t, tr, te)
    res = e2_sensitivity(t, tr, te, plan, cuts)
    e2_report(res)
    e6_c_sweep(t, tr, te, plan)
    e7_retransform()
    e8_log_ols_vs_glm()
    out = wb2.DATA_DIR / "audit" / "scaling_evidence.json"
    out.write_text(json.dumps(EV, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
