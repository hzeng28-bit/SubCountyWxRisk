"""The scaling decisions of the pipeline, with their basis, and the generator of docs/SCALING_PLAN.md.

What is decided here is data (`PILOT_PLAN`, `TARGETS`); why is in the `basis` fields and in `PRINCIPLES`, whose numbers are filled in from
the evidence files written by scripts/42_scaling_audit.py and scripts/43_scaling_kmeans.py.

Roles used to pick a transform (the rule, applied to the TRAINING rows; see PRINCIPLES[2]):
  rain / amount / depth / count   non-negative, skew > 1.5      -> log1p, then standardise
  signed heavy tail               skew > 1.5 and values < 0      -> asinh, then standardise
  coordinate, bounded share, already-log column                  -> standardise only
  calendar month                                                 -> sin / cos of 2 pi m / 12
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from .. import wb2
from .scaling import FeaturePlan, FeatureRule, TargetSpec

R = FeatureRule
_RAIN = "rain/amount, skew {s}: log1p"
PILOT_PLAN = FeaturePlan((
    # county-level rain of the episode (constant inside an episode, so within-event AUC ignores them alone; they set the level for interactions)
    R("cty_f1_max", "log1p", basis="rain depth (mm), non-negative, skew 2.95, excess kurtosis 12.9"),
    R("cty_f1_mean", "log1p", basis="rain depth (mm), skew 2.19"),
    R("cty_s_mean", "log1p", basis="rain depth (mm), skew 2.39"),
    R("cty_a14_mean", "log1p", basis="antecedent 14-day rain (mm), skew 2.30"),
    R("dur", "log1p", basis="episode length in days, skew 3.32, median 1"),
    R("month", cyclic=12, basis="calendar month: 12 and 1 are neighbours; the linear 1..12 puts them furthest apart for a linear learner"),
    R("cty_nbg", basis="block groups per county, skew 0.74: standardise only"),
    # exposure
    R("log_pol", basis="already log1p of the policy count: standardise only"),
    R("sfha_share", basis="share in [0,1], 43% exact zeros: standardise; supervised bins are a CANDIDATE extra for linear learners (PRINCIPLES[4])"),
    R("single_share", basis="share in [0,1]"),
    R("mean_cov", "asinh", basis="mean building coverage per policy, signed (rounding gives -0.000), skew 45, max 187x the 99th percentile"),
    # block-group rain
    R("bg_f1", "log1p", basis="rain depth (mm), skew 2.45; top decile event rate 0.0436 against 0.0054 for the ninth"),
    R("bg_f3", "log1p", basis="rain depth (mm), skew 2.27"),
    R("bg_s", "log1p", basis="rain depth (mm), skew 2.45"),
    R("f1_vs_cty", "asinh", basis="cell rain minus county mean (mm), signed, excess kurtosis 17.4. It is the one hand-made WITHIN-GROUP feature already in the table"),
    # neighbourhood
    *[R(c, "log1p", basis="neighbourhood rain (mm), skew 2.1-2.7") for c in ("nb3_mean", "nb7_mean", "nb15_mean", "nb7_max", "nb15_max")],
    *[R(c, basis="neighbourhood exceedance fraction in [0,1]") for c in ("nep7_25", "nep7_50", "nep15_25", "nep15_50")],
    # static
    R("lat", basis="coordinate: standardise only (a heavy-tail rule would be meaningless on an angle)"),
    R("lon", basis="coordinate: standardise only"),
    R("log_area", basis="already log of the area: standardise only"),
    R("bg_a14", "log1p", basis="antecedent 14-day rain (mm), skew 2.51"),
    # terrain
    R("elev", "asinh", basis="elevation (m) with a few values below 0, skew 3.32"),
    R("rel1", "log1p", basis="height above the neighbourhood, skew 9.3, excess kurtosis 202"),
    R("rel3", "log1p", basis="height above the neighbourhood, skew 21.5, excess kurtosis 692"),
    R("rel10", "log1p", basis="height above the neighbourhood, skew 7.4"),
    R("slope", "log1p", basis="slope, skew 4.0"),
    R("pos3", basis="position index in [0,1]"),
))

TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec("flood occurrence y (pilot)", "scripts/26, 29", "probability", "identity", False, "none: the learner outputs a probability",
               "Absolute probabilities do not transfer across years (label drift, stage 8), so skill is reported as within-event AUC; Brier only against the train prevalence."),
    TargetSpec("SCS3 occurrence y0/y10k/y100k", "scripts/10, 12, 13, 14", "probability", "identity", False,
               "none; the readout p0 x lift is formed and clipped in probability space",
               "logit(p0) enters some models as a FEATURE; the logistic learner returns a probability, so no inverse is pending."),
    TargetSpec("new purchases N (stock-flow)", "scripts/37", "log of the mean (Poisson, log link)", "log link", True,
               "exp(alpha + x beta) inside predict_flow / project_stock: already applied",
               "The prediction is a conditional mean on the count scale. It is scored on log1p of the prediction vs log1p N (a loss different from the training "
               "likelihood) and as state-total WAPE on the count scale; both take the original-scale prediction, so no inverse is missing."),
    TargetSpec("claim rate (event features)", "scripts/36", "log rate (Poisson, offset)", "log link", True,
               "coefficients are reported as log-rate; exp(coef) is a rate ratio of MEANS", "Valid reading, because the model is Poisson and not a regression on log y."),
    TargetSpec("damage ratio (compliance, building attributes)", "scripts/30, 33", "log damage ratio (OLS) / raw ratio (medians)", "log", True,
               "exp(coef) of a log-OLS is a ratio of GEOMETRIC means, not of means: use a log-link GLM, or Duan smearing PER GROUP",
               "RESULTS stage 9 ('elevated lose 0.27x slab') is a geometric-mean ratio. Script 33 reports medians, which commute with any monotone transform, so no inverse is needed there."),
    TargetSpec("take-up rate (descriptive)", "scripts/34", "log10", "log10", True,
               "any threshold read from the log10 histogram / mixture must be quoted as 10**x", "Used for a histogram and a BIC comparison only."),
    TargetSpec("normalised features z", "all", "z", "standardise / asinh / log1p", False,
               "never returned by a model; inverse only for DISPLAY (cluster centres: z * std + group mean)",
               "A feature plan needs no inverse; a target transform does. Do not confuse them."),
)


@dataclass(frozen=True)
class Principle:
    title: str
    body: Callable[[dict, dict], str]


def f4(x: float) -> str:
    return f"{x:+.4f}"


def ci(d: list[float]) -> str:
    return f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]"


def _p1(e, k):
    r = e["E2_E3.within_event_auc"]["results"]
    c6 = e["E6.regularisation_x_scaling"]["results"]
    kb = k["B (T-3D..T-3h) | county x month anomaly, no scaling"]
    kc = k["C (<T-3h) | county x month anomaly, no scaling"]
    return (
        "**The learner decides.** Trees are invariant to monotone transforms and to scale: boosting on transformed features moves within-event AUC by "
        f"{ci(r['GBM transformed + cyclic']['diff_vs_reference'])} (reference: raw). They get no scaling; any gain must come from new information, not rescaling.\n\n"
        "Logistic regression with an L2 penalty, K-means/GMM and Poisson fits by BFGS are scale-dependent and always get standardised inputs, because:\n\n"
        f"- *K-means:* with no scaling the column standard deviations differ by a factor of {kb['column_std_max_over_min']:.0f} and the skill collapses "
        f"(tier B, y0: BSS {kb['y0']:+.4f} against {k['B (T-3D..T-3h) | county x month anomaly + z (current)']['y0']:+.4f}; difference {ci(kb['y0_diff_vs_current'])}; "
        f"tier C difference {ci(kc['y0_diff_vs_current'])}). One column dominates the distance.\n"
        f"- *Logistic regression:* the AUC of the unscaled fit is about the same ({r['LR raw, no scaling']['within_event_auc']} against "
        f"{r['LR raw + standardise (current)']['within_event_auc']}), which hides the problem: it did not converge within 400 iterations at any C "
        f"({c6['raw, unscaled C=0.1']['iterations']} iterations used), against {c6['raw, standardised C=0.1']['iterations']}-"
        f"{c6['transformed+cyclic, standardised C=0.1']['iterations']} when standardised. Convergence is therefore a check in the plan: an unconverged fit is a scaling failure even when the score looks fine.\n"
        "- With standardised and transformed inputs the C sweep is flat from C = 0.1 up "
        f"({c6['transformed+cyclic, standardised C=0.1']['within_event_auc']} at 0.1, 1 and 10) and only slightly lower below it "
        f"({c6['transformed+cyclic, standardised C=0.001']['within_event_auc']} at 0.001): the data, not the penalty, set the weights, so C = 0.1 is kept.")


def _p2(e, k):
    r = e["E2_E3.within_event_auc"]["results"]
    return (
        "**Transform by role, from the training distribution, never by looking at the test years.** Rules: non-negative amounts, depths, rain and counts with skew > 1.5 "
        "get `log1p`; signed heavy tails get `asinh`; coordinates, bounded shares and columns that are already logs are only standardised; the month is encoded as sin/cos. "
        "The role beats a pure statistic: the skew rule alone would `asinh` the longitude (skew -1.8) and give a meaningless transform of an angle.\n\n"
        f"Measured effect on the logistic regression (within-event AUC, test 2017-2023, difference to raw+standardised): transforms {ci(r['LR transformed + standardise']['diff_vs_reference'])}, "
        f"plus cyclic month {ci(r['LR transformed + cyclic month']['diff_vs_reference'])}. Neither interval excludes 0: the transforms are kept for conditioning, weight balance and the "
        "cyclic month (a modelling error, not an accuracy trick), not because they won on this test period.")


def _p3(e, k):
    v = e["E4.variance_share_between_groups"]["columns"]
    r = e["E2_E3.within_event_auc"]["results"]
    kb = k["B (T-3D..T-3h) | global z only"]
    kc = k["C (<T-3h) | global z only"]
    rain = ["bg_f1", "bg_f3", "bg_s", "nb3_mean", "nb7_mean", "nb15_mean", "nb7_max", "nb15_max"]
    within = [1 - v[c]["ep"] for c in rain]
    return (
        "**Group-aware statistics only where the metric is within-group and the evidence shows a gain.** Two kinds exist (`FeaturePlan`): `fitted` (per-group statistics from the "
        "training rows, shrunk toward the global value for small groups, global fallback for an unseen group) and `sample` (statistics of the group inside the batch being "
        "transformed, e.g. within-episode z-score or percentile; no label, no other group, but the whole group must be present at prediction time).\n\n"
        f"Where the variance sits (share between episodes): rain {min(v[c]['ep'] for c in rain):.2f}-{max(v[c]['ep'] for c in rain):.2f} for block-group and neighbourhood rain, {v['log_pol']['ep']:.2f} for log policy count, "
        f"{v['mean_cov']['ep']:.2f} for mean coverage; between years at most {max(x['year'] for x in v.values()):.2f} for any feature. So a *year* normalisation has nothing to remove "
        f"and is forbidden anyway (its statistics would use the test years). The within-event metric looks only at the {min(within):.0%}-{max(within):.0%} of rain variance that is within an episode.\n\n"
        f"Measured: within-episode z-score and percentile of the rain, exposure and county-relative terrain features added to the logistic regression: {ci(r['LR transformed + cyclic + group-aware']['diff_vs_reference'])} "
        f"(with transforms and cyclic month alone: {ci(r['LR transformed + cyclic month']['diff_vs_reference'])}); added to boosting: {ci(r['GBM raw + group-aware']['diff_vs_reference'])}. "
        "No gain beyond the interval, so the pilot plan carries **no** group-aware column; the existing `f1_vs_cty` (cell rain minus county mean) is the one such feature and stays.\n\n"
        f"SCS K-means: removing the county x month mean against a plain global z-score: tier B {ci(kb['y0_diff_vs_current'])}, tier C {ci(kc['y0_diff_vs_current'])} (difference of the "
        "alternative minus the current anomalies, y0). The county x month anomaly is not demonstrably better than the global z-score; the evidence does not require it. It is kept "
        "to leave the published stage 2-5 results unchanged, with its group means fitted on the training years only; a group never seen in training must fall back to the global mean.")


def _p4(e, k):
    r = e["E2_E3.within_event_auc"]["results"]
    b = e["E5b.bins_ablation"]["results"]
    sh = e["E5.single_feature_shape"]["columns"]
    x = e["E9.exposure_bins_blocked_cv"]
    return (
        "**Discretise only to give a linear or distance-based learner the thresholds a tree has, and only as an extra column.** Cut points come from a shallow tree on the "
        "training rows (target-based, at most 8 bins, at least 1% of the rows per bin). Never for trees; never instead of the continuous column: one feature's within-event AUC "
        f"is invariant to monotone transforms but binning ties the rows inside a bin (bg_f1: {sh['bg_f1']['single_feature_within_auc_raw_or_any_monotone']} continuous against "
        f"{sh['bg_f1']['single_feature_within_auc_bins']} binned).\n\n"
        f"Measured, added to the transformed + cyclic logistic regression ({b['reference_auc']}): all 12 tested columns {ci(r['LR transformed + cyclic + group-aware + supervised bins']['diff_vs_reference'])} "
        f"(that run also carries the group-aware columns), by role: exposure columns {ci(b['exposure (log_pol, sfha_share, mean_cov)']['diff_vs_reference'])}, "
        f"rain {ci(b['rain (bg_f1, cty_f1_max, nb7_mean)']['diff_vs_reference'])}, geography {ci(b['geography (elev, rel3, slope, cty_nbg, log_area, bg_a14)']['diff_vs_reference'])}; "
        f"single columns: sfha_share {ci(b['sfha_share']['diff_vs_reference'])}, log_pol {ci(b['log_pol']['diff_vs_reference'])}, mean_cov {ci(b['mean_cov']['diff_vs_reference'])}.\n\n"
        "The gain sits in the exposure columns, where the response has thresholds (share of policies in the special flood hazard area: none against some). Rain and geography gain nothing. "
        f"**Status: confirmed, small.** The ablation above made 15 comparisons on one test period; the exposure bins were therefore re-tested with blocked-by-year cross-validation "
        f"(7 folds of two test years, cut points refitted on the other years): pooled within-event AUC {x['reference_auc']} -> {x['with_bins_auc']}, difference {ci(x['diff'])}, positive in "
        f"{sum(f['diff'] > 0 for f in x['folds'])} of {len(x['folds'])} folds. The gain is about 0.003 AUC and only for linear / distance-based learners; use the bins as extra columns there, "
        "never for trees. The discretisation of the target-free analysis bins (depth classes in script 33, claim-rate classes in script 35) is descriptive and is not part of any model input.")


def _p5(e, k):
    g = e["E8.log_ols_vs_loglink_glm"]["results"]
    t = e["E7.retransformation"]
    return (
        "**Targets that were transformed must come back before they are scored or reported; features never come back.** `TargetTransform` keeps the transform, its inverse and the "
        "retransformation bias; `Scaled` tags a model output with its space; `scaling.mse / brier / mape_weighted` raise on anything not in original units. A log-scale prediction read "
        "back with `exp` is a *geometric mean*, and the mean is larger by the smearing factor; that factor differs by group when the residual spread does "
        f"(damage ratio by foundation: {', '.join(f'{x} {v}' for x, v in t['duan_smearing_factor'].items())}), so it must be estimated per group.\n\n"
        f"Measured on the damage ratio with the same controls (5 depth bins, loss year), foundation vs slab: ratio of geometric means (log-OLS, what script 30 reports) "
        f"basement {g['bsmt']['exp_coef_log_ols_ratio_of_geometric_means']}, crawlspace {g['crawl']['exp_coef_log_ols_ratio_of_geometric_means']}, elevated {g['elev']['exp_coef_log_ols_ratio_of_geometric_means']}; "
        f"ratio of means (log-link GLM) basement {g['bsmt']['exp_coef_loglink_glm_ratio_of_means']}, crawlspace {g['crawl']['exp_coef_loglink_glm_ratio_of_means']}, elevated {g['elev']['exp_coef_loglink_glm_ratio_of_means']}. "
        f"Stage 9's 'elevated lose 0.27x slab' (state-month fixed effects, other sample) is the first kind too; in this comparison the reduction in dollars is smaller (about {g['elev']['exp_coef_loglink_glm_ratio_of_means']}x).")


PRINCIPLES = (Principle("1. Which learners need scaling", _p1), Principle("2. Which columns are transformed, and how", _p2),
              Principle("3. Group-aware statistics", _p3), Principle("4. Discretisation", _p4), Principle("5. Targets and the inverse transform", _p5))


def render(out) -> None:
    e = json.loads((wb2.DATA_DIR / "audit" / "scaling_evidence.json").read_text())
    k = json.loads((wb2.DATA_DIR / "audit" / "scaling_kmeans.json").read_text())
    md = ["# Scaling plan", "",
          "Generated by `python scripts/40_prep.py scaling-plan` from `src/wxrisk/prep/scaling_plan.py` and the evidence of `scripts/42_scaling_audit.py` and "
          "`scripts/43_scaling_kmeans.py`. Do not edit by hand.", "",
          "Evaluation rules (AGENTS.md): split by year (fit 2010-2016 / 2000-2016, test later), skill against a climatological or prevalence baseline, "
          "event- or year-clustered bootstrap intervals; a step is kept only if its gain exceeds the interval. Every parameter below is fitted on the training rows only.", ""]
    for p in PRINCIPLES:
        md += [f"## {p.title}", "", p.body(e, k), ""]
    md += ["## Column plan: Texas flood pilot (`PILOT_PLAN`)", "",
           "Applied by `FeaturePlan.fit(df, train_mask, ...)` then `.apply(...)`; parameters are stored with a hash and the plan's hash, and `apply` refuses edited parameters or a changed plan.", "",
           "| Column | Transform | Scale | Cyclic | Group-aware | Bins | Basis |", "|---|---|---|---|---|---|---|"]
    for r in PILOT_PLAN.rules:
        md.append(f"| `{r.col}` | {r.transform} | {r.scale if not r.cyclic else '-'} | {f'period {r.cyclic:g}' if r.cyclic else '-'} | {r.group or 'none'} | "
                  f"{'yes' if r.bins else 'no'} | {r.basis} |")
    d = e["E1.feature_diagnostics"]["columns"]
    md += ["", "Diagnostics behind the plan (training rows 2010-2016): skew / excess kurtosis / exact-zero share / max over p99.", "",
           "| Column | Skew | Excess kurtosis | Zeros | Max / p99 |", "|---|---|---|---|---|"]
    for c, v in d.items():
        md.append(f"| `{c}` | {v['skew']} | {v['excess_kurtosis']} | {v['zero_share']} | {v['max_over_p99']} |")
    md += ["", "## Targets and where the inverse transform is applied", "",
           "| Target | Where | Model space | Transform | Inverse needed | Where the inverse is applied | Note |", "|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        md.append(f"| {t.name} | {t.where} | {t.model_space} | {t.transform} | {'**yes**' if t.inverse_needed else 'no'} | {t.inverse_point} | {t.basis} |")
    md += ["", "## Open items", "",
           "- The stage 9 foundation effect: script 30 (`--sections glm`) now reports both the geometric-mean ratio and the log-link GLM ratio of means; quote the second for dollars.",
           "- The SCS features go through `scs_eval.anomalise`, now a `FeaturePlan` (shrunk county x month means, global fallback for an unseen cell); the older version is `anomalise_plain`. "
           "Skill of scripts 10 and 14 with the two versions is compared in RESULTS stage 14.",
           "- Tree models receive raw features; if boosting is ever replaced by a linear or neural learner, apply `PILOT_PLAN` first and re-run `scripts/42_scaling_audit.py`.", ""]
    out.write_text("\n".join(md))
