"""Audit of the flood pilot's model assumptions against the task (stage 34).

Task (DESIGN.md section 0 and 3): R = Pr[hazard occurs | forecast], judged by Brier skill against climatology, reliability and AUC-PR.
The pilot instead scores the ranking of block groups inside an observed rain episode (within-event AUC). This script measures how far apart
the two are, without changing the model:
  1 metrics    the stage 8 L2 model's stored test predictions under the task's metrics (Brier, BSS against train / test / yearly prevalence,
               calibration slope, reliability by decile, AP) and under different aggregations of the within-event AUC
  2 selection  which days and claims the pilot data cover (episode days only; GEFS crops exist only for those days)
  3 dur        whether `dur` (observed episode duration, in BASE of the forecast pilot) leaks the future: forecast pilot with and without it,
               within-event and between-episode
  4 trigger    (--t20) the same L2 model on a table built with a 20 mm trigger instead of 30 mm
Inputs : flood_pilot_test_preds.npz, flood_pilot_table.parquet, flood_pilot_fc_features.parquet, gefs/gefs_meta.npz, county_daily.npz [table_T20.parquet]
Output : $WXRISK_DATA/audit/model_assumption_audit.txt
Run    : python scripts/73_model_assumption_audit.py [--t20]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e26, e48 = _load("e26", "26_flood_pilot_experiment.py"), _load("e48", "48_flood_forecast_pilot.py")
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def episode_aucs(ep, y, s, min_bg=10):
    rows = []
    df = pd.DataFrame({"ep": ep, "y": y, "s": s})
    for e, g in df.groupby("ep", sort=False):
        n1, n0 = int(g.y.sum()), len(g) - int(g.y.sum())
        if len(g) < min_bg or n1 == 0 or n0 == 0:
            continue
        r = rankdata(g.s.to_numpy())
        rows.append((e, (r[g.y.to_numpy() == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), n1 * n0, n1, len(g)))
    return pd.DataFrame(rows, columns=["ep", "auc", "pairs", "n1", "n"])


def part_metrics() -> None:
    say("== 1. the stage 8 L2 model under the task's metrics (test 2017-2023)")
    d = np.load(fp.PILOT / "flood_pilot_test_preds.npz")
    p, y, ep, yr = d["L2 + cell rain"], d["y"], d["ep"], d["year"]
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["year", "y"])
    prev_tr = t.y[t.year <= 2016].mean()
    brier = np.mean((p - y) ** 2)
    ref = lambda q: np.mean((q - y) ** 2)  # noqa: E731
    say(f"  rows {len(y):,}, positives {int(y.sum()):,}, prevalence train {prev_tr:.4f} test {y.mean():.4f}, mean prediction {p.mean():.4f}")
    say(f"  Brier {brier:.5f}; BSS vs train prevalence {1 - brier / ref(prev_tr):+.3f}; vs test prevalence {1 - brier / ref(y.mean()):+.3f}")
    yr_prev = pd.Series(y).groupby(yr).transform("mean").to_numpy()
    say(f"  BSS vs the prevalence of the same year (oracle level) {1 - brier / np.mean((yr_prev - y) ** 2):+.3f}")
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    m = LogisticRegression(C=1e6, max_iter=500).fit(lg[:, None], y)
    say(f"  calibration on the test set: slope {m.coef_[0][0]:.2f}, intercept {m.intercept_[0]:+.2f} (1 and 0 would be calibrated)")
    pc = p * (y.mean() / p.mean())
    say(f"  after rescaling the mean to the test prevalence: BSS vs test prevalence {1 - np.mean((pc - y) ** 2) / ref(y.mean()):+.3f}")
    say("  by year: observed rate / mean prediction: " + ", ".join(f"{int(a)} {y[yr == a].mean() * 100:.2f}/{p[yr == a].mean() * 100:.2f}%" for a in np.unique(yr)))
    dec = pd.qcut(p, 10, labels=False, duplicates="drop")
    rel = pd.DataFrame({"p": p, "y": y}).groupby(dec).mean()
    say("  reliability by decile of the prediction (mean p -> observed rate, %): " + ", ".join(f"{a * 100:.3f}->{b * 100:.3f}" for a, b in zip(rel.p, rel.y, strict=True)))
    say(f"  overall AUC {roc_auc_score(y, p):.4f}; AP {average_precision_score(y, p):.4f} against base rate {y.mean():.4f} (lift {average_precision_score(y, p) / y.mean():.1f}x)")
    r = episode_aucs(ep, y, p)
    n_ep = len(np.unique(ep))
    say(f"  within-event AUC: pair weighted {np.average(r.auc, weights=r.pairs):.4f}; mean of episodes {r.auc.mean():.4f}; median {r.auc.median():.4f}; "
        f"weighted by positives {np.average(r.auc, weights=r.n1):.4f}")
    say(f"  episodes that count: {len(r)} of {n_ep} test episodes ({len(r) / n_ep:.1%}); they hold {r.n1.sum() / y.sum():.1%} of the test positives")
    top = r.sort_values("pairs", ascending=False)
    say(f"  the 10 largest episodes carry {top.pairs.head(10).sum() / r.pairs.sum():.1%} of the pairs and have a pair-weighted AUC of "
        f"{np.average(top.auc.head(10), weights=top.pairs.head(10)):.4f}; the other episodes {np.average(top.auc.iloc[10:], weights=top.pairs.iloc[10:]):.4f}")
    big = r.n >= 100
    say(f"  episodes with >= 100 block groups: {big.sum()} ({np.average(r.auc[big], weights=r.pairs[big]):.4f}); fewer: {(~big).sum()} ({np.average(r.auc[~big], weights=r.pairs[~big]):.4f})")


def part_selection() -> None:
    say("\n== 2. which days and claims the pilot data cover")
    d = np.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    D1 = d["D1"]
    days = pd.date_range("2000-01-01", periods=D1.shape[0])
    m = (days >= "2010-01-01") & (days <= "2019-12-31")
    n_trig = int((D1[m] >= 30).sum())
    n_gefs = len(list(e48.GEFS.glob("apcp_*.npy")))
    tot_days = int(m.sum())
    say(f"  county-days 2010-2019 with a county 1-day maximum >= 30 mm: {n_trig:,} of {D1[m].size:,} ({n_trig / D1[m].size:.2%})")
    say(f"  GEFS reforecast days stored: {n_gefs:,} of {tot_days:,} calendar days ({n_gefs / tot_days:.1%}); chosen as the day before an observed episode starts")
    say("  => the stored forecast days are the days before ANY county's observed episode, and the rows are county x observed episode: a county-day on which")
    say("     the forecast said rain and no 30 mm episode followed (a false alarm) or on which nothing was forecast is not a row. The pilot scores the ranking")
    say("     of block groups given that an observed episode happened, not the probability a forecast should issue")
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "y", "n_claims", "year"])
    say(f"  claims inside the pilot windows: {int(t.n_claims.sum()):,} of 34,989 Texas inland non-tropical claims 2010-2023 (88.8%); "
        f"episodes {t.ep.nunique():,}, with a positive {int((t.groupby('ep').y.max() > 0).sum()):,} ({(t.groupby('ep').y.max() > 0).mean():.1%})")


def part_dur() -> None:
    say("\n== 3. is `dur` (observed episode duration) a forecast-time input? forecast pilot with and without it")
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t = t[t.year <= e48.LAST_YEAR].merge(pd.read_parquet(e48.FEATS), on=["ep", "bg"], how="inner")
    t["log_pol"] = np.log1p(t.n_pol)
    y = t.y.to_numpy().astype(float)
    yr = t.year.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    fc = ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"]
    per = {}
    for name, base in {"with dur (stage 16)": e48.BASE, "without dur": [c for c in e48.BASE if c != "dur"]}.items():
        p, n = e26.fit_predict(t[base + fc].to_numpy(np.float32), y, tr, te, va)
        per[name] = e48.per_episode(t.ep.to_numpy()[te], y[te], p)
        say(f"  within-event AUC, forecast rain, {name}: {per[name][1].sum() / per[name][2].sum():.4f} ({n} iterations)")
    say("  difference without - with dur: {:+.4f} [{:+.4f}, {:+.4f}]".format(*e48.boot_diff(per["with dur (stage 16)"], per["without dur"])))
    g = t.groupby("ep")
    ee = g.agg(year=("year", "first"), month=("month", "first"), dur=("dur", "first"), nbg=("cty_nbg", "first"), anyc=("y", "max"), fcmax=("fc_cty_max", "first"),
               fcmean=("fc_cty_mean", "first"), fcsd=("fc_sd", "mean"), fcv25=("fc_v25", "max"), ocmax=("oc_cty_max", "first"))
    for label, cols in {"with dur": ["month", "dur", "nbg", "fcmax", "fcmean", "fcsd", "fcv25"], "without dur": ["month", "nbg", "fcmax", "fcmean", "fcsd", "fcv25"]}.items():
        pr = np.zeros(len(ee))
        for yv in sorted(ee.year.unique()):
            m = (ee.year == yv).to_numpy()
            pr[m] = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=40, random_state=0).fit(
                ee.loc[~m, cols], ee.anyc[~m]).predict_proba(ee.loc[m, cols])[:, 1]
        say(f"  episode level (any claim, leave-one-year-out), forecast rain, {label}: AUC {roc_auc_score(ee.anyc, pr):.4f}")
    say(f"  Spearman of dur with the number of claimed block groups per episode: {pd.Series(ee.dur).corr(g.y.sum(), method='spearman'):+.3f}")


def part_t20() -> None:
    say("\n== 4. trigger 20 mm instead of 30 mm (same L2 model, same split)")
    f = fp.PILOT / "table_T20.parquet"
    t = pd.read_parquet(f)
    t["log_pol"] = np.log1p(t.n_pol)
    base = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "y", "n_claims"])
    say(f"  T=30: {len(base):,} rows, {base.ep.nunique():,} episodes, {int(base.y.sum()):,} positives, claims in windows {int(base.n_claims.sum()):,}")
    say(f"  T=20: {len(t):,} rows, {t.ep.nunique():,} episodes, {int(t.y.sum()):,} positives, claims in windows {int(t.n_claims.sum()):,}")
    cols = e26.COUNTY + e26.EXPO + e26.CELL
    y = t.y.to_numpy().astype(float)
    yr = t.year.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    p, n = e26.fit_predict(t[cols].to_numpy(np.float32), y, tr, te, va)
    r = episode_aucs(t.ep.to_numpy()[te], y[te], p)
    say(f"  within-event AUC {np.average(r.auc, weights=r.pairs):.4f} over {len(r)} episodes; overall AUC {roc_auc_score(y[te], p):.4f}; "
        f"test positives {int(y[te].sum())}; prevalence train {y[tr].mean():.4f} test {y[te].mean():.4f}")
    # recall of claims by trigger: share of all claims inside windows at the two triggers
    say("  (share of the 34,989 claims inside windows: T=30 88.8%; T=20 above)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t20", action="store_true")
    a = ap.parse_args()
    part_metrics()
    part_selection()
    part_dur()
    if a.t20:
        part_t20()
    p = wb2.cache_path("audit") / "model_assumption_audit.txt"
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
