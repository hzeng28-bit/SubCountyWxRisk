"""Claim history as a pilot feature: full evaluation of base vs base + history (stage 19).

Stage 17 showed +0.026 within-event AUC from hist00 (claims 2000-2009 in the block group) and prior3 (claims in the 3 years before the
episode), both known ahead. This script checks that the gain is not an artefact of one split or one learner:
  1. main split (train 2010-2016, test 2017-2023), boosting (stage 8 learner) and logistic regression on the scaling plan (S12)
  2. blocked cross-validation over 7 year pairs, boosting; per-fold differences and a pooled event-clustered paired bootstrap
  3. overall AUC and AP next to within-event AUC (evaluation rules)

Inputs : flood_pilot_table.parquet, clean NFIP claims (through script 49 `history`)
Output : $WXRISK_DATA/audit/pilot_with_history.txt
Run    : python scripts/53_pilot_with_history.py          # about 15 min on a Mac CPU; on Sol: sbatch hpc/run.sbatch scripts/53_pilot_with_history.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402
from wxrisk.prep.scaling import FeaturePlan, FeatureRule  # noqa: E402
from wxrisk.prep.scaling_plan import PILOT_PLAN  # noqa: E402

FOLDS = [(2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023)]
HIST = ["hist00", "prior3"]
OUT = []


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def pooled(parts):
    """Concatenate per-episode (ids, num, den) of disjoint test sets into one sorted triple for the bootstrap."""
    ids, num, den = (np.concatenate(x) for x in zip(*parts, strict=True))
    o = np.argsort(ids)
    return ids[o], num[o], den[o]


def logistic(t, cols, tr, te):
    """Logistic regression on the pilot scaling plan; history columns are already log1p, so they are standardised only."""
    rules = [r for r in PILOT_PLAN.rules if r.col in cols] + [FeatureRule(c, basis="log1p of a claim count already") for c in cols if c in HIST]
    plan = FeaturePlan(tuple(rules))
    prm = plan.fit(t, tr, "train rows")
    Z = np.nan_to_num(plan.apply(t, prm).to_numpy())
    return LogisticRegression(C=0.1, max_iter=300).fit(Z[tr], t.y.to_numpy()[tr]).predict_proba(Z[te])[:, 1]


def main() -> None:
    e26, e48, e49 = _load("e26", "26_flood_pilot_experiment.py"), _load("e48", "48_flood_forecast_pilot.py"), _load("e49", "49_information_audit.py")
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t = t.join(e49.history(t))
    base = e26.COUNTY + e26.EXPO + e26.CELL
    y = t.y.to_numpy().astype(np.float64)
    yr = t.year.to_numpy()
    ep = t.ep.to_numpy()

    def metrics(mask, p):
        ids, num, den = e48.per_episode(ep[mask], y[mask], p)
        return (ids, num, den), num.sum() / den.sum(), roc_auc_score(y[mask], p), average_precision_score(y[mask], p)

    say("== 1. main split, train 2010-2016, test 2017-2023")
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    res = {}
    for name, cols in {"boosting base": base, "boosting + history": base + HIST}.items():
        p, n = e26.fit_predict(t[cols].to_numpy(np.float32), y, tr, te, va)
        res[name] = metrics(te, p)
        say(f"    fitted {name} ({n} iterations)")
    for name, cols in {"logistic base": base, "logistic + history": base + HIST}.items():
        res[name] = metrics(te, logistic(t, cols, tr, te))
    say(f"  {'model':22s} within-event AUC   overall AUC     AP   difference to the base of the same learner [95% interval]")
    for name, (per, w, auc, ap) in res.items():
        ref = res[name.split()[0] + " base"][0]
        d = "" if name.endswith("base") else "{:+.4f} [{:+.4f}, {:+.4f}]".format(*e48.boot_diff(ref, per))
        say(f"  {name:22s} {w:.4f}           {auc:.4f}      {ap:.4f}   {d}")

    say("\n== 2. blocked cross-validation, boosting, 7 year-pair folds (training on the other 12 years)")
    parts = {"base": [], "history": []}
    rows = []
    for f in FOLDS:
        hold = np.isin(yr, f)
        trn = ~hold
        row = {"fold": f"{f[0]}-{f[1]}", "test positives": int(y[hold].sum())}
        for name, cols in {"base": base, "history": base + HIST}.items():
            p, _ = e26.fit_predict(t[cols].to_numpy(np.float32), y, trn, hold, None, 200)
            per, w, _, _ = metrics(hold, p)
            parts[name].append(per)
            row[name] = w
        row["difference"] = row["history"] - row["base"]
        rows.append(row)
        say(f"    fold {f} done")
    cv = pd.DataFrame(rows)
    say(cv.round(4).to_string(index=False))
    a, b = pooled(parts["base"]), pooled(parts["history"])
    dlt, lo, hi = e48.boot_diff(a, b)
    say(f"  pooled over folds: base {a[1].sum() / a[2].sum():.4f}, + history {b[1].sum() / b[2].sum():.4f}, "
        f"difference {dlt:+.4f} [{lo:+.4f}, {hi:+.4f}]; folds with a positive difference: {(cv.difference > 0).sum()} of {len(cv)}")
    p = wb2.cache_path("audit") / "pilot_with_history.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
