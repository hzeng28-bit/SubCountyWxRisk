"""Collect the learner searches of script 54: test table, learning curve, label-randomness ceiling (stage 20).

The claim being tested: if the tuned families agree within the event-clustered interval, more capacity or another classical family will
not help, and what is left is information (stage 17), not a learner.

Inputs : $WXRISK_DATA/audit/learner_sweep/<full|quick>/*.json, *_test.npy, the pilot table
Output : $WXRISK_DATA/audit/learner_convergence.txt
Run    : python scripts/55_learner_convergence.py [--quick]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("m54", Path(__file__).with_name("54_learner_search.py"))
    m54 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m54)
    o = m54.out_dir(a.quick)
    d = m54.prepare(a.quick)
    e48, y, ep, te = d.e48, d.y, d.ep, d.te
    found = {f: json.loads((o / f"{f}.json").read_text()) for f in m54.FAMILIES if (o / f"{f}.json").exists()}
    res = {f: np.load(o / f"{f}_test.npy") for f in found}
    if {"logit", "hgb", "mlp"} <= set(res):
        res["average"] = (res["logit"] + res["hgb"] + res["mlp"]) / 3
    say(f"rows {len(d.t):,}; test positives {int(y[te].sum()):,}" + (" (quick)" if a.quick else ""))
    say("\n== searched families (rolling-origin validation, same trial budget)")
    for f, j in found.items():
        say(f"  {f:7s} validation {j['best_validation']:.4f}  {j['trials']} trials, {j['minutes']:.0f} min  {j['best_params']}")

    say("\n== test 2017-2023, parameters fixed on the validation folds, scored once")
    per = {k: e48.per_episode(ep[te], y[te], p) for k, p in res.items()}
    rl, rh = per.get("logit"), per.get("hgb")
    say(f"  {'learner':8s} within-event AUC   overall AUC     AP   vs logit [95% interval]        vs hgb [95% interval]")
    for k, p in res.items():
        w = per[k][1].sum() / per[k][2].sum()
        dl = "" if k == "logit" or rl is None else "{:+.4f} [{:+.4f}, {:+.4f}]".format(*e48.boot_diff(rl, per[k]))
        dh = "" if k == "hgb" or rh is None else "{:+.4f} [{:+.4f}, {:+.4f}]".format(*e48.boot_diff(rh, per[k]))
        say(f"  {k:8s} {w:.4f}           {roc_auc_score(y[te], p):.4f}      {average_precision_score(y[te], p):.4f}   {dl:28s}  {dh}")

    if "logit" in found and "hgb" in found:
        say("\n== learning curve: within-event AUC by share of the training episodes (3 draws, test 2017-2023)")
        eps = np.unique(ep[d.tr])
        for frac in (0.1, 0.25, 0.5, 1.0):
            row = {"logit": [], "hgb": []}
            for k in range(1 if frac == 1.0 else 3):
                sub = np.isin(ep, np.random.default_rng(k).choice(eps, int(len(eps) * frac), replace=False)) & d.tr
                for f in row:
                    _, n, m = e48.per_episode(ep[te], y[te], m54.fit_score(f, found[f]["best_params"], d, sub, te))
                    row[f].append(n.sum() / m.sum())
            say(f"  {int(frac * 100):3d}%  logit {np.mean(row['logit']):.4f} (sd {np.std(row['logit']):.4f})   hgb {np.mean(row['hgb']):.4f} (sd {np.std(row['hgb']):.4f})")

    say("\n== label-randomness ceiling of the best model")
    best = max(per, key=lambda k: per[k][1].sum() / per[k][2].sum())
    d.e49.say = say
    d.e49.ceiling(pd.DataFrame({"s": 1 / (1 + np.exp(-res[best])), "y": y[te], "ep": ep[te]}))
    say(f"  (model: {best})")
    p = wb2.cache_path("audit") / ("learner_convergence_quick.txt" if a.quick else "learner_convergence.txt")
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
