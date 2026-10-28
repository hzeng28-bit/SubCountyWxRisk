"""Bayesian hyper-parameter search of one learner family on the flood pilot (stage 20; collected by script 55).

Feature set: stage 8 L2 features + claim history (stage 19). Objective of every study: within-event AUC (pair weighted) pooled over three
rolling-origin validation folds (train <= 2011 / validate 2012-13, <= 2013 / 2014-15, <= 2015 / 2016); the test years 2017-2023 are not
touched until the best parameters are refitted on 2010-2016 and scored once. One TPE study per family with the same trial budget, so the
comparison between families is not tilted by where the sampler spends its trials.
Families: logit, logit2 (pairwise products), hgb, rf, et (random / extra trees on negatives downsampled to 5%: uniform downsampling
shifts the logit by a constant, which leaves a ranking inside an episode unchanged), mlp (sklearn).

Inputs : flood_pilot_table.parquet, clean NFIP claims (script 49 `history`)
Outputs: $WXRISK_DATA/audit/learner_sweep/<full|quick>/<family>.json (best parameters, validation score), <family>_trials.csv,
         <family>_test.npy and <family>.journal (the study; rerunning the command continues an unfinished search)
Run    : python scripts/54_learner_search.py --family hgb --trials 40 [--quick]
         sbatch hpc/learner_sweep.sbatch                     # all families, then script 55 collects
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402
from wxrisk.prep.scaling import FeaturePlan, FeatureRule  # noqa: E402
from wxrisk.prep.scaling_plan import PILOT_PLAN  # noqa: E402

FAMILIES = ["logit", "logit2", "hgb", "rf", "et", "mlp"]
HIST = ["hist00", "prior3"]
KEEP_NEG = 0.05
FOLDS = [(2011, (2012, 2013)), (2013, (2014, 2015)), (2015, (2016, 2016))]  # (last training year, validation years)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def logit_of(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def out_dir(quick: bool) -> Path:
    d = wb2.cache_path("audit") / "learner_sweep" / ("quick" if quick else "full")
    d.mkdir(parents=True, exist_ok=True)
    return d


def prepare(quick: bool) -> SimpleNamespace:
    e26, e48, e49 = _load("e26", "26_flood_pilot_experiment.py"), _load("e48", "48_flood_forecast_pilot.py"), _load("e49", "49_information_audit.py")
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t = t.join(e49.history(t))
    if quick:
        eps = t.ep.drop_duplicates().sample(frac=0.15, random_state=0)
        t = t[t.ep.isin(eps)].reset_index(drop=True)
    cols = e26.COUNTY + e26.EXPO + e26.CELL + HIST
    d = SimpleNamespace(t=t, cols=cols, e48=e48, e49=e49)
    d.y, d.yr, d.ep = t.y.to_numpy().astype(np.float64), t.year.to_numpy(), t.ep.to_numpy()
    d.X = t[cols].to_numpy(np.float32)
    d.Xn = np.nan_to_num(d.X)
    plan = FeaturePlan(tuple([r for r in PILOT_PLAN.rules if r.col in cols] + [FeatureRule(c, basis="log1p count") for c in HIST]))
    d.tr, d.te = d.yr <= 2016, d.yr >= 2017
    d.Z = np.nan_to_num(plan.apply(t, plan.fit(t, d.tr, "train rows")).to_numpy()).astype(np.float32)
    d.keep = (d.y == 1) | (np.random.default_rng(0).random(len(d.y)) < KEEP_NEG)
    d.folds = [(d.yr <= last, np.isin(d.yr, v)) for last, v in FOLDS]
    d.P = None
    d.jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or -1
    return d


def poly(d):
    if d.P is None:  # all pairwise products and squares of the plan features, standardised on the training years
        P = PolynomialFeatures(2, include_bias=False).fit_transform(d.Z)
        d.P = StandardScaler().fit(P[d.tr]).transform(P).astype(np.float32)
    return d.P


def suggest(fam, tr):
    if fam == "logit":
        return {"C": tr.suggest_float("C", 1e-3, 10, log=True)}
    if fam == "logit2":
        return {"C": tr.suggest_float("C", 1e-4, 1, log=True)}
    if fam == "hgb":
        return {"max_depth": tr.suggest_int("max_depth", 2, 10), "min_samples_leaf": tr.suggest_int("min_samples_leaf", 50, 3000, log=True),
                "learning_rate": tr.suggest_float("learning_rate", 0.02, 0.2, log=True), "l2": tr.suggest_float("l2", 0.1, 30, log=True),
                "max_iter": tr.suggest_int("max_iter", 100, 600), "max_leaf_nodes": tr.suggest_int("max_leaf_nodes", 8, 64)}
    if fam in ("rf", "et"):
        return {"min_samples_leaf": tr.suggest_int("min_samples_leaf", 5, 300, log=True), "max_features": tr.suggest_float("max_features", 0.1, 0.8),
                "max_depth": tr.suggest_categorical("max_depth", [8, 12, 16, 24, None])}
    return {"hidden": tr.suggest_categorical("hidden", ["32", "64-32", "128-64", "128-64-32"]), "alpha": tr.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "lr": tr.suggest_float("lr", 1e-4, 1e-2, log=True)}


def fit_score(fam, p, d, trm, sm):
    """Scores (logits, higher = more likely) of rows `sm` from a model of family `fam` fitted on rows `trm`."""
    y = d.y
    if fam in ("logit", "logit2"):
        M = d.Z if fam == "logit" else poly(d)
        return LogisticRegression(C=p["C"], max_iter=400).fit(M[trm], y[trm]).decision_function(M[sm])
    if fam == "hgb":
        m = HistGradientBoostingClassifier(max_iter=p["max_iter"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                                           min_samples_leaf=p["min_samples_leaf"], l2_regularization=p["l2"], max_leaf_nodes=p["max_leaf_nodes"],
                                           early_stopping=False, random_state=0).fit(d.X[trm], y[trm])
        return logit_of(m.predict_proba(d.X[sm])[:, 1])
    if fam in ("rf", "et"):
        cls = RandomForestClassifier if fam == "rf" else ExtraTreesClassifier
        k = trm & d.keep
        m = cls(n_estimators=200, min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"], max_depth=p["max_depth"],
                n_jobs=d.jobs, random_state=0).fit(d.Xn[k], y[k])
        return logit_of(m.predict_proba(d.Xn[sm])[:, 1])
    k = trm & d.keep
    m = MLPClassifier(tuple(int(x) for x in p["hidden"].split("-")), alpha=p["alpha"], learning_rate_init=p["lr"], early_stopping=True,
                      n_iter_no_change=8, max_iter=200, random_state=0).fit(d.Z[k], y[k])
    return logit_of(m.predict_proba(d.Z[sm])[:, 1])


D = None  # the prepared data of a fold worker, loaded once by `init_worker`


def init_worker(quick, thr):
    global D
    D = prepare(quick)
    D.jobs = thr  # n_jobs of the forests inside a worker


def fold_job(arg):
    fam, p, i, thr = arg
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=thr):
        trm, vam = D.folds[i]
        _, n, m = D.e48.per_episode(D.ep[vam], D.y[vam], fit_score(fam, p, D, trm, vam))
    return n.sum(), m.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout-h", type=float, default=18.0, help="stop the search after this many hours (the job limit is 24 h)")
    ap.add_argument("--quick", action="store_true", help="15%% of the episodes (smoke test)")
    ap.add_argument("--fold-workers", type=int, default=3, help="processes that fit the validation folds of one trial side by side (1 = in turn)")
    a = ap.parse_args()
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    d = prepare(a.quick)
    print(f"{a.family}: rows {len(d.t):,}, positives {int(d.y.sum()):,}, features {len(d.cols)}", flush=True)
    t0 = time.time()

    # One trial = three folds. Boosting does not scale past a few threads inside one process (bench_threads.py: 4 processes x 2 threads
    # give 2.4x the fits per second of 1 process x 4 threads), so the folds run in separate processes while the sampler stays sequential.
    # Workers are spawned and load the data themselves: a fork after BLAS or OpenMP has run can leave the children hanging (seen with
    # logit2 and mlp on macOS), and nothing big has to be pickled.
    global D
    D = d
    cpus = d.jobs if d.jobs > 0 else (os.cpu_count() or 1)
    nw = min(a.fold_workers, len(d.folds))
    thr = max(cpus // nw, 1)
    if nw > 1:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(nw, initializer=init_worker, initargs=(a.quick, thr))

    def objective(trial):
        p = suggest(a.family, trial)
        jobs = [(a.family, p, i, thr) for i in range(len(d.folds))]
        res = pool.map(fold_job, jobs) if nw > 1 else [fold_job(j) for j in jobs]  # without a pool the parent's own data is used
        num, den = sum(r[0] for r in res), sum(r[1] for r in res)
        print(f"  trial {trial.number}: {num / den:.4f}  {p}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
        return num / den

    # a journal file keeps the finished trials: a job that hits its wall time (or is cancelled) is resubmitted and carries on
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    storage = JournalStorage(JournalFileBackend(str(out_dir(a.quick) / f"{a.family}.journal")))
    done = len(optuna.load_study(study_name=a.family, storage=storage).trials) if a.family in {x.study_name for x in optuna.get_all_study_summaries(storage)} else 0
    # the seed follows the number of trials already in the journal: a fixed seed would replay the same random start points after a restart
    study = optuna.create_study(study_name=a.family, storage=storage, load_if_exists=True, direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=done, multivariate=True))
    done = sum(x.state == optuna.trial.TrialState.COMPLETE for x in study.trials)
    print(f"  {done} trials already finished", flush=True)
    if done < a.trials:
        study.optimize(objective, n_trials=a.trials - done, timeout=a.timeout_h * 3600)
    if sum(x.state == optuna.trial.TrialState.COMPLETE for x in study.trials) < a.trials:
        print("budget not reached before the time limit: resubmit the same job to continue", flush=True)
        return
    best = study.best_params
    if nw > 1:
        pool.close()
    s = fit_score(a.family, best, d, d.tr, d.te)  # the only time the test years are scored
    o = out_dir(a.quick)
    np.save(o / f"{a.family}_test.npy", s.astype(np.float32))
    study.trials_dataframe().to_csv(o / f"{a.family}_trials.csv", index=False)
    (o / f"{a.family}.json").write_text(json.dumps({"family": a.family, "best_params": best, "best_validation": study.best_value,
                                                    "trials": sum(x.state == optuna.trial.TrialState.COMPLETE for x in study.trials), "minutes": (time.time() - t0) / 60}, indent=1))
    print(f"best validation {study.best_value:.4f} after {len(study.trials)} trials: {best}", flush=True)


if __name__ == "__main__":
    main()
