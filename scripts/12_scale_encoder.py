"""How much does a scale-resolved (wavelet) encoder add over point features, on existing data?

Same label table, split and p0 as scripts/10. Three tiers, nested in what they may see:
  A3 (T-3D+)       : the 12Z frame of day D-3 only (what is known three days ahead)
  B  (T-3D..T-3h)  : the 12Z frame of day D
  C  (<T-3h)       : 12Z + 18Z of day D, 12->18Z motion per scale, tendencies, 6 h precipitation
ERA5 stands in for forecasts, so B and C are upper bounds; A3 is plain persistence of the state.

Methods, every one scored as BSS vs p0 (county x month) with year-grouped bootstrap CIs:
  KM-point   K-means on point values at the county's grid cell          (the scripts/10 baseline)
  KM-pooled  K-means on all scale features concatenated
  Vote-EW    one K-means expert per scale (+ a low-pass/context expert), log-lifts summed with a
             single shared weight chosen on out-of-fold train rows
  Vote-LR    same experts, weights learned per scale (logistic regression on out-of-fold lifts)
  RandVote-LR / ChanVote-LR  controls: the SAME columns regrouped at random / by channel, so a
             win of Vote-LR over them is due to grouping by scale, not to averaging K-means
  HGB-point / HGB-scale  gradient boosting on point vs scale features: the learner is held fixed,
             so the gap is the feature effect, and the gap to K-means is the learner effect

K is fixed (no tuning) so the comparison is not a tuning contest. Test years are used once.

    python scripts/12_scale_encoder.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.scale_features import CHANNELS  # noqa: E402
from wxrisk.scs_data import HEADS, STATIC, load, tier_spec  # noqa: E402
from wxrisk.scs_eval import (  # noqa: E402
    anomalise,
    brier,
    bss_ci,
    climatology,
    cluster_stats,
    fit_predict,
)

K, N_FOLDS = 32, 4
PAIRS = [("KM-pooled", "KM-point"), ("Vote-LR", "KM-point"), ("Vote-LR", "RandVote-LR"),
         ("Vote-LR", "ChanVote-LR"), ("ChanVote-LR", "RandVote-LR"), ("HGB-point", "KM-point"),
         ("HGB-scale", "HGB-point"), ("HGB-scale", "KM-point")]


def kmeans_lifts(X, fit, pred, p0, y, seed=0):
    """Fit K-means on the `fit` rows; per-head lift for the `pred` rows."""
    lab = fit_predict(X, fit, K, seed=seed, n_init=2, max_fit=100_000)
    out = {}
    for h in HEADS:
        _, lift = cluster_stats(lab, fit, p0[h], K, y[h])
        out[h] = lift[lab[pred]]
    return out


def oof_expert_lifts(X, fit, years, p0, y):
    """Out-of-fold (year-grouped) lifts for the fit rows, full-fit lifts for everything else."""
    fold = years % N_FOLDS
    oof = {h: np.zeros(len(X), np.float32) for h in HEADS}
    for f in range(N_FOLDS):
        held = fit & (fold == f)
        lifts = kmeans_lifts(X, fit & ~held, held, p0, y, seed=f)
        for h in HEADS:
            oof[h][held] = lifts[h]
    full = kmeans_lifts(X, fit, ~fit, p0, y)
    for h in HEADS:
        oof[h][~fit] = full[h]
    return oof


def choose_hgb_iters(Xa, ya, tr, va, max_iter):
    m = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.05, max_depth=4, min_samples_leaf=300,
        l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xa[tr], ya[tr])
    sc = [brier(p[:, 1], ya[va]) for p in m.staged_predict_proba(Xa[va])]
    return int(np.argmin(sc)) + 1


def hgb_predict(Xa, ya, fit, val_from, years, max_iter):
    tr, va = fit & (years < val_from), fit & (years >= val_from)
    n = choose_hgb_iters(Xa, ya, tr, va, max_iter)
    m = HistGradientBoostingClassifier(
        max_iter=n, learning_rate=0.05, max_depth=4, min_samples_leaf=300,
        l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xa[fit], ya[fit])
    return m.predict_proba(Xa)[:, 1], n


def vote(sub, fit, years, p0, y, groups):
    """Per-group K-means experts -> (equal-weight, learned-weight) predictions per head."""
    expert = {g: oof_expert_lifts(sub(cols_g), fit, years, p0, y) for g, cols_g in groups.items()}
    ew, lr_pred, weights = {}, {}, {}
    for h in HEADS:
        L = np.stack([np.log(expert[g][h]) for g in expert], axis=1)  # (n, n_experts)
        grid = np.arange(0.05, 1.0001, 0.05)
        bs = [brier(np.clip(p0[h][fit] * np.exp(w * L[fit].sum(1)), 0, 1), y[h][fit]) for w in grid]
        w = float(grid[int(np.argmin(bs))])
        ew[h] = np.clip(p0[h] * np.exp(w * L.sum(1)), 0, 1)
        Z = np.column_stack([np.log(p0[h] / (1 - p0[h])), L])
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[fit], y[h][fit])
        lr_pred[h] = lr.predict_proba(Z)[:, 1]
        weights[h] = {"EW_w": w, **dict(zip(expert, np.round(lr.coef_[0][1:], 3), strict=True))}
    return ew, lr_pred, weights


def control_groups(spec, n_groups):
    """Same columns as the scale experts, regrouped: random partition, and by channel."""
    cols = [c for g in spec["groups"].values() for c in g]
    perm = np.random.default_rng(0).permutation(len(cols))
    rand = {f"r{i}": [cols[j] for j in part]
            for i, part in enumerate(np.array_split(perm, n_groups))}
    chan = {c: [x for x in cols if x.startswith(c + "_")] for c in CHANNELS}
    rest = [x for x in cols if not any(x.startswith(c + "_") for c in CHANNELS)]
    if rest:
        chan["extra"] = rest
    return rand, chan


def run_tier(df, tier, a, rng, skip_hgb):
    spec = tier_spec(tier)
    fit = (df.year <= a.train_to).to_numpy()
    test = (df.year >= a.test_from).to_numpy()
    years = df.year.to_numpy()
    p0 = climatology(df, pd.Series(fit))
    y = {h: df[h].to_numpy().astype(np.float64) for h in HEADS}

    cols = sorted({c for g in [spec["point"], *spec["groups"].values()] for c in g} | set(STATIC))
    A = anomalise(df, cols, pd.Series(fit))
    ix = {c: i for i, c in enumerate(cols)}
    sub = lambda names: A[:, [ix[c] for c in [*names, *STATIC]]]  # noqa: E731
    X_point = sub(spec["point"])
    X_pool = sub([c for g in spec["groups"].values() for c in g])
    t0 = time.time()

    preds: dict[str, dict[str, np.ndarray]] = {
        m: {} for m in ("KM-point", "KM-pooled", "Vote-EW", "Vote-LR", "RandVote-LR",
                        "ChanVote-LR", "HGB-point", "HGB-scale")}
    for name, X in (("KM-point", X_point), ("KM-pooled", X_pool)):
        lifts = kmeans_lifts(X, fit, np.ones(len(df), bool), p0, y)
        for h in HEADS:
            preds[name][h] = np.clip(p0[h] * lifts[h], 0, 1)

    preds["Vote-EW"], preds["Vote-LR"], weights = vote(sub, fit, years, p0, y, spec["groups"])
    print(f"  [{tier}] scale vote done {time.time() - t0:.0f}s", flush=True)
    rand, chan = control_groups(spec, len(spec["groups"]))
    _, preds["RandVote-LR"], _ = vote(sub, fit, years, p0, y, rand)
    _, preds["ChanVote-LR"], _ = vote(sub, fit, years, p0, y, chan)
    print(f"  [{tier}] controls done {time.time() - t0:.0f}s", flush=True)

    if not skip_hgb:
        for name, X in (("HGB-point", X_point), ("HGB-scale", X_pool)):
            for h in HEADS:
                Xa = np.column_stack([X, np.log(p0[h] / (1 - p0[h]))])
                preds[name][h], n = hgb_predict(Xa, y[h], fit, a.val_from, years, a.hgb_iter)
                print(f"  [{tier}] {name} {h}: {n} iters  ({time.time() - t0:.0f}s)", flush=True)

    pair_rows = []
    for m, ref in PAIRS:
        if preds[m] and preds[ref]:
            for h in HEADS:
                d, lo, hi = bss_ci(preds[m][h][test], preds[ref][h][test], y[h][test],
                                   years[test], rng)
                pair_rows.append({"tier": tier, "method": m, "ref": ref, "head": h,
                                  "delta_BSS": d, "lo": lo, "hi": hi})
    rows = []
    for method, by_head in preds.items():
        for h, p in by_head.items():
            for st in ("ALL", "CO", "NM", "AZ"):
                m = test if st == "ALL" else test & (df.state == st).to_numpy()
                if y[h][m].sum() == 0:
                    continue
                row = {"tier": tier, "method": method, "head": h, "state": st,
                       "n_pos": int(y[h][m].sum()),
                       "BSS_p0": 1 - brier(p[m], y[h][m]) / brier(p0[h][m], y[h][m])}
                if st == "ALL":
                    _, lo, hi = bss_ci(p[test], p0[h][test], y[h][test], years[test], rng)
                    ref = preds["KM-point"][h]
                    d, dlo, dhi = bss_ci(p[test], ref[test], y[h][test], years[test], rng)
                    row.update(ci_lo=lo, ci_hi=hi, BSS_vs_KMpoint=d, d_lo=dlo, d_hi=dhi,
                               AUCPR=average_precision_score(y[h][test], p[test]))
                rows.append(row)
    return pd.DataFrame(rows), weights, pd.DataFrame(pair_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2000, 2023)))
    ap.add_argument("--val-from", type=int, default=2014)  # HGB iteration count only
    ap.add_argument("--train-to", type=int, default=2016)
    ap.add_argument("--test-from", type=int, default=2017)
    ap.add_argument("--tiers", nargs="+", default=["A3", "B", "C", "Cnm"])
    ap.add_argument("--hgb-iter", type=int, default=250)
    ap.add_argument("--skip-hgb", action="store_true")
    ap.add_argument("--out", default="scale_encoder_results.csv")
    a = ap.parse_args()

    df = load(a.years)
    print(f"{len(df)} county-days, {df.shape[1]} columns, years {df.year.min()}-{df.year.max()}",
          flush=True)
    rng = np.random.default_rng(0)
    res, wts, pairs = [], {}, []
    for tier in a.tiers:
        r, w, pr = run_tier(df, tier, a, rng, a.skip_hgb)
        res.append(r)
        pairs.append(pr)
        wts[tier] = w
        pd.concat(res).to_csv(wb2.cache_path(a.out), index=False)
        pd.concat(pairs).to_csv(wb2.cache_path(a.out.replace(".csv", "_pairs.csv")), index=False)
    res = pd.concat(res)
    pd.set_option("display.width", 220)
    show = res[res.state == "ALL"].drop(columns="state")
    print(show.round(4).to_string(index=False))
    print("\npaired differences in BSS (year-grouped bootstrap 95% CI)")
    print(pd.concat(pairs).round(4).to_string(index=False))
    print("\nper-state BSS vs p0")
    print(res[res.state != "ALL"].pivot_table(index=["tier", "method", "head"], columns="state",
                                              values="BSS_p0").round(4).to_string())
    print("\nVote-LR weights (log-lift coefficients):")
    for t, w in wts.items():
        print(t, pd.DataFrame(w).T.round(3).to_string())


if __name__ == "__main__":
    main()
