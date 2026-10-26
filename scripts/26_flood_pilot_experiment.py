"""Texas sub-county flood pilot: does going below the county add skill, and which scheme gets it?

Unit: block group x rain episode (see 25_build_flood_pilot_table.py). Outcome: any non-tropical inland
NFIP claim in the block group during the episode window. Train 2010-2016, test 2017-2023; a blocked
cross-validation over year pairs (all 14 years held out once) gives the stability check, because the
official test years hold only 3.7k positives.

Scores that need no training (the "vote" family, observed-rain analogue of ensemble voting):
  county            the county's peak rain (identical for every block group of the episode)
  cell              rain at the block group's own 4 km cell
  nep7_25 ...       fraction of cells within 7x7 / 15x15 of the block group above 25 / 50 mm
Learned models: HistGradientBoosting on nested feature groups
  L0 county rain only      L1 + exposure (policies, flood-zone share, coverage)
  L2 + cell rain           L3 + neighbourhood rain (mean, max, exceedance fractions)
  L4 + location, antecedent rain, optional terrain      L5 + wavelet scale features (if present)

Metrics: AUC and AP over all rows; WITHIN-EPISODE AUC pooled over block-group pairs of the same
episode (this is the sub-county skill: a county-level model scores exactly 0.5); Brier skill vs the
training prevalence. Year-block bootstrap for intervals.

    python scripts/26_flood_pilot_experiment.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402

COUNTY = ["cty_f1_max", "cty_f1_mean", "cty_s_mean", "cty_a14_mean", "dur", "month", "cty_nbg"]
EXPO = ["log_pol", "sfha_share", "single_share", "mean_cov"]
CELL = ["bg_f1", "bg_f3", "bg_s", "f1_vs_cty"]
NEIGH = ["nb3_mean", "nb7_mean", "nb15_mean", "nb7_max", "nb15_max", "nep7_25", "nep7_50", "nep15_25", "nep15_50"]
STATIC = ["lat", "lon", "log_area", "bg_a14"]
SCORES = {"county peak rain": "cty_f1_max", "cell rain (point vote)": "bg_f1",
          "nep 7x7 >=25mm": "nep7_25", "nep 15x15 >=25mm": "nep15_25",
          "nep 7x7 >=50mm": "nep7_50", "nep 15x15 >=50mm": "nep15_50", "neighbourhood max 15x15": "nb15_max"}


def groups(extra: dict[str, list[str]]) -> dict[str, list[str]]:
    g = {"L0 county rain": COUNTY}
    g["L1 + exposure"] = g["L0 county rain"] + EXPO
    g["L2 + cell rain"] = g["L1 + exposure"] + CELL
    g["L3 + neighbourhood"] = g["L2 + cell rain"] + NEIGH
    g["L4 + location/antecedent"] = g["L3 + neighbourhood"] + STATIC
    prev = g["L4 + location/antecedent"]
    for name, cols in extra.items():
        prev = prev + cols
        g[name] = prev
    return g


def within_auc(ep: np.ndarray, y: np.ndarray, s: np.ndarray, min_bg: int = 10) -> tuple[float, float, int]:
    """Pooled (pair-weighted) and mean-of-episodes AUC over block groups of the same episode."""
    df = pd.DataFrame({"ep": ep, "y": y, "s": s})
    num = den = 0.0
    per = []
    for _, g in df.groupby("ep", sort=False):
        n1 = int(g.y.sum())
        n0 = len(g) - n1
        if len(g) < min_bg or n1 == 0 or n0 == 0:
            continue
        r = rankdata(g.s.to_numpy())
        auc = (r[g.y.to_numpy() == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        num += auc * n1 * n0
        den += n1 * n0
        per.append(auc)
    return (num / den if den else np.nan), (float(np.mean(per)) if per else np.nan), len(per)


def fit_predict(X, y, tr, te, va=None, max_iter=300):
    def mk(n):
        return HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300,
                                              l2_regularization=1.0, early_stopping=False, random_state=0)
    if va is not None:
        m = mk(max_iter).fit(X[tr & ~va], y[tr & ~va])
        sc = [np.mean((p[:, 1] - y[va]) ** 2) for p in m.staged_predict_proba(X[va])]
        n = int(np.argmin(sc)) + 1
    else:
        n = max_iter
    return mk(n).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1], n


def score_row(name, ep, y, s, base_rate, bs_ref):
    w, mean_w, n_ep = within_auc(ep, y, s)
    return {"model": name, "AUC": roc_auc_score(y, s), "AP": average_precision_score(y, s),
            "within_AUC": w, "within_AUC_mean": mean_w, "n_episodes": n_ep,
            "BSS": 1 - np.mean((s - y) ** 2) / bs_ref if s.max() <= 1 else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="flood_pilot_table.parquet")
    ap.add_argument("--cv", action="store_true", help="blocked cross-validation over year pairs")
    ap.add_argument("--out", default="flood_pilot_results.csv")
    ap.add_argument("--only", nargs="*", help="run only the feature groups whose name contains one of these")
    ap.add_argument("--drop", nargs="*", default=[], help="feature columns to remove from every group")
    ap.add_argument("--no-votes", action="store_true")
    a = ap.parse_args()

    t = pd.read_parquet(fp.PILOT / a.table)
    t["log_pol"] = np.log1p(t.n_pol)
    t["log_area"] = np.log(t.area_km2.clip(lower=0.01))
    extra = {}
    ter = fp.PILOT / "bg_terrain.parquet"
    if ter.exists():
        tt = pd.read_parquet(ter)
        t = t.merge(tt, on="bg10", how="left")
        extra["L4b + terrain"] = [c for c in tt.columns if c != "bg10"]
    wav = fp.PILOT / "flood_pilot_wavelet.parquet"
    if wav.exists():
        w = pd.read_parquet(wav)
        t = t.merge(w, on=["ep", "bg"], how="left")
        extra["L5 + wavelet scales"] = [c for c in w.columns if c not in ("ep", "bg")]
    edy = fp.PILOT / "flood_pilot_expodyn.parquet"
    if edy.exists():
        # mand_share drifts with the year (0.7% in 2010 to 12% in 2021, a reporting change) and would teach the model the year
        e = pd.read_parquet(edy, columns=["ep", "bg", "new12_share", "mid_share"])
        t = t.merge(e, on=["ep", "bg"], how="left")
        extra["L6 + exposure dynamics"] = ["new12_share", "mid_share"]
    G = groups(extra)
    if a.only:
        G = {k: v for k, v in G.items() if any(o in k for o in a.only)}
    G = {k: [c for c in v if c not in a.drop] for k, v in G.items()}
    y = t.y.to_numpy().astype(np.float64)
    yr = t.year.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    prev = y[tr].mean()
    bs_ref = np.mean((prev - y[te]) ** 2)
    print(f"{len(t):,} rows; train {tr.sum():,} ({int(y[tr].sum()):,} pos), test {te.sum():,} ({int(y[te].sum()):,} pos); "
          f"prevalence train {prev:.4f}", flush=True)

    rows = []
    for name, col in ({} if a.no_votes else SCORES).items():
        s = t[col].to_numpy().astype(np.float64)
        rows.append(score_row("vote: " + name, t.ep.to_numpy()[te], y[te], s[te], prev, bs_ref))
    preds = {}
    for name, cols in G.items():
        X = t[cols].to_numpy(np.float32)
        p, n = fit_predict(X, y, tr, te, va)
        preds[name] = p
        rows.append(score_row(f"learned: {name}", t.ep.to_numpy()[te], y[te], p, prev, bs_ref))
        print(f"  {name}: {n} iterations", flush=True)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("\nTEST 2017-2023")
    print(res.round(4).to_string(index=False))
    res.to_csv(fp.PILOT / a.out, index=False)
    # a non-default --out gets its own predictions file so a side experiment cannot overwrite the ones scripts/29 bootstraps
    pred_name = "flood_pilot_test_preds.npz" if a.out == "flood_pilot_results.csv" else Path(a.out).stem + "_test_preds.npz"
    np.savez_compressed(fp.PILOT / pred_name, **{k: v for k, v in preds.items()},
                        ep=t.ep.to_numpy()[te], y=y[te], bg=t.bg.to_numpy()[te], year=yr[te])

    if a.cv:
        print("\nBLOCKED CROSS-VALIDATION (year pairs held out; all 14 years scored)")
        folds = [(2010, 2011), (2012, 2013), (2014, 2015), (2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023)]
        out = {k: [] for k in G}
        for f in folds:
            hold = np.isin(yr, f)
            for name, cols in G.items():
                X = t[cols].to_numpy(np.float32)
                p, _ = fit_predict(X, y, ~hold, hold, None, 200)
                w, _, _ = within_auc(t.ep.to_numpy()[hold], y[hold], p)
                out[name].append((roc_auc_score(y[hold], p), w, int(y[hold].sum())))
            print(f"  fold {f} done", flush=True)
        cv = pd.DataFrame({k: {"AUC_mean": np.mean([x[0] for x in v]), "AUC_min": np.min([x[0] for x in v]),
                               "within_mean": np.nanmean([x[1] for x in v]), "within_min": np.nanmin([x[1] for x in v]),
                               "within_max": np.nanmax([x[1] for x in v])} for k, v in out.items()}).T
        print(cv.round(4).to_string())
        cv.to_csv(fp.PILOT / "flood_pilot_cv.csv")


if __name__ == "__main__":
    main()
