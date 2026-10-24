"""Diagnostics for the Texas sub-county flood pilot (run after 26_flood_pilot_experiment.py).

  paired     episode-cluster bootstrap of the within-episode AUC differences between feature groups
  shuffle    labels permuted within each episode: the within-episode AUC must fall to 0.5
  curve      learning curve: train on 2, 4, 7 years (random draws) and score the test years
  sweep      learner capacity: boosting depth 3 / 5 / 8, logistic regression
  drift      claim rate per policy-episode by year, and train-to-test shift of the main features
  strata     within-episode AUC by county size (number of block groups) and by episode size

    python scripts/29_flood_pilot_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402

exp = __import__("importlib").util.spec_from_file_location(
    "exp26", Path(__file__).resolve().parent / "26_flood_pilot_experiment.py")
m26 = __import__("importlib").util.module_from_spec(exp)
exp.loader.exec_module(m26)


def load():
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t["log_area"] = np.log(t.area_km2.clip(lower=0.01))
    extra = {}
    ter = pd.read_parquet(fp.PILOT / "bg_terrain.parquet")
    t = t.merge(ter, on="bg10", how="left")
    extra["L4b + terrain"] = [c for c in ter.columns if c != "bg10"]
    w = pd.read_parquet(fp.PILOT / "flood_pilot_wavelet.parquet")
    t = t.merge(w, on=["ep", "bg"], how="left")
    extra["L5 + wavelet scales"] = [c for c in w.columns if c not in ("ep", "bg")]
    return t, m26.groups(extra)


def episode_pairs(ep, y, s, min_bg=10):
    """Per-episode (auc * pairs, pairs) so pooled AUCs can be bootstrapped by episode."""
    df = pd.DataFrame({"ep": ep, "y": y, "s": s})
    rows = {}
    for e, g in df.groupby("ep", sort=False):
        n1 = int(g.y.sum())
        n0 = len(g) - n1
        if len(g) < min_bg or n1 == 0 or n0 == 0:
            continue
        r = rankdata(g.s.to_numpy())
        rows[e] = ((r[g.y.to_numpy() == 1].sum() - n1 * (n1 + 1) / 2), n1 * n0)
    return rows


def paired_bootstrap(pred_file, models, n_boot=1000):
    z = np.load(pred_file)
    ep, y = z["ep"], z["y"]
    per = {m: episode_pairs(ep, y, z[m]) for m in models}
    eps = sorted(set.intersection(*[set(v) for v in per.values()]))
    num = {m: np.array([per[m][e][0] for e in eps]) for m in models}
    den = np.array([per[models[0]][e][1] for e in eps], float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(eps), (n_boot, len(eps)))
    pooled = {m: num[m][idx].sum(1) / den[idx].sum(1) for m in models}
    print(f"episodes in the comparison: {len(eps)}")
    for m in models:
        pt = num[m].sum() / den.sum()
        lo, hi = np.percentile(pooled[m], [2.5, 97.5])
        print(f"  {m:34s} within-episode AUC {pt:.4f}  95% CI ({lo:.4f}, {hi:.4f})")
    print("paired differences (later minus earlier):")
    for a, b in zip(models[:-1], models[1:], strict=True):
        d = pooled[b] - pooled[a]
        pt = num[b].sum() / den.sum() - num[a].sum() / den.sum()
        print(f"  {b:34s} - {a:30s} {pt:+.4f}  ({np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f})")
    return eps


def main() -> None:
    t, G = load()
    y = t.y.to_numpy().astype(np.float64)
    yr = t.year.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    ep = t.ep.to_numpy()
    names = list(G)

    print("== PAIRED BOOTSTRAP (test 2017-2023, episodes with >=10 block groups and both classes)")
    paired_bootstrap(fp.PILOT / "flood_pilot_test_preds.npz", names)

    print("\n== SHUFFLE: labels permuted within each episode, model L4b trained on the real labels")
    cols = G[names[-2] if len(names) > 5 else names[-1]]
    X = t[cols].to_numpy(np.float32)
    p, _ = m26.fit_predict(X, y, tr, te, va)
    rng = np.random.default_rng(0)
    ys = y[te].copy()
    e_te = ep[te]
    order = np.argsort(e_te, kind="stable")
    bounds = np.flatnonzero(np.diff(e_te[order])) + 1
    for part in np.split(order, bounds):
        ys[part] = rng.permutation(ys[part])
    print(f"  real labels:     within AUC {m26.within_auc(e_te, y[te], p)[0]:.4f}")
    print(f"  shuffled labels: within AUC {m26.within_auc(e_te, ys, p)[0]:.4f}   (should be ~0.5)")

    print("\n== LEARNING CURVE (feature set L4b, boosting; train on random year subsets of 2010-2016)")
    rng = np.random.default_rng(1)
    rows = []
    for n in (2, 4, 7):
        for rep in range(3 if n < 7 else 1):
            yrs = rng.choice(np.arange(2010, 2017), n, replace=False)
            trm = np.isin(yr, yrs)
            p, _ = m26.fit_predict(X, y, trm, te, None, 200)
            rows.append({"years": n, "rep": rep, "within_AUC": m26.within_auc(e_te, y[te], p)[0]})
    c = pd.DataFrame(rows)
    print(c.groupby("years").within_AUC.agg(["mean", "std"]).round(4).to_string())

    print("\n== LEARNER SWEEP (feature set L4b)")
    out = []
    for d in (3, 5, 8):
        mdl = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=d, min_samples_leaf=300,
                                             l2_regularization=1.0, random_state=0).fit(X[tr], y[tr])
        out.append((f"boosting depth {d}", m26.within_auc(e_te, y[te], mdl.predict_proba(X[te])[:, 1])[0]))
    sc = StandardScaler().fit(np.nan_to_num(X[tr]))
    lr = LogisticRegression(C=0.1, max_iter=300).fit(sc.transform(np.nan_to_num(X[tr])), y[tr])
    out.append(("logistic regression", m26.within_auc(e_te, y[te], lr.predict_proba(sc.transform(np.nan_to_num(X[te])))[:, 1])[0]))
    print(pd.DataFrame(out, columns=["learner", "within_AUC"]).round(4).to_string(index=False))

    print("\n== DRIFT")
    g = t.groupby("year").agg(rows=("y", "size"), pos=("y", "sum"), eps=("ep", "nunique"), pol=("n_pol", "sum"))
    g["pos_per_100k_rows"] = g.pos / g.rows * 1e5
    g["claims_per_1M_policy_episodes"] = t.groupby("year").n_claims.sum() / g.pol * 1e6
    print(g[["rows", "pos", "eps", "pos_per_100k_rows", "claims_per_1M_policy_episodes"]].round(1).T.to_string())
    print("train-to-test shift of main features (standardised mean difference):")
    sh = {}
    for c in ["bg_f1", "cty_f1_max", "log_pol", "sfha_share", "mean_cov", "elev", "rel3", "bg_a14"]:
        a, b = t.loc[tr, c], t.loc[te, c]
        sh[c] = round(float((b.mean() - a.mean()) / a.std()), 3)
    print(" ", sh)

    print("\n== STRATA (feature set L4b, test)")
    d = pd.DataFrame({"ep": e_te, "y": y[te], "s": p if False else m26.fit_predict(X, y, tr, te, va)[0]})
    nbg = t.groupby("ep").cty_nbg.first()
    dur = t.groupby("ep").dur.first()
    for name, key, bins in (("block groups in the county", nbg, [0, 20, 50, 150, 10_000]), ("episode duration (days)", dur, [0, 1, 2, 4, 100])):
        cat = pd.cut(key, bins)
        for lab in cat.cat.categories:
            sel = set(key.index[cat == lab])
            mm = d.ep.isin(sel).to_numpy()
            if mm.sum() == 0:
                continue
            w, _, n = m26.within_auc(d.ep.to_numpy()[mm], d.y.to_numpy()[mm], d.s.to_numpy()[mm])
            print(f"  {name} {str(lab):14s} within AUC {w:.3f} ({n} episodes)")


if __name__ == "__main__":
    main()
