"""Audit: is the weak lift a data/normalisation problem, a label ceiling, or a learner problem?

Sections (run all, or pick with --sections):
  features  distributions, scale spread, tails, train->test drift, collinearity (tier C columns)
  labels    report-rate drift by year, zone-vs-county records, damage blanks, event hour vs frame time
  shuffle   pipeline leak check: shuffle features within county x month, score should be ~0
  curve     learning curve: BSS vs number of training years (is it data-limited?)
  sweep     learner capacity sweep with train-vs-test BSS (is it learner-limited or noise-limited?)
  oracle    ceilings: how much does knowing the same-day neighbour reports buy? (not forecastable)
  norm      normalisation variants (county-month anomaly, raw z, quantile-normal, clipped)

All on tier B (12Z of the target day), scale-feature and point-feature sets, scored as BSS vs p0
on 2017-2022 with train 2000-2016. Nothing here is tuned on the test years.

    python scripts/13_data_audit.py | tee ~/wxrisk_data/audit/report.txt
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.scs_data import HEADS, STATIC, load, tier_spec  # noqa: E402
from wxrisk.scs_eval import (  # noqa: E402
    anomalise,
    brier,
    climatology,
    cluster_stats,
    fit_predict,
)

K = 32
STATES = {"08": "CO", "35": "NM", "04": "AZ"}


def bss(p, p0, y):
    return 1 - brier(p, y) / brier(p0, y)


def logit(p):
    return np.log(p / (1 - p))


def title(s):
    print(f"\n{'=' * 8} {s} {'=' * 8}", flush=True)


class Ctx:
    """Everything the sections share, built once for tier B."""

    def __init__(self, df, a):
        self.df, self.a = df, a
        self.fit = (df.year <= a.train_to).to_numpy()
        self.test = (df.year >= a.test_from).to_numpy()
        self.val = self.fit & (df.year >= a.val_from).to_numpy()
        self.tr = self.fit & ~self.val  # iteration counts are chosen with train=tr, val=val
        self.years = df.year.to_numpy()
        self.p0 = climatology(df, pd.Series(self.fit))
        self.y = {h: df[h].to_numpy().astype(np.float64) for h in HEADS}
        spec = tier_spec("B")
        self.point_cols = spec["point"]
        self.pool_cols = [c for g in spec["groups"].values() for c in g]
        self.cols = sorted({*self.point_cols, *self.pool_cols, *STATIC})
        self.A = anomalise(df, self.cols, pd.Series(self.fit))
        self.ix = {c: i for i, c in enumerate(self.cols)}

    def X(self, names, A=None):
        A = self.A if A is None else A
        return A[:, [self.ix[c] for c in [*names, *STATIC]]]


def hgb(Xa, y, p0, c, depth=4, max_iter=250, leaf=300, lr=0.05, tr=None, va=None):
    """Fit on tr, choose iterations on va, return best-iteration BSS on test and (sub)train."""
    tr = c.tr if tr is None else tr
    va = c.val if va is None else va
    m = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=lr, max_depth=depth, min_samples_leaf=leaf,
        l2_regularization=1.0, early_stopping=False, random_state=0).fit(Xa[tr], y[tr])
    sub = np.flatnonzero(tr)[::5]
    best, out = np.inf, None
    for pv, pt, ps in zip(m.staged_predict_proba(Xa[va]), m.staged_predict_proba(Xa[c.test]),
                          m.staged_predict_proba(Xa[sub]), strict=True):
        b = brier(pv[:, 1], y[va])
        if b < best:
            best, out = b, (pt[:, 1], ps[:, 1])
    n = int(np.argmin([brier(p[:, 1], y[va]) for p in m.staged_predict_proba(Xa[va])])) + 1
    return {"iters": n, "BSS_test": bss(out[0], p0[c.test], y[c.test]),
            "BSS_train": bss(out[1], p0[sub], y[sub])}


def section_features(c):
    title("FEATURES: tier C columns, statistics on the 2000-2016 rows")
    spec = tier_spec("C")
    cols = sorted({x for g in [spec["point"], *spec["groups"].values()] for x in g})
    raw = c.df.loc[c.fit, cols].astype(np.float64)
    A = anomalise(c.df, cols, pd.Series(c.fit))
    an = pd.DataFrame(A[c.fit], columns=cols)
    print(f"columns {len(cols)}, rows {c.fit.sum()}, NaN {int(raw.isna().sum().sum())}, "
          f"constant {(raw.std() == 0).sum()}")
    sd = raw.std()
    print(f"raw std spans {sd.min():.3g} .. {sd.max():.3g}  (ratio {sd.max() / sd.min():.2g}) "
          "-> unnormalised Euclidean K-means would be dominated by the largest-variance columns")
    fam = lambda s: ("mv" if s.startswith("mv_") else "lp" if s.endswith("_lp") else  # noqa: E731
                     "band" if "_b" in s else "amp" if "_e" in s else "point")
    t = pd.DataFrame({"fam": [fam(x) for x in cols], "raw_skew": raw.skew().to_numpy(),
                      "raw_kurt": raw.kurt().to_numpy(), "an_skew": an.skew().to_numpy(),
                      "an_kurt": an.kurt().to_numpy(), "an_max": an.abs().max().to_numpy()},
                     index=cols)
    print(t.groupby("fam").agg(n=("raw_skew", "size"), raw_skew_med=("raw_skew", "median"),
                               an_skew_med=("an_skew", "median"), raw_kurt_med=("raw_kurt", "median"),
                               an_kurt_med=("an_kurt", "median"), an_max_med=("an_max", "median")
                               ).round(2).to_string())
    print(f"|skew|>2: raw {(t.raw_skew.abs() > 2).sum()}, anomalised {(t.an_skew.abs() > 2).sum()}; "
          f"kurtosis>20: raw {(t.raw_kurt > 20).sum()}, anomalised {(t.an_kurt > 20).sum()}")
    print("worst anomalised tails:\n", t.sort_values("an_kurt", ascending=False).head(8)[
        ["an_skew", "an_kurt", "an_max"]].round(1).to_string())
    te = pd.DataFrame(A[c.test], columns=cols)
    drift = ((te.mean() - an.mean()) / an.std()).abs()
    sdr = (te.std() / an.std() - 1).abs()
    print(f"train->test drift in anomalised mean (std units): median {drift.median():.3f}, "
          f"max {drift.max():.2f} ({drift.idxmax()}); std change median {sdr.median():.3f}, "
          f"max {sdr.max():.2f} ({sdr.idxmax()})")
    samp = an.sample(40_000, random_state=0).corr().abs().to_numpy()
    iu = np.triu_indices_from(samp, 1)
    print(f"collinearity among anomalised columns: pairs |r|>0.95: {(samp[iu] > 0.95).sum()}, "
          f">0.99: {(samp[iu] > 0.99).sum()} of {len(iu[0])}")


def section_labels(c):
    title("LABELS")
    df = c.df
    r = df.groupby("year")[["y0", "y10k", "y100k"]].mean() * 1e4
    r["n_y0"] = df.groupby("year").y0.sum()
    print("rate per 1e4 county-days by year:\n", r.round(1).T.to_string())
    for h in ("y0", "y10k"):
        s = df.groupby("year")[h].mean()
        print(f"{h}: first-half/second-half rate ratio (2000-2010 vs 2012-2022) "
              f"{s.loc[:2010].mean() / s.loc[2012:].mean():.2f}")
    print("rate per 1e4 by state:\n", (df.groupby("state")[["y0", "y10k", "y100k"]].mean() * 1e4)
          .round(1).to_string())

    files = sorted(glob.glob(str(wb2.cache_path("storm_events")) + "/StormEvents_details*.csv.gz"))
    cols = ["BEGIN_YEARMONTH", "BEGIN_DAY", "BEGIN_TIME", "STATE_FIPS", "CZ_TYPE", "CZ_FIPS",
            "EVENT_TYPE", "DAMAGE_PROPERTY", "CZ_TIMEZONE"]
    ev = pd.concat([pd.read_csv(f, usecols=cols, dtype=str, low_memory=False) for f in files])
    ev = ev[ev.EVENT_TYPE.isin(["Hail", "Thunderstorm Wind", "Tornado"])
            & ev.STATE_FIPS.str.zfill(2).isin(STATES) & ev.BEGIN_YEARMONTH.str[4:6].isin(
                ["05", "06", "07", "08", "09"]) & (ev.BEGIN_YEARMONTH.str[:4].astype(int) <= 2022)]
    print(f"\nSCS3 records, May-Sep 2000-2022, CO/NM/AZ: {len(ev)}")
    print("record type:", ev.CZ_TYPE.value_counts(normalize=True).round(3).to_dict(),
          "<- Z (zone) records carry no county label and are not in y0")
    print("zone share by state:", ev.assign(st=ev.STATE_FIPS.str.zfill(2).map(STATES))
          .groupby("st").CZ_TYPE.apply(lambda s: round((s == "Z").mean(), 3)).to_dict())
    blank = ev.DAMAGE_PROPERTY.isna() | (ev.DAMAGE_PROPERTY.str.strip() == "")
    print(f"records with blank property damage: {blank.mean():.3f}; by type:",
          ev.assign(b=blank).groupby("EVENT_TYPE").b.mean().round(3).to_dict())
    ev["yr"] = ev.BEGIN_YEARMONTH.str[:4].astype(int)
    print("share of records with blank damage by period:",
          ev.assign(b=blank).groupby(pd.cut(ev.yr, [1999, 2005, 2011, 2017, 2022])).b.mean()
          .round(3).to_dict())

    ev = ev[ev.CZ_TYPE == "C"].copy()
    off = ev.CZ_TIMEZONE.str.extract(r"(-?\d+)")[0].astype(float)
    ev["utc"] = ev.BEGIN_TIME.astype(float) // 100 + (-off).fillna(7)
    ev["cty"] = ev.STATE_FIPS.str.zfill(2) + ev.CZ_FIPS.str.zfill(3)
    ev["date"] = pd.to_datetime(ev.BEGIN_YEARMONTH + ev.BEGIN_DAY.str.zfill(2), format="%Y%m%d")
    first = ev.groupby(["cty", "date"]).utc.min()
    print(f"\nearliest report per positive county-day, hour UTC (n={len(first)}):")
    print("  quantiles 5/25/50/75/95%:", first.quantile([.05, .25, .5, .75, .95]).round(1).tolist())
    print(f"  before 12Z (tier B frame is after the event): {(first < 12).mean():.3f}")
    print(f"  before 18Z (tier C frame is after the event): {(first < 18).mean():.3f}")
    print(f"  18Z-24Z: {((first >= 18) & (first < 24)).mean():.3f}, next UTC day (>=24Z): "
          f"{(first >= 24).mean():.3f}")


def section_shuffle(c):
    title("SHUFFLE: features shuffled within county x month -> BSS should be ~0")
    rng = np.random.default_rng(0)
    key = (c.df.cty + "_" + c.df.month.astype(str)).to_numpy()
    perm = np.arange(len(key))
    for k in pd.unique(key):
        idx = np.flatnonzero(key == k)
        perm[idx] = rng.permutation(idx)
    for name, names in (("scale", c.pool_cols), ("point", c.point_cols)):
        X = c.X(names)
        for lab, Xs in (("real", X), ("shuffled", X[perm])):
            Xa = np.column_stack([Xs, logit(c.p0["y0"])])
            r = hgb(Xa, c.y["y0"], c.p0["y0"], c)
            print(f"  {name:6s} {lab:9s} y0: test BSS {r['BSS_test']:+.4f}  train BSS "
                  f"{r['BSS_train']:+.4f}  ({r['iters']} iters)", flush=True)


def section_curve(c):
    title("LEARNING CURVE: train on a random subset of 2000-2013 years (val 2014-2016)")
    rng = np.random.default_rng(1)
    pool = np.arange(2000, 2014)
    rows = []
    for n in (2, 4, 7, 14):
        for rep in range(3 if n < 14 else 1):
            yrs = rng.choice(pool, n, replace=False)
            tr = np.isin(c.years, yrs)
            for name, names in (("scale", c.pool_cols), ("point", c.point_cols)):
                Xa = np.column_stack([c.X(names), logit(c.p0["y0"])])
                r = hgb(Xa, c.y["y0"], c.p0["y0"], c, tr=tr)
                rows.append({"n_years": n, "rep": rep, "feat": name, **r})
    t = pd.DataFrame(rows)
    print(t.groupby(["feat", "n_years"])[["BSS_test", "BSS_train"]].agg(["mean", "std"]).round(4)
          .to_string())


def section_sweep(c):
    title("LEARNER SWEEP (y0, tier B): capacity vs train/test BSS")
    y, p0 = c.y["y0"], c.p0["y0"]
    rows = []
    for name, names in (("scale", c.pool_cols), ("point", c.point_cols)):
        X = np.clip(c.X(names), -5, 5)
        Xa = np.column_stack([X, logit(p0)])
        for d in (2, 3, 4, 6, 8):
            rows.append({"feat": name, "learner": f"HGB depth {d}",
                         **hgb(Xa, y, p0, c, depth=d, max_iter=400)})
        for C in (0.01, 0.1, 1.0):
            lr = LogisticRegression(C=C, max_iter=500).fit(Xa[c.fit], y[c.fit])
            sub = np.flatnonzero(c.fit)[::5]
            rows.append({"feat": name, "learner": f"LogReg C={C}", "iters": 0,
                         "BSS_test": bss(lr.predict_proba(Xa[c.test])[:, 1], p0[c.test], y[c.test]),
                         "BSS_train": bss(lr.predict_proba(Xa[sub])[:, 1], p0[sub], y[sub])})
        mlp = MLPClassifier((128, 64), alpha=1e-3, batch_size=512, early_stopping=True,
                            n_iter_no_change=5, max_iter=60, random_state=0).fit(Xa[c.fit], y[c.fit])
        sub = np.flatnonzero(c.fit)[::5]
        rows.append({"feat": name, "learner": f"MLP 128-64 ({mlp.n_iter_} ep)", "iters": mlp.n_iter_,
                     "BSS_test": bss(mlp.predict_proba(Xa[c.test])[:, 1], p0[c.test], y[c.test]),
                     "BSS_train": bss(mlp.predict_proba(Xa[sub])[:, 1], p0[sub], y[sub])})
        print(f"  [{name}] done", flush=True)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


def section_oracle(c):
    title("ORACLE: same-day neighbour reports as a feature (a ceiling on regional coherence)")
    df = c.df
    ctys = np.sort(df.cty.unique())
    ci = df.cty.map({k: i for i, k in enumerate(ctys)}).to_numpy()
    di = df.date.map({k: i for i, k in enumerate(np.sort(df.date.unique()))}).to_numpy()
    ll = df.drop_duplicates("cty").set_index("cty").loc[ctys, ["lat", "lon"]].to_numpy()
    la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la[:, None])
                                     * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2))
    state = np.array([x[:2] for x in ctys])
    W = {"same state": (state[:, None] == state).astype(float), "all box": np.ones_like(d),
         "<150 km": (d < 150).astype(float), "<300 km": (d < 300).astype(float)}
    for w in W.values():
        np.fill_diagonal(w, 0.0)
    nd, nc = di.max() + 1, len(ctys)
    rows = []
    for h in ("y0", "y10k"):
        Y = np.zeros((nd, nc))
        P = np.zeros((nd, nc))
        Y[di, ci], P[di, ci] = c.y[h], c.p0[h]
        base = {"feature set": "p0 only", "head": h,
                "BSS_test": 0.0}
        rows.append(base)
        for name, w in W.items():
            O, E = Y @ w.T, P @ w.T
            f = np.log((O[di, ci] + 0.5) / (E[di, ci] + 0.5))
            Z = np.column_stack([logit(c.p0[h]), f])
            lr = LogisticRegression(C=10.0, max_iter=500).fit(Z[c.fit], c.y[h][c.fit])
            p = lr.predict_proba(Z[c.test])[:, 1]
            rows.append({"feature set": f"neighbours {name}", "head": h,
                         "BSS_test": bss(p, c.p0[h][c.test], c.y[h][c.test])})
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("(reference: HGB-scale tier B reached y0 0.068, y10k 0.043; tier C 0.108 / 0.042)")


def section_norm(c):
    title("NORMALISATION variants (y0 / y10k BSS vs p0, tier B)")
    df = c.df
    g = (df[c.cols].astype(np.float64)).copy()
    mu, sd = g[c.fit].mean(), g[c.fit].std()
    raw_z = ((g - mu) / sd).to_numpy(np.float32)
    qt = QuantileTransformer(n_quantiles=500, subsample=200_000, output_distribution="normal",
                             random_state=0).fit(c.A[c.fit])
    variants = {"county-month anomaly z (current)": c.A, "global z, no anomaly": raw_z,
                "anomaly + quantile->normal": qt.transform(c.A).astype(np.float32),
                "anomaly z clipped +-3": np.clip(c.A, -3, 3),
                "raw values, no scaling at all": df[c.cols].to_numpy(np.float32)}
    rows = []
    for vname, M in variants.items():
        for fname, names in (("point", c.point_cols), ("scale", c.pool_cols)):
            X = c.X(names, M)
            lab = fit_predict(X, c.fit, K, n_init=2, max_fit=100_000)
            for h in ("y0", "y10k"):
                _, lift = cluster_stats(lab, c.fit, c.p0[h], K, c.y[h])
                p = np.clip(c.p0[h] * lift[lab], 0, 1)
                rows.append({"variant": vname, "feat": fname, "learner": "K-means", "head": h,
                             "BSS_test": bss(p[c.test], c.p0[h][c.test], c.y[h][c.test])})
        if vname in ("county-month anomaly z (current)", "global z, no anomaly"):
            for fname, names in (("point", c.point_cols), ("scale", c.pool_cols)):
                X = c.X(names, M)
                for h in ("y0", "y10k"):
                    Xa = np.column_stack([X, logit(c.p0[h])])
                    r = hgb(Xa, c.y[h], c.p0[h], c)
                    rows.append({"variant": vname, "feat": fname, "learner": "HGB", "head": h,
                                 "BSS_test": r["BSS_test"]})
        print(f"  [{vname}] done", flush=True)
    t = pd.DataFrame(rows)
    print(t.pivot_table(index=["learner", "variant"], columns=["feat", "head"], values="BSS_test")
          .round(4).to_string())


SECTIONS = {"features": section_features, "labels": section_labels, "shuffle": section_shuffle,
            "curve": section_curve, "sweep": section_sweep, "oracle": section_oracle,
            "norm": section_norm}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2000, 2023)))
    ap.add_argument("--val-from", type=int, default=2014)
    ap.add_argument("--train-to", type=int, default=2016)
    ap.add_argument("--test-from", type=int, default=2017)
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS))
    a = ap.parse_args()
    df = load(a.years)
    print(f"{len(df)} county-days, years {df.year.min()}-{df.year.max()}, "
          f"train<= {a.train_to}, test>= {a.test_from}", flush=True)
    c = Ctx(df, a)
    for s in a.sections:
        SECTIONS[s](c)


if __name__ == "__main__":
    main()
