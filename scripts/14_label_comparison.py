"""Is the label the problem? Same features and learners, different county-day label definitions.

For each label we report, on the 2017-2022 test years (train 2000-2016):
  rate, test positives   how common and how scoreable the label is
  BSS                    boosted trees on scale features / on point features / logistic regression,
                         each vs that label's own county x month climatology p0
  AUC, AUC_p0            ranking quality of the boosted model vs p0 alone (base-rate free)
  oracle                 BSS of an unforecastable feature: same-day reports of neighbours within
                         150 km (a ceiling on how regionally coherent the label is)
  frac                   BSS / oracle, i.e. how much of that ceiling the model reaches

Labels (see wxrisk.labels_alt): any report, NWS-severe, significant severe, hail >= 1 in,
wind >= 50 kt, tornado, damage >= $10k and >= $100k, plus two smoother supports:
any report within 150 km, and any report in the state.

    python scripts/14_label_comparison.py --tiers B C
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wxrisk import labels_alt, wb2  # noqa: E402
from wxrisk.scs_data import STATIC, load, tier_spec  # noqa: E402
from wxrisk.scs_eval import anomalise  # noqa: E402

_spec = importlib.util.spec_from_file_location("audit13", ROOT / "scripts" / "13_data_audit.py")
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

FLOOD_LABELS = {"f_ff", "f_any", "f_10k", "f_100k", "nfip", "g_rise", "h_decl"}
LABELS = ["a_any", "a_sev", "a_sig", "a_hail", "a_wind", "a_tor", "y10k", "y100k",
          "a_nbr150", "a_state"]


class TierCtx:
    """Features and split for one tier; labels are attached separately."""

    def __init__(self, df, a, tier):
        self.df, self.a = df, a
        self.fit = (df.year <= a.train_to).to_numpy()
        self.test = (df.year >= a.test_from).to_numpy()
        self.val = self.fit & (df.year >= a.val_from).to_numpy()
        self.tr = self.fit & ~self.val
        self.years = df.year.to_numpy()
        spec = tier_spec(tier)
        self.point_cols = spec["point"]
        self.pool_cols = [c for g in spec["groups"].values() for c in g]
        cols = sorted({*self.point_cols, *self.pool_cols, *STATIC})
        self.A = anomalise(df, cols, pd.Series(self.fit))
        self.ix = {c: i for i, c in enumerate(cols)}

    def X(self, names):
        return self.A[:, [self.ix[c] for c in [*names, *STATIC]]]


def climatology_1(df, fit, y):
    """County rate x month rate / overall rate for one 0/1 label."""
    s = pd.Series(y)
    base = s[fit].mean()
    r_c = s[fit].groupby(df.cty[fit]).mean()
    r_m = s[fit].groupby(df.month[fit]).mean()
    return np.clip(df.cty.map(r_c).to_numpy() * df.month.map(r_m).to_numpy() / base, 1e-5, 1 - 1e-5)


def neighbour_maps(df, radius=150.0):
    ctys = np.sort(df.cty.unique())
    ci = df.cty.map({k: i for i, k in enumerate(ctys)}).to_numpy()
    di = df.date.map({k: i for i, k in enumerate(np.sort(df.date.unique()))}).to_numpy()
    ll = df.drop_duplicates("cty").set_index("cty").loc[ctys, ["lat", "lon"]].to_numpy()
    la, lo = np.radians(ll[:, 0]), np.radians(ll[:, 1])
    d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la[:, None] - la) / 2) ** 2 + np.cos(la[:, None])
                                     * np.cos(la) * np.sin((lo[:, None] - lo) / 2) ** 2))
    W = (d < radius).astype(float)
    np.fill_diagonal(W, 0.0)
    state = np.array([x[:2] for x in ctys])
    S = (state[:, None] == state).astype(float)
    return ci, di, W, S, len(ctys), di.max() + 1


def add_support_labels(df):
    """a_nbr150 / a_state: any report within 150 km (incl. own county) / in the state that day."""
    ci, di, W, S, nc, nd = neighbour_maps(df)
    Y = np.zeros((nd, nc))
    Y[di, ci] = df.a_any.to_numpy()
    df["a_nbr150"] = ((Y @ (W + np.eye(nc)).T)[di, ci] > 0).astype(np.int8)
    df["a_state"] = ((Y @ (S).T)[di, ci] > 0).astype(np.int8)  # S includes the county itself
    return df


def oracle(df, c, y, p0):
    ci, di, W, _, nc, nd = neighbour_maps(df)
    Y, P = np.zeros((nd, nc)), np.zeros((nd, nc))
    Y[di, ci], P[di, ci] = y, p0
    f = np.log(((Y @ W.T)[di, ci] + 0.5) / ((P @ W.T)[di, ci] + 0.5))
    Z = np.column_stack([audit.logit(p0), f])
    lr = LogisticRegression(C=10.0, max_iter=500).fit(Z[c.fit], y[c.fit])
    return audit.bss(lr.predict_proba(Z[c.test])[:, 1], p0[c.test], y[c.test])


def hgb_with_pred(Xa, y, p0, c):
    """Like audit.hgb but also returns the test predictions (for AUC)."""
    from sklearn.ensemble import HistGradientBoostingClassifier as H

    mk = lambda n: H(max_iter=n, learning_rate=0.05, max_depth=4, min_samples_leaf=300,  # noqa: E731
                     l2_regularization=1.0, early_stopping=False, random_state=0)
    m = mk(250).fit(Xa[c.tr], y[c.tr])
    sc = [audit.brier(p[:, 1], y[c.val]) for p in m.staged_predict_proba(Xa[c.val])]
    n = int(np.argmin(sc)) + 1
    m = mk(n).fit(Xa[c.fit], y[c.fit])
    return m.predict_proba(Xa[c.test])[:, 1], n


def run(df, c, tier, labels):
    rows = []
    for label in labels:
        y = df[label].to_numpy().astype(np.float64)
        p0 = climatology_1(df, c.fit, y)
        row = {"tier": tier, "label": label, "rate_1e4": y[c.test].mean() * 1e4,
               "n_pos": int(y[c.test].sum())}
        for name, names in (("scale", c.pool_cols), ("point", c.point_cols)):
            Xa = np.column_stack([c.X(names), audit.logit(p0)])
            p, n = hgb_with_pred(Xa, y, p0, c)
            row[f"BSS_hgb_{name}"] = audit.bss(p, p0[c.test], y[c.test])
            if name == "scale":
                row["AUC"] = roc_auc_score(y[c.test], p)
                row["AUC_p0"] = roc_auc_score(y[c.test], p0[c.test])
                row["AUCPR_lift"] = average_precision_score(y[c.test], p) / max(
                    average_precision_score(y[c.test], p0[c.test]), 1e-9)
        Xa = np.column_stack([c.X(c.pool_cols), audit.logit(p0)])
        Xa = np.clip(Xa, -6, 6)
        lr = LogisticRegression(C=0.1, max_iter=500).fit(Xa[c.fit], y[c.fit])
        row["BSS_logreg_scale"] = audit.bss(lr.predict_proba(Xa[c.test])[:, 1], p0[c.test], y[c.test])
        # the smooth supports already contain the neighbours' reports, so the oracle is circular
        row["oracle150"] = np.nan if label in ("a_nbr150", "a_state") else oracle(df, c, y, p0)
        row["frac_of_oracle"] = (row["BSS_hgb_scale"] / row["oracle150"]
                                 if row["oracle150"] > 0 else np.nan)
        rows.append(row)
        print(f"  [{tier}] {label}: BSS {row['BSS_hgb_scale']:+.4f}  AUC {row['AUC']:.3f} "
              f"(p0 {row['AUC_p0']:.3f})  oracle {row['oracle150']:+.4f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2000, 2023)))
    ap.add_argument("--val-from", type=int, default=2014)
    ap.add_argument("--train-to", type=int, default=2016)
    ap.add_argument("--test-from", type=int, default=2017)
    ap.add_argument("--tiers", nargs="+", default=["B", "C"])
    ap.add_argument("--out", default="label_comparison.csv")
    ap.add_argument("--labels", nargs="+", default=LABELS)
    ap.add_argument("--gauged-only", action="store_true",
                    help="keep only counties with a USGS gauge (needed for g_rise)")
    a = ap.parse_args()
    labels = a.labels

    df = load(a.years)
    alt = labels_alt.county_day_labels(sorted(df.cty.unique()), range(2000, 2023))
    df = df.merge(alt.drop(columns=["a_any"]), on=["cty", "date"], how="left")
    df[[c for c in alt.columns if c.startswith("a_") and c != "a_any"]] = df[
        [c for c in alt.columns if c.startswith("a_") and c != "a_any"]].fillna(0)
    df["a_any"] = df["y0"]
    df = add_support_labels(df)
    if any(x.startswith("w_") for x in labels):
        # NWS storm-based warnings exist from 2002 on, so run with --years 2002 2003 ...
        w = pd.read_parquet(wb2.cache_path("warnings_county_day.parquet"))
        df = df.merge(w, on=["cty", "date"], how="left").fillna({"n_sv": 0, "n_to": 0, "n_ff": 0})
        df["w_svto"] = ((df.n_sv + df.n_to) > 0).astype(np.int8)
        df["w_to"] = (df.n_to > 0).astype(np.int8)
        df["w_ff"] = (df.n_ff > 0).astype(np.int8)
        t = df[df.year >= a.test_from]
        print("agreement, test years: P(warning|report) =",
              round(t.loc[t.a_any == 1, "w_svto"].mean(), 3), " P(report|warning) =",
              round(t.loc[t.w_svto == 1, "a_any"].mean(), 3),
              " P(tornado warning|tornado report) =", round(t.loc[t.a_tor == 1, "w_to"].mean(), 3),
              " P(tornado report|tornado warning) =", round(t.loc[t.w_to == 1, "a_tor"].mean(), 3),
              flush=True)
    flood = [x for x in labels if x in FLOOD_LABELS]
    if flood:
        # built by scripts/18_flood_sources.py; flood labels reuse the same features and split
        fl = pd.read_parquet(wb2.cache_path("flood_labels_county_day.parquet"),
                             columns=["cty", "date", "has_gauge", *flood]).drop_duplicates(["cty", "date"])
        df = df.merge(fl, on=["cty", "date"], how="left")
        df[flood] = df[flood].fillna(0).astype(np.int8)
        if a.gauged_only:
            df = df[df.has_gauge.fillna(False)].reset_index(drop=True)
    print(f"{len(df)} county-days; test rate per 1e4:",
          {k: round(df.loc[df.year >= a.test_from, k].mean() * 1e4, 1) for k in labels}, flush=True)

    out = []
    for tier in a.tiers:
        c = TierCtx(df, a, tier)
        out.append(run(df, c, tier, labels))
        pd.concat(out).to_csv(wb2.cache_path(a.out), index=False)
    res = pd.concat(out)
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
