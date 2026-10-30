"""How noisy is the flood label, and does a cleaner or noise-aware label help? (stage 23)

Part A measures the label (claim coverage of the rain episodes, zero-payment claims, exposure dependence, concentration in a few
episodes). Part B retrains the stage 8 L2 model on label variants and scores every variant on the ORIGINAL test label and on a strict
label (paid > 0 inland claims), so the rows are comparable (scores on each variant's own label are not):
  orig      any inland non-tropical claim in [start, end + 2 d]                      (stage 8)
  paid      only claims with a building payment > 0 (drops closed-without-payment)
  cause4    only rainfall-ponding claims (cause 4)
  win0/win5 claim window [start, end] / [start, end + 5 d]
  ge2       positive only with >= 2 claims in the block group (training only; test stays on the original label)
  big_pol   training rows restricted to block groups with >= 3 policies
  rate      Poisson boosting on claims / policies weighted by policies, scored by rate x policies (the exposure-aware target)
  cap_ev    original label, event weights 1 / sqrt(positives of the event) (a few episodes carry most positives)
Inputs : flood_pilot_table.parquet, clean NFIP claims
Output : $WXRISK_DATA/audit/label_noise_audit.txt
Run    : python scripts/60_label_noise_audit.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import bg_crosswalk  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402
from wxrisk.prep import build as prep  # noqa: E402


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


def claims() -> pd.DataFrame:
    s = (prep.scan("claims_nfip")
         .filter((pl.col("state") == "TX") & pl.col("cause").is_in(["2", "3", "4"]) & pl.col("geoid").is_not_null()
                 & ~pl.col("flood_event").fill_null("").str.contains(fp.TC_PATTERN))
         .select("id", pl.col("geoid").alias("bg"), pl.col("loss_date").alias("date"), "cause", "paid_building")
         .collect().to_pandas())
    s["date"] = pd.to_datetime(s.date)
    s = s[(s.date >= "2010-01-01") & (s.date <= "2023-12-31")]
    cw = bg_crosswalk.build().set_index("code")
    s["bg10"] = s.bg.map(cw.bg10)
    s = s[s.bg10.notna()].copy()
    s["cty"] = s.bg10.str[:5]
    return s


def counts(t: pd.DataFrame, cl: pd.DataFrame, before: int = 0, after: int = 2) -> np.ndarray:
    """Claims per (block group, episode) row for a claim subset and window [start - before, end + after]."""
    eps = t.drop_duplicates("ep")[["cty", "ep", "start", "end"]]
    m = cl.merge(eps, on="cty")
    m = m[(m.date >= m.start - pd.Timedelta(days=before)) & (m.date <= m.end + pd.Timedelta(days=after))]
    c = m.groupby(["bg10", "ep"]).size().rename("n")
    return t[["bg10", "ep"]].join(c, on=["bg10", "ep"]).n.fillna(0).to_numpy()


def main() -> None:
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    cl = claims()
    cols = e26.COUNTY + e26.EXPO + e26.CELL
    X = t[cols].to_numpy(np.float32)
    yr, ep, npol = t.year.to_numpy(), t.ep.to_numpy(), t.n_pol.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr

    say("== A. the label")
    n0 = counts(t, cl)
    say(f"  rebuilt label matches the stored one: {(((n0 > 0) == (t.y.to_numpy() > 0)).mean()):.6f}")
    say(f"  claims with paid_building missing or <= 0: {(cl.paid_building.fillna(0) <= 0).mean():.3f}  "
        f"(missing {cl.paid_building.isna().mean():.3f})")
    y0 = (n0 > 0).astype(np.float64)
    sh = pd.Series(y0).groupby(pd.cut(npol, [0, 2, 9, 99, 1e9])).agg(["size", "sum", "mean"])
    say("  positive rate by policies in the block group:\n" + sh.to_string())
    ev = pd.Series(y0).groupby(ep).sum().sort_values(ascending=False)
    say(f"  episodes with a positive: {(ev > 0).sum()} of {len(ev)}; top 1% of episodes hold {ev.head(int(len(ev) * .01)).sum() / ev.sum():.3f} "
        f"of positives, top 5% {ev.head(int(len(ev) * .05)).sum() / ev.sum():.3f}")

    strict = counts(t, cl[cl.paid_building.fillna(0) > 0]) > 0
    yo = y0
    variants: dict[str, tuple] = {}
    variants["orig"] = (y0, None)
    variants["paid"] = ((counts(t, cl[cl.paid_building.fillna(0) > 0]) > 0).astype(float), None)
    variants["cause4"] = ((counts(t, cl[cl.cause == "4"]) > 0).astype(float), None)
    variants["win0"] = ((counts(t, cl, 0, 0) > 0).astype(float), None)
    variants["win5"] = ((counts(t, cl, 0, 5) > 0).astype(float), None)
    variants["ge2"] = ((n0 >= 2).astype(float), None)
    variants["big_pol"] = (y0, npol >= 3)
    ev_pos = pd.Series(y0).groupby(ep).transform("sum").to_numpy()
    variants["cap_ev"] = (y0, ("w", 1 / np.sqrt(np.maximum(ev_pos, 1))))

    per = {}
    say("\n== B. label variants (train 2010-2016, test 2017-2023; scored on the original label | strict paid>0 label)")
    for name, (y, extra) in variants.items():
        trm = tr.copy()
        w = None
        if isinstance(extra, tuple):
            w = extra[1]
        elif extra is not None:
            trm = tr & extra
        mk = lambda n: HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300,
                                                      l2_regularization=1.0, early_stopping=False, random_state=0)
        fit_rows = trm & ~va
        m = mk(300).fit(X[fit_rows], y[fit_rows], sample_weight=None if w is None else w[fit_rows])
        sc = [np.mean((p[:, 1] - y[va]) ** 2) for p in m.staged_predict_proba(X[va])]
        n = int(np.argmin(sc)) + 1
        p = mk(n).fit(X[trm], y[trm], sample_weight=None if w is None else w[trm]).predict_proba(X[te])[:, 1]
        per[name] = (e48.per_episode(ep[te], yo[te], p), e48.per_episode(ep[te], strict[te].astype(float), p), int(y[trm].sum()))
        say(f"  {name:8s} train positives {per[name][2]:6d}  iterations {n}")

    # exposure-aware rate target
    rate = n0 / np.maximum(npol, 1)
    mkr = lambda n: HistGradientBoostingRegressor(loss="poisson", max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300,
                                                  l2_regularization=1.0, early_stopping=False, random_state=0)
    fit_rows = tr & ~va
    m = mkr(300).fit(X[fit_rows], rate[fit_rows], sample_weight=npol[fit_rows])
    sc = [np.mean(npol[va] * (q - rate[va]) ** 2) for q in m.staged_predict(X[va])]
    n = int(np.argmin(sc)) + 1
    lam = mkr(n).fit(X[tr], rate[tr], sample_weight=npol[tr]).predict(X[te])
    p = 1 - np.exp(-lam * npol[te])
    per["rate"] = (e48.per_episode(ep[te], yo[te], p), e48.per_episode(ep[te], strict[te].astype(float), p), int(y0[tr].sum()))
    say(f"  rate     (Poisson, weighted by policies) iterations {n}")

    say(f"\n  {'variant':8s}  vs original label: within-AUC  diff to orig [95%]      vs strict label: within-AUC  diff to orig [95%]")
    for name, (a, b, _) in per.items():
        wa, wb = a[1].sum() / a[2].sum(), b[1].sum() / b[2].sum()
        if name == "orig":
            say(f"  {name:8s}  {wa:.4f}                                     {wb:.4f}")
        else:
            d1 = e48.boot_diff(per["orig"][0], a)
            d2 = e48.boot_diff(per["orig"][1], b)
            say(f"  {name:8s}  {wa:.4f}       {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}]      {wb:.4f}       {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}]")
    p = wb2.cache_path("audit") / "label_noise_audit.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
