"""Does an 'extreme weather' regime feature help between events? (stage 27, part C, Texas)

Within-event AUC cannot show it (a regime feature is constant inside an event). The question is event-level: given a county rain
episode, does any block group have a claim (all inland claims incl. named tropical), and how many.
Episode table: table_notc.parquet (no tropical exclusion). Features:
  rain       county peak 1-day rain, mean accumulation, duration, month, block groups with policies
  + TC       the episode lies in a tropical-cyclone window (800 km, +-2 d)
  + extreme  county peak rain relative to the county's own training-period distribution of episode peaks (percentile) and the 3-day sum likewise
Train episodes 2010-2016, test 2017-2023; metrics: AUC of 'any positive', Spearman of predicted vs observed positives, event-clustered bootstrap
over counties-years is not done (episodes of one storm are correlated), so differences are read against the spread over test years.
Inputs : table_notc.parquet, clean NFIP claims, county_daily.npz, tc_flag_800_2.npy
Run    : python scripts/66_regime_feature_texas.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402

spec = importlib.util.spec_from_file_location("e64", Path(__file__).with_name("64_label_scope_relaxation.py"))
e64 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e64)
D0 = pd.Timestamp("2000-01-01")


def main() -> None:
    t = pd.read_parquet(fp.PILOT / "table_notc.parquet")
    cl = e64.all_claims()
    cl = cl[cl.cause.isin(["2", "3", "4"])]
    t["n_all"] = e64.counts(t, cl)
    d = np.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    ci = {c: i for i, c in enumerate(d["cty"])}
    tc = np.load(fp.PILOT / "tc_flag_800_2.npy")
    csum = np.concatenate([np.zeros((1, tc.shape[1]), np.int32), np.cumsum(tc, axis=0, dtype=np.int32)])
    e = t.groupby("ep").agg(cty=("cty", "first"), start=("start", "first"), end=("end", "first"), year=("year", "first"), month=("month", "first"),
                            dur=("dur", "first"), peak=("cty_f1_max", "first"), msum=("cty_s_mean", "first"), nbg=("cty_nbg", "first"),
                            npos=("n_all", lambda x: int((x > 0).sum())), nclaims=("n_all", "sum")).reset_index()
    si, ei = (e.start - D0).dt.days.to_numpy(), (e.end - D0).dt.days.to_numpy()
    cj = e.cty.map(ci).to_numpy().astype(int)
    e["tc"] = ((csum[ei + 3, cj] - csum[si, cj]) > 0).astype(int)
    tr = e.year <= 2016
    for c in ("peak", "msum"):  # percentile within the county's own training episodes
        ref = e[tr].groupby("cty")[c].apply(lambda s: np.sort(s.to_numpy()))
        glob = np.sort(e.loc[tr, c].to_numpy())
        e[c + "_pct"] = [np.searchsorted(ref.get(k, glob), v) / max(len(ref.get(k, glob)), 1) for k, v in zip(e.cty, e[c], strict=True)]
    e["any"] = (e.npos > 0).astype(int)
    e["lognpos"] = np.log1p(e.npos)
    base = ["peak", "msum", "dur", "month", "nbg"]
    sets = {"rain": base, "rain + TC flag": base + ["tc"], "rain + extremeness percentile": base + ["peak_pct", "msum_pct"],
            "rain + TC + extremeness": base + ["tc", "peak_pct", "msum_pct"]}
    te = e.year >= 2017
    print(f"episodes {len(e):,}; train {tr.sum():,} test {te.sum():,}; episodes with a claim: train {e[tr]['any'].mean():.4f} test {e[te]['any'].mean():.4f}; "
          f"TC episodes {e.tc.mean():.3f}, share of positives in TC episodes {e.npos[e.tc == 1].sum() / e.npos.sum():.3f}")
    print(f"  P(any positive | TC) = {e[e.tc == 1]['any'].mean():.3f}, | not TC = {e[e.tc == 0]['any'].mean():.3f}")
    res = []
    for name, cols in sets.items():
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=50, random_state=0).fit(e.loc[tr, cols], e.loc[tr, "any"])
        reg = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=50, random_state=0).fit(e.loc[tr, cols], e.loc[tr, "lognpos"])
        p, q = clf.predict_proba(e.loc[te, cols])[:, 1], reg.predict(e.loc[te, cols])
        by = {}
        for y in sorted(e.loc[te, "year"].unique()):
            m = (e.year[te] == y).to_numpy()
            if e.loc[te, "any"][m].nunique() == 2:
                by[y] = roc_auc_score(e.loc[te, "any"][m], p[m])
        res.append({"features": name, "AUC any positive": roc_auc_score(e.loc[te, "any"], p),
                    "Spearman log positives": spearmanr(q, e.loc[te, "lognpos"])[0],
                    "AUC by test year (min-max)": f"{min(by.values()):.3f}-{max(by.values()):.3f}"})
        # TC episodes only
        m = (e.tc[te] == 1).to_numpy()
        res[-1]["AUC any positive, TC test episodes"] = roc_auc_score(e.loc[te, "any"][m], p[m]) if e.loc[te, "any"][m].nunique() == 2 else np.nan
    print(pd.DataFrame(res).round(4).to_string(index=False))
    # extremeness: share of positives by percentile bins
    e["bin"] = pd.cut(e.peak_pct, [0, .5, .8, .95, .99, 1.0001], include_lowest=True)
    print(e.groupby("bin", observed=True).agg(episodes=("any", "size"), p_any=("any", "mean"), claims=("nclaims", "sum"), tc_share=("tc", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
