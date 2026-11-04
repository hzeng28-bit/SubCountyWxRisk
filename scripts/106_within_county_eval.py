"""Within-county ranking: does the sub-county probability tell block groups apart inside one county? (scenario report, RESULTS stage 57 addendum)

For every (event day, county) with >= 20 block groups and >= 3 claimed block groups, in events with >= 10 claimed block groups (Texas, S0): AUC of the score across the county's block groups
(pair weighted over the pairs), share of the county's claimed block groups inside the county's top 10% block groups, for the forecast-mode network `fc + expo` (2017-2019, real GEFS run),
the observed-rain network `rain + expo` (same rows) and the observed-rain CNN (2017-2023), against policies only and a random score. The county-level dispatch (send to the whole county) captures
every claimed block group of the county by construction and ranks none of its block groups, so the within-county metrics are the sub-county information on its own.
Interval: bootstrap over event days.
Inputs : script 96 builders      Output : audit/within_county_eval.txt
Run    : python scripts/106_within_county_eval.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

OUT: list[str] = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pairs(T: pd.DataFrame, cols: dict[str, str], e75) -> pd.DataFrame:
    ev = T.assign(y=T.n > 0).groupby("sysid").y.sum()
    big = set(ev[ev >= 10].index)
    rows = []
    for (sid, cty), g in T[T.sysid.isin(big)].groupby(["sysid", "cty"], sort=False):
        y = (g.n.to_numpy() > 0).astype(int)
        if len(g) < 20 or y.sum() < 3 or y.sum() == len(y):
            continue
        r = {"event": sid, "cty": cty, "nbg": len(g), "ncl": int(y.sum())}
        m = max(int(np.ceil(0.1 * len(g))), 1)
        for name, c in cols.items():
            s = g[c].to_numpy().astype(float)
            a, w = e75.pair_auc(s, y)
            r[f"auc_{name}"], r["w"] = a, w
            r[f"cap_{name}"] = y[e75.top_idx(s, m)].sum() / y.sum()
        rows.append(r)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, names: list[str], label: str, B: int = 400) -> None:
    rng = np.random.default_rng(0)
    ev = df.event.unique()
    say(f"\n== {label}: {len(df)} (event, county) pairs in {len(ev)} event days; counties {df.cty.nunique()}; median block groups {int(df.nbg.median())}, median claimed {int(df.ncl.median())}")
    g = {e: d for e, d in df.groupby("event")}

    def stat(sel):
        d = pd.concat([g[e] for e in sel])
        out = {}
        for n in names:
            out[f"auc_{n}"] = float((d[f"auc_{n}"] * d.w).sum() / d.w.sum())
            out[f"cap_{n}"] = float(d[f"cap_{n}"].mean())
        return out

    pt = stat(ev)
    bs = [stat(rng.choice(ev, len(ev))) for _ in range(B)]
    for n in names:
        lo, hi = np.percentile([b[f"auc_{n}"] for b in bs], [2.5, 97.5])
        lo2, hi2 = np.percentile([b[f"cap_{n}"] for b in bs], [2.5, 97.5])
        say(f"   {n:14s} pair-weighted AUC {pt[f'auc_{n}']:.3f} [{lo:.3f}, {hi:.3f}] | share of the county's claimed block groups in its top 10%: {pt[f'cap_{n}']:.3f} [{lo2:.3f}, {hi2:.3f}] (random 0.10) | pairs with AUC > 0.5: {np.mean(df[f'auc_{n}'] > 0.5):.0%}")
    for n in names:
        if n in ("policies only", "blind guess"):
            continue
        for ref in ("policies only",):
            da = [b[f"auc_{n}"] - b[f"auc_{ref}"] for b in bs]
            dc = [b[f"cap_{n}"] - b[f"cap_{ref}"] for b in bs]
            say(f"   {n} - {ref}: AUC {pt[f'auc_{n}'] - pt[f'auc_{ref}']:+.3f} [{np.percentile(da, 2.5):+.3f}, {np.percentile(da, 97.5):+.3f}], top-10% capture {pt[f'cap_{n}'] - pt[f'cap_{ref}']:+.3f} [{np.percentile(dc, 2.5):+.3f}, {np.percentile(dc, 97.5):+.3f}]")


def main() -> None:
    e75, s96 = _load("e75", "75_task_metrics.py"), _load("s96", "96_task_scorecard.py")
    T, _, _ = s96.build_forecast()
    cols = {"blind guess": "rand", "policies only": "n_pol", "forecast (fc+expo)": "fc", "observed (rain+expo)": "obs"}
    df = pairs(T, cols, e75)
    summarize(df, list(cols), "forecast mode 2017-2019 (real GEFS run)")
    df.to_parquet(wb2.cache_path("audit") / "within_county_pairs_fc.parquet")
    del T
    T, _, _ = s96.build_observed(_load("e85", "85_state_group_baseline.py"))
    cols = {"blind guess": "rand", "policies only": "n_pol", "boosting": "boost", "CNN (observed)": "cnn"}
    df2 = pairs(T, cols, e75)
    summarize(df2, list(cols), "observed rain 2017-2023")
    (wb2.cache_path("audit") / "within_county_eval.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
