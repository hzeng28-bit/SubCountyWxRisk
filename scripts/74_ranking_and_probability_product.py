"""The ranking product and its probability companion, measured on the stage 8 L2 model (stage 35).

Task (Impact Forecasting.pdf): for each event, rank the units (counties; here block groups and counties of Texas) by forecast impact so that a limited
number of resources go to the highest; the probability is an auxiliary product that tells the emergency manager the level behind the ranking
("you rank this county first, yet the others cannot all be 0%").

An event here = all county episodes that start on the same day (one weather system or day of rain); units are ranked ACROSS counties inside it,
which the stage 8 metric (within county) never does.

  1 ranking, block groups   AUC across counties inside an event; top-k capture (share of the claimed block groups inside the top 1/5/10/20% of the
                            block groups of the event) for the model, for cell rain alone, for policy count alone, and at random
  2 ranking, counties       county score = expected claimed block groups (sum of p) against: any claim, number of claimed block groups; AUC, top-k capture
  3 probability companion   what the probabilities look like in an event: top-ranked vs the rest, how many are ~0, expected vs observed claimed
                            block groups per event, calibration by size of event
  4 level                   the level as predicted, recentred with the TRAILING three-year prevalence (deployable) and with the same-year prevalence (oracle)
Inputs : flood_pilot_test_preds.npz (key 'L2 + cell rain'), flood_pilot_table.parquet
Output : $WXRISK_DATA/audit/ranking_and_probability_product.txt
Run    : python scripts/74_ranking_and_probability_product.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

OUT = []
KS = (0.01, 0.05, 0.10, 0.20)


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def pair_auc(score: np.ndarray, y: np.ndarray):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), n1 * n0


def capture(score: np.ndarray, y: np.ndarray, k: float, rng: np.random.Generator) -> float:
    m = max(int(np.ceil(k * len(y))), 1)
    jitter = rng.random(len(y)) * 1e-12  # ties broken at random
    top = np.argsort(-(score + jitter))[:m]
    return y[top].sum() / y.sum()


def system_metrics(df: pd.DataFrame, cols: dict[str, str], rng) -> dict:
    """AUC and top-k capture per system for several scores, pooled with the pair / positive weights."""
    res = {n: {"auc": [], "w": [], "cap": {k: [] for k in KS}, "pos": []} for n in cols}
    for _, g in df.groupby("sysid", sort=False):
        y = g.y.to_numpy()
        if y.sum() == 0 or y.sum() == len(y) or g.cty.nunique() < 2:
            continue
        for n, c in cols.items():
            a = pair_auc(g[c].to_numpy(), y)
            if a:
                res[n]["auc"].append(a[0])
                res[n]["w"].append(a[1])
            res[n]["pos"].append(y.sum())
            for k in KS:
                res[n]["cap"][k].append(capture(g[c].to_numpy(), y, k, rng))
    return res


def main() -> None:
    d = np.load(fp.PILOT / "flood_pilot_test_preds.npz")
    p, y, ep, bg, yr = d["L2 + cell rain"], d["y"], d["ep"], d["bg"], d["year"]
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "bg", "cty", "start", "year", "y", "n_pol", "bg_f1", "cty_f1_max"])
    te = t[t.year >= 2017].reset_index(drop=True)
    assert len(te) == len(p) and (te.ep.to_numpy() == ep).all() and (te.bg.to_numpy() == bg).all(), "prediction rows do not align with the table"
    df = te.assign(p=p, sysid=te.start.dt.strftime("%Y%m%d"), log_pol=np.log1p(te.n_pol)).drop(columns=["y"]).assign(y=y)
    rng = np.random.default_rng(0)
    nsys = df.sysid.nunique()
    multi = df.groupby("sysid").cty.nunique()
    say(f"== events = county episodes starting the same day; test 2017-2023: {nsys:,} events, {int((multi >= 2).sum()):,} with >= 2 counties; "
        f"rows {len(df):,}, positives {int(df.y.sum()):,}")

    say("\n== 1. ranking block groups across counties inside an event")
    cols = {"model L2": "p", "cell rain only": "bg_f1", "policy count only": "log_pol", "county peak rain only": "cty_f1_max"}
    r = system_metrics(df, cols, rng)
    say(f"  {'score':24s} AUC across counties (pair weighted | mean of events)   share of claimed block groups in the top 1% | 5% | 10% | 20% (mean of events)")
    for n, v in r.items():
        w = np.array(v["w"])
        say(f"  {n:24s} {np.average(v['auc'], weights=w):.4f} | {np.mean(v['auc']):.4f}   "
            + " | ".join(f"{np.mean(v['cap'][k]):.3f}" for k in KS) + f"   (events {len(v['auc'])})")
    say("  random: " + " | ".join(f"{k:.3f}" for k in KS))
    # within-county reference
    wc = []
    for (_, _), g in df.groupby(["sysid", "cty"], sort=False):
        a = pair_auc(g.p.to_numpy(), g.y.to_numpy()) if len(g) >= 10 else None
        if a:
            wc.append(a)
    wc = np.array(wc)
    say(f"  reference, stage 8 metric (within one county, >= 10 block groups): {np.average(wc[:, 0], weights=wc[:, 1]):.4f}")

    say("\n== 2. ranking counties inside an event (score = expected claimed block groups, sum of p)")
    cty = df.groupby(["sysid", "cty"]).agg(score=("p", "sum"), nb=("p", "size"), npos=("y", "sum"), rain=("cty_f1_max", "first"), pol=("n_pol", "sum")).reset_index()
    cty["y"] = (cty.npos > 0).astype(int)
    cty["pmax"] = df.groupby(["sysid", "cty"]).p.max().to_numpy()
    cty["pany"] = 1 - np.exp(df.assign(l=np.log1p(-df.p.clip(upper=1 - 1e-9))).groupby(["sysid", "cty"]).l.sum().to_numpy())
    ccols = {"expected claimed BGs (sum p)": "score", "P(any claim) = 1 - prod(1 - p)": "pany", "county peak rain only": "rain", "policies in the county only": "pol"}
    res = {n: {"auc": [], "w": [], "cap": {k: [] for k in (0.05, 0.10, 0.20)}, "rho": []} for n in ccols}
    for _, g in cty.groupby("sysid", sort=False):
        if g.y.sum() == 0 or g.y.sum() == len(g) or len(g) < 5:
            continue
        for n, c in ccols.items():
            a = pair_auc(g[c].to_numpy(), g.y.to_numpy())
            res[n]["auc"].append(a[0])
            res[n]["w"].append(a[1])
            for k in res[n]["cap"]:
                m = max(int(np.ceil(k * len(g))), 1)
                top = np.argsort(-(g[c].to_numpy() + rng.random(len(g)) * 1e-12))[:m]
                res[n]["cap"][k].append(g.npos.to_numpy()[top].sum() / g.npos.sum())
            if g.npos.nunique() > 1:
                res[n]["rho"].append(spearmanr(g[c], g.npos)[0])
    say(f"  {'score':34s} county AUC (any claim; pair weighted | mean)   share of claimed block groups in the top 5% | 10% | 20% of counties   Spearman with claimed BGs")
    for n, v in res.items():
        say(f"  {n:34s} {np.average(v['auc'], weights=v['w']):.4f} | {np.mean(v['auc']):.4f}   " + " | ".join(f"{np.mean(v['cap'][k]):.3f}" for k in v["cap"])
            + f"   {np.nanmean(v['rho']):+.3f}   (events {len(v['auc'])})")

    say("\n== 3. the probability companion: what the other units look like")
    big = df.groupby("sysid").y.sum()
    top_ids = big[big >= 5].index
    rows = []
    for sid in top_ids:
        g = df[df.sysid == sid]
        m = max(int(np.ceil(0.05 * len(g))), 1)
        o = np.argsort(-g.p.to_numpy())
        top, rest = g.iloc[o[:m]], g.iloc[o[m:]]
        rows.append((len(g), g.p.max(), top.p.mean(), top.y.mean(), rest.p.mean(), rest.y.mean(), (g.p < 0.001).mean(), rest.y.sum() / max(g.y.sum(), 1),
                     g.p.sum(), g.y.sum()))
    q = pd.DataFrame(rows, columns=["n", "pmax", "top_p", "top_rate", "rest_p", "rest_rate", "share_p_lt_0.1pct", "pos_in_rest", "sum_p", "sum_y"])
    say(f"  events with >= 5 claimed block groups: {len(q)}; median over events (rows are block groups):")
    say(f"    highest predicted probability {q.pmax.median() * 100:.1f}%; top 5%: mean p {q.top_p.median() * 100:.1f}% observed {q.top_rate.median() * 100:.1f}%")
    say(f"    the other 95%: mean p {q.rest_p.median() * 100:.2f}% observed {q.rest_rate.median() * 100:.2f}%; share of units with p < 0.1%: {q['share_p_lt_0.1pct'].median():.0%}; "
        f"share of the event's claimed block groups outside the top 5%: {q.pos_in_rest.median():.0%}")
    say(f"    expected claimed block groups (sum p) against observed, per event: median ratio {np.median(q.sum_p / q.sum_y):.2f}; pooled {q.sum_p.sum() / q.sum_y.sum():.2f}; "
        f"Spearman of sum p with observed {spearmanr(q.sum_p, q.sum_y)[0]:.2f}")
    q["bin"] = pd.qcut(q.sum_y, 4, duplicates="drop")
    say("  expected / observed claimed block groups by size of event (quartiles of observed):")
    say(q.groupby("bin", observed=True).apply(lambda z: pd.Series({"events": len(z), "observed": z.sum_y.sum(), "expected": z.sum_p.sum(), "ratio": z.sum_p.sum() / z.sum_y.sum()}),
                                              include_groups=False).round(2).to_string())

    say("\n== 4. level of the probabilities")
    full = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["year", "y"])
    prev = full.groupby("year").y.mean()
    lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))

    def shift(target: float, z: np.ndarray) -> np.ndarray:
        lo, hi = -10.0, 10.0
        for _ in range(60):
            b = (lo + hi) / 2
            if (1 / (1 + np.exp(-(z + b)))).mean() > target:
                hi = b
            else:
                lo = b
        return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    out = {"as predicted": np.zeros(len(p)), "trailing 3-year prevalence": np.zeros(len(p)), "same-year prevalence (oracle)": np.zeros(len(p))}
    ref = {"train prevalence": np.zeros(len(p)), "trailing 3-year prevalence": np.zeros(len(p))}
    for a in np.unique(yr):
        m = yr == a
        out["as predicted"][m] = p[m]
        out["trailing 3-year prevalence"][m] = shift(prev.loc[a - 3:a - 1].mean(), lg[m])
        out["same-year prevalence (oracle)"][m] = shift(prev.loc[a], lg[m])
        ref["train prevalence"][m] = prev.loc[2010:2016].mean()
        ref["trailing 3-year prevalence"][m] = prev.loc[a - 3:a - 1].mean()
    bs = lambda q: np.mean((q - y) ** 2)  # noqa: E731
    say(f"  {'level':34s} Brier      skill vs a constant forecast at the train prevalence | at the trailing 3-year prevalence | at the same-year prevalence")
    same = np.zeros(len(p))
    for a in np.unique(yr):
        same[yr == a] = prev.loc[a]
    for n, q_ in out.items():
        say(f"  {n:34s} {bs(q_):.6f}   {1 - bs(q_) / bs(ref['train prevalence']):+.3f} | {1 - bs(q_) / bs(ref['trailing 3-year prevalence']):+.3f} | {1 - bs(q_) / bs(same):+.3f}")
    say("  yearly prevalence (%): " + ", ".join(f"{a} {v * 100:.2f}" for a, v in prev.items()))
    pth = wb2.cache_path("audit") / "ranking_and_probability_product.txt"
    pth.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
