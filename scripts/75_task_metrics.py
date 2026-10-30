"""The task metrics of docs/MODEL_AUDIT.md section 5, measured (stage 36).

Task (Impact Forecasting.pdf): inside an event, rank units by forecast impact so that a limited number of resources go to the highest;
the probability is an auxiliary product. Event = county episodes starting the same day. Units = block groups, and counties (score = expected claimed
block groups, sum of p; for the single-score baselines the county peak rain and the policies in the county).

Relevance per block group and event, from the clean claims in [start, end + 2 d]: claim count and paid building amount.
Primary, per event, resampled over events (500 bootstrap draws):
  block groups : AUC across counties (pair weighted and mean of events); share of the claimed block groups, of the claim count and of the paid
                 amount in the top 1 / 5 / 10 / 20% of the block groups; NDCG at 5% with claim count and with paid amount as gain
  counties     : AUC (any claim); share of claim count and paid amount in the top 5 / 10 / 20% of counties and, for events with >= 10 counties,
                 in the top 3 / 5 / 10; Spearman with the claim count
Auxiliary (probability): expected / observed claimed block groups per event, calibration slope, share of claims outside the top 5%, mean p of the others,
Brier skill as predicted / trailing three years / same-year (oracle).
Runs: (A) observed rain, stage 8 L2 model, scope S0, test 2017-2023; (B) forecast rain (GEFSv12), stage 16 setting WITHOUT `dur`, test 2017-2019;
(C) observed rain, scope S2 (tropical windows kept, named tropical claims included), L2 model retrained on table_notc.parquet, test 2017-2023.
Inputs : flood_pilot_test_preds.npz, flood_pilot_table.parquet, flood_pilot_fc_features.parquet, table_notc.parquet, clean NFIP claims
Output : $WXRISK_DATA/audit/task_metrics.txt
Run    : python scripts/75_task_metrics.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import rankdata, spearmanr

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
KB = (0.01, 0.05, 0.10, 0.20)
KCP = (0.05, 0.10, 0.20)
KCN = (3, 5, 10)
B = 500
RNG = np.random.default_rng(0)


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def claims(scope: str) -> pd.DataFrame:
    s = (prep.scan("claims_nfip").filter((pl.col("state") == "TX") & pl.col("cause").is_in(["2", "3", "4"]) & pl.col("geoid").is_not_null())
         .select("id", pl.col("geoid").alias("bg"), pl.col("loss_date").alias("date"), "flood_event", pl.col("paid_building").alias("paid")).collect().to_pandas())
    s["date"] = pd.to_datetime(s.date)
    s = s[(s.date >= "2010-01-01") & (s.date <= "2023-12-31")]
    cw = bg_crosswalk.build().set_index("code")
    s["bg10"] = s.bg.map(cw.bg10)
    s = s[s.bg10.notna()].copy()
    s["cty"] = s.bg10.str[:5]
    s["paid"] = s.paid.fillna(0.0).clip(lower=0)
    if scope == "S0":
        s = s[~s.flood_event.fillna("").str.contains(fp.TC_PATTERN)]
    return s


def relevance(t: pd.DataFrame, cl: pd.DataFrame):
    eps = t.drop_duplicates("ep")[["cty", "ep", "start", "end"]]
    m = cl.merge(eps, on="cty")
    m = m[(m.date >= m.start) & (m.date <= m.end + pd.Timedelta(days=2))]
    g = m.groupby(["bg10", "ep"]).agg(n=("id", "size"), paid=("paid", "sum"))
    j = t[["bg10", "ep"]].join(g, on=["bg10", "ep"])
    return j.n.fillna(0).to_numpy(), j.paid.fillna(0).to_numpy()


def pair_auc(score, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan, 0.0
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), float(n1 * n0)


def top_idx(score, m):
    return np.argsort(-(score + RNG.random(len(score)) * 1e-12))[:m]


def ndcg(score, rel, m):
    if rel.sum() <= 0:
        return np.nan
    disc = 1 / np.log2(np.arange(2, m + 2))
    top = top_idx(score, m)
    ideal = np.sort(rel)[::-1][:m]
    return float((rel[top] * disc).sum() / (ideal * disc).sum())


def event_table(df: pd.DataFrame, scores: dict[str, tuple[str, str]]) -> dict[str, pd.DataFrame]:
    """Per-event metric rows for every score. scores[name] = (block-group column, county aggregation 'sum' | 'max')."""
    cty = df.groupby(["sysid", "cty"]).agg(n=("n", "sum"), paid=("paid", "sum"), nbg=("n", "size")).reset_index()
    for name, (col, agg) in scores.items():
        cty[name] = df.groupby(["sysid", "cty"])[col].agg(agg).to_numpy()
    rows = {n: [] for n in scores}
    cty_by = {k: g for k, g in cty.groupby("sysid", sort=False)}
    for sid, g in df.groupby("sysid", sort=False):
        c = cty_by[sid]
        y = (g.n.to_numpy() > 0).astype(int)
        nbg_ok = g.cty.nunique() >= 2 and 0 < y.sum() < len(y)
        yc = (c.n.to_numpy() > 0).astype(int)
        ncty_ok = len(c) >= 5 and 0 < yc.sum() < len(c)
        if not (nbg_ok or ncty_ok):
            continue
        for name, (col, _) in scores.items():
            r = {"event": sid}
            if nbg_ok:
                s = g[col].to_numpy()
                a, w = pair_auc(s, y)
                r["bg_auc"], r["bg_w"] = a, w
                for k in KB:
                    m = max(int(np.ceil(k * len(y))), 1)
                    top = top_idx(s, m)
                    r[f"bg_cap_n_{k}"] = g.n.to_numpy()[top].sum() / g.n.sum()
                    r[f"bg_cap_p_{k}"] = g.paid.to_numpy()[top].sum() / g.paid.sum() if g.paid.sum() > 0 else np.nan
                m5 = max(int(np.ceil(0.05 * len(y))), 1)
                r["bg_ndcg_n"] = ndcg(s, g.n.to_numpy().astype(float), m5)
                r["bg_ndcg_p"] = ndcg(s, g.paid.to_numpy(), m5)
            if ncty_ok:
                s = c[name].to_numpy()
                a, w = pair_auc(s, yc)
                r["c_auc"], r["c_w"] = a, w
                for k in KCP:
                    m = max(int(np.ceil(k * len(c))), 1)
                    top = top_idx(s, m)
                    r[f"c_cap_n_{k}"] = c.n.to_numpy()[top].sum() / c.n.sum()
                    r[f"c_cap_p_{k}"] = c.paid.to_numpy()[top].sum() / c.paid.sum() if c.paid.sum() > 0 else np.nan
                if len(c) >= 10:
                    for m in KCN:
                        top = top_idx(s, m)
                        r[f"c_top{m}_n"] = c.n.to_numpy()[top].sum() / c.n.sum()
                if c.n.nunique() > 1:
                    r["c_rho"] = spearmanr(s, c.n)[0]
            rows[name].append(r)
    return {n: pd.DataFrame(v).set_index("event") for n, v in rows.items()}


def pooled(tab: pd.DataFrame, col: str, wcol: str | None, idx: np.ndarray | None = None) -> float:
    v = tab[col].to_numpy()
    w = tab[wcol].to_numpy() if wcol else np.ones(len(v))
    if idx is not None:
        v, w = v[idx], w[idx]
    ok = ~np.isnan(v)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else np.nan


def boot(tabs: dict[str, pd.DataFrame], col: str, wcol: str | None, ref: str | None = None):
    events = sorted(set().union(*[set(t.index) for t in tabs.values()]))
    n = len(events)
    aligned = {k: t.reindex(events) for k, t in tabs.items()}
    idx = RNG.integers(0, n, (B, n))
    out = {}
    for k, t in aligned.items():
        out[k] = np.array([pooled(t, col, wcol, i) for i in idx])
    point = {k: pooled(t, col, wcol) for k, t in aligned.items()}
    res = {k: (point[k], *np.nanpercentile(out[k], [2.5, 97.5])) for k in aligned}
    diff = {}
    if ref:
        for k in aligned:
            if k != ref:
                d = out[k] - out[ref]
                diff[k] = (point[k] - point[ref], *np.nanpercentile(d, [2.5, 97.5]))
    return res, diff


def report(title: str, df: pd.DataFrame, scores: dict[str, tuple[str, str]], model: str, base: str) -> dict:
    say(f"\n--- {title}")
    tabs = event_table(df, scores)
    ev_bg = int(tabs[model].bg_auc.notna().sum())
    ev_c = int(tabs[model].c_auc.notna().sum())
    ev_c10 = int(tabs[model].c_top3_n.notna().sum()) if "c_top3_n" in tabs[model] else 0
    say(f"events scored: block-group metrics {ev_bg}, county metrics {ev_c}, fixed top-k counties (>= 10 counties) {ev_c10}")
    say("  block groups across counties (point [95% interval over events]); capture of the claimed block groups / claim count / paid amount")
    hdr = f"  {'score':26s} {'AUC pair-w':>22s} {'AUC mean':>8s} | cap5% claimed BGs {'':>14s} claim count | paid | NDCG5% count | paid"
    say(hdr)
    for nme in scores:
        t = tabs[nme]
        r_auc, _ = boot({nme: t}, "bg_auc", "bg_w")
        r5, _ = boot({nme: t}, "bg_cap_n_0.05", None)
        say(f"  {nme:26s} {r_auc[nme][0]:.3f} [{r_auc[nme][1]:.3f}, {r_auc[nme][2]:.3f}] {pooled(t, 'bg_auc', None):.3f} | "
            f"top5% claim count {r5[nme][0]:.3f} [{r5[nme][1]:.3f}, {r5[nme][2]:.3f}], paid {pooled(t, 'bg_cap_p_0.05', None):.3f}; "
            f"top 1/10/20% count {pooled(t, 'bg_cap_n_0.01', None):.3f}/{pooled(t, 'bg_cap_n_0.1', None):.3f}/{pooled(t, 'bg_cap_n_0.2', None):.3f}; "
            f"NDCG count {pooled(t, 'bg_ndcg_n', None):.3f}, paid {pooled(t, 'bg_ndcg_p', None):.3f}")
    say("  counties inside an event")
    for nme in scores:
        t = tabs[nme]
        r_auc, _ = boot({nme: t}, "c_auc", "c_w")
        r10, _ = boot({nme: t}, "c_cap_n_0.1", None)
        say(f"  {nme:26s} AUC {r_auc[nme][0]:.3f} [{r_auc[nme][1]:.3f}, {r_auc[nme][2]:.3f}] | top 10% of counties claim count {r10[nme][0]:.3f} "
            f"[{r10[nme][1]:.3f}, {r10[nme][2]:.3f}], paid {pooled(t, 'c_cap_p_0.1', None):.3f}; top 5/20%: {pooled(t, 'c_cap_n_0.05', None):.3f}/{pooled(t, 'c_cap_n_0.2', None):.3f}; "
            f"top 3/5/10 counties: {pooled(t, 'c_top3_n', None):.3f}/{pooled(t, 'c_top5_n', None):.3f}/{pooled(t, 'c_top10_n', None):.3f}; Spearman {pooled(t, 'c_rho', None):+.3f}")
    say(f"  paired differences, model minus {base} (point [95%]):")
    for col, w, lab in (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "block-group top 5% claim-count capture"), ("bg_ndcg_p", None, "block-group NDCG 5% (paid)"),
                        ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top 10% claim-count capture")):
        _, d = boot({model: tabs[model], base: tabs[base]}, col, w, ref=base)
        say(f"    {lab:40s} {d[model][0]:+.3f} [{d[model][1]:+.3f}, {d[model][2]:+.3f}]")
    return tabs


def auxiliary(title: str, df: pd.DataFrame, p: np.ndarray, prev_by_year: pd.Series) -> None:
    say(f"\n--- auxiliary probability product: {title}")
    y = (df.n.to_numpy() > 0).astype(float)
    yr = df.year.to_numpy()
    d = df.assign(p=p, y=y)
    rows = []
    for _, g in d.groupby("sysid", sort=False):
        if g.y.sum() < 5:
            continue
        m = max(int(np.ceil(0.05 * len(g))), 1)
        o = np.argsort(-g.p.to_numpy())
        rest = g.iloc[o[m:]]
        rows.append((g.p.sum() / g.y.sum(), rest.p.mean(), rest.y.mean(), rest.y.sum() / g.y.sum(), (g.p < 0.001).mean()))
    q = pd.DataFrame(rows, columns=["ratio", "rest_p", "rest_y", "claims_outside", "p_lt_01"])
    lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(C=1e6, max_iter=500).fit(lg[:, None], y)
    say(f"  events with >= 5 claimed block groups: {len(q)}; expected / observed claimed block groups: median {q.ratio.median():.2f}; others (outside the top 5%): "
        f"mean p {q.rest_p.median() * 100:.2f}% observed {q.rest_y.median() * 100:.2f}%, share of claimed block groups outside the top 5% {q.claims_outside.median():.0%}, "
        f"units with p < 0.1% {q.p_lt_01.median():.0%}")
    say(f"  calibration slope {cal.coef_[0][0]:.2f}, intercept {cal.intercept_[0]:+.2f}; mean prediction {p.mean() * 100:.3f}% against observed {y.mean() * 100:.3f}%")

    def shift(target, z):
        lo, hi = -12.0, 12.0
        for _ in range(60):
            b = (lo + hi) / 2
            lo, hi = (lo, b) if (1 / (1 + np.exp(-(z + b)))).mean() > target else (b, hi)
        return 1 / (1 + np.exp(-(z + (lo + hi) / 2)))
    trail, same, ref_tr = np.zeros(len(p)), np.zeros(len(p)), np.zeros(len(p))
    for a in np.unique(yr):
        m = yr == a
        trail[m] = shift(prev_by_year.loc[a - 3:a - 1].mean(), lg[m])
        same[m] = shift(y[m].mean(), lg[m])
        ref_tr[m] = prev_by_year.loc[a - 3:a - 1].mean()
    bs = lambda x: np.mean((x - y) ** 2)  # noqa: E731
    say(f"  Brier skill against a constant forecast at the trailing 3-year prevalence: as predicted {1 - bs(p) / bs(ref_tr):+.3f}; re-centred on the trailing 3 years "
        f"{1 - bs(trail) / bs(ref_tr):+.3f}; re-centred on the same year (oracle) {1 - bs(same) / bs(ref_tr):+.3f}")


def to_df(t: pd.DataFrame, n: np.ndarray, paid: np.ndarray, p: np.ndarray | None = None) -> pd.DataFrame:
    d = t.assign(n=n, paid=paid, sysid=t.start.dt.strftime("%Y%m%d"), log_pol=np.log1p(t.n_pol))
    if p is not None:
        d["p"] = p
    return d


def main() -> None:
    cl0, cl2 = claims("S0"), claims("S2")
    scores_obs = {"model": ("p", "sum"), "cell rain only": ("bg_f1", "max"), "county peak rain only": ("cty_f1_max", "max"), "policies only": ("n_pol", "sum")}
    cols_t = ["ep", "bg", "bg10", "cty", "start", "end", "year", "y", "n_pol", "bg_f1", "cty_f1_max"]

    say("== A. observed rain, stage 8 L2 model, scope S0 (tropical windows excluded), test 2017-2023")
    d = np.load(fp.PILOT / "flood_pilot_test_preds.npz")
    full = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=cols_t)
    te = full[full.year >= 2017].reset_index(drop=True)
    assert (te.ep.to_numpy() == d["ep"]).all() and (te.bg.to_numpy() == d["bg"]).all()
    n, paid = relevance(full, cl0)
    A = to_df(te, *[x[(full.year >= 2017).to_numpy()] for x in (n, paid)], d["L2 + cell rain"])
    say(f"claim count and paid amount per block group: claimed block groups {int((A.n > 0).sum()):,}, claims {int(A.n.sum()):,}, paid ${A.paid.sum() / 1e6:.1f} M in the test years")
    report("A: block groups and counties", A, scores_obs, "model", "policies only")
    auxiliary("A", A, A.p.to_numpy(), full.groupby("year").y.mean())

    say("\n  model by test year (block-group AUC pair weighted | top-5% claim-count capture | county AUC):")
    tabs = event_table(A, {"model": ("p", "sum")})["model"]
    tabs["year"] = [int(str(i)[:4]) for i in tabs.index]
    for yv, g in tabs.groupby("year"):
        say(f"    {yv}: {pooled(g, 'bg_auc', 'bg_w'):.3f} | {pooled(g, 'bg_cap_n_0.05', None):.3f} | {pooled(g, 'c_auc', 'c_w'):.3f}  ({int(g.bg_auc.notna().sum())} / {int(g.c_auc.notna().sum())} events)")

    say("\n== B. forecast rain (GEFSv12, stage 16 setting) WITHOUT `dur`, scope S0, test 2017-2019")
    t16 = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t16 = t16[t16.year <= e48.LAST_YEAR].merge(pd.read_parquet(e48.FEATS), on=["ep", "bg"], how="inner")
    t16["log_pol"] = np.log1p(t16.n_pol)
    n16, p16 = relevance(t16, cl0)
    yv = t16.y.to_numpy().astype(float)
    yr16 = t16.year.to_numpy()
    tr, te16 = yr16 <= 2016, yr16 >= 2017
    va = (yr16 >= 2014) & tr
    base_nodur = [c for c in e48.BASE if c != "dur"]
    preds = {}
    for name, cols in {"forecast model": base_nodur + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"],
                       "observed-rain model (same days, reference)": base_nodur + ["oc_f1", "oc_s", "oc_cty_max", "oc_cty_mean"]}.items():
        pr, it = e26.fit_predict(t16[cols].to_numpy(np.float32), yv, tr, te16, va)
        preds[name] = pr
        say(f"  fitted {name} ({it} iterations)")
    Bd = to_df(t16[te16].reset_index(drop=True), n16[te16], p16[te16])
    Bd["pf"], Bd["po"] = preds["forecast model"], preds["observed-rain model (same days, reference)"]
    sc_b = {"forecast model": ("pf", "sum"), "observed-rain model (reference)": ("po", "sum"), "forecast cell rain only": ("fc_f1", "max"),
            "forecast county peak only": ("fc_cty_max", "max"), "policies only": ("n_pol", "sum")}
    report("B: block groups and counties", Bd, sc_b, "forecast model", "policies only")
    auxiliary("B (forecast model)", Bd, Bd.pf.to_numpy(), full.groupby("year").y.mean())

    say("\n== C. observed rain, scope S2 (tropical windows kept, named tropical claims included), L2 model retrained, test 2017-2023")
    tn = pd.read_parquet(fp.PILOT / "table_notc.parquet")
    tn["log_pol"] = np.log1p(tn.n_pol)
    n2, p2 = relevance(tn, cl2)
    y2 = (n2 > 0).astype(float)
    yr2 = tn.year.to_numpy()
    tr2, te2 = yr2 <= 2016, yr2 >= 2017
    va2 = (yr2 >= 2014) & tr2
    cols = e26.COUNTY + e26.EXPO + e26.CELL
    pr2, it2 = e26.fit_predict(tn[cols].to_numpy(np.float32), y2, tr2, te2, va2)
    say(f"  fitted S2 model ({it2} iterations); positives train {int(y2[tr2].sum())}, test {int(y2[te2].sum())}")
    C = to_df(tn[te2].reset_index(drop=True), n2[te2], p2[te2], pr2)
    report("C: block groups and counties", C, scores_obs, "model", "policies only")
    prev2 = pd.Series(y2).groupby(yr2).mean()
    auxiliary("C", C, C.p.to_numpy(), prev2)
    pth = wb2.cache_path("audit") / "task_metrics.txt"
    pth.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
