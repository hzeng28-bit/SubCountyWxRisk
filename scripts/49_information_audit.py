"""Where is the information bottleneck of the flood models, and which data would lift it? (stage 17)

Three diagnostics on the existing pilot, all within-event AUC (pair weighted), paired event-clustered bootstrap:
  1. strata    - the stage 8 model (L5 test predictions) by episode length, county size, rain level, year, positives per episode
  2. misses    - what the positives that the model ranks last look like (rain at the cell, exposure, duration)
  3. oracles   - the base model (stage 8 L2 features) with one extra piece of information at a time:
       hist00   claims 2000-2009 in the block group (static flood history, known ex ante)
       prior3   claims in the 3 years before the episode (rolling history, known ex ante)
       hist     hist00 + prior3
       nn5      share of positives among the 5 nearest OTHER block groups of the same episode (not knowable ahead; what
                spatial coherence of the outcome alone could add)
     (a first version used the leave-one-out share of the whole episode; that is anti-correlated with the block group's own
      label by construction, so it was replaced)
  4. ceiling   - within-event AUC that the model's own probabilities would reach if they were the true probabilities
                (labels redrawn as Bernoulli(p), p re-centred to the test prevalence): the part of the gap that is label randomness
       terrain  elevation / relative height / slope at the centroid (null in observed-rain mode in stage 8; retested here because
                a coarse forecast leaves more of the within-county ranking to static susceptibility)
     run on the observed-rain table (train 2010-2016, test 2017-2023) and on the forecast subset of stage 16
     (fcmean rain, test 2017-2019)

Inputs : flood_pilot_table.parquet, flood_pilot_test_preds.npz, flood_pilot_fc_features.parquet, clean NFIP claims
Output : $WXRISK_DATA/audit/information_audit.txt
Run    : python scripts/49_information_audit.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e26 = _load("e26", "26_flood_pilot_experiment.py")
e48 = _load("e48", "48_flood_forecast_pilot.py")
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def history(t: pd.DataFrame) -> pd.DataFrame:
    cl = fp.claims_tx()
    bgs = fp.bg_geometry().bg10.tolist()
    idx = {b: i for i, b in enumerate(bgs)}
    c = cl[cl.date >= "2000-01-01"]
    bi = c.bg10.map(idx).to_numpy()
    day = (c.date - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    keys = np.sort(bi.astype(np.int64) * 20000 + day)
    hist00 = np.bincount(bi[day < 3653], minlength=len(bgs))
    start = (pd.to_datetime(t.start) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    b = t.bg.to_numpy().astype(np.int64) * 20000
    prior3 = (np.searchsorted(keys, b + start) - np.searchsorted(keys, b + start - 1095)).astype(np.float32)
    return pd.DataFrame({"hist00": np.log1p(hist00[t.bg.to_numpy()]), "prior3": np.log1p(prior3)}, index=t.index)


def nn5(t: pd.DataFrame) -> np.ndarray:
    from scipy.spatial import cKDTree

    out = np.zeros(len(t), np.float32)
    y = t.y.to_numpy()
    xy = np.column_stack([t.lat.to_numpy(), t.lon.to_numpy() * 0.87])  # ~cos(31 deg): degrees of longitude are shorter
    for rows in t.groupby("ep", sort=False).indices.values():
        n = len(rows)
        if n < 2:
            continue
        k = min(6, n)
        _, nb = cKDTree(xy[rows]).query(xy[rows], k=k)
        nb = nb[:, 1:]  # column 0 is the point itself
        out[rows] = y[rows][nb].mean(1)
    return out


def ceiling(d):
    say("\n== 4. label-randomness ceiling of the stage 8 model (test 2017-2023)")
    p = d.s.to_numpy().clip(1e-6, 1 - 1e-6)
    lo, hi = -20.0, 20.0
    target = d.y.mean()
    logit = np.log(p / (1 - p))
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(logit + mid)))).mean() < target else (lo, mid)
    q = 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(10):
        ys = (rng.random(len(q)) < q).astype(int)
        ids, num, den = e48.per_episode(d.ep.to_numpy(), ys, q)
        vals.append(num.sum() / den.sum())
    ids, num, den = e48.per_episode(d.ep.to_numpy(), d.y.to_numpy(), d.s.to_numpy())
    say(f"  observed within-event AUC {num.sum() / den.sum():.4f}; if the predicted probabilities were true: "
        f"{np.mean(vals):.4f} (10 redraws, sd {np.std(vals):.4f})")


def strata(t, p, te):
    say("== 1. strata of the stage 8 model (L5 test predictions), within-event AUC")
    d = t[te].copy()
    d["s"] = p
    d["nbg"] = d.groupby("ep").bg.transform("size")
    d["npos"] = d.groupby("ep").y.transform("sum")

    def cut(col, bins, labels):
        return pd.cut(d[col], bins, labels=labels, right=False)

    d["g_dur"] = cut("dur", [1, 2, 3, 5, 100], ["1", "2", "3-4", ">=5"])
    d["g_nbg"] = cut("nbg", [0, 20, 60, 150, 10_000], ["<=20", "21-60", "61-150", ">150"])
    d["g_rain"] = pd.qcut(d.cty_f1_max, 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    d["g_pos"] = cut("npos", [1, 2, 4, 10, 100_000], ["1", "2-3", "4-9", ">=10"])
    d["g_year"] = d.year
    for g in ["g_dur", "g_nbg", "g_rain", "g_pos", "g_year"]:
        rows = []
        for k, sub in d.groupby(g, observed=True):
            ids, num, den = e48.per_episode(sub.ep.to_numpy(), sub.y.to_numpy(), sub.s.to_numpy())
            if den.sum():
                rows.append((k, len(ids), num.sum() / den.sum(), int(sub.y.sum())))
        say(f"  {g[2:]:5s} " + "   ".join(f"{k}: {a:.3f} (ep {n}, pos {pos})" for k, n, a, pos in rows))
    return d


def misses(d):
    say("\n== 2. positives ranked last inside their episode")
    d = d.copy()
    d["rk"] = d.groupby("ep").s.rank(pct=True)
    pos = d[(d.y == 1) & (d.nbg >= 10)]
    low, high = pos[pos.rk <= 0.25], pos[pos.rk >= 0.75]
    say(f"  positives {len(pos):,}; in the bottom quarter of their episode {len(low) / len(pos):.1%}, top quarter {len(high) / len(pos):.1%}")
    cols = ["bg_f1", "cty_f1_max", "n_pol", "sfha_share", "dur", "mean_cov", "area_km2"]
    say("  column                bottom-quarter positives   top-quarter positives   (median)")
    for c in cols:
        say(f"  {c:20s} {low[c].median():>14.2f} {high[c].median():>24.2f}")
    r = (pos.bg_f1 < 0.25 * pos.cty_f1_max)
    say(f"  positives whose cell rain is below a quarter of the county peak: {r.mean():.1%}; with cell rain < 10 mm: {(pos.bg_f1 < 10).mean():.1%}")
    say(f"  ... among bottom-quarter positives: {(low.bg_f1 < 10).mean():.1%} have cell rain < 10 mm, {(low.n_pol < d.n_pol.median()).mean():.1%} have fewer policies than the median block group")


def fit_eval(t, tr, te, base_cols, extras, label):
    y = t.y.to_numpy().astype(np.float64)
    va = (t.year.to_numpy() >= 2014) & tr
    ep = t.ep.to_numpy()[te]
    res = {}
    for name, cols in extras.items():
        X = t[base_cols + cols].to_numpy(np.float32)
        p, n = e26.fit_predict(X, y, tr, te, va)
        res[name] = e48.per_episode(ep, y[te], p)
        say(f"    fitted {name} ({n} iterations)")
    base = res["base"]
    say(f"\n  {label}: test positives {int(y[te].sum()):,}")
    say(f"  {'information added':28s} within-event AUC   difference to base [95% interval]")
    for name, v in res.items():
        a = v[1].sum() / v[2].sum()
        if name == "base":
            say(f"  {name:28s} {a:.4f}")
        else:
            dlt, lo, hi = e48.boot_diff(base, v)
            say(f"  {name:28s} {a:.4f}            {dlt:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def main() -> None:
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t["log_area"] = np.log(t.area_km2.clip(lower=0.01))
    h = history(t)
    t = t.join(h)
    t["nn5"] = nn5(t)
    tt = pd.read_parquet(fp.PILOT / "bg_terrain.parquet")
    terr = [c for c in tt.columns if c != "bg10"]
    t = t.merge(tt, on="bg10", how="left")
    static = ["lat", "lon", "log_area", "bg_a14"]

    pr = np.load(fp.PILOT / "flood_pilot_test_preds.npz")
    yr = t.year.to_numpy()
    te = yr >= 2017
    d = strata(t, pr["L5 + wavelet scales"], te)
    misses(d)
    ceiling(d)

    say("\n== 3. information added to the base model (stage 8 L2 features)")
    base = e26.COUNTY + e26.EXPO + e26.CELL
    ex = {"base": [], "+ hist00": ["hist00"], "+ prior3": ["prior3"], "+ hist00 + prior3": ["hist00", "prior3"],
          "+ terrain": terr, "+ hist + terrain + static": ["hist00", "prior3", *terr, *static],
          "oracle: 5 nearest block groups": ["nn5"]}
    fit_eval(t, yr <= 2016, te, base, ex, "observed rain, test 2017-2023")

    f = pd.read_parquet(e48.FEATS)
    u = t[t.year <= 2019].merge(f, on=["ep", "bg"], how="inner")
    yu = u.year.to_numpy()
    fb = e48.BASE + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"]
    fit_eval(u, yu <= 2016, yu >= 2017, fb, ex, "forecast rain (fcmean), test 2017-2019")
    p = wb2.cache_path("audit") / "information_audit.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
