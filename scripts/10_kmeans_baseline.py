"""K-means baseline on discretised SCS3 county-day targets, three lead-time tiers.

Tiers are nested feature sets (ERA5 stands in for a perfect forecast, so B and C are upper bounds):
  A  T-3D and beyond : calendar (sin/cos doy) + county lat/lon. No field from the target day.
  B  T-3D .. T-3h    : A + ERA5 convective environment at 12Z of the target day.
  C  T-3h and below  : B + 18Z environment + 12->18Z tendencies + 6 h precipitation windows.

Unsupervised K-means on the features, then each cluster carries empirical rates for the four
nested heads (occurrence, >=$10k, >=$100k, >=$1M). Two readouts:
  rate  p = shrunk cluster base rate
  lift  p = p0(county x month) * cluster lift
K is chosen on --val-from..--train-to (default 2014-2016, fit before it); the final model is
refit up to --train-to (default 2016) and scored from --test-from (default 2017) against p0
with year-grouped bootstrap CIs on BSS.

    python scripts/10_kmeans_baseline.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import county_env, wb2  # noqa: E402
from wxrisk.scs_eval import (  # noqa: E402
    HEADS,
    anomalise,
    brier,
    bss_ci,
    climatology,
    fit_predict,
    readouts,
)

KS = (4, 8, 16, 32, 64, 128)
SKEWED = ("cape", "cin")  # log1p before anomalising


def build_table(years: list[int]) -> pd.DataFrame:
    lab = pd.read_parquet(wb2.cache_path("labels_scs3_conm_az.parquet"))
    cty = lab.drop_duplicates("cty")[["cty", "lat", "lon"]].reset_index(drop=True)
    env = county_env.load(years, cty)
    df = lab.merge(env, on=["cty", "date"], how="inner")
    df["year"], df["month"] = df.date.dt.year, df.date.dt.month
    doy = df.date.dt.dayofyear.to_numpy()
    df["doy_s"], df["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for c in df.columns:
        if c.startswith(("cape_", "cin_", "tp_")):
            df[c] = np.log1p(df[c].clip(lower=0))
    df["state"] = df.cty.str[:2].map({"08": "CO", "35": "NM", "04": "AZ"})
    return df.dropna().reset_index(drop=True)


def tier_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    a = ["doy_s", "doy_c", "lat", "lon"]
    b = [f"{k}_12" for k in county_env.ENV]
    env18 = [f"{k}_18" for k in county_env.ENV]
    extra = ["d_cape", "d_t2m", "d_d2m", "d_pw", "tp_am", "tp_pm"]
    return {"A (T-3D+)": a, "B (T-3D..T-3h)": a + b, "C (<T-3h)": a + b + env18 + extra}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kmeans_results.csv")
    ap.add_argument("--years", type=int, nargs="+", default=list(range(2000, 2023)))
    ap.add_argument("--val-from", type=int, default=2014)  # K chosen on [val-from, train-to]
    ap.add_argument("--train-to", type=int, default=2016)
    ap.add_argument("--test-from", type=int, default=2017)
    args = ap.parse_args()
    VAL_FROM, TRAIN_TO, TEST_FROM = args.val_from, args.train_to, args.test_from
    df = build_table(args.years)
    print(f"{len(df)} county-days with features; years {df.year.min()}-{df.year.max()}", flush=True)
    tiers = tier_columns(df)
    rng = np.random.default_rng(0)
    rows, k_rows = [], []

    # ---- choose K per tier on the validation years (fit on 2000-2013 only)
    fit_v = (df.year < VAL_FROM).to_numpy()
    val = ((df.year >= VAL_FROM) & (df.year <= TRAIN_TO)).to_numpy()
    p0_v = climatology(df, pd.Series(fit_v))
    best_k = {}
    for tier, cols in tiers.items():
        Xv = anomalise(df, cols, pd.Series(fit_v))
        scores = {}
        for k in KS:
            lab = fit_predict(Xv, fit_v, k)
            sc = []
            for h in HEADS[:3]:  # y1m too sparse to steer K
                y = df[h].to_numpy().astype(float)
                _, lift = readouts(lab, fit_v, p0_v[h], k, y)
                sc.append(1 - brier(lift[val], y[val]) / brier(p0_v[h][val], y[val]))
            scores[k] = float(np.mean(sc))
            k_rows.append({"tier": tier, "K": k, "val_BSS_mean_y0_10k_100k": scores[k]})
        best_k[tier] = max(scores, key=scores.get)
        print(f"[{tier}] val BSS by K: { {k: round(v, 4) for k, v in scores.items()} } -> K={best_k[tier]}",
              flush=True)

    # ---- final: refit on 2000-2016, score 2017-2022
    fit = (df.year <= TRAIN_TO).to_numpy()
    test = (df.year >= TEST_FROM).to_numpy()
    p0 = climatology(df, pd.Series(fit))
    yrs = df.year.to_numpy()[test]
    for tier, cols in tiers.items():
        k = best_k[tier]
        X = anomalise(df, cols, pd.Series(fit))
        lab = fit_predict(X, fit, k)
        for h in HEADS:
            y = df[h].to_numpy().astype(float)
            rate, lift = readouts(lab, fit, p0[h], k, y)
            for name, p in (("rate", rate), ("lift", lift)):
                bss, lo, hi = bss_ci(p[test], p0[h][test], y[test], yrs, rng)
                rows.append({"tier": tier, "K": k, "head": h, "readout": name, "state": "ALL",
                             "n_pos_test": int(y[test].sum()), "BSS_vs_p0": bss, "ci_lo": lo,
                             "ci_hi": hi, "AUCPR": average_precision_score(y[test], p[test]),
                             "AUCPR_p0": average_precision_score(y[test], p0[h][test])})
            for st in ("CO", "NM", "AZ"):
                m = test & (df.state == st).to_numpy()
                if y[m].sum() == 0:
                    continue
                rows.append({"tier": tier, "K": k, "head": h, "readout": "lift", "state": st,
                             "n_pos_test": int(y[m].sum()),
                             "BSS_vs_p0": 1 - brier(lift[m], y[m]) / brier(p0[h][m], y[m])})
    res = pd.DataFrame(rows)
    res.to_csv(wb2.cache_path(args.out), index=False)
    pd.pivot_table(pd.DataFrame(k_rows), index="K", columns="tier").to_csv(
        wb2.cache_path("kmeans_k_sweep.csv"))
    pd.set_option("display.width", 200)
    print(res[res.state == "ALL"].round(4).to_string(index=False))
    print(res[res.state != "ALL"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
