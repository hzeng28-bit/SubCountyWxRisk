"""Scaling for the distance-based learner: K-means on the SCS county-day environment, under several normalisations.

K-means is the learner where scaling decides the answer: Euclidean distance lets the largest-variance columns dominate, and group structure
(county, season) can swamp the signal. Variants, all fitted on 2000-2016 only, same K, same test 2017-2022, lift readout (script 10):

  county x month anomaly + z   the current `scs_eval.anomalise`
  global z only                no group-aware centring
  county anomaly + z           remove the county mean only
  month anomaly + z            remove the calendar-month mean only (shared by all counties)
  county x month, no scaling   anomalies left in physical units
  county x month, robust       divide by the IQR instead of the std

Skill = BSS against the climatology p0 (county rate x month rate), year-grouped paired bootstrap against the current normalisation.

    python scripts/43_scaling_kmeans.py | tee $WXRISK_DATA/audit/scaling_kmeans.txt
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.scs_eval import STATIC, climatology, fit_predict, readouts  # noqa: E402

spec = importlib.util.spec_from_file_location("km10", Path(__file__).resolve().parent / "10_kmeans_baseline.py")
km10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(km10)

HEADS = ["y0", "y10k", "y100k"]
RNG = np.random.default_rng(0)


def normalise(df: pd.DataFrame, cols: list[str], fit: np.ndarray, centre: str, scale: str) -> np.ndarray:
    """centre: none | county | month | county_month ; scale: z | robust | none. Statistics from `fit` rows only."""
    out = np.empty((len(df), len(cols)), np.float32)
    key = {"county": df.cty, "month": df.month.astype(str), "county_month": df.cty + "_" + df.month.astype(str)}.get(centre)
    for j, c in enumerate(cols):
        x = df[c].astype(float)
        if c in STATIC or key is None:
            z = x - x[fit].mean()
        else:
            mu = x[fit].groupby(key[fit]).mean()
            z = x - key.map(mu).fillna(x[fit].mean()).to_numpy()  # a group unseen in training falls back to the global mean
        if scale == "z":
            z = z / z[fit].std()
        elif scale == "robust":
            q1, q3 = np.quantile(z[fit], [0.25, 0.75])
            z = z / (q3 - q1)
        out[:, j] = z
    return out


VARIANTS = {
    "county x month anomaly + z (current)": ("county_month", "z"),
    "global z only": ("none", "z"),
    "county anomaly + z": ("county", "z"),
    "month anomaly + z": ("month", "z"),
    "county x month anomaly, no scaling": ("county_month", "none"),
    "county x month anomaly, robust": ("county_month", "robust"),
}


def se_by_year(p: np.ndarray, y: np.ndarray, years: np.ndarray) -> np.ndarray:
    return np.array([((p - y)[years == v] ** 2).sum() for v in np.unique(years)])


def main() -> None:
    df = km10.build_table(list(range(2000, 2023)))
    fit = (df.year <= 2016).to_numpy()
    test = (df.year >= 2017).to_numpy()
    yrs = df.year.to_numpy()[test]
    tiers = km10.tier_columns(df)
    kres = pd.read_csv(wb2.cache_path("kmeans_results.csv"))
    bestk = kres.groupby("tier").K.first().to_dict()
    p0 = climatology(df, pd.Series(fit))
    out = {}
    print(f"{len(df):,} county-days; fit {fit.sum():,}, test {test.sum():,}")
    for tier in ("B (T-3D..T-3h)", "C (<T-3h)"):
        cols, k = tiers[tier], int(bestk[tier])
        print(f"\n== {tier}: K={k}, {len(cols)} columns", flush=True)
        se = {}
        for name, (centre, scale) in VARIANTS.items():
            X = normalise(df, cols, fit, centre, scale)
            lab = fit_predict(X, fit, k)
            row = {}
            for h in HEADS:
                y = df[h].to_numpy(float)
                _, lift = readouts(lab, fit, p0[h], k, y)
                se[(name, h)] = se_by_year(lift[test], y[test], yrs)
                se[("p0", h)] = se_by_year(p0[h][test], y[test], yrs)
                row[h] = round(1 - se[(name, h)].sum() / se[("p0", h)].sum(), 4)
            # spread of the column scales entering the distance (largest / smallest std)
            sd = X[fit].std(axis=0)
            row["column_std_max_over_min"] = round(float(sd.max() / sd.min()), 1)
            out[f"{tier} | {name}"] = row
        ref = "county x month anomaly + z (current)"
        for name in VARIANTS:
            for h in HEADS:
                a, b, s0 = se[(name, h)], se[(ref, h)], se[("p0", h)]
                n = len(a)
                idx = RNG.integers(0, n, (1000, n))
                d = (b[idx].sum(1) - a[idx].sum(1)) / s0[idx].sum(1)  # >0 means `name` is better than the reference
                out[f"{tier} | {name}"][f"{h}_diff_vs_current"] = [round(float((b.sum() - a.sum()) / s0.sum()), 4), *[round(float(v), 4) for v in np.percentile(d, [2.5, 97.5])]]
            print(f"  {name}: {out[f'{tier} | {name}']}", flush=True)
    path = wb2.DATA_DIR / "audit" / "scaling_kmeans.json"
    path.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
