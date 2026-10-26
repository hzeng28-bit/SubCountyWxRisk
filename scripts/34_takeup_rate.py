"""NFIP take-up rate by Texas block group: is voluntary insurance in non-SFHA areas a mixture?

CANDIDATE-pool analysis (see docs/FLOOD_PRIORS.md, "Candidate features"). Hypothesis from the user: in
non-SFHA areas the people who insure are two kinds -- owners with professional property management, and
people frightened by hearing about NFIP -- while most owners skip it ("100-year flood" misconception).
That would make block-group take-up bimodal. This script only checks the observable part: the
distribution of take-up across block groups. A bimodal block-group distribution does not prove owner types.

  take-up = single-family policies in force on the snapshot day, rated zone X/B/C/D
            / ACS 5-year housing units (B25001), 2020 block groups

Restricted to block groups with < 5% of policies in the SFHA, so X-zone policies are most of the housing stock
(ACS has no flood-zone split of housing units; elsewhere the ratio understates take-up).
Policies carry 2020 block groups from 2022 on (see AGENTS.md trap 5), so the default snapshot is 2022-12-31.

    python scripts/34_takeup_rate.py | tee $WXRISK_DATA/audit/takeup_rate.txt
    python scripts/34_takeup_rate.py --snapshot 2023-06-30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch, wb2  # noqa: E402
from wxrisk.prep import build as prep  # noqa: E402

FLOOD = wb2.cache_path("labels_extra") / "flood"
ACS = wb2.cache_path("labels_extra") / "census" / "acs5_2022_b25001.dat"
ACS_URL = "https://www2.census.gov/programs-surveys/acs/summary_file/2022/table-based-SF/data/5YRData/acsdt5y2022-b25001.dat"


def housing_units() -> pd.Series:
    fetch.https(ACS_URL, ACS)
    a = pd.read_csv(ACS, sep="|", dtype={"GEO_ID": str})
    a = a[a.GEO_ID.str.startswith("1500000US")]
    return pd.Series(a.B25001_E001.values, index=a.GEO_ID.str[-12:])


def snapshot_policies(day: str) -> pd.DataFrame:
    d = pd.Timestamp(day).date()
    return (prep.scan("policies_tx")
            .filter(pl.col("occupancy").is_in([1, 11]) & pl.col("geoid").is_not_null() & (pl.col("eff") <= d) & (pl.col("coverage_end").is_null() | (pl.col("coverage_end") > d)))
            .select("geoid", pl.col("n_policies").alias("policyCount"), pl.col("mandatory").alias("mandatoryPurchaseFlag"),
                    pl.col("zone").str.slice(0, 1).is_in(["A", "V"]).fill_null(False).alias("sfha"))
            .collect().to_pandas())


def gmm_bic(x: np.ndarray) -> dict[int, float]:
    from sklearn.mixture import GaussianMixture

    return {k: float(GaussianMixture(k, n_init=5, random_state=0).fit(x[:, None]).bic(x[:, None])) for k in (1, 2, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="2022-12-31")
    a = ap.parse_args()
    hu = housing_units()
    p = snapshot_policies(a.snapshot)
    w = p.policyCount.astype(float)
    g = pd.DataFrame({"n_x": (w * ~p.sfha).groupby(p.geoid).sum(), "n_sfha": (w * p.sfha).groupby(p.geoid).sum(),
                      "n_mand": (w * p.mandatoryPurchaseFlag.astype(bool)).groupby(p.geoid).sum()})
    g["hu"] = hu.reindex(g.index)
    print(f"snapshot {a.snapshot}: {len(p):,} single-family policy rows, {len(g):,} block groups, "
          f"{int(g.hu.isna().sum())} without ACS housing units")
    g = g.dropna(subset=["hu"])
    g = g[(g.hu >= 200) & (g.n_sfha / (g.n_x + g.n_sfha).clip(lower=1) < 0.05)].copy()
    g["takeup"] = g.n_x / g.hu
    g["mand_share"] = g.n_mand / (g.n_x + g.n_sfha).clip(lower=1)
    print(f"kept {len(g):,} block groups (hu >= 200, SFHA policy share < 5%)")
    print("take-up quantiles:", g.takeup.quantile([.05, .25, .5, .75, .9, .95, .99]).round(4).to_dict())
    print("share of block groups with zero policies is excluded by construction; with >=1 policy only")
    h, e = np.histogram(np.log10(g.takeup.clip(lower=1e-4)), bins=14)
    print("log10(take-up) histogram:")
    for n, lo in zip(h, e[:-1]):
        print(f"  {lo:5.2f}  {'#' * max(1, int(60 * n / h.max()))} {n}")
    x = np.log10(g.takeup.clip(lower=1e-4)).values
    bic = gmm_bic(x)
    print("GMM BIC on log10(take-up), 1/2/3 components:", {k: round(v, 1) for k, v in bic.items()},
          "-> best", min(bic, key=bic.get))
    print("mandatory-purchase share of policies in these block groups (median):", round(float(g.mand_share.median()), 3))


if __name__ == "__main__":
    main()
