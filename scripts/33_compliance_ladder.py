"""Compliance ladder: does flood loss fall monotonically with how much flood-resistant design a building has?

Latent-variable idea (ASCE 24 / NFIP as a hidden "design level"): we cannot observe code compliance, only
a few proxies, so buildings are assigned to observable tiers ordered from most to least protected:

  T0  post-FIRM, elevated, SFHA zone (A*/V*)   closest to an engineered, code-compliant building
  T1  post-FIRM, not elevated, SFHA            NFIP minimum (slab/enclosure allowed)
  T2  pre-FIRM, SFHA                           old building, built before flood maps
  T3  non-SFHA zone (X/B/C/D), any             owner unaware; often insured at sale (mandatory purchase)

Not observable at all: flood-damage-resistant materials, equipment elevation, flood openings, freeboard height
(`elevationDifference` exists only on policies, ~17% filled), the ASCE 24 edition a community adopted.

  loss   within exterior-depth bins: damage ratio and absolute damage quantiles by tier (the q10 is the
         "residual loss floor" given a claim), ratio to T2, event-clustered bootstrap CI
  freq   Texas claims per policy-year by tier (post-FIRM x SFHA only, no elevation on the policy side),
         rate ratio to pre-FIRM SFHA per loss year
  mono   share of depth bins in which the tier medians are ordered T0 <= T1 <= T2

Claims below the deductible never enter the file, so low quantiles are left-truncated; use buildingDamageAmount
and read q10 as a floor on *reported* claims only. Tiers are selected (insured) populations: no causal reading.

    python scripts/33_compliance_ladder.py | tee $WXRISK_DATA/audit/compliance_ladder.txt
    python scripts/33_compliance_ladder.py --sections loss --scope national --boot 100
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.prep import build as prep  # noqa: E402

warnings.filterwarnings("ignore")
FLOOD = wb2.cache_path("labels_extra") / "flood"
TIERS = ["T0", "T1", "T2", "T3"]
DEPTH_BINS = [0, 6, 12, 24, 48, 1e4]
DEPTH_LAB = ["1-6", "7-12", "13-24", "25-48", ">48"]
QS = [0.1, 0.5]


def is_sfha(zone: pd.Series) -> pd.Series:
    z = zone.astype(str).str.upper()
    return z.str.startswith("A") | z.str.startswith("V")


def assign_tier(sfha: pd.Series, post: pd.Series, elev: pd.Series | None) -> pd.Series:
    t = pd.Series("T3", index=sfha.index)
    t[sfha & ~post] = "T2"
    t[sfha & post] = "T1"
    if elev is not None:
        t[sfha & post & elev] = "T0"
    return t


def load_claims(scope: str, since: int = 2010) -> pd.DataFrame:
    """Single-family claims with a usable depth and damage ratio, from the clean table `claims_nfip`.
    `depth_ext` is null where the source says 0 (default, FLOOD_PRIORS trap 2), 999 (unknown code) or > 600 in."""
    q = (prep.scan("claims_nfip")
         .filter((pl.col("loss_year") >= since) & pl.col("occupancy").is_in([1, 11]) & (pl.col("depth_ext") > 0)
                 & (pl.col("cov_building") > 0) & pl.col("damage_building").is_not_null()))
    if scope == "tx":
        q = q.filter(pl.col("state") == "TX")
    c = q.select("state", pl.col("loss_year").alias("yearOfLoss"), pl.col("loss_date").alias("dateOfLoss"),
                 pl.col("zone").alias("ratedFloodZone"), pl.col("elevated").alias("elevatedBuildingIndicator"),
                 pl.col("post_firm").alias("postFIRMConstructionIndicator"), pl.col("depth_ext").alias("exteriorWaterDepth"),
                 pl.col("damage_building").alias("buildingDamageAmount"),
                 pl.col("cov_building").cast(pl.Float64).alias("totalBuildingInsuranceCoverage")).collect().to_pandas()
    c["dr"] = (c.buildingDamageAmount / c.totalBuildingInsuranceCoverage).clip(0, 1.5)
    c["tier"] = assign_tier(is_sfha(c.ratedFloodZone), c.postFIRMConstructionIndicator.astype(bool),
                            c.elevatedBuildingIndicator.astype(bool))
    c["bin"] = pd.cut(c.exteriorWaterDepth, DEPTH_BINS, labels=DEPTH_LAB)
    c["ev"] = c.state.astype(str) + "_" + pd.to_datetime(c.dateOfLoss).dt.strftime("%Y-%m")
    return c


def cell_stats(c: pd.DataFrame) -> pd.DataFrame:
    g = c.groupby(["bin", "tier"], observed=True)
    out = pd.DataFrame({"n": g.size(), "dr_q10": g.dr.quantile(0.1), "dr_med": g.dr.median(),
                        "usd_q10": g.buildingDamageAmount.quantile(0.1),
                        "usd_med": g.buildingDamageAmount.median()})
    return out


def boot_ratio(c: pd.DataFrame, nboot: int, seed: int = 0) -> pd.DataFrame:
    """Per depth bin: median damage ratio of each tier over T2, event-clustered bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    evs = c.ev.unique()
    groups = {e: d for e, d in c.groupby("ev")}
    rows = []
    for _ in range(nboot):
        pick = rng.choice(evs, len(evs))
        b = pd.concat([groups[e] for e in pick])
        m = b.groupby(["bin", "tier"], observed=True).dr.median().unstack("tier")
        rows.append(m.div(m["T2"], axis=0))
    r = pd.concat(rows, keys=range(nboot))
    pt = cell_stats(c).dr_med.unstack("tier")
    pt = pt.div(pt["T2"], axis=0)
    lo = r.groupby(level=1).quantile(0.025)
    hi = r.groupby(level=1).quantile(0.975)
    return pt, lo, hi


def sec_loss(scope: str, nboot: int) -> None:
    c = load_claims(scope)
    print(f"\n## loss: {scope} single-family claims 2010+, {len(c):,} rows, {c.ev.nunique()} state-months")
    print("tier shares:", c.tier.value_counts(normalize=True).round(3).to_dict())
    print("\ncell stats (dr = damage / building coverage; usd = building damage amount):")
    print(cell_stats(c).round(3).to_string())
    pt, lo, hi = boot_ratio(c, nboot)
    print(f"\nmedian damage ratio relative to T2 (pre-FIRM SFHA), {nboot} event-clustered bootstrap draws:")
    for t in ["T0", "T1", "T3"]:
        print(f"  {t}/T2")
        for b in pt.index:
            print(f"    depth {b:>6}: {pt.loc[b, t]:.2f}  [{lo.loc[b, t]:.2f}, {hi.loc[b, t]:.2f}]")
    ok = [(pt.loc[b, "T0"] <= pt.loc[b, "T1"] <= 1.0) for b in pt.index]
    print(f"\nmono: bins with T0 <= T1 <= T2 in median ratio: {sum(ok)}/{len(ok)}")


def policy_years(t: pd.DataFrame, year: int) -> pd.Series:
    lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
    start = t.policyEffectiveDate.values.astype("datetime64[D]")
    end = t.policyTerminationDate.fillna(pd.Timestamp("2100-01-01")).values.astype("datetime64[D]")
    days = (np.minimum(end, hi) - np.maximum(start, lo)).astype(int).clip(0)
    return pd.Series(days / 365.25 * t.policyCount.values, index=t.index)


def sec_freq() -> None:
    p = (prep.scan("policies_tx").filter(pl.col("occupancy").is_in([1, 11]))
         .select(pl.col("zone").alias("ratedFloodZone"), pl.col("post_firm").alias("postFIRMConstructionIndicator"),
                 pl.col("eff").alias("policyEffectiveDate"), pl.col("coverage_end").alias("policyTerminationDate"),
                 pl.col("n_policies").alias("policyCount")).collect().to_pandas())
    p["policyEffectiveDate"] = pd.to_datetime(p.policyEffectiveDate).astype("datetime64[ns]")
    p["policyTerminationDate"] = pd.to_datetime(p.policyTerminationDate).astype("datetime64[ns]")
    p["tier"] = assign_tier(is_sfha(p.ratedFloodZone), p.postFIRMConstructionIndicator.astype(bool), None)
    c = load_claims("tx")
    c["tier"] = assign_tier(is_sfha(c.ratedFloodZone), c.postFIRMConstructionIndicator.astype(bool), None)
    print(f"\n## freq: Texas single-family, claims per 1000 policy-years (tier without elevation), {len(p):,} policy rows")
    print("year  " + "  ".join(f"{t}_rate" for t in ["T1", "T2", "T3"]) + "   T1/T2   T3/T2")
    for y in range(2010, 2024):
        py = p.groupby("tier").apply(lambda d: policy_years(d, y).sum())
        cl = c[c.yearOfLoss == y].groupby("tier").size()
        r = (cl / py * 1000).reindex(["T1", "T2", "T3"])
        print(f"{y}  " + "  ".join(f"{v:7.2f}" for v in r) + f"   {r['T1'] / r['T2']:.2f}    {r['T3'] / r['T2']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=["loss", "freq"])
    ap.add_argument("--scope", choices=["tx", "national"], default="tx")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    if "loss" in a.sections:
        sec_loss(a.scope, a.boot)
    if "freq" in a.sections:
        sec_freq()


if __name__ == "__main__":
    main()
