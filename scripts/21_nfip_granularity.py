"""How fine can a flood outcome be, using NFIP claims? (national, 2000 onward)

Reports, from the claims files written by scripts/20_fetch_nfip_claims.py:
  completeness   share of claims with a block group, water depth, duration, event name
  scope          cause-of-damage codes and what an "inland flood" definition keeps or drops
  tropical       claims and dollars in named tropical-cyclone events vs the rest
  units          how many block groups / tracts / ZIPs / counties the non-tropical inland claims touch
  episodes       claim clusters (county, gaps <= 3 days): sizes, and how many are big enough to study
                 spatial variation inside one event
  intensity      does building-level water depth track dollars paid? (the SHELDUS-style outcome did not)

    python scripts/21_nfip_granularity.py
"""

from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

FLOOD = wb2.cache_path("labels_extra") / "flood"
# word boundaries matter: a bare "ida" or "ian" would flag Florida, Idaho, Louisiana, Indiana
TC = re.compile(r"\b(hurricane|tropical|typhoon|superstorm|post-tropical|"
                r"katrina|harvey|irma|florence|michael|gustav|wilma|dorian|sandy|rita|ike|ian|ida)\b", re.I)
INLAND = {"2", "3", "4"}  # stream/river/lake overflow, alluvial fan, rainfall or snowmelt accumulation


def load() -> pd.DataFrame:
    d = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(str(FLOOD / "nfip_national" / "nfip_*.parquet")))],
                  ignore_index=True)
    d["date"] = pd.to_datetime(d.dateOfLoss).dt.tz_localize(None).dt.normalize()
    d["bg"] = d.censusGeoid.astype("string").str.zfill(12).where(d.censusGeoid.notna())
    d["tract"], d["cty"] = d.bg.str[:11], d.bg.str[:5]
    num = ["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim", "waterDepth", "interiorWaterDepth",
           "exteriorWaterDepth", "floodWaterDuration", "buildingDamageAmount", "contentsDamageAmount"]
    for c in num:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["paid"] = d.amountPaidOnBuildingClaim.fillna(0) + d.amountPaidOnContentsClaim.fillna(0)
    d["cause"] = d.causeOfDamage.astype("string").str.strip()
    d["tc"] = d.floodEvent.fillna("").str.contains(TC)
    d["named"] = d.floodEvent.notna() & (d.floodEvent.str.strip() != "")
    return d


def section_completeness(d: pd.DataFrame) -> None:
    print(f"claims {len(d):,} ({d.date.min():%Y-%m-%d} to {d.date.max():%Y-%m-%d}); paid ${d.paid.sum() / 1e9:.1f}B; "
          f"{d.state.nunique()} states, {d.cty.nunique()} counties, {d.bg.nunique():,} block groups, "
          f"{d.reportedZipCode.nunique():,} ZIPs")
    by = d.groupby("yearOfLoss").agg(
        claims=("id", "size"), bg=("bg", lambda s: s.notna().mean()), depth=("waterDepth", lambda s: s.notna().mean()),
        dur=("floodWaterDuration", lambda s: s.notna().mean()), named=("named", "mean"),
        dmg=("buildingDamageAmount", lambda s: s.notna().mean()))
    print(by.round(3).T.to_string())


def section_scope(d: pd.DataFrame) -> None:
    print("\n== cause of damage (code: claims share, paid share)")
    g = d.groupby(d.cause.fillna("(blank)")).agg(n=("id", "size"), paid=("paid", "sum"))
    g["claims"], g["paid_share"] = g.n / g.n.sum(), g.paid / g.paid.sum()
    print(g.sort_values("n", ascending=False).head(10)[["n", "claims", "paid_share"]].round(3).to_string())
    print("codes: 1 tidal overflow, 2 river/lake overflow, 3 alluvial fan, 4 rainfall or snowmelt accumulation, "
          "0 other, B/C/D expedited handling (cause not recorded)")


def section_tropical(d: pd.DataFrame) -> None:
    print("\n== named events and tropical cyclones")
    print(f"claims with an event name {d.named.mean():.3f}; named tropical-cyclone claims {d.tc.mean():.3f} of all, "
          f"{d.paid[d.tc].sum() / d.paid.sum():.3f} of dollars paid")
    top = d[d.named].groupby("floodEvent").agg(n=("id", "size"), paid=("paid", "sum"), states=("state", "nunique"))
    print("largest events by claims:\n", top.sort_values("n", ascending=False).head(12).round(0).to_string())
    print("\ntropical vs not, by cause code (claims):")
    print(pd.crosstab(d.cause.fillna("(blank)").where(lambda s: s.isin(["0", "1", "2", "3", "4", "B", "C", "D"]), "other"),
                      d.tc.map({True: "tropical-named", False: "not tropical-named"}), normalize="index").round(3).to_string())


def section_units(d: pd.DataFrame) -> pd.DataFrame:
    print("\n== non-tropical inland scope (cause 2/3/4, not a tropical-named event)")
    s = d[d.cause.isin(INLAND) & ~d.tc & d.bg.notna()].copy()
    print(f"claims {len(s):,} ({len(s) / len(d):.3f} of all), dollars ${s.paid.sum() / 1e9:.1f}B ({s.paid.sum() / d.paid.sum():.3f})")
    print(f"touches {s.bg.nunique():,} block groups, {s.tract.nunique():,} tracts, {s.reportedZipCode.nunique():,} ZIPs, "
          f"{s.cty.nunique():,} counties in {s.state.nunique()} states")
    print("claims by state (top 10):", s.state.value_counts().head(10).to_dict())
    bgd = s.groupby(["bg", "date"]).size()
    print(f"claims per block-group-day: mean {bgd.mean():.2f}, share of block-group-days with >=2 claims {(bgd >= 2).mean():.3f}; "
          f"block-group-days {len(bgd):,}")
    print("paid per claim ($): median %.0f, 90th %.0f, 99th %.0f" % tuple(s.paid.quantile([.5, .9, .99])))
    return s


def section_episodes(s: pd.DataFrame) -> None:
    print("\n== claim episodes (county, gaps <= 3 days)")
    s = s.sort_values(["cty", "date"]).copy()
    new = (s.cty != s.cty.shift()) | ((s.date - s.date.shift()).dt.days > 3)
    s["ep"] = new.cumsum()
    e = s.groupby("ep").agg(n=("id", "size"), bgs=("bg", "nunique"), paid=("paid", "sum"), yr=("yearOfLoss", "first"))
    print(f"episodes {len(e):,}; claims per episode median {e.n.median():.0f}, 90th {e.n.quantile(.9):.0f}, 99th {e.n.quantile(.99):.0f}")
    for n0, b0 in ((10, 5), (30, 10), (100, 20)):
        k = e[(e.n >= n0) & (e.bgs >= b0)]
        print(f"episodes with >= {n0} claims in >= {b0} block groups: {len(k):,} "
              f"(train 2000-2016 {int((k.yr <= 2016).sum())}, test 2017-2023 {int((k.yr >= 2017).sum())}); "
              f"hold {k.n.sum() / e.n.sum():.2f} of claims")


def section_intensity(s: pd.DataFrame) -> None:
    print("\n== does building-level water depth track dollars paid?")
    s = s.copy()
    s["cover"] = pd.to_numeric(s.totalBuildingInsuranceCoverage, errors="coerce")
    s["ratio"] = s.amountPaidOnBuildingClaim / s.cover.where(s.cover > 0)
    sp = lambda df, a, b: df[[a, b]].corr("spearman").iloc[0, 1]  # noqa: E731
    x = s[(s.waterDepth > 0) & (s.waterDepth <= 240) & (s.paid > 0)]
    print(f"any depth field: n={len(x):,}  Spearman(depth, paid) = {sp(x, 'waterDepth', 'paid'):.3f}")
    b = x.groupby(pd.cut(x.waterDepth, [0, 3, 6, 12, 24, 48, 240]), observed=True).paid.median()
    print("  median paid by depth bin (inches):", {str(k): int(v) for k, v in b.items()})
    i = s[(s.interiorWaterDepth > 0) & (s.interiorWaterDepth <= 240) & (s.paid > 0)]
    print(f"interior depth: n={len(i):,}  Spearman(interior depth, paid) = {sp(i, 'interiorWaterDepth', 'paid'):.3f}")
    r = s[(s.interiorWaterDepth > 0) & (s.interiorWaterDepth <= 240) & (s.ratio > 0) & (s.ratio <= 1.0)
          & (pd.to_numeric(s.occupancyType, errors="coerce") == 1)]  # 1 = single-family; values arrive as '1.0'
    print(f"single-family, interior depth vs paid/coverage: n={len(r):,}  Spearman = {sp(r, 'interiorWaterDepth', 'ratio'):.3f}")
    print("  median paid/coverage by interior depth (inches):",
          {str(k): round(float(v), 3) for k, v in r.groupby(pd.cut(r.interiorWaterDepth, [0, 6, 12, 24, 48, 240]),
                                                           observed=True).ratio.median().items()})
    for c, name in (("2", "river/lake overflow"), ("4", "rain/snowmelt accumulation")):
        t = s[(s.cause == c) & (s.waterDepth > 0) & (s.waterDepth <= 240) & (s.paid > 0)]
        print(f"  cause {c} ({name}): n={len(t):,}  Spearman(depth, paid) = {sp(t, 'waterDepth', 'paid'):.3f}")
    z = s[(s.buildingDamageAmount > 0) & (s.paid > 0)]
    print(f"Spearman(estimated damage, paid) = {sp(z, 'buildingDamageAmount', 'paid'):.3f} (n={len(z):,})")
    print(f"share of paid claims at the $250k building cap: {(s.amountPaidOnBuildingClaim >= 249000).mean():.4f}")


def main() -> None:
    d = load()
    section_completeness(d)
    section_scope(d)
    section_tropical(d)
    s = section_units(d)
    section_episodes(s)
    section_intensity(s)
    s.to_parquet(wb2.cache_path("nfip_inland_nontropical.parquet"))


if __name__ == "__main__":
    main()
