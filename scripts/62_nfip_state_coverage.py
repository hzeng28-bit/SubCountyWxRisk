"""How unevenly does the NFIP cover the US, state by state? (stage 24)

Three views, all from NFIP's own files:
  1. penetration   OpenFEMA NfipResidentialPenetrationRates (county): contracts in force / residential structures, all and SFHA only,
                   aggregated to state; dispersion of county rates inside each state
  2. policy flow   policies written per effective year and state (bulk policy file, 2009+), to see whether the state ranking is stable in time
  3. claims        inland non-tropical claims 2010-2023 per 1,000 policy-years, by state (the label rate the pilot would face in each state)
Inputs : labels_extra/flood/nfip_penetration.parquet, nfip_policies_v3.parquet, nfip_national/nfip_20*.parquet
Output : $WXRISK_DATA/audit/nfip_state_coverage.txt
Run    : python scripts/62_nfip_state_coverage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

FL = wb2.cache_path("labels_extra") / "flood"
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def main() -> None:
    pen = pd.read_parquet(FL / "nfip_penetration.parquet")
    say(f"== 1. penetration table: {len(pen)} counties, asOfDate {pen.asOfDate.astype(str).str[:10].unique()[:3]}")
    for c in ["resContractsInForce", "totalResStructures", "resContractsInForceSfha", "totalResStructuresSfha"]:
        pen[c] = pd.to_numeric(pen[c], errors="coerce")
    g = pen.groupby("state").agg(counties=("fipsCode", "size"), pol=("resContractsInForce", "sum"), struct=("totalResStructures", "sum"),
                                 pol_sfha=("resContractsInForceSfha", "sum"), struct_sfha=("totalResStructuresSfha", "sum"))
    g["pen"] = g.pol / g.struct
    g["pen_sfha"] = g.pol_sfha / g.struct_sfha
    g["share_pol_sfha"] = g.pol_sfha / g.pol
    pen["rate"] = pen.resContractsInForce / pen.totalResStructures
    g["county_median"] = pen.groupby("state").rate.median()
    g["share_cty_lt1pct"] = pen.groupby("state").rate.apply(lambda x: (x < 0.01).mean())
    g = g.sort_values("pen", ascending=False)
    say(g.round(4).to_string())
    tot = g[["pol", "struct"]].sum()
    say(f"\n  national penetration {tot.pol / tot.struct:.4f}; top 5 states hold {g.pol.head(5).sum() / tot.pol:.3f} of policies "
        f"({g.struct.head(5).sum() / tot.struct:.3f} of structures)")
    say(f"  states (>= 100k structures) with penetration < 1%: {(g[g.struct >= 1e5].pen < 0.01).sum()} of {(g.struct >= 1e5).sum()}; "
        f"> 5%: {(g[g.struct >= 1e5].pen > 0.05).sum()}")
    say(f"  ratio of 90th to 10th percentile of state penetration (>= 100k structures): "
        f"{g[g.struct >= 1e5].pen.quantile(.9) / g[g.struct >= 1e5].pen.quantile(.1):.1f}")
    # between-state share of the variance of county log-rate
    pc = pen[(pen.rate > 0) & (pen.totalResStructures >= 1000)].copy()
    pc["lr"] = np.log(pc.rate)
    within = pc.groupby("state").lr.transform(lambda x: x - x.mean())
    say(f"  share of the variance of county log-penetration (counties >= 1000 structures, n={len(pc)}) that is between states: "
        f"{1 - within.var() / pc.lr.var():.3f}")

    say("\n== 2. policies written per effective year, by state (contracts, millions of rows aggregated in DuckDB)")
    con = duckdb.connect()
    q = f"""select propertyState st, year(cast(policyEffectiveDate as date)) yr, sum(coalesce(policyCount,1)) n
            from read_parquet('{FL / 'nfip_policies_v3.parquet'}')
            where propertyState is not null and year(cast(policyEffectiveDate as date)) between 2010 and 2023 group by 1, 2"""
    p = con.execute(q).df().pivot(index="st", columns="yr", values="n").fillna(0)
    top = p.sum(axis=1).sort_values(ascending=False).head(12).index
    say((p.loc[top] / 1000).round(0).astype(int).to_string())
    share = p.div(p.sum(axis=0), axis=1)
    say("\n  state share of national policies written, 2010 vs 2023 (top 8 states by 2023):")
    say(pd.DataFrame({"2010": share[2010], "2023": share[2023]}).sort_values("2023", ascending=False).head(8).round(4).to_string())
    say(f"  rank correlation of state policy counts 2010 vs 2023 (Spearman): {p[2010].rank().corr(p[2023].rank()):.3f}")

    say("\n== 3. inland non-tropical claims per 1,000 policy-years, 2010-2023, by state")
    frames = [pd.read_parquet(FL / "nfip_national" / f"nfip_{y}.parquet", columns=["state", "dateOfLoss", "causeOfDamage"]) for y in range(2010, 2024)]
    cl = pd.concat(frames)
    cl = cl[cl.causeOfDamage.astype(str).isin(["2", "3", "4"])]
    cc = cl.groupby("state").size()
    polyears = p.sum(axis=1)  # policies written per year summed over 2010-2023 ~ policy-years (terms are ~1 year)
    r = pd.DataFrame({"claims": cc, "policy_years": polyears}).dropna()
    r["per_1000_polyears"] = 1000 * r.claims / r.policy_years
    r = r.sort_values("policy_years", ascending=False).head(15)
    say(r.round(2).to_string())
    p_ = FL.parent.parent / "audit" / "nfip_state_coverage.txt"
    p_.parent.mkdir(exist_ok=True)
    p_.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
