"""Inventory of the candidate states before the scale-up: block groups, policies and claims per state, and how many of them
land on a 2010 block group through the crosswalk (the match rate is the label coverage the model can use).

Inputs : pilot/policies_<st>.parquet (scripts/24), nfip_national claims (clean table claims_nfip), TIGER 2010 block groups and the
         Census relationship file per state (scripts/22, scripts/fetch/fetch_state_geography.py)
Output : stdout, tee to $WXRISK_DATA/audit/state_inventory.txt
Run    : python scripts/83_state_inventory.py TX LA FL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import bg_crosswalk, states  # noqa: E402
from wxrisk.prep import build as prep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("states", nargs="+")
    a = ap.parse_args()
    claims = (prep.scan("claims_nfip").filter(pl.col("cause").is_in(["2", "3", "4"]))
              .select("state", pl.col("geoid"), pl.col("loss_date")).collect())
    for ab in a.states:
        st = states.get(ab)
        cw = bg_crosswalk.build(st.relationship if ab != "TX" else None)
        ok = set(cw.code)
        n_bg10 = cw.bg10.nunique()
        pol = pl.read_parquet(st.policies, columns=["censusGeoid", "policyEffectiveDate", "policyCount"])
        pol = pol.with_columns(pl.col("censusGeoid").cast(pl.Utf8).str.zfill(12).alias("g"))
        pol_has = pol.filter(pl.col("censusGeoid").is_not_null())
        pol_ok = pol_has.filter(pl.col("g").is_in(list(ok)))
        c = claims.filter(pl.col("state") == ab)
        c_has = c.filter(pl.col("geoid").is_not_null())
        c_ok = c_has.filter(pl.col("geoid").is_in(list(ok)))
        yrs = c_ok.group_by(pl.col("loss_date").dt.year().alias("y")).len().sort("y")
        print(f"\n== {st.name} ({ab}) ==")
        print(f"2010 block groups in crosswalk {n_bg10:,}; codes {len(ok):,}")
        print(f"policy rows {pol.height:,}; with block group {pol_has.height:,} ({pol_has.height / pol.height:.1%}); "
              f"on a known code {pol_ok.height:,} ({pol_ok.height / max(pol_has.height, 1):.1%}); "
              f"policy count sum {pol['policyCount'].sum():,.0f}")
        print(f"inland claims (cause 2/3/4) {c.height:,}; with block group {c_has.height:,}; on a known code {c_ok.height:,} "
              f"({c_ok.height / max(c_has.height, 1):.1%})")
        y = dict(zip(yrs["y"].to_list(), yrs["len"].to_list(), strict=True))
        print("claims with BG by year:", {k: v for k, v in y.items() if 2010 <= k <= 2023})


if __name__ == "__main__":
    main()
