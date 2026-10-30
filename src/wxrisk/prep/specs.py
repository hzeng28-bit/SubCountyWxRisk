"""Cleaning specs of the tables the pipeline uses. Add a table by adding a TableSpec here and running scripts/40_prep.py."""

from __future__ import annotations

from dataclasses import replace

from .spec import Col, TableSpec

POLICIES_TX = TableSpec(
    name="policies_tx",
    source="labels_extra/flood/pilot/policies_tx.parquet",
    derived=("coverage_end",),
    cols=(
        # join key: 2010-vintage block group. A null / malformed code (0.34% of rows) makes the row unlocatable but it is still a
        # real policy (state-level frequencies need it in the denominator), so it is kept and flagged; block-group analyses
        # filter on `geoid_missing`. Never stringify this column before that check.
        Col("censusGeoid", "key", name="geoid", width=12, nulls="flag"),
        Col("ratedFloodZone", "str", name="zone", upper=True, nulls="flag"),
        Col("occupancyType", "int", name="occupancy"),
        Col("policyEffectiveDate", "date", name="eff", nulls="drop"),
        # structural null: no termination date = still running (consumers read `term_missing`, they do not guess a date)
        Col("policyTerminationDate", "date", name="term", nulls="flag"),
        Col("cancellationDateOfFloodPolicy", "date", name="cancelled"),
        Col("policyCount", "int", name="n_policies", lo=1),
        Col("totalBuildingInsuranceCoverage", "int", name="cov_building"),
        # negative premium = refund (OpenFEMA dictionary), a valid value: no range rule
        Col("totalInsurancePremiumOfThePolicy", "int", name="premium"),
        Col("buildingReplacementCost", "int", name="replacement_cost", lo=1, nulls="flag"),
        # docs/DATA_SOURCES trap 1: '0' = unknown, and its share collapses from 86% (2010) to ~0 (2022+). Null + flag keeps
        # the unknown share available as an explicit feature instead of a category that leaks the year.
        Col("foundationType", "str", name="foundation", sentinel=("0",), nulls="flag"),
        # a CODE, not a count (dictionary): 1 one floor, 2 two, 3 three or more, 4 split-level, 5 manufactured home, 6 townhouse
        Col("numberOfFloorsInInsuredBuilding", "int", name="floors_code", lo=1, hi=6, nulls="flag"),
        # dictionary: 9999 = not reported/used (none in Texas); values beyond +-1000 ft (26 rows, min -9986) are entry errors
        Col("elevationDifference", "int", name="elev_diff", sentinel=(9999,), lo=-1000, hi=1000, nulls="flag"),
        Col("mandatoryPurchaseFlag", "bool", name="mandatory"),
        Col("postFIRMConstructionIndicator", "bool", name="post_firm"),
        Col("primaryResidenceIndicator", "bool", name="primary_res"),
        # dates outside the plausible range are junk: NFIP began in 1968 (rows from year 0001 / 1010 exist); construction dates
        # carry the placeholder 1492-10-12 (14,094 rows) and years 0, 900-1100, 9999. Range -> null, `_missing` keeps the fact.
        Col("originalNBDate", "date", name="first_policy_date", lo="1968-01-01", hi="2027-01-01", nulls="flag"),
        Col("originalConstructionDate", "date", name="built_date", lo="1800-01-01", hi="2027-01-01", nulls="flag"),
    ),
)

# Claims keep rows whose location is missing (state-level analyses use them), so geoid/county are flagged, not dropped:
# a consumer that needs a block group filters on `geoid_missing`. `id` is unique, so it also fixes the row order.
CLAIMS_NFIP = TableSpec(
    name="claims_nfip",
    source="labels_extra/flood/nfip_national/nfip_*.parquet",
    expect_unique=("id",),
    cols=(
        Col("id", "long"),
        Col("state", "str", upper=True),
        Col("countyCode", "key", name="county", width=5, nulls="flag"),
        Col("censusGeoid", "key", name="geoid", width=12, nulls="flag"),
        Col("dateOfLoss", "date", name="loss_date", parse="%Y-%m-%d", nulls="drop"),
        Col("yearOfLoss", "int", name="loss_year"),
        Col("causeOfDamage", "str", name="cause", nulls="flag"),
        Col("floodEvent", "str", name="flood_event", nulls="flag"),  # ~81% filled, ~72 names; used only to drop tropical-named claims
        # docs/DATA_SOURCES trap 2: depth 0 is a default, not "dry" (large damage at depth 0). 999 is a second unknown code:
        # it holds essentially all depths above 600 in (99th percentile of the positive values is 999). Anything else above 600 in is
        # out of range (a few 9600 / 7000 entries). floodWaterDuration (~0 everywhere) is left out.
        # waterDepth is NOT carried: the dictionary says "there are instances where measurements were provided in feet", and
        # 99 is a placeholder (8,164 rows vs 124 at 98 and 213 at 100). exteriorWaterDepth / interiorWaterDepth are defined in inches.
        Col("exteriorWaterDepth", "float", name="depth_ext", sentinel=(0.0, 999.0), hi=600, nulls="flag"),
        Col("interiorWaterDepth", "float", name="depth_int", sentinel=(0.0, 999.0), hi=600, nulls="flag"),
        Col("buildingDamageAmount", "float", name="damage_building", nulls="flag"),
        Col("contentsDamageAmount", "float", name="damage_contents", nulls="flag"),
        # negative = a check that was not cashed and had to be re-issued (dictionary): valid, no range rule
        Col("amountPaidOnBuildingClaim", "float", name="paid_building", nulls="flag"),
        # coverage 0 is a TRUE value: a contents-only policy (93-99% of these claims have no building damage, 98-99.9% have contents
        # damage). Kept as 0; a damage/coverage ratio must filter cov_building > 0 itself.
        Col("totalBuildingInsuranceCoverage", "long", name="cov_building"),
        Col("ratedFloodZone", "str", name="zone", upper=True, nulls="flag"),
        Col("occupancyType", "int", name="occupancy"),
        Col("foundationType", "str", name="foundation", sentinel=("0",), nulls="flag"),
        Col("elevatedBuildingIndicator", "bool", name="elevated"),
        Col("postFIRMConstructionIndicator", "bool", name="post_firm"),
        Col("policyCount", "int", name="n_policies"),
    ),
)

# Same cleaning rules for the candidate states (extracts of scripts/24); the name differs, so the sidecars and clean files do too.
POLICIES_LA = replace(POLICIES_TX, name="policies_la", source="labels_extra/flood/pilot/policies_la.parquet")
POLICIES_FL = replace(POLICIES_TX, name="policies_fl", source="labels_extra/flood/pilot/policies_fl.parquet")

SPECS = {s.name: s for s in (POLICIES_TX, POLICIES_LA, POLICIES_FL, CLAIMS_NFIP)}
