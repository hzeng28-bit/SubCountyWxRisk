"""Statistical evidence behind every cleaning rule (docs/CLEANING_REGISTER.md quotes these numbers).

Audits the RAW files, not the clean tables: the point is to justify the rules that make the clean tables.

For each rule the question is one of three:
  provider-specified  the OpenFEMA dictionary / FAQ says so          -> the evidence is a consistency check on our data
  undocumented code   a value the provider does not define            -> spike test (count at v vs its neighbours) plus a
                      (0, 999, 99, 1492-10-12, foundation '0')           profile test (does the outcome at v look like its neighbours?)
  entry error         sporadic junk values                            -> missingness diagnostics (is it related to anything we observe?)
                                                                         and, where imputing is on the table, a masked-value validation

    python scripts/41_cleaning_evidence.py | tee $WXRISK_DATA/audit/cleaning_evidence.txt
Writes $WXRISK_DATA/audit/cleaning_evidence.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

FLOOD = wb2.cache_path("labels_extra") / "flood"
POL = f"read_parquet('{FLOOD / 'pilot' / 'policies_tx.parquet'}')"
CLM = f"read_parquet('{FLOOD / 'nfip_national' / '*.parquet'}', union_by_name=true)"
SEED = 0
EV: dict = {}


def q(sql: str) -> list:
    return duckdb.sql(sql).fetchall()


def _plain(x):
    """JSON-safe copy: dict keys become strings (None and ints mix), numpy scalars become Python numbers."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x.item() if hasattr(x, "item") else x


def put(key: str, **kw) -> None:
    EV[key] = _plain(kw)
    print(f"\n[{key}]")
    for k, v in kw.items():
        print(f"  {k}: {v}")


# ------------------------------------------------------------------ spike test


def spike_scan() -> None:
    """Every numeric column: is there a value v in {-9999..99999999 of the form 9, 99, 999, ...} whose count is >= 5x its neighbours?"""
    found = {}
    for tag, src in (("policies_tx", POL), ("claims", CLM)):
        cols = [r[0] for r in duckdb.sql(f"describe select * from {src}").fetchall()
                if r[1] in ("DOUBLE", "BIGINT", "INTEGER", "SMALLINT", "FLOAT") or r[1].startswith("DECIMAL")]
        for c in cols:
            if c in ("id", "yearOfLoss", "policyCount"):
                continue
            for v in (9, 99, 999, 9999, 99999, 999999, 9999999, 99999999, -99, -999, -9999):
                n, a, b = q(f"select count(*) filter (where {c}={v}), count(*) filter (where {c}={v - 1}), "
                            f"count(*) filter (where {c}={v + 1}) from {src}")[0]
                if n >= 20 and n >= 5 * max((a + b) / 2, 1):
                    found[f"{tag}.{c}"] = {"value": v, "count": n, "neighbours": [a, b]}
        found[f"{tag}._columns_scanned"] = len(cols)
    put("spike_scan_99xx", rule="count(v) >= 20 and >= 5x mean(count(v-1), count(v+1)) for v in 9, 99, ..., 99999999, -99, -999, -9999",
        found=found)


# ------------------------------------------------------------------ claims depth


def depth() -> None:
    sf = "occupancyType in (1,11) and totalBuildingInsuranceCoverage>0 and buildingDamageAmount is not null"
    ratio = "buildingDamageAmount/totalBuildingInsuranceCoverage"
    prof = q(f"""select case when exteriorWaterDepth<0 then 'negative' when exteriorWaterDepth=0 then '0'
        when exteriorWaterDepth<=6 then '1-6' when exteriorWaterDepth<=24 then '7-24' when exteriorWaterDepth<=48 then '25-48'
        when exteriorWaterDepth<999 then '49-998' when exteriorWaterDepth=999 then '999' else '>999' end k,
        count(*), round(median({ratio}),3) from {CLM} where {sf} and exteriorWaterDepth is not null group by 1 order by 1""")
    freq = dict(q(f"select exteriorWaterDepth, count(*) from {CLM} where exteriorWaterDepth in (0,1,2,3,4,5,6,998,999) group by 1 order by 1"))
    by_year = q(f"select yearOfLoss, round(avg((exteriorWaterDepth is null)::int),2), "
                f"round(avg((exteriorWaterDepth=0)::int) filter (where exteriorWaterDepth is not null),2) from {CLM} group by 1 order by 1")
    other = q(f"select exteriorWaterDepth, count(*) from {CLM} where exteriorWaterDepth>600 and exteriorWaterDepth<>999 "
              f"group by 1 order by 2 desc limit 6")
    wd = q(f"select count(*), round(avg((abs(waterDepth*12-exteriorWaterDepth)<=6)::int),3), "
           f"round(avg((abs(waterDepth-exteriorWaterDepth)<=2)::int),3) from {CLM} where waterDepth>0 and exteriorWaterDepth>0")[0]
    put("claims.exterior_depth", count_at=freq,
        median_damage_ratio_by_exterior_depth_class={k: {"n": n, "median": m} for k, n, m in prof},
        null_share_and_zero_share_by_loss_year={y: [a, b] for y, a, b in by_year},
        values_above_600_excluding_999=other)
    put("claims.waterDepth_units", rows_both_positive=wd[0], share_wd_times_12_close_to_exterior=wd[1],
        share_wd_close_to_exterior=wd[2])


def claims_other() -> None:
    cov = q(f"select yearOfLoss, round(avg((totalBuildingInsuranceCoverage=0)::int),3) from {CLM} where yearOfLoss>=2005 group by 1 order by 1")
    z = q(f"select count(*), round(avg((buildingDamageAmount is null)::int),3), round(avg((contentsDamageAmount>0)::int),3), "
          f"round(avg((amountPaidOnBuildingClaim>0)::int),3) from {CLM} where totalBuildingInsuranceCoverage=0")[0]
    pos = q(f"select round(avg((buildingDamageAmount is null)::int),3), round(avg((contentsDamageAmount>0)::int),3), "
            f"round(avg((amountPaidOnBuildingClaim>0)::int),3) from {CLM} where totalBuildingInsuranceCoverage>0")[0]
    put("claims.coverage_zero", zero_share_by_year={y: s for y, s in cov}, rows=z[0],
        zero_coverage={"building_damage_null": z[1], "contents_damage_positive": z[2], "building_paid_positive": z[3]},
        positive_coverage={"building_damage_null": pos[0], "contents_damage_positive": pos[1], "building_paid_positive": pos[2]})
    neg = q(f"select count(*) filter (where amountPaidOnBuildingClaim<0), count(*) filter (where exteriorWaterDepth<0), "
            f"count(*) filter (where interiorWaterDepth<0) from {CLM}")[0]
    put("claims.negative_values_are_valid", negative_paid=neg[0], negative_exterior_depth=neg[1], negative_interior_depth=neg[2])
    key = q(f"select count(*) filter (where censusGeoid is null), count(*) filter (where countyCode is null), count(*) from {CLM}")[0]
    put("claims.keys", null_geoid=key[0], null_county=key[1], rows=key[2])


# ------------------------------------------------------------------ policies


def policies() -> None:
    fy = q(f"select year(policyEffectiveDate), round(avg((foundationType='0')::int),2) from {POL} group by 1 order by 1")
    legal = dict(q(f"select foundationType, count(*) from {POL} group by 1 order by 1"))
    cy = q(f"select yearOfLoss, round(avg((foundationType='0')::int),2) from {CLM} where yearOfLoss>=2005 group by 1 order by 1")
    within = q(f"select yearOfLoss, round(median(buildingDamageAmount/totalBuildingInsuranceCoverage) filter (where foundationType='0'),3), "
               f"round(median(buildingDamageAmount/totalBuildingInsuranceCoverage) filter (where foundationType<>'0'),3) "
               f"from {CLM} where occupancyType in (1,11) and totalBuildingInsuranceCoverage>0 and yearOfLoss in (2012,2016,2018,2020) "
               f"group by 1 order by 1")
    occ_year = q(f"select year(policyEffectiveDate), occupancyType, round(avg((foundationType='0')::int), 3), count(*) from {POL} "
                 f"where occupancyType in (1, 11) and year(policyEffectiveDate) in (2010, 2013, 2016, 2019, 2021, 2023) group by 1, 2 order by 1, 2")
    claims_occ = q(f"select occupancyType, count(*), round(avg((foundationType = '0')::int), 4), round(avg((foundationType in ('1','2','3','4','5','6'))::int), 4) "
                   f"from {CLM} group by 1 order by 2 desc limit 12")
    put("policies.foundationType_0", counts=legal,
        policies_share_zero_by_year_and_occupancy={f"{y}/{int(o)}": [z, n] for y, o, z, n in occ_year},
        claims_by_occupancy_rows_share_zero_share_known={str(o): [n, z, k] for o, n, z, k in claims_occ}, share_zero_by_policy_effective_year={y: s for y, s in fy},
        claims_share_zero_by_loss_year={y: s for y, s in cy},
        median_damage_ratio_zero_vs_known_same_year={y: [a, b] for y, a, b in within})
    rc = q(f"select year(policyEffectiveDate), round(avg((buildingReplacementCost<1)::int),3), round(avg((buildingReplacementCost is null)::int),3) "
           f"from {POL} group by 1 order by 1")
    put("policies.replacementCost_zero", share_zero_and_null_by_year={y: [a, b] for y, a, b in rc})
    dt = q(f"select count(*) filter (where originalConstructionDate=date '1492-10-12'), "
           f"count(*) filter (where originalConstructionDate<date '1800-01-01' and originalConstructionDate<>date '1492-10-12'), "
           f"count(*) filter (where originalConstructionDate>date '2027-01-01'), "
           f"count(*) filter (where originalNBDate<date '1968-01-01'), count(*) filter (where originalNBDate>date '2027-01-01') from {POL}")[0]
    top = q(f"select originalConstructionDate, count(*) from {POL} group by 1 order by 2 desc limit 3")
    put("policies.dates", construction_1492_10_12=dt[0], construction_other_before_1800=dt[1], construction_after_2027=dt[2],
        nb_before_1968=dt[3], nb_after_2027=dt[4], most_frequent_construction_dates=[(str(d), n) for d, n in top])
    nb = q(f"select rolloverTransferCode, count(*), round(avg((originalNBDate>policyEffectiveDate)::int),3) from {POL} group by 1 order by 2 desc")
    put("policies.nb_after_effective", by_rollover_transfer_code={str(c): [n, s] for c, n, s in nb})
    c = q(f"select count(*) filter (where cancellationDateOfFloodPolicy is not null), "
          f"count(*) filter (where cancellationDateOfFloodPolicy < policyTerminationDate), "
          f"count(*) filter (where cancellationDateOfFloodPolicy < policyEffectiveDate), "
          f"quantile_cont(date_diff('day',policyEffectiveDate,policyTerminationDate),[0.01,0.5,0.99]) from {POL}")[0]
    inf = q(f"select sum(policyCount) filter (where policyEffectiveDate<=date '2017-07-31' and policyTerminationDate>date '2017-07-31'), "
            f"sum(policyCount) filter (where policyEffectiveDate<=date '2017-07-31' and "
            f"least(policyTerminationDate, coalesce(cancellationDateOfFloodPolicy, policyTerminationDate))>date '2017-07-31') "
            f"from {POL} where occupancyType in (1,11)")[0]
    put("policies.mid_term_cancellation", with_cancellation_date=c[0], cancelled_before_nominal_expiry=c[1], cancelled_before_effective=c[2],
        nominal_term_days_p1_p50_p99=c[3], single_family_in_force_2017_07_31_nominal=inf[0], single_family_in_force_2017_07_31_with_cancellation=inf[1])
    fl = dict(q(f"select numberOfFloorsInInsuredBuilding, count(*) from {POL} group by 1 order by 1"))
    oc = dict(q(f"select occupancyType, count(*) from {POL} group by 1 order by 1"))
    ed = q(f"select count(*) filter (where elevationDifference=9999), count(*) filter (where elevationDifference<-1000), "
           f"count(*) filter (where elevationDifference>1000), min(elevationDifference), max(elevationDifference) from {POL}")[0]
    put("policies.codes", floors_code_counts=fl, occupancy_counts=oc,
        elevationDifference={"equal_9999": ed[0], "below_-1000": ed[1], "above_1000": ed[2], "min": ed[3], "max": ed[4]})
    neg = q(f"select count(*) filter (where totalInsurancePremiumOfThePolicy<0) from {POL}")[0][0]
    put("policies.negative_premium_is_valid", rows=neg)


# ------------------------------------------------------------------ missingness diagnostics


def smd(a: pd.Series, b: pd.Series) -> float:
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def cramers_v(x: pd.Series, flag: pd.Series) -> float:
    t = pd.crosstab(x, flag)
    if t.shape[0] < 2 or t.shape[1] < 2:
        return 0.0
    from scipy.stats import chi2_contingency

    chi2 = chi2_contingency(t, correction=False)[0]
    return float(np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))))


def diagnose(name: str, df: pd.DataFrame, flag: str, num: list[str], cat: list[str]) -> None:
    m, o = df[df[flag]], df[~df[flag]]
    s = {c: round(smd(m[c].dropna(), o[c].dropna()), 3) for c in num if m[c].notna().any() and o[c].notna().any()}
    v = {c: round(cramers_v(df[c].astype(str), df[flag]), 3) for c in cat}
    mx = max([abs(x) for x in s.values()] + list(v.values()) + [0.0])
    put(f"missingness.{name}", n_missing=int(len(m)), share=round(len(m) / len(df), 4), smd=s, cramers_v=v, max_abs_effect=round(mx, 3),
        verdict="compatible with MCAR (all effects < 0.1)" if mx < 0.1 else "NOT MCAR: depends on observed variables (MAR or worse)")


def diagnostics() -> None:
    p = duckdb.sql(f"""select buildingReplacementCost<1 as missing, ln(greatest(totalBuildingInsuranceCoverage,1)) as log_cov,
        (upper(substr(ratedFloodZone,1,1)) in ('A','V'))::int as sfha, (occupancyType in (1,11))::int as single_family,
        coalesce(mandatoryPurchaseFlag,false)::int as mandatory, year(policyEffectiveDate) as year
        from {POL} where buildingReplacementCost is not null using sample reservoir(1000000 rows) repeatable ({SEED})""").df()
    p["missing"] = p["missing"].astype(bool)
    diagnose("policies.replacementCost_zero", p, "missing", ["log_cov", "sfha", "single_family", "mandatory"], ["year"])
    g = duckdb.sql(f"""select (censusGeoid is null) as missing, ln(greatest(totalBuildingInsuranceCoverage,1)) as log_cov,
        (upper(substr(ratedFloodZone,1,1)) in ('A','V'))::int as sfha, (occupancyType in (1,11))::int as single_family,
        yearOfLoss as year, state from {CLM} using sample reservoir(600000 rows) repeatable ({SEED})""").df()
    diagnose("claims.geoid_null", g, "missing", ["log_cov", "sfha", "single_family"], ["year", "state"])
    c = duckdb.sql(f"""select (totalBuildingInsuranceCoverage=0) as missing, (upper(substr(ratedFloodZone,1,1)) in ('A','V'))::int as sfha,
        (occupancyType in (1,11))::int as single_family, yearOfLoss as year from {CLM}""").df()
    diagnose("claims.coverage_zero", c, "missing", ["sfha", "single_family"], ["year"])


def impute_check() -> None:
    """Masked-value validation: can buildingReplacementCost be imputed better than a naive guess? (Only a feature-stage question:
    the clean table never contains imputed values.)"""
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = duckdb.sql(f"""select ln(buildingReplacementCost) as y, ln(greatest(totalBuildingInsuranceCoverage,1)) as log_cov,
        (upper(substr(ratedFloodZone,1,1)) in ('A','V'))::int as sfha, occupancyType as occ, year(policyEffectiveDate) as year,
        coalesce(numberOfFloorsInInsuredBuilding,0) as floors, coalesce(try_cast(foundationType as int),0) as fnd,
        coalesce(mandatoryPurchaseFlag,false)::int as mand, substr(censusGeoid,1,5) as cty
        from {POL} where buildingReplacementCost>=1 and totalBuildingInsuranceCoverage>0 and censusGeoid is not null
        using sample reservoir(600000 rows) repeatable ({SEED})""").df()
    d["cty"] = pd.Categorical(d.cty).codes
    rng = np.random.default_rng(SEED)
    test = rng.random(len(d)) < 0.2
    X = d.drop(columns="y")
    base = d[~test].groupby("year").y.median()
    pred_base = d[test].year.map(base).fillna(d[~test].y.median())
    lin_beta = np.polyfit(d[~test].log_cov, d[~test].y, 1)
    pred_lin = np.polyval(lin_beta, d[test].log_cov)
    gbm = HistGradientBoostingRegressor(random_state=SEED, max_iter=200).fit(X[~test], d[~test].y)
    pred_gbm = gbm.predict(X[test])
    rm = lambda p: float(np.sqrt(np.mean((d[test].y - p) ** 2)))  # noqa: E731
    r = {"baseline_year_median": rm(pred_base), "linear_on_log_coverage": rm(pred_lin), "gradient_boosting": rm(pred_gbm)}
    put("imputation.replacementCost", n=int(len(d)), test_share=0.2, rmse_log=r,
        rmse_reduction_vs_baseline=round(1 - r["gradient_boosting"] / r["baseline_year_median"], 3),
        typical_multiplicative_error_of_best=round(float(np.exp(r["gradient_boosting"])), 2))


def main() -> None:
    spike_scan()
    depth()
    claims_other()
    policies()
    diagnostics()
    impute_check()
    out = wb2.DATA_DIR / "audit" / "cleaning_evidence.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(EV, indent=1, sort_keys=True, default=str) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
