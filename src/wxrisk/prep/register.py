"""The cleaning register: why each rule exists, who defined the value, and what the data say.

Rules live in specs.py (what is done); this file says WHY, in terms a reviewer can audit, and `render()` writes
docs/CLEANING_REGISTER.md with the numbers filled in from the build records (`<table>.prep.json`) and from
scripts/41_cleaning_evidence.py (`audit/cleaning_evidence.json`). tests/test_prep.py fails if a rule in a spec has no entry here.

Origins
  provider-documented   the OpenFEMA data dictionary / FAQ states it; the evidence is a consistency check on our data
  provider-placeholder  the FAQ says '99XX' placeholders exist; the spike test finds which value in which column
  undocumented-code     the provider does not define the value; its data signature (spike + different profile) shows it is a code
  impossible-value      excluded by the provider's own definition of the field
  entry-error           sporadic implausible values with no signature: unrecoverable, so the value (never the row) is nulled
  structural            true by construction of the data (kept as is)
  design                our decision about what the clean table is
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import wb2
from .build import CLEAN_DIR
from .specs import SPECS

ORIGINS = ("provider-documented", "provider-placeholder", "undocumented-code", "impossible-value", "entry-error", "structural", "design")
IMPUTE_MIN_GAIN = 0.30  # masked-value RMSE reduction vs the naive baseline needed before imputation is even considered
IMPUTE_MAX_ERROR = 1.25  # ... and the typical multiplicative error of the imputer must not exceed this


@dataclass(frozen=True)
class Ctx:
    ev: dict
    sc: dict  # table -> sidecar report

    def n(self, table: str, col: str, field: str = "sentinel_or_out_of_range") -> int:
        return self.sc[table]["columns"][col][field]

    def rows(self, table: str) -> int:
        return self.sc[table]["rows_in"]


@dataclass(frozen=True)
class Rule:
    id: str
    table: str
    col: str  # output column name, or a derived column, or "-" for a table-level decision
    origin: str
    action: str
    why: Callable[[Ctx], str]
    quote: str = ""  # the provider's words, when the provider is the authority

    def __post_init__(self):
        assert self.origin in ORIGINS, self


def f(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


P, C = "policies_tx", "claims_nfip"
RULES: tuple[Rule, ...] = (
    # ------------------------------------------------------------------ policies_tx
    Rule("P01", P, "geoid", "design", "keep the row, geoid null, `geoid_missing` flag",
         lambda c: f"{c.n(P, 'geoid', 'src_null'):,} rows ({f(c.n(P, 'geoid', 'src_null') / c.rows(P), 2)}) have no usable block group. "
                   "They are real policies (state totals need them) so they are not dropped; block-group analyses filter on the flag. "
                   "Dropping them would also silently shrink the denominator of any state-level frequency. Never stringify the key first: "
                   "pandas `astype(str)` turns null into a phantom group that earlier scripts counted as a block group (stage 11)."),
    Rule("P02", P, "coverage_end", "provider-documented", "derived column: the earlier of the nominal expiry and the cancellation date",
         lambda c: (lambda m: f"{m['with_cancellation_date']:,} policies carry a cancellation date and {m['cancelled_before_nominal_expiry']:,} of them "
                    f"precede the nominal expiry, which is always effective date + 1 year (p1/p50/p99 = {m['nominal_term_days_p1_p50_p99']} days). "
                    f"Single-family policies in force on 2017-07-31: {m['single_family_in_force_2017_07_31_nominal']:,.0f} by nominal expiry, "
                    f"{m['single_family_in_force_2017_07_31_with_cancellation']:,.0f} with cancellations "
                    f"({f(1 - m['single_family_in_force_2017_07_31_with_cancellation'] / m['single_family_in_force_2017_07_31_nominal'])} lower). "
                    f"{m['cancelled_before_effective']:,} cancellations precede the effective date (zero-length coverage; they drop out of every in-force count).")(
                    c.ev["policies.mid_term_cancellation"]),
         quote="One other consideration is to check to see whether policies have had their policy period cancelled mid-term. "
               "The date of cancellation can be found in the new policy termination date field. (OpenFEMA FAQ)"),
    Rule("P03", P, "premium", "provider-documented", "no range rule: negative values are kept",
         lambda c: f"{c.ev['policies.negative_premium_is_valid']['rows']} negative premiums are refunds, not errors.",
         quote="Negative values indicate a refund. (dictionary, totalInsurancePremiumOfThePolicy)"),
    Rule("P04", P, "replacement_cost", "impossible-value", "values < 1 become null, `replacement_cost_missing` flag; no imputation",
         lambda c: (lambda d, m, i: f"A building cannot cost <= 0 dollars to rebuild: {c.n(P, 'replacement_cost'):,} rows. The share falls from "
                    f"{f(d['2009'][0])} (2009) to {f(d['2020'][0])} (2020), i.e. it follows reporting practice. Diagnostics: "
                    f"{m['verdict']} (largest effect {m['max_abs_effect']}: the zeros sit on low-coverage policies). "
                    f"Masked-value test: gradient boosting cuts the log-RMSE by {f(i['rmse_reduction_vs_baseline'])} against a per-year median, "
                    f"but its typical multiplicative error is x{i['typical_multiplicative_error_of_best']}, above the x{IMPUTE_MAX_ERROR} limit: "
                    f"{'imputation allowed at the feature stage' if i['rmse_reduction_vs_baseline'] >= IMPUTE_MIN_GAIN and i['typical_multiplicative_error_of_best'] <= IMPUTE_MAX_ERROR else 'NOT imputed; use the flag with coverage as the feature'}.")(
                    c.ev["policies.replacementCost_zero"]["share_zero_and_null_by_year"], c.ev["missingness.policies.replacementCost_zero"],
                    c.ev["imputation.replacementCost"]),
         quote="Estimated cost in whole dollars to replace the building as reported by the insurer. (dictionary, buildingReplacementCost)"),
    Rule("P05", P, "foundation", "undocumented-code", "'0' becomes null, `foundation_missing` flag (the unknown-share feature)",
         lambda c: (lambda e: f"The dictionary lists codes 1-6 only; '0' covers {c.n(P, 'foundation'):,} rows. Its share falls with the effective year "
                    f"({f(e['share_zero_by_policy_effective_year']['2009'], 0)} of 2009 policies, {f(e['share_zero_by_policy_effective_year']['2022'], 0)} of 2022, "
                    f"{f(e['share_zero_by_policy_effective_year']['2023'], 0)} of 2023), and the fall happens WITHIN the legacy occupancy code 1 "
                    f"({', '.join(f'{k.split(chr(47))[0]}: {v[0]:.0%}' for k, v in e['policies_share_zero_by_year_and_occupancy'].items() if k.endswith('/1'))}), "
                    f"while occupancy 11 has none ({', '.join(f'{k.split(chr(47))[0]}: {v[0]:.0%}' for k, v in e['policies_share_zero_by_year_and_occupancy'].items() if k.endswith('/11'))}), "
                    "so it is a filling-in process plus a change of coding scheme, not a property of buildings. "
                    f"Within the same loss year the unknown group has the higher damage ratio ({e['median_damage_ratio_zero_vs_known_same_year']}: unknown vs known), "
                    "but in claims that comparison is between occupancy-code families (see C10), so it cannot be read as 'unknown buildings are different'. "
                    "Note: code 3 is CrawlSpace, not 'elevated' (earlier inference in AGENTS.md trap 8 and script 30 was wrong).")(
                    c.ev["policies.foundationType_0"]),
         quote="Type of foundation of the building or structure. 1 - Slab 2 - Basement 3 - CrawlSpace 4 - Elevated Without Enclosure Post Pile or Pier "
               "5 - Elevated With Enclosure Post Pile or Pier 6 - Elevated With Enclosure Not Post Pile or Pier (dictionary)"),
    Rule("P06", P, "floors_code", "provider-documented", "kept as a code (1-6); out-of-set values would become null",
         lambda c: f"Not a floor count: 1 one floor, 2 two, 3 three or more, 4 split-level, 5 manufactured home, 6 townhouse. Observed codes "
                   f"{sorted(k for k in c.ev['policies.codes']['floors_code_counts'] if k != 'None')}, no value outside the set; "
                   f"{c.n(P, 'floors_code', 'src_null'):,} nulls flagged.",
         quote="Code that indicates the number of floors in the insured structure. 1 - One floor; 2 - Two floors; 3 - Three or more floors; ... (dictionary)"),
    Rule("P07", P, "elev_diff", "provider-documented", "9999 -> null (none in Texas); |x| > 1000 ft -> null, `elev_diff_missing` flag",
         lambda c: (lambda e: f"The dictionary reserves 9999 for 'not reported' ({e['equal_9999']} rows in Texas). {e['below_-1000']} values lie below -1000 ft "
                    f"(min {e['min']}), impossible for a floor-to-flood elevation difference: entry errors, unrecoverable, value nulled.")(
                    c.ev["policies.codes"]["elevationDifference"]),
         quote="The value of 9999.0 indicates the field is not reported and/or used for this policy (dictionary, elevationDifference)"),
    Rule("P08", P, "first_policy_date", "entry-error", "outside 1968-2027 -> null, `first_policy_date_missing` flag",
         lambda c: f"{c.n(P, 'first_policy_date'):,} dates fall before 1968 (the NFIA year; years 0001 and 1010 occur) or after 2027. Scattered, no repeated value: "
                   "entry errors. Separate finding, no rule: the first-policy date is later than the term's effective date for 4-6% of "
                   "new-business and renewal rows but almost never for transfers, so it is a booking convention, not an error "
                   f"({c.ev['policies.nb_after_effective']['by_rollover_transfer_code']}: rows, share by rollover/transfer code)."),
    Rule("P09", P, "built_date", "undocumented-code", "outside 1800-2027 -> null, `built_date_missing` flag",
         lambda c: (lambda d: f"{d['construction_1492_10_12']:,} rows hold exactly 1492-10-12, a placeholder (a single date; not in the dictionary). "
                    f"A further {d['construction_other_before_1800'] + d['construction_after_2027']} rows are scattered implausible years (entry errors). "
                    f"Total nulled: {c.n(P, 'built_date'):,}.")(c.ev["policies.dates"]),
         ),
    Rule("P10", P, "eff", "design", "null effective date would drop the row", lambda c: f"{c.n(P, 'eff', 'src_null')} rows affected.  A policy without a start date cannot be placed in time for any analysis."),
    Rule("P11", P, "n_policies", "design", "values < 1 would become null (guard)", lambda c: f"{c.n(P, 'n_policies')} rows affected; kept as a guard."),
    Rule("P12", P, "-", "design", "columns with `nulls=flag` keep the source null and add `<col>_missing`",
         lambda c: "Applies to zone, term, replacement_cost, foundation, floors_code, elev_diff, first_policy_date, built_date. The source null is not filled; "
                   "the flag makes missingness available as a feature (and as a stratifier in the write-up)."),
    # ------------------------------------------------------------------ claims_nfip
    Rule("C01", C, "id", "provider-documented", "uniqueness is asserted at build time", lambda c: "`id` is the provider's unique record id; the build fails on duplicates.",
         quote="Unique ID assigned to the record (dictionary)"),
    Rule("C02", C, "geoid", "design", "keep the row, geoid null, `geoid_missing` flag",
         lambda c: (lambda m: f"{c.n(C, 'geoid', 'src_null'):,} claims ({f(c.n(C, 'geoid', 'src_null') / c.rows(C), 2)}) have no block group. Missingness diagnostics: {m['verdict']}; "
                    f"largest effects {m['smd']} and {m['cramers_v']} (state, year): missing locations are not a random subset, so the rows are not "
                    "dropped from state-level counts and the flag stays available.")(c.ev["missingness.claims.geoid_null"])),
    Rule("C03", C, "depth_ext", "undocumented-code", "0 -> null, `depth_ext_missing` flag (also interior depth)",
         lambda c: (lambda e: f"The dictionary defines the depth in inches relative to the lowest adjacent grade and does not mention 0 as a code. "
                    f"Spike: {e['count_at']['0.0']:,} rows at 0 against {e['count_at']['1.0']:,} at 1 and {e['count_at']['2.0']:,} at 2. Profile: the median damage ratio at 0 is "
                    f"{e['median_damage_ratio_by_exterior_depth_class']['0']['median']} versus {e['median_damage_ratio_by_exterior_depth_class']['1-6']['median']} at 1-6 in: "
                    "a dry building would show less damage, not more. 0 is a default for 'not measured' (trap 2 in AGENTS.md), mixed with some true at-grade readings, "
                    "which cannot be separated; null + flag keeps the fact. The field is >95% null in every loss year up to "
                    f"{max(int(y) for y, v in e['null_share_and_zero_share_by_loss_year'].items() if v[0] > 0.95)} (not collected before), so depth analyses are limited to later years.")(
                    c.ev["claims.exterior_depth"])),
    Rule("C04", C, "depth_ext", "provider-placeholder", "999 -> null (also interior depth)",
         lambda c: (lambda s, e: f"The FAQ says 99XX values are placeholders. The spike scan over {s['claims._columns_scanned']} numeric columns finds 999 in both depth columns "
                    f"({e['count_at']['999.0']:,} rows against 0 and 12 at the neighbours); the median damage ratio at 999 is "
                    f"{e['median_damage_ratio_by_exterior_depth_class']['999']['median']} against {e['median_damage_ratio_by_exterior_depth_class']['49-998']['median']} at 49-998 in, "
                    "so it is not a real depth.")(c.ev["spike_scan_99xx"]["found"], c.ev["claims.exterior_depth"]),
         quote="'99XX' is often used as placeholder or default value. (OpenFEMA FAQ, partial sentence as retrieved)"),
    Rule("C05", C, "depth_ext", "entry-error", "> 600 in (other than 999) -> null",
         lambda c: f"Values above 600 in other than 999: {c.ev['claims.exterior_depth']['values_above_600_excluding_999']} (value, rows): 1000, 6000, 8000 look like "
                   "a mistyped extra digit; the true value is unrecoverable, so the value is nulled and the row kept."),
    Rule("C06", C, "depth_ext", "provider-documented", "negative values are kept",
         lambda c: f"{c.ev['claims.negative_values_are_valid']['negative_exterior_depth']:,} negative exterior and {c.ev['claims.negative_values_are_valid']['negative_interior_depth']:,} "
                   "negative interior depths are water marks below the lowest adjacent grade.",
         quote="The depth of water, in inches, ... relative to the lowest adjacent grade (dictionary)"),
    Rule("C07", C, "-", "provider-placeholder", "waterDepth is NOT carried into the clean table",
         lambda c: (lambda e: f"The dictionary warns that some measurements are in feet; {f(e['share_wd_times_12_close_to_exterior'], 0)} of rows with both depths agree with "
                    f"'feet' (x12) and {f(e['share_wd_close_to_exterior'], 0)} with 'inches'. The column also has a 99 placeholder "
                    f"({c.ev['spike_scan_99xx']['found']['claims.waterDepth']['count']:,} rows against {c.ev['spike_scan_99xx']['found']['claims.waterDepth']['neighbours']} at the neighbours). "
                    "The unit cannot be recovered per row, so the column is excluded; use exteriorWaterDepth.")(c.ev["claims.waterDepth_units"]),
         quote="Depth of flood water in inches. Note: there are instances where measurements were provided in feet. (dictionary, waterDepth)"),
    Rule("C08", C, "cov_building", "structural", "0 is kept as a true value",
         lambda c: (lambda e: f"{e['rows']:,} claims have building coverage 0: {f(e['zero_coverage']['building_damage_null'], 0)} have no building damage and "
                    f"{f(e['zero_coverage']['contents_damage_positive'], 0)} have contents damage (positive-coverage claims: {f(e['positive_coverage']['building_damage_null'], 0)} and "
                    f"{f(e['positive_coverage']['contents_damage_positive'], 0)}). These are contents-only policies, not missing values. "
                    "A damage / coverage ratio is undefined for them, so ratio analyses filter `cov_building > 0` themselves (an earlier version of this rule nulled the zeros as a placeholder; the evidence showed that was wrong).")(
                    c.ev["claims.coverage_zero"])),
    Rule("C09", C, "paid_building", "provider-documented", "no range rule: negative values are kept",
         lambda c: f"{c.ev['claims.negative_values_are_valid']['negative_paid']} negative payments.",
         quote="In some instances, a negative amount may appear which occurs when a check issued to a policy holder is not cashed and has to be re-issued (dictionary)"),
    Rule("C10", C, "foundation", "undocumented-code", "'0' -> null, `foundation_missing` flag (equivalent to the legacy occupancy codes)",
         lambda c: (lambda o: "In claims '0' is a deterministic function of the occupancy code: " + "; ".join(
             f"occupancy {int(float(k))}: {100 * v[1]:.0f}% '0', {100 * v[2]:.0f}% known (n={v[0]:,})" for k, v in sorted(o.items(), key=lambda kv: float(kv[0]))[:8])
             + ". Legacy codes (1, 2, 3, 4, 6) never carry a foundation and the two-digit codes (11, 12, 15, 16, 18, ...) always do (checked year by year for codes 1 and 11: every loss year 2000-2023). "
             "`foundation_missing` is therefore the same information as the occupancy-code family, and a damage-ratio difference between 'unknown' and 'known' is a difference between "
             "code families (national single-family claims, state-month and depth-bin fixed effects: occupancy 11 has 0.74x the loss ratio of occupancy 1 before any foundation effect; scripts/30 --sections occ). "
             "Foundation comparisons are only possible among the two-digit-coded claims.")(c.ev["policies.foundationType_0"]["claims_by_occupancy_rows_share_zero_share_known"])),
    Rule("C11", C, "-", "design", "floodWaterDuration and floodCharacteristicsIndicator are not carried",
         lambda c: "floodWaterDuration is ~0 everywhere (AGENTS.md trap 2) and carries a 999 placeholder; floodCharacteristicsIndicator is 99% null. "
                   "Neither is usable as a feature."),
    Rule("C13", C, "-", "design", "columns with `nulls=flag` keep the source null and add `<col>_missing`",
         lambda c: "Applies to county, geoid, cause, depth_ext, depth_int, damage_building, damage_contents, paid_building, zone, foundation. "
                   "The source null is not filled."),
    Rule("C14", C, "depth_int", "undocumented-code", "same rules as depth_ext (C03-C06): 0 and 999 -> null, > 600 -> null, negatives kept",
         lambda c: (lambda e: f"{c.n(C, 'depth_int'):,} values nulled in interior depth ({c.n(C, 'depth_ext'):,} in exterior). "
                    f"999 appears {c.ev['spike_scan_99xx']['found']['claims.interiorWaterDepth']['count']:,} times in interior depth "
                    f"(neighbours {c.ev['spike_scan_99xx']['found']['claims.interiorWaterDepth']['neighbours']}).")(None)),
    Rule("C12", C, "loss_date", "provider-documented", "text -> date; a null date would drop the row",
         lambda c: f"{c.n(C, 'loss_date', 'src_null')} rows affected; the time part is a default and is discarded.",
         quote="When time is not specified in the field, it will default to T00:00:00.000Z (dictionary)"),
)

DECISION = f"""\
## How a rule is decided

1. **The provider decides when it has spoken.** The OpenFEMA data dictionary and FAQ define codes, units and placeholders
   (for example "99XX is often used as placeholder", "negative values indicate a refund", "9999 = not reported"). We follow them and
   check that our data are consistent with the statement.
2. **A value the provider does not define is tested, not guessed.** *Spike test*: the count at v is >= 20 and >= 5x the mean count at v-1 and v+1
   (a code is entered on purpose, an error is not). *Profile test*: the outcome at v (damage ratio, share by reporting year) must differ from its legitimate
   neighbours. Both positive: an undocumented code, handled as unknown (null + flag). Spike but a profile that fits a true value: structural, kept.
3. **Sporadic implausible values with no signature are entry errors.** The true value is unrecoverable, so the *value* is nulled (never the row) and counted.
   A row is dropped only when it is unusable for every analysis (no event date), and the count is reported.
4. **Missingness diagnostics decide whether complete-case analysis is safe.** For a missing flag we compare observed covariates between missing and observed rows
   (standardised mean difference, Cramer's V). All effects < 0.1: compatible with MCAR, dropping the rows is acceptable. Otherwise the rows stay and the flag is
   a feature.
5. **Imputation never enters a clean table.** It is a feature-stage choice, fitted on the training period only, and allowed only when a masked-value test
   beats a naive baseline by at least {IMPUTE_MIN_GAIN:.0%} in RMSE *and* the typical multiplicative error is at most x{IMPUTE_MAX_ERROR}. Imputed values are estimates; the flag is kept.
6. **Every rule has an entry below**, and the build tests fail when a rule has none.
"""


def _load() -> Ctx:
    ev = json.loads((wb2.DATA_DIR / "audit" / "cleaning_evidence.json").read_text())
    sc = {n: json.loads((CLEAN_DIR / f"{n}.prep.json").read_text())["report"] for n in SPECS}
    return Ctx(ev, sc)


def render(out: Path) -> None:
    c = _load()
    md = ["# Cleaning register", "",
          "Generated by `python scripts/40_prep.py register` from `src/wxrisk/prep/register.py`, the build records and "
          "`scripts/41_cleaning_evidence.py`. Do not edit by hand.", "",
          "Sources: OpenFEMA data dictionary for NfipClaims v3 and NfipPolicies v3 (fetched by `scripts/fetch/fetch_openfema_dictionary.py`) and the OpenFEMA FAQ "
          "(https://www.fema.gov/about/openfema/faq). The FAQ was read through a web-fetch tool that summarises pages, so its quotations are partial.", "",
          DECISION]
    for t in SPECS:
        r = c.sc[t]
        md += [f"## {t}", "", f"{r['rows_in']:,} rows in, {r['rows_out']:,} out; dropped by rule: {r['dropped_null_or_malformed_key']}.", "",
               "| Id | Column | Origin | Action | Why (evidence) |", "|---|---|---|---|---|"]
        for rule in sorted((x for x in RULES if x.table == t), key=lambda x: x.id):
            why = rule.why(c).replace("|", "/").replace("\n", " ")
            q = f' Provider: "{rule.quote}"' if rule.quote else ""
            md.append(f"| {rule.id} | `{rule.col}` | {rule.origin} | {rule.action} | {why}{q} |")
        md.append("")
    md += ["## Not retrievable", "",
           "- The OpenFEMA FAQ PDF link from the dataset page redirects to the HTML FAQ in a browser and returned HTTP 403 to our downloader (`wxrisk.fetch`); "
           "the HTML page was read with a web-fetch tool instead. The per-dataset 'Data Fields' pages and the legacy manuals the FAQ points to were not read.", ""]
    out.write_text("\n".join(md))
