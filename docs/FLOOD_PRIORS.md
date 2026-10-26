# Flood prior features audit (building structure / foundation / flood zones)

Audit date 2026-10-23. Reproducible script: `scripts/30_building_attribute_audit.py`, with output archived in
`$WXRISK_DATA/audit/building_attrs.txt`. This page records **which data exist, how good they are, and which hypotheses were tested**;
it is not a model result. For model results see `RESULTS.md`.

## Decision (confirmed by the user)

Building-structure information is used only as **auxiliary labels / vulnerability priors**, not as forecast inputs. Reason: how long water stood,
seepage, and where equipment was placed cannot be determined even by claims adjusters, and the data has no reliable field for them; all we can
confirm is that there is "possibly a connection".

## What the data has and does not have

| Information wanted | NFIP field | Status |
|---|---|---|
| Foundation type | `foundationType` (policies and claims) | Present, but largely "unknown" before 2022 (see below) |
| Whether elevated | `elevatedBuildingIndicator` (policies, claims) | Complete, robust |
| Basement / enclosure / crawlspace | `basementEnclosureCrawlspaceType`, `enclosureTypeCode` (policies only) | Fill rate 40% / 26%, not directly usable |
| Lowest floor relative to base flood elevation | `elevationDifference`, `lowestFloorElevation`, `baseFloodElevation` (policies only) | Fill rate about 17%, incomplete |
| Number of floors, occupancy, year built | `numberOfFloorsInInsuredBuilding`, `occupancyType`, `postFIRMConstructionIndicator`, `originalConstructionDate` | Mostly complete |
| Building purpose (main house / garage / apartment…) | `buildingDescriptionCode` (policies only) | 81% |
| **Wood / brick / mixed construction** | **None** | `construction` is only an "under construction" boolean (0.25% true) |
| How long the water stood | `floodWaterDuration` (claims) | Almost all 0, **unusable** |
| Seepage, backflow-protection design, equipment location | None | Not present in the public data |

`foundationType` codes (**inferred from cross-tabs, not from an official dictionary**; verify before use): 0 unknown; 1 no basement / slab; 2 basement (strongly
overlapping with "finished / unfinished basement"); 4–6 the various forms of elevated (93–95% have `elevatedBuildingIndicator` true); **3 is a crawlspace** per the OpenFEMA dictionary (65% elevated; mixed class, corrected 2026-10-24).

## Interpretation hypotheses tested

All results come from `scripts/30` (Texas single-family claims from 2010 on; "loss ratio" = building damage / building coverage; records with exterior water depth 0 removed).

| Hypothesis | Result | Verdict |
|---|---|---|
| **A** Elevated buildings lose less at the same water depth | After controlling for depth bin and year, about **0.27×** a slab building (Texas); **0.32** nationally with state-month fixed effects; monotone across depth bins | **Holds, the most robust** |
| **B** Elevated buildings have a lower interior/exterior depth ratio | Median 0.35 (elevated) vs 0.56 (slab) | Direction consistent; the reference level of interior depth may differ, so interpret with care |
| **C** Basements are more vulnerable (backflow) | Only 62 Texas basement claims, untestable; nationally with state-month fixed effects basements are **0.43×** slab, and state-by-state estimates in 14 states are all below 1 (0.26–0.79) | **Not supported**. The reported loss is lower, but see below |
| **D** "Unknown" foundation is just a reporting habit, unrelated to the building | Texas, same years 2017–2021: median loss ratio 0.39 for the unknown group vs 0.32 for known slab | **Does not hold**, but the contrast is confounded: in claims 'unknown' is exactly occupancy code 1 (legacy) and known is code 11; code 11 has 0.74× the loss ratio of code 1 with state-month and depth-bin fixed effects (`scripts/30 --sections occ`). The elevated effect replicates in both families (0.42×, 0.34×) |
| **E** Year/event effects are smaller than structure effects | Annual median loss ratio of slab buildings 0.04–0.35 (highest in 2017, Harvey) | **Does not hold**. Year/event must be controlled |
| **F** Basement equipment is flooded together with the basement (building-design prior raised by the user) | National single-family: basement losses are nearly flat (about $10–20 k, ~$10 k already at 0–3 inches of interior depth), while slab rises to about $100 k with depth | Consistent with a threshold-type loss shape, but **cannot be distinguished** from "NFIP covers only limited basement items and caps payouts"; the commercial vs residential comparison has no data (non-residential claims almost never have a foundation type) |

**Limits on interpreting C/F**: NFIP coverage of basements is limited, so "loss / coverage" understates basement loss, and "basement losses are lower" must not be read as
"basements are safer". Basement samples are concentrated in states such as NJ, NY, PA, IL, MA, MI, CT, OH, WI (50–63% of claims); in NC and KY the share of "foundation type = 2"
is only 2–18%, which may mean something else, and the code needs to be checked.

## Data-quality traps (these directly affect feature construction)

1. **The share of unknown `foundationType` changes drastically by year**: Texas policies 86% in 2010, 58% in 2016, 32% in 2020, about 0% from 2022. Using it directly
   would teach the model "year" as "foundation" and worsen the known era drift. **Only compute shares within the known subset and keep the "unknown share" as a separate feature**, or use 2022 onward only.
2. **Exterior/interior water depth of 0 is unreliable** (the 0 bin has large losses): treat as missing.
3. **`floodEvent` covers only about 81% of records and only 72 events** (essentially named hurricanes), so it cannot serve as an event fixed effect; we use "state × year-month of loss" as a proxy.
4. Slab buildings are rare in basement-heavy states, so the reference group there is thin.

## Proposed prior features (not yet built)

| Feature | Granularity | Source | Usage |
|---|---|---|---|
| Elevated share, `elevationDifference` distribution, number of floors, post-FIRM share | Block group (weighted by in-force policies) | Policies | Use directly; follow `flood_pilot.exposure_matrices` |
| Foundation-type shares (slab / basement / elevated) + **unknown share** | Block group | Policies | Use only shares within the known subset |
| Threshold-type loss shape (about $10 k as soon as water gets in, plus a slowly depth-increasing part) | Global or per state | Claims | A shape prior on the loss side; not written up as causal |
| Basement × small-residential interaction | Block group | Policies | Flag separately; "basement × non-residential" has no data at present, leave it empty, **do not fill it from the former** |
| Flood-zone area share, SFHA / 500-year-zone share, BFE | Block group | FEMA NFHL (**not USGS**) | To download |
| HAND (height above nearest drainage) statistics | Block group | NOAA/ORNL HAND (based on USGS 3DEP + NHDPlus); Texas also has a 3 m version (CC BY 4.0) | To download; reflects the river network better than the local relative height in `27_terrain_features.py` |

Policies already carry `ratedFloodZone` / `floodZoneCurrent`, and `n_sfha` in `expo_2010_2023.npz` is already a policy-level SFHA exposure;
but "inside the SFHA" is subject to selection bias from mandatory-purchase rules, whereas the area share from NFHL polygons is independent of insurance-buying behaviour, which is its extra value.
USGS flood inundation maps (FIM) cover only a few gaged river reaches and are not suited to statewide modelling.

## To do (in order)

1. Redo C and F with absolute damage amounts (instead of loss / coverage) to avoid the confound from the scope of coverage.
2. Build block-group-level building exposure summaries (elevated share, known-subset foundation shares, unknown share), attach them to `flood_pilot_table`, and run an ablation
   with and without the features (year-blocked cross-validation, test 2017–2023).
3. Download NFHL and HAND and add the flood-zone features.
4. Verify the `foundationType` codes and the anomalous shares in NC/KY.

## Candidate features (unproven selection and compliance proxies; stage 10)

These come from reasoning plus weak statistics, not from a proof. They live in the candidate pool and enter a model only through an ablation that beats the year-grouped bootstrap interval.

| Candidate | Source | Why it might help | Why it is only a candidate |
|---|---|---|---|
| Compliance tier T0–T3 (post-FIRM x elevated x SFHA) | claims / policies | T0 loses 0.05–0.24× of T2 at equal depth | elevation is geometry too; materials, equipment, openings unobserved |
| Block-group take-up rate (policies / ACS housing units) | policies + `scripts/34` | encodes who selects into NFIP; bimodal across block groups | bimodality is mostly Gulf-coast geography, not owner type |
| Inverse-probability weight from a take-up model | policies | push the insured sample toward the whole housing stock | no instrument; selection model is not identified |
| Zone-change flag (`ratedFloodZone` vs `floodZoneCurrent`) | policies | newly mapped owners did not choose to insure | only 8% flagged mandatory; not clean |
| Latent-class mixture (professional / frightened / mandated owners) | model | explains the bimodal take-up | classes cannot be identified without anchors |
| Post-incident take-up (local claim intensity of the last big event, years since) | claims + policies, `scripts/35` | Harvey: hit block groups bought 12-20 points more than the control, decaying with half-life ~2.5 yr | one event; control also moved; not per-household |
| Loss inside a declared (known) event | claims + FEMA declarations, `scripts/36` | known-event losses are heavy-tailed (mean $74k vs $22k) | ex-post: label/stratifier only, leakage if used as input |
| Post-event buyer share (NB within 24 months after last declared county event) | policies + declarations, `scripts/36` | exposure composition and selection signal, ex-ante | county-wide declarations too coarse; Harvey test was null. Local (tract) version: raw association +1.80 vanishes to +0.10 [-0.49, +2.80] after hazard controls, placebo-consistent: no detectable effect for Harvey (18 counties), but Imelda 2019 gives +1.72 [+0.10, +3.72] over a placebo range of [-0.59, +0.78] (6 counties): mixed, mechanism (word of mouth vs hazard vs outreach) untested |
| Expected stock and new-purchase share from a stock-flow model (season + decaying event pulse + age lapse) | policies + claims, `scripts/37` | exposure composition given the realised hazard; +0.18 [+0.13, +0.25] one-step skill over season-only (2020-2023) | pulse not learnable from pre-2017 events (2016Q4 origin: +0.02 [-0.01, +0.05]); regime break at Risk Rating 2.0 untested; economic covariates (rate, home value, permits, premium, home sales) gave no stable gain (permits and sales change sign with the training window) |
| Exposure composition in the pilot table (`new12_share`, `mid_share`) | policies, `scripts/38` | stock-flow ages of the insured population | tested: within-episode AUC 0.7839 -> 0.7829 (test), CV +0.0012: no gain. `mand_share` drifts by year (0.7% to 11.6%) and is excluded |
