# Modeling decisions: what the evidence settles, what it does not (2026-10-26)

Synthesis of `RESULTS.md` stages 0-15. Nothing here is new evidence; each row points at the stage that carries it. "Settled" means the
result survived the year-grouped / event-clustered interval and at least one robustness check. Everything else is labelled as a proposal.

## 1. One-paragraph answer

The index is a **three-layer chain**: hazard q (what the atmosphere does) -> exposure / impact tau (who and what is there, how it fails) -> loss psi.
The evidence supports building the **hazard layer on objective labels with a simple learner on top of a climatological baseline**, the
**flood line at block-group x rain-episode** resolution as the main product, and the **impact layer from NFIP damage ratios**, not from dollar thresholds.
The evidence does **not** yet cover the one thing the project is named for: every learned result uses *observed* fields in place of forecasts
(perfect-prog). The next decision is therefore an experiment, not a model choice: replace observed rain with ensemble-derived rain in the pilot table.

## 2. Settled

| # | Decision | Evidence | Strength |
|---|---|---|---|
| S1 | Always report skill against a climatological baseline with year-grouped / event-clustered bootstrap; split by year | stages 0, 2, 4 | rule |
| S2 | Baseline p0 = county rate x month rate; learned part is a **correction on top of p0**, never a free-standing cluster rate (rate read-out loses to p0 by 0.013 BSS) | stage 2 | strong |
| S3 | p0 is itself not optimal: about 0.005 BSS of every SCS result is static recalibration (lat/lon/month/logit p0), subtract it when attributing gain to weather | stage 4 | strong |
| S4 | Equal-weight ensemble vote is already near-calibrated; recalibrating it (isotonic, per band) ties on Brier. A pooled linear combiner with spread features ranks better (AUC 0.990) but has worse Brier (reliability 5x worse) | stage 0 | one train year, one test year, one lead: weak on generality |
| S5 | Learner for SCS: logistic regression or shallow boosting on scale features, **not K-means** (K-means reads out half the linear model: 0.034 vs 0.064). Depth 2 to 8 does not move test BSS, extra capacity only overfits | stages 3-4 | strong |
| S6 | K-means survives only as a **quantiser** (no natural clusters: silhouette 0.09, permutation null 0.05), justified by out-of-sample label information, not as a discovery of regimes | stage 13 | strong |
| S7 | SCS ceiling: tier B 0.068, tier C 0.108 BSS (perfect-prog); the 150 km neighbouring-report oracle reaches 0.138. A genuine beyond-T-3D tier (A3) is about 0.01. More capacity or finer encoding cannot break this | stages 3, 4 | strong |
| S8 | Labels: train the hazard layer on **objective / smooth labels** (MESH, warnings, intensity thresholds, QPE); never on a dollar threshold. Dollar labels have AUC lift 55-60% of occurrence labels and Spearman 0.03-0.15 with intensity | stages 5, 6 | strong |
| S9 | Flood line scope: inland (NFIP cause 2/3/4), tropical cyclone windows excluded, **block group x rain episode, Texas, 2010-2023** | stages 7, 8 | strong, one state |
| S10 | Flood metric: **within-event AUC** (county-level model is 0.5 by construction), plus overall AUC and AP. Absolute probabilities are not comparable across years (positive rate 0.58% train vs 0.22% test) | stage 8 | rule |
| S11 | Flood skill comes from two inputs only: **exposure** (policy count, sfha share, coverage; +0.147) and **the block group's own rainfall** (+0.139). Within-event AUC 0.786 [0.765, 0.805], worst CV fold 0.751 | stage 8 | strong |
| S12 | Flood learner: **logistic regression on `PILOT_PLAN`** (0.7858) equals boosting (0.7859). Pick the linear model as default, keep boosting as a check | stages 8, 14 | strong |
| S13 | Scaling is a pipeline step: standardise for linear / distance learners, raw for trees, fit on training rows only, never year-wise. Exposure bins are a small candidate gain (+0.0028 [+0.0003, +0.0051]) for linear learners only | stages 12, 14 | small, confirmed |
| S14 | Cleaning: model on `clean/` tables (cancellations, phantom keys, undocumented codes). Magnitudes moved 1-5%, the deepest compliance bin more | stage 11 | rule |
| S15 | Impact layer: model **damage ratio = building damage / coverage** against exterior depth, not raw dollars. Elevated buildings: 0.27x (geometric means) / 0.47x (means, Texas); say which one. Group-specific retransformation factors (Duan 1.96-2.86) | stages 9, 12, 14 | strong for elevated; others weak |
| S16 | Exposure dynamics (take-up) are modelled as **season + decaying event pulse (about 4-5 quarters) + age-dependent lapse**; economic covariates add nothing stable | stage 10 addenda | moderate, one state |
| S17 | Sparse-to-dense: with point observations at >= 2% of cells, **use the observations alone** (kriging beats inverse-distance by +0.04 RMSE skill at 8%). An advected old field is a better prior than persistence but does not add to observations | stage 15 | moderate: 8 events |

## 3. Tested and rejected (do not rebuild without a new reason)

| Idea | Result | Stage |
|---|---|---|
| Vote by wavelet scale | ties a random-grouping control; variable-grouped control is better by 0.011-0.016 | 3 |
| Motion features as SCS predictors | C vs C-without-motion differs by about 0.003 | 3 |
| Wavelet encoder, terrain, location, neighbourhood rain in the flood pilot | all intervals span 0 on top of L2 | 8 |
| Exposure composition (new vs mid-age policy share) | +0.001 against fold spread 0.07 | 10 addendum |
| Group-aware scaling (within-episode z, county x month) | not demonstrable (+0.004 [-0.000, +0.009]) | 12 |
| `foundation_missing` as a building property in claims | it is the occupancy-code family flag | 13 |
| Flood zone share removal as explanation for terrain null | flood-zone dropped, terrain still adds nothing | 8 |
| Advected prior + observations for squared error at >= 2% density | worse than observations alone, also with kriging | 15 |
| Economic covariates for NFIP purchases (rates, prices, permits, sales, premium) | sign flips with training window | 10 |

## 4. What each layer should be (proposal built on the settled rows)

**Hazard layer q.**
- Flood: block-group rainfall (log1p) as the main driver; the daily-gridMET value at the centroid is what was tested. Forecast mode replaces it with ensemble-derived values (see section 5).
- SCS: p0 x correction from logistic regression / shallow boosting on county-anomaly environment features (tier B / C), fitted on an objective label (warning, MESH-derived, intensity-thresholded report). Tier A3 carries no signal, so lead times beyond three days need forecast fields or climate indices, not a better encoder.
- Output is a **ranking-quality probability**. Calibrate the level separately (next bullet).

**Level vs shape (proposal, untested).** Within-event AUC is stable across eras while the absolute rate swings by an order of magnitude (positive rate 0.58% vs 0.22%; Texas claims from <500 to 10k a year). So split the probability into
an **episode-level factor** (how many claims this event produces, a count-type model on episode rainfall and insured stock) times a **within-event share** (what the logistic model ranks). The within-event part is what S11-S12 validated; the level part has no validated model yet.

**Impact layer tau.**
- Damage ratio by depth, with elevated indicator, occupancy-family flag and state-month / event controls (S15). Basement items are partly uncovered, so lower basement ratios do not mean safer.
- Foundation ratios are occupancy-11 estimates; the elevated indicator replicates in both occupancy families.
- Treat building priors as a vulnerability prior, not a forecast input (stage 9 decision). They have **not** been tested as pilot-table features (open item below).

**Exposure.** Policy counts are known ex ante (day before the episode). Take-up given the realised hazard follows S16; do not forecast take-up from economics.

**Loss psi.** No label supports it in CO/NM/AZ (NFIP penetration 0.74%, RMA 69 county-months). In Texas the NFIP claim amount is the usable loss series. Wind loss needs EAGLE-I (not obtained).

**Ensemble / forecast handling.** Start from the equal-weight vote (S4). Use neighbourhood exceedance probability for position error (stage 8 note: point beats neighbourhood only for *observed* rain). Add spread features only through a per-cell / per-block mechanism, never a pooled linear combiner.

## 5. Not settled (each is a concrete experiment)

| Open item | Why it matters | Cost / blocker |
|---|---|---|
| **O1. Forecast-mode pilot (done, stage 16: 40% of the rain gain survives, ensemble extras add nothing; next: finer/longer-lead/more members)**: swap observed gridMET rain for IFS ENS (or HRES) rain at the block-group point; compare point value, ensemble mean, vote probability, NEP at 1-3 day leads, same within-event AUC | Everything learned is perfect-prog; neighbourhood features and the encoder may matter once rain has position error | Need ENS precipitation for the 44k Texas episodes (download size to check); highest priority |
| O2. Level model (episode-level claim count / rate) | The ranking model cannot give usable absolute probabilities across eras | Needs a model of episode claims on rain and stock; positives are dominated by 2015-2016 |
| O3. Flood history (claims 2000-2009, prior 3 years) and building priors (elevated share, known-foundation shares + unknown share, post-FIRM share) in the pilot table | History: +0.026 observed / +0.051 forecast mode (stage 17); building priors still untested | History is cheap and ready (script 49 `history`); building priors follow script 38 |
| O4. FEMA NFHL + HAND flood-zone features | Terrain centroid features added nothing; zone geometry may | Download + processing; only after O1 |
| O5. Clean-input pilot table as default for stages 8-10 | Stored table carries 376 changed rows and the old 366-day term | Regenerate wavelet and terrain features for those rows |
| O6. Impact-layer model fit end to end (depth distribution given rain, then damage ratio) | Only the pieces exist; rain-to-depth is not modelled | Exterior depth exists mostly from 2016-2017 onward (Harvey-heavy) |
| O7. A second state | All flood results are Texas; reporting regimes differ | Louisiana, Florida, South Carolina have usable NFIP volume |
| O8. SCS objective label retrain (MESH 2021-2023, warnings) | Decision S8 says do it; stage 5 only compared labels | MESH only 3 seasons; warnings drift (5x growth 2002-2022) |
| O9. Sparse-to-dense follow-ups: ensemble of priors, non-stationary error model, risk (not weather) extrapolation to a neighbouring cell | The ensemble-of-priors run was started in conversation and interrupted; it has not been run | Expected small after stage 15; run only on a specific reason |

## 6. Recommended order

1. ~~O1~~ done (stage 16). Follow-ups: a finer or more-member forecast than 5 members at 0.25 deg, other leads, and 2020-2023 with IFS ENS.
2. **O3 + O5** together (cheap, close two stage-8/9 loose ends on one rebuilt table).
3. **O2** (level model) once O1 fixes which rain feature is the driver.
4. **O4, O7, O8** by data availability.

## 7. What not to claim yet

- No forecast skill for the flood or SCS product (all perfect-prog or observed-rain).
- No payout or loss forecast (occurrence only).
- No evidence for scale decomposition or motion as predictors; they are valid tools for the sparse-to-dense and motion-estimation questions only.
- No cross-state or cross-era absolute probability.

## 8. Data needs from the information audit (stage 17)

| Priority | Data | Evidence | Status |
|---|---|---|---|
| 1 | Flood history features (claims 2000-2009, rolling prior claims, per-policy rates) into the pilot table | +0.026 observed, +0.051 forecast mode | in hand; add to the table and the model |
| 2 | Building-level susceptibility: share of buildings / area in the FEMA NFHL flood zone, HAND or distance to channel at building footprints, imperviousness, soils | centroid terrain is null in both modes; history (a proxy) already gives +0.026, so the extra from a real layer is unknown | needs download (NFHL, building footprints, NLCD) |
| 3 | A finer forecast for forecast mode: HRRR / HREF / NAM-nest hourly rain, or more members; Stage IV or MRMS hourly for sub-daily intensity | forecast mode loses 0.024 (resolution) + 0.045 (error); outcome oracle leaves +0.085 | needs download; sub-daily intensity in observed mode is not supported by the miss analysis (3.9% of last-ranked positives have cell rain < 10 mm) |
| 4 | More outcome data: a second state (LA, FL, SC) and later years | test is 497-1015 episodes, intervals +-0.03; small-exposure block groups dominate misses | NFIP claims and policies are already national |
| 5 | Soil moisture beyond 14-day rain | antecedent rain was null in stage 8 | untested; low prior |
