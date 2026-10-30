# Model assumption audit and target-metric specification (revised 2026-10-29, after reading Impact Forecasting.pdf)

**Revision note.** The first version of this audit took `DESIGN.md` as the task definition (a probability issued every day, judged by Brier skill). `DESIGN.md` describes the
earlier video-encoder testbed on the coarse 64x32 ensemble grid. The task statement for the current work is `Impact Forecasting.pdf` (the group's slides): the product is a
**ranking of units by forecast impact within an event**, and the **probability is an auxiliary product** (owner's decision, 2026-10-29). The audit below is redone for that task.
Numbers: `scripts/73_model_assumption_audit.py` (checks that survive) and `scripts/74_ranking_and_probability_product.py` (new), stage 35 in `RESULTS.md`.

## 1. The task

**From the slides.** Before an event a hazard forecast q(w) is issued; emergency managers send resources to a **limited number of counties**; the event happens; losses are recorded.
Goal: for every county, forecast its loss from hazard forecasts, **used to rank counties**. Unit: county s x event t (a hurricane, flood, wildfire or heat wave), 2000-2023. Loss Y (SHELDUS,
inflation-adjusted dollars), hazard intensity W, exposure and vulnerability X, climatology C, adaptation A (for floods: NFIP policies, Community Rating System). Impact model
tau(w; x, c) learned from **observed** hazards, then chained with the **forecast** q(w); errors compound. Evaluation: learn from events 2000-2016, test on 2017-2023, for each test event
forecast every county's loss from pre-event information only; check the impact model on simulated losses with a known answer. Open question on the slides: county level, or a spatial grid.

**Decisions of 2026-10-29.**
1. **Primary product**: ranking of units inside an event, from pre-event information.
2. **Probability is kept as an auxiliary product**, to answer the emergency manager's doubt: "you rank this county first, but surely the others are not all 0%". It shows the level
   behind the ranking (what the others look like), not a separate forecast to be judged on its own.
3. Event-extreme scope (storm and tropical floods together), candidate group TX + LA + FL (stages 26, 27).
4. Unit stays block group (the grid-like answer to the slides' open question), aggregated to counties for the county product.

## 2. The assumptions of the pilot, re-evaluated for this task

| # | Assumption (where) | Evidence | Verdict for the ranking task |
|---|---|---|---|
| A1 | **Event = county rain episode** from observed rain (county 1-day maximum >= 30 mm, gaps <= 1 day merged); GEFS days = day before an episode | trigger 20 / 30 / 50 mm: within-county AUC 0.766 / 0.786 / 0.802; 20 mm adds 40% more episodes and 2.8% more positives | **Acceptable**: the slides give the event (the forecast exists before it). Weakness: events are identified from observed rain, not forecast; the headline depends on the trigger, so report 20 / 30 / 50 |
| A2 | Rows = block groups with >= 1 NFIP policy at the episode start; no subsampling | rate rises from 0.07% (<= 2 policies) to 0.74% (> 99) | Acceptable: loss needs something insured; the claim is an insured-building outcome |
| A3 | Label = >= 1 inland (cause 2/3/4) claim in [start, end + 2 d]; tropical windows removed | cause filter separates rain support (0.94 vs 0.69-0.82); window and claim-count variants no better (stage 23) | **Partly**: the slides' Y is a loss amount; a 0/1 label cannot rank two units that both claimed. Use claim count and paid amount as graded relevance. **The tropical exclusion contradicts decision 3** |
| A4 | Split by year, train <= 2016, test 2017-2023 | stages 8, 19 | **Matches the slides' protocol** |
| A5 | Learner: boosting depth 5 / logistic (tied) | stages 8, 14, 20 | Fine for prediction. The slides' estimand is causal (tau under do(W = w), with C and A as confounders); a predictive learner is enough for the ranking inside the training regime, not for extrapolating to new regions or what-if intensities |
| A6 | Features L2: county rain, exposure (log policies, SFHA share, single-family share, mean coverage), the block group's own rain | exposure +0.147, own rain +0.139 (stage 8) | Observed-mode features are right for the learning stage. Policies and SFHA share are the slides' **adaptation A** and also decide whether a claim can exist; claim history (`hist00`) is the slides' C / A proxy (+0.026). Forecast mode contains **`dur`**, which is not known before the event |
| A7 | Plug-in of the forecast: ensemble mean and member votes at the cell (stage 16) | fcens ties fcmean | The slides' q gives a probability of each intensity; integrating the observed-trained model over members (mean of the members' impacts) was not tested |
| A8 | **Main metric = within-county AUC (pair weighted)**; cross-county ranking excluded because "a county-level model is 0.5" (rule S10) | below | **Wrong metric for this task**: it discards the between-county information, which is the ranking the slides ask for |
| A9 | Level of the probability not transferable across years | stages 8, 34 | Now an auxiliary product; see section 3.3 |
| A10 | Rain driver: gridMET daily at the centroid; forecast GEFSv12 36-84 h | stage 16 | A stand-in for the slides' National Water Model flood forecast; sub-daily structure missing |
| A11 | Texas only | stage 24, 25 | Group (TX, LA, FL) not built |
| A12 | Event for ranking = county episodes that start the same day | script 74 | Heuristic for "one weather system"; storms spanning several days are cut |

## 3. Measured under the revised task (script 74, stage 8 L2 model, test 2017-2023)

### 3.1 Ranking across counties inside an event (events with >= 2 counties)

Block groups ranked across counties (408 events with a claim); share of the claimed block groups inside the top 1 / 5 / 10 / 20% of an event's block groups (random: 1 / 5 / 10 / 20%):

| Score | AUC pair weighted | AUC event mean | top 1% | top 5% | top 10% | top 20% |
|---|---|---|---|---|---|---|
| model L2 | 0.865 | 0.830 | 0.175 | 0.410 | 0.552 | 0.693 |
| cell rain only | 0.834 | 0.783 | 0.159 | 0.342 | 0.485 | 0.642 |
| county peak rain only | 0.747 | 0.682 | 0.066 | 0.197 | 0.296 | 0.474 |
| policy count only | 0.715 | 0.687 | 0.050 | 0.161 | 0.253 | 0.415 |

Counties ranked by expected claimed block groups (sum of p; 386 events):

| Score | county AUC pair weighted | county AUC event mean | claimed block groups in top 5% of counties | top 10% | top 20% | Spearman with claimed block groups |
|---|---|---|---|---|---|---|
| expected claimed block groups | 0.922 | 0.923 | 0.60 | 0.74 | 0.89 | +0.39 |
| county peak rain only | 0.782 | 0.739 | 0.29 | 0.40 | 0.56 | +0.22 |
| policies in the county only | 0.855 | 0.858 | 0.48 | 0.58 | 0.75 | +0.33 |

- The stage 8 metric (within one county) is 0.786: the product ranks **better across counties (0.865 block groups, 0.922 counties)** than the metric reported so far.
- Part of the county ranking is exposure: policies alone give county AUC 0.855; the model adds 0.067 above exposure and 0.14 above county rain alone.

### 3.2 Reading the ranking for an event

Median over the 146 test events with >= 5 claimed block groups: the highest block group has p = 24%; **the top 5% of block groups** (mean p 4.8%, observed rate 3.2%) hold only about one third of the event's
claimed block groups (**64% lie outside the top 5%**). A short list therefore misses most of the claims of the same event; it is a list of where the density is highest.

### 3.3 The probability companion: do the others read 0%?

- **No.** Outside the top 5% the mean predicted probability is 0.31% and the observed rate 0.22%; 40% of block groups sit below 0.1%. So the others are small, not zero, and not all negligible: a
  minority of them claim (64% of the event's claimed block groups).
- **The level is too high by about 1.5x**: expected claimed block groups (sum of p) against observed per event: median ratio 1.50, pooled 1.58 (small events 2.0, large 1.5); Spearman of the
  event's expected count with the observed count 0.66.
- **The level cannot be fixed from the past**: recentring each year's probabilities with the trailing three-year prevalence (deployable) gives Brier skill -0.018, worse than as predicted (+0.002); with the
  same-year prevalence (oracle, not available) +0.057. Yearly prevalence: 2015 1.16%, 2016 0.84%, 2017 0.19%, 2020 0.10%, 2023 0.13%. The trailing level is wrong because the level follows the size of the
  events, not the calendar.
- Until an event-level model of the level exists, show the probabilities as **relative risk** (times the event's median, or the share of the event's expected claims) and as the event's
  **expected number of affected units**, with the stated overprediction, rather than as an absolute percentage.

## 4. Mismatches to fix before the measurement

| Priority | Item | Fix |
|---|---|---|
| 1 | Metric A8 | Cross-county ranking inside an event as primary (section 5) |
| 2 | `dur` in forecast mode (A6; episode-level AUC 0.8627 to 0.8398 without it, within-county AUC -0.0097) | Remove from forecast-mode feature sets; features known at issue time only |
| 3 | Binary label (A3) | Graded relevance: number of claims and paid building amount per block group, for top-k and NDCG; keep the 0/1 label for AUC |
| 4 | Tropical exclusion (A3) | Scope S2: tropical windows kept, named tropical claims included |
| 5 | Trigger dependence (A1) | Report 20 / 30 / 50 mm; fix 30 as the headline |
| 6 | Level of the probability (A9) | Event-level model of the expected count (the part of O2 that matters now); until then relative risk display |
| 7 | Plug-in of the ensemble (A7) | Test the mean of member-wise impacts against the impact of the mean rain |
| 8 | Impact model check on simulated losses with a known answer (slides) | Not done for any layer |
| 9 | Group and county aggregation (A11, A12) | LA and FL pipelines; event = system, not same-day |

## 5. Target-metric specification

**Unit and event.** Block group x event (county episodes starting the same day, 30 mm headline, 20 / 50 mm sensitivity); counties by aggregating block groups (sum of p, and 1 - prod(1 - p)).
**Data.** Observed mode: train 2010-2016, test 2017-2023. Forecast mode: GEFSv12 36-84 h (stage 16 setting), test 2017-2019, features known at issue time (no `dur`, no observed rain).
**Label and relevance.** Occurrence (any claim) for AUC; claim count and paid building amount as graded relevance for capture and NDCG.

**Primary (ranking).**
1. Cross-county AUC inside an event, at block-group and at county level; pair weighted and mean of events.
2. Top-k capture: share of the event's claimed units (and of the claim count, and of the paid amount) inside the top k = 1 / 5 / 10 / 20% of units, and in the top 5 / 10 / 20% of counties, against random and against the two single-score baselines (policies only, rain only).
3. Spearman with the claimed count; NDCG at k with graded relevance.
4. Event-clustered bootstrap, results by year; the same table for the forecast-mode model.

**Auxiliary (probability).**
1. Expected / observed claimed units per event (ratio, Spearman), by size of event; calibration slope and reliability by decile.
2. "Others are not zero" summary: mean p, observed rate and share of claims outside the top 5%, per event.
3. Brier skill with the level as predicted and re-centred with the trailing three years (deployable); the same-year re-centring is reported as an oracle only. Not a gate.
4. Display check: relative risk and expected count against the absolute percentage.

**Check of the impact layer.** Simulated losses with a known answer, as on the slides (open).

## 6. Decisions on the four open points (2026-10-29, "as recommended")

1. **k**: 5 / 10 / 20% of the units of an event, and the top 3 / 5 / 10 counties (the fixed-k county numbers use events with at least 10 counties; 1% of block groups is also shown).
2. **Relevance**: claim count **and** paid building amount are both reported (capture of each, NDCG with each as gain).
3. **Event**: county episodes that start on the same day (heuristic); a system defined by connected days and neighbouring counties is a later refinement.
4. **GEFS all-days archive**: postponed (about 23 GB transferred); the measurement uses the stored 1,640 forecast days (events given, as the slides state).

Measured by `scripts/75_task_metrics.py` (stage 36).

## 7. First measurement (stage 36, `scripts/75_task_metrics.py`)

| Mode | block-group AUC across counties | county AUC | top 10% of counties: share of claim count | versus policies only |
|---|---|---|---|---|
| observed rain, scope S0 | 0.865 | 0.922 | 0.74 | +0.151 / +0.066, intervals clear of 0 |
| observed rain, scope S2 (tropical kept) | 0.886 | 0.924 | 0.76 | +0.137 / +0.068, clear of 0 |
| **forecast rain (GEFS, no `dur`)** | **0.650** | **0.852** | **0.59** | **-0.066 / +0.005, no gain beyond exposure** |

The ranking works with observed hazard; with the current forecast input it adds nothing to the exposure ranking across counties. Probability companion: others outside the top 5% have a mean p of 0.3-0.4%
(not 0%), and 60-91% of an event's claimed block groups lie outside the top 5%.

**Diagnosis of the forecast row (stage 37).** The forecast result is largely a sample effect: the pilot's county set is defined by observed rain >= 30 mm, which removes the forecast's between-county skill
(Spearman with the observed county peak 0.74 over all county-days, 0.29 among the counties that reached 30 mm; AUC for "reached 30 mm" 0.92). With all counties of a day included, county AUC for any claim is
policies 0.881, forecast + policies 0.889, observed + policies 0.898. Day alignment, the duration cut and bias correction are not the cause. **Open specification point:** the event's county set must be fixed
before the event (a forecast footprint or a region per day), and the forecast-mode ranking re-measured on it.

**Forecast mode re-measured on an ex-ante unit (stage 38, `scripts/77_forecast_ranking_all_counties.py`).** All block groups and counties of every forecast day, issue-time features only:

| Mode | block-group AUC across counties | county AUC | top 10% of counties: share of claim count | versus policies only (county AUC / top-10% capture) |
|---|---|---|---|---|
| forecast rain (GEFS, no `dur`), all counties of a day | 0.713 | 0.891 | 0.80 | +0.010 [+0.005, +0.014] / +0.029 [+0.014, +0.044] |
| observed rain on the same rows (reference) | 0.794 | 0.902 | 0.82 | |
| policies only | 0.747 | 0.881 | 0.77 | |

This replaces the forecast row of the table above (0.650 / 0.852, a sample effect, stage 37). The forecast adds a small, clear increment over exposure at county level and recovers about half of what observed rain adds;
at block-group level the pooled AUC stays below the exposure ranking. The probabilities trained on this unit are calibrated in aggregate (events: expected / observed 0.91) and the Brier skill against the trailing
prevalence is about 0; 86% of an event's claimed block groups lie outside the top 5%.

**By event size (stage 39, `scripts/78_ranking_by_event_size.py`).** The forecast model's gain over exposure does not grow with the event (county AUC +0.009, +0.011, +0.012, +0.006 for 1-2, 3-9, 10-29, >= 30 claimed
block groups); the observed-rain model's does (+0.007, +0.013, +0.024, +0.030; top-5% capture +0.057 to +0.159). In the 76 largest days (78% of the claims) the forecast model equals exposure at county level
(+0.006 [-0.003, +0.015]). Ex ante, on days with a forecast county peak >= 25 mm it improves county AUC by +0.014 to +0.018 and paid-amount NDCG by +0.017 to +0.022; below 25 mm it equals exposure. The shortfall of the
ranking product in forecast mode is in the extreme events, where the five-member mean underestimates the rain.

**Ensemble extremes (stage 40, `scripts/79_ensemble_extremes.py`).** Member maximum, exceedance shares (>= 25 / 50 mm), spread and the county versions did not lift the forecast ranking in the big events: county AUC -0.003
overall and -0.005 in the days with >= 30 claimed block groups, top-5% block-group capture -0.026 overall and -0.055 on days with a forecast peak >= 50 mm (all intervals below zero), against the mean model. The missing
information is not in the spread of a five-member, 0.25-degree, 36-84 h ensemble. Untested levers: a shorter lead (the forecast of the day itself), more members, a finer or different model.
