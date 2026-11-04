# Scenario report: a Texas city forecaster after an NWS warning (2026-11-02)

**Setting.** A weather forecaster at a Texas city government receives a National Weather Service flash-flood warning. The county emergency department wants to prepare resources
or adjust the duty roster. Three questions: (1) how early, and how accurately, can the model help the forecaster propose a dispatch order; (2) when can the emergency department
first see the sub-county map, shown as discrete points or as a geographic interpolation; (3) what happens when the clock reaches T-0 with the model running on a laptop.
Everything below is a retrospective test on Texas, scope S0, test years 2017-2019 (forecast mode) and 2017-2023 (observed rain); method and caveats are in `RESULTS.md` stages 53-56.

**Clock.** T0 = 12Z of day S, the start of the 2-day rain window the model reads. The forecast mode uses the GEFSv12 00Z run of day S-1 (leads 36-84 h), so the run is initialised
T0-36 h. Hours are relative to T0; "T-31" means 31 hours before the rain window opens.

## 1. Short answer

| Question | Answer | Basis |
|---|---|---|
| Earliest the model can rank | **T-31 h** (ranking on screen about 5 h after the 00Z initialisation, plus about 10 seconds of computing) | assumed 5 h publication delay (not verified, sensitivity 4-8 h); computing measured |
| Accuracy of the forecast-mode ranking inside the counties that hold a warning | block-group AUC **0.75** (policies only 0.67), top-5% of block groups hold **20%** of the claims (12%), county ranking no better than exposure (AUC 0.82 vs 0.82) | 180 event days, 2017-2019 |
| Accuracy after the rain, with observed rain | block-group AUC **0.86**, top-5% capture **43%**, county AUC 0.89 | 332 event days, 2017-2023 |
| Lead over the first warning for an affected county | median **28 h** earlier; **24%** of events had the warning before the model run was ready | 163 event days; see the caveat on windows |
| Sub-county map: points or interpolation | **show the discrete points**; use the interpolated surface only as context | points AUC 0.78 / 0.84 vs smoothed 0.71-0.73 / 0.79-0.81 |
| At T-0 | the forecast does not change by itself; the observed-rain ranking arrives about **T+54 h** and is a damage-assessment tool, not a warning | assumed 6 h QPE delay |

*Part 2 (below) replaces the single clock of this part by two T-0s and simulates how often updates can arrive; where the two parts differ, part 2 is the more careful one.*

## 2. When can the model help, and how accurately

### 2.1 Timeline of one event
Replay of a typical event (2019-06-04, Houston area, 54 claimed block groups; chosen because its forecast-mode AUC of 0.85 is the most common value among large events, not the best):
[docs/media/countdown_20190604.mp4](media/countdown_20190604.mp4). The clip runs the clock from T-36 h to T+120 h, shows each processing step with its measured cost, the real NWS warnings of that
period as they were issued, the model's county ranking, block-group points and interpolated surface, and the real claims at the end. A harder event (2019-06-09, Dallas-Fort Worth, 16 claimed
block groups, forecast AUC 0.40, no warning covered an affected county) is in [docs/media/countdown_20190609.mp4](media/countdown_20190609.mp4); use it next to the first clip so the audience sees the spread.

| Clock | What exists | Source of the number |
|---|---|---|
| T-36 h | GEFS 00Z run of S-1 initialised | design of the test |
| T-31 h | ranking on screen: 5 h publication delay + 9 s download (5 members, Texas crop) + 3 ms features + 30 ms network + 0.1 s map | delay assumed; rest measured on this Apple M4 Pro (script 97) |
| T-31 h to T0 | the forecast is static; the next 06Z cycle would refresh it (not evaluated: the reforecast has only 00Z runs) | |
| median T0-2 h to T0+8 h | first NWS warning for an affected county (see 2.3) | script 98, 2017-2023 |
| T0 | rain window opens | |
| T+54 h | observed-rain ranking possible (48 h window + 6 h assumed QPE delay) | assumed |
| days to weeks | NFIP claims arrive (the labels) | |

### 2.2 Accuracy at the earliest time (forecast mode, T-31 h)
Inside the counties that hold an NWS Flash Flood Warning on the event day (the set the forecaster would apply the ranking to), 180 event days 2017-2019 (95% interval over events):

| metric | policies only | model, forecast | model, observed rain |
|---|---|---|---|
| block-group AUC | 0.674 [0.604, 0.734] | **0.746** [0.697, 0.786] (+0.072 [+0.048, +0.098]) | 0.856 (same 180 days) |
| top-5% block groups, share of claims | 0.122 [0.095, 0.150] | **0.204** [0.168, 0.239] (+0.081 [+0.045, +0.113]) | 0.403 |
| top-10% block groups, share of claims | 0.257 | 0.334 (+0.077 [+0.034, +0.117]) | 0.541 |
| county AUC | 0.823 | 0.819 (-0.004 [-0.016, +0.008]) | 0.876 |

Reading: after a warning, the early model helps to choose **where inside the warned counties** to look (sub-county), not **which county** (no gain over exposure). Over all
events (not only warned counties) the forecast-mode block-group AUC is 0.787 and the day is flagged (>= 10 claimed block groups) with AUC 0.78 against 0.48 for exposure alone (stage 54).
About a fifth of the claims fall inside the top 5% of block groups; four fifths are elsewhere, so the list is a priority order, not a complete picture.

### 2.3 Is the ranking earlier than the warning?
Texas Flash Flood Warnings 2017-2023: 2,559 (median duration 3.0 h). For the event days with >= 10 claimed block groups, 87-90% had a Texas warning in the window T0-48 h .. T0+48 h
and 76-78% had a warning covering a county that had claims. Hours between the ranking being ready (T-31 h) and the first warning for an affected county:

| window and test years | median | p10 / p25 / p75 / p90 | ranking earlier than the warning |
|---|---|---|---|
| warnings from T0-48 h, forecast mode 2017-2019 (n = 163) | 28 h | -10 / 2 / 50 / 68 | 76% (55% by more than 24 h) |
| warnings from T0-48 h, 2017-2023 (n = 299) | 27 h | -11 / -1 / 48 / 67 | 75% |
| warnings from T0-24 h, 2017-2019 | 39 h | 12 / 20 / 56 / 69 | 100% (by construction, earlier warnings are cut off) |

The wide window is the conservative one but it can count a warning for a storm of the day before, so the truth lies between the two rows; the narrow window cannot show warnings
issued before the ranking was ready. In about a quarter of the events the warning came first: there the model confirms and orders what the warning already says; it does not
give an earlier alert. For the event days without a warning covering an affected county (22-24%), a warning-triggered use would never see the model; those days are where an
always-on display (section 4) matters. The archive holds warnings only; Flash Flood **Watches** and WPC outlooks, which come earlier, are not in the data, so the lead over the whole NWS chain is not measured.

## 3. The sub-county map for the emergency department

### 3.1 When
The same moment as the ranking: T-31 h. Nothing in the pipeline waits for the human; the block-group probabilities are a by-product of the same run. The first usable sub-county display
therefore appears with the first model run, more than a day before the rain window opens and, in three quarters of the events, hours to days before the first warning for an affected county.

### 3.2 Discrete points or geographic interpolation
Block-group points (one probability per block group with a policy) against a distance-kernel surface of the same probabilities (5, 10, 20, 40 km), scored at the block groups, 150 sampled event days with >= 10 claimed block groups:

| display | BG AUC forecast | top-5% capture forecast | BG AUC observed | top-5% capture observed |
|---|---|---|---|---|
| **points as they are** | **0.775** | **0.281** | **0.836** | **0.431** |
| smoothed 5 km | 0.729 | 0.243 | 0.805 | 0.390 |
| smoothed 10 km | 0.719 | 0.237 | 0.799 | 0.377 |
| smoothed 20 km / 40 km | 0.712 / 0.706 | 0.232 / 0.228 | 0.793 / 0.789 | 0.369 / 0.362 |
| policies only | 0.723 | 0.175 | 0.724 | 0.189 |

Smoothing loses 0.03-0.07 AUC and a fifth of the head-of-list capture, and in forecast mode a 10 km surface ranks no better than exposure alone. The points carry the exposure-weighted
information that the surface washes out. Interpolation is useful as a **filler**: hiding 30% of the block groups and rebuilding them from the neighbours gives an AUC of 0.795 (observed) and 0.701
(forecast), better than the nearest visible point (0.767, 0.672); but in forecast mode exposure alone ranks the hidden units better (0.733). The recommendation: **points are the primary layer** (sized by exposure,
colour by probability); the interpolated surface is a context layer for the unpoliced land, labelled as a hazard surface, not as a ranking. The three-panel layout of the clip puts both side by side.

## 4. T-0 with a laptop running

What the measurements say about running this on a forecaster's MacBook or a borrowed machine (script 97, Apple M4 Pro, CPU unless stated):

| item | value |
|---|---|
| network | 60,137 parameters; one day (12,590 block groups): features 3 ms + forward 28 ms on CPU (6 ms on the Apple GPU); 95th percentile 32 ms |
| download of the forecast | 9 s for 5 members of the Texas crop (2.4 MB each) from the public S3 bucket |
| map rendering | about 0.1 s per frame in the clip (matplotlib, not optimised) |
| memory | 2.7 GB resident in the research harness (it keeps 10 years of day tensors in memory; a one-day service needs much less, not measured) |
| agreement with the evaluation | rebuilt pipeline reproduces the stored predictions to 1e-7 |

What happens as the clock runs down (the clip shows each step):
1. **T-31 h:** the run takes about 10 seconds of laptop time, so a battery-powered machine is not the constraint; the constraint is the publication of the ensemble and the network connection.
2. **T-31 h to T0:** the screen is static between model cycles. A refresh every 6 hours is cheap, but whether the ranking becomes better or only different between cycles was not tested; the evaluation has one 00Z run per day. Treat a changed ranking as new information to read, not as a correction of a mistake.
3. **T0:** nothing in the model changes; the forecast was issued for this window. Warnings accumulate (15 in the window of the clip). Observed rain starts to fall but the observed-rain model reads a 2-day window, so it cannot update at T0.
4. **T+54 h:** the observed-rain ranking (block-group AUC 0.86 in warned counties against 0.75 for the forecast; top-5% capture 43% against 20%) is available if a daily QPE product arrives within 6 h. The model was validated with daily gridMET only; Stage IV or MRMS as a substitute is not validated. Use it to prioritise damage assessment and resource re-allocation, not as a warning.
5. **Days later:** claims arrive; the ranking can be audited against them (the clip ends with the real claims).

## 5. What the audience should take from it
- The model is early (T-31 h) and cheap; the early value is **inside the warned area** and **sub-county**, where it raises the share of claims found in the top 5% of block groups from 12% to 20% (about 1.7 times) compared with ranking by insurance exposure.
- It does not replace the warning: in a quarter of the events the warning came first, and at county level the early ranking adds nothing over exposure.
- The numbers that sell the system (AUC 0.86, 43%) belong to the **after-the-rain** mode, not to the T-31 h mode (0.75, 20%).
- Show points, not a smoothed surface.

## 6. Limits of this report
- Texas only, GEFSv12 reforecast with 5 members (not the 31-member operational ensemble), three forecast years, two seeds; no intervals for the lead-time medians; events inside a storm are dependent.
- The 5 h publication delay and the 6 h QPE delay are assumptions; the GEFS delay could not be confirmed from the sources found (NCEP gives no single figure for the 00Z cycle).
- The accuracy numbers are conditional on a day with at least one claim (and in 2.2 on counties that hold a warning issued in T0-24 h .. T0+48 h; counties warned by day-ahead products are not distinguished).
- The warning archive contains storm-based warnings only (no watches, no WPC outlooks), so the comparison is against the later, shorter-fuse product.
- The dispatch rankings are expected claimed block groups under NFIP policies: a flood-insurance outcome, not a measure of all flood impact or of resource need; no resource-allocation or cost model is involved.
- The clip is a replay of stored outputs with real warnings and claims, built for demonstration; it is not a live product and does not simulate later GEFS cycles.

Reproduce: `python scripts/97_scenario_pipeline.py`, `98_scenario_analysis.py`, `100_scenario_figures.py` (figure `docs/figures/scenario_summary.png`), `99_countdown_video.py --event 20190604` (and `20190609`).

---

# Part 2: realistic update cadence and two T-0s (added 2026-11-02)

Part 1 used one 00Z forecast and one clock. Two refinements: (a) how often a government organisation can actually get new information through the NWS interfaces, simulated on synthetic fields with a known answer
(the task sheet checks the impact model on simulated fields with known answers), and (b) splitting T-0 into the one that matters for emergency management and the one that matters for the insurer.
Video (recommended): [docs/media/countdown4_20190511.mp4](media/countdown4_20190511.mp4) (86 s, 1920x1080, HEVC `hvc1` encoded with Apple VideoToolbox; an H.264 copy for players without HEVC: [countdown4_20190511_h264.mp4](media/countdown4_20190511_h264.mp4)). The screen is split in two groups. **Model output** (left): county ranking, the sub-county probability of every block group, a run counter with a scrolling log (the measured cost of each run), the quality of each run. **Real-world consequence** (right): the dispatch set (top 5% of block groups) with rings for block groups that entered it and crosses for those that left, a map of how the ranking moved since the previous run, the two T-0 clocks, and lanes for warnings, model runs (bar height = churn of the dispatch set) and opened claims. The last 14 s are a featured-county act (section 12). The scan sweep of the earlier cuts is gone. Two clocks and five update lanes as before; claims appear as their real open dates pass.
Figure: [docs/figures/cadence_summary.png](figures/cadence_summary.png).
Method and caveats: `RESULTS.md` stage 57; code `scripts/101-106`.

## 7. Definitions of the two T-0s

| | Hazard T-0 (emergency management) | Claims T-0 (insurer) |
|---|---|---|
| Meaning | the moment the flash-flood threat reaches the affected block groups; every mitigation decision, including a wireless alert, has to be finished before it | the moment the insurer's own claim stream becomes usable; preliminary estimates and survey reservations (commercial satellite imagery, shared aerial-flight schedules) have to be made before it |
| How it is set here | first NWS Flash Flood Warning covering an affected county in [T0-24 h, T0+48 h] (72% of the 210 events), otherwise T0 = 12Z of the first loss day S. Not independent of the NWS (a warning is the NWS's own call that the threat is imminent), but it is the only real-time marker available for all test events. | day by which **10%** of the event's claims have been opened (OpenFEMA `openDate`, day resolution, end of day); the 50% day is reported too |
| Distribution (hours from T0) | median **0 h**, p10 -15.5 h, p90 +36 h | median **+66 h** (S+2), p10 +18 h (S+0), p90 +114 h (S+4); 50% opened: median +90 h (S+3), p10 +42 h, p90 +162 h |

Two facts behind the claims clock (Texas NFIP flood claims, loss years 2017-2023, 78,573 claims with an open date): only 16% of claims were opened on or before the loss day, the median lag from loss to opening is 2 days (p10 0, p90 13).
Per event (394 events): 0% of the event's claims opened by day S (p25-p75 0-6%), **7% by S+1, 25% by S+2, 47% by S+3, 77% by S+7, 87% by S+14** (medians). The insurer is blind for the first two days.
Wireless alerts: the NWS sends a Wireless Emergency Alert only for flash-flood warnings tagged considerable or catastrophic damage threat (policy of 2020); in Texas 2020-2023 that is **305 of 1,397 warnings (22%)**,
4 of them emergencies. So for most warnings no phone alert exists at the hazard T-0; whether a local authority adds its own alert is outside this analysis.

## 8. How often can an organisation get updates

### 8.1 Interfaces (what is verified, what is assumed)
| Channel | Update behaviour | Status |
|---|---|---|
| NWWS-OI (NWS text-product push, needs NWS credentials, XMPP, CAP added 2024) | within 10 s of issuance (NWS's own figure); not every product is guaranteed on it | sourced: [weather.gov/nwws](https://www.weather.gov/nwws), [SCN 24-87](https://weather.gov/media/notification/pdf_2023_24/scn24-87_NWWS_OI_support_for_CAP_messages.pdf) |
| api.weather.gov `/alerts/active` (poll) | no official polling interval found; the docs say content expires by its life cycle; a community guide suggests caching 5-10 min | sourced: [NWS API docs](https://www.weather.gov/documentation/services-web-API), community [caching note](https://alex-gekas.github.io/nws-api-reference/caching) |
| HRRR-class rapid-refresh run | every hour, 18 h forecasts (48 h at 00/06/12/18Z), appears on NOMADS about 1 h 50 min to 2 h after the cycle (third-party figures) | sourced: [NWS notice](https://www.weather.gov/media/notification/tins/tin14-32hrrr_aaa.pdf), third-party [LuckGrib](https://www.luckgrib.com/models/hrrr_extended/) |
| MRMS multi-sensor QPE | hourly at the top of the hour; Pass 1 latency 20 min, Pass 2 60 min (NSSL change log); radar-only product on a 2-min cycle (dated briefing) | sourced: [NSSL](https://inside.nssl.noaa.gov/mrms/?p=482), [NOAA briefing](https://roc.noaa.gov/public-documents/field-requirements-branch/tac/meetings/2019/MRMS%20Briefing.pdf) |
| GEFS 00/06/12/18Z cycles | 6-hourly; publication delay **not found** in the sources (assumed 5 h) | assumption |
| WEA | only considerable / catastrophic flash-flood warnings | sourced: [NOAA](https://noaa.gov/node/3504), [NWS SCN 19-101](https://www.weather.gov/media/notification/pdf2/scn19-101wea_floods_aac.pdf) |

### 8.2 Simulated detection delay and information age (the real warnings of Texas 2020-2023 and the cadences above)
| | median | 95th percentile |
|---|---|---|
| warning seen via NWWS-OI push | 5 s | 9 s |
| via api.weather.gov, poll every 1 / 5 / 10 / 15 / 30 min (+5-60 s propagation, assumed) | 62 / 181 / 331 / 481 / 927 s | 102 / 319 / 602 / 886 / 1,737 s |
| age of the newest GEFS-like cycle (6 h, +5 h assumed) | 8.0 h | 10.7 h |
| age of the newest rapid-refresh-like run (hourly, +1.9 h) | 2.4 h | 2.9 h |
| age of the newest MRMS QPE, Pass 1 / Pass 2 | 0.83 h / 1.5 h | 1.3 h / 1.95 h |

A 30-minute poll at its worst uses 16% of the typical 3-hour lifetime of a warning; a 5-minute poll is under 3%. The interface choice matters little for warnings (minutes against a 3-hour product) and a lot for
the forecast side (hours of age). Practical reading: use the push feed where credentials are available, otherwise poll at 1-5 min; rely on hourly rain forecasts rather than 6-hourly ones once the hazard is within a day.

### 8.3 What a fresher update is worth (synthetic fields with a known answer)
Setup: the observed 4 km rain field of a test day is the truth; a synthetic forecast issued l hours before the rain is that field with a displacement, a correlated amplitude error and a blur that grow with l
(eps ~ l^alpha); the observed-rain network ranks the block groups from it; the real claims are the answer (210 event days with >= 10 claimed block groups, 2017-2019). The error scale at the real GEFS-mode lead
(60 h) is calibrated so the network reproduces the real forecast-mode skill: block-group AUC **0.778** (real run 0.778) and top-5% capture 0.265 (real 0.274, not fitted). The growth law is an assumption (alpha 0.5 / 0.75 / 1 shown).

| lead of the information | 84 h | 60 h | 36 h | 24 h | 12 h | 6 h | 1 h | observed rain |
|---|---|---|---|---|---|---|---|---|
| block-group AUC (alpha 0.75) | 0.769 | 0.778 | 0.790 | 0.798 | 0.810 | 0.818 | 0.825 | 0.826 |
| top-5% capture (alpha 0.75) | 0.246 | 0.266 | 0.292 | 0.316 | 0.353 | 0.379 | 0.413 | 0.416 |
| (policies only: AUC 0.731, top-5% 0.184) | | | | | | | | |

Across alpha 0.5-1.0 the top-5% capture at a 12 h lead is 0.328-0.375 and at 1 h 0.394-0.414.
On the timeline before the hazard T-0 (events with their own hazard T-0, expected values):

| decision time before the hazard T-0 | 48 h | 24 h | 12 h | 6 h | 1 h |
|---|---|---|---|---|---|
| 6-hourly stream (+5 h), AUC / top-5% | 0.774 / 0.258 | 0.784 / 0.283 | 0.792 / 0.302 | 0.795 / 0.313 | 0.799 / 0.323 |
| hourly stream (+2 h), AUC / top-5% | 0.776 / 0.261 | 0.787 / 0.290 | 0.795 / 0.311 | 0.798 / 0.322 | 0.802 / 0.331 |

Between T-48 h and the hazard T-0 the ranking improves by about +0.025 AUC and +0.06-0.07 top-5% capture (alpha 0.5-1: +0.018 to +0.031 AUC), whichever stream is used; the **hourly stream is worth at most +0.003 AUC and
+0.01 capture over the 6-hourly one** at the same moment, although it delivers 37 updates (p10 21, p90 73) against 6 (p10 3, p90 12) between T0-36 h and the hazard T-0. The value is in the age of the newest information (about 8 h
against 2.4 h), not in the number of updates. The decision budget: the first ranking (T-31 h) is on screen a median **31 h** before the hazard T-0 (p10 15.5 h, p90 67 h) and at least 12 h before it in 93% of the events
(24 h in 77%); this is conditional on the hazard T-0 definition above, whose window starts at T0-24 h; with a window that also admits earlier warnings, about a quarter of the events had a warning before the ranking (section 2.3).

## 9. The insurer: before the claims T-0
- **When the model information exists relative to the claims T-0.** Forecast ranking at T-31 h: a median **97 h** (T0-31 h to T0+66 h) before the 10%-opened day. Observed-rain ranking at about T0+54 h: a median **12 h** before it; in 46% of the events the 10% day came first (day S+1 or earlier).
- **Survey reservation.** What is sourced: a satellite vendor states SAR flood extent and depth within 24 h of the flood peak ([SpaceRef](https://spaceref.com/?p=202998), vendor claim), a reinsurance broker pitched initial loss estimates "in a matter of days" ([Strategic Risk](https://www.strategic-risk-global.com/catastrophe-risk/guy-carpenter-launches-satellite-based-catastrophe-evaluation-service-/1407245.article)); no independent tasking or aerial-deployment timeline was found. The slack below therefore uses assumed survey lead times (6 / 24 / 72 h):

| assumed lead needed to reserve imagery or flights | slack using the forecast ranking (T-31 h), median / share of events with slack >= 0 | slack using the observed-rain ranking (T+54 h), median / share >= 0 |
|---|---|---|
| 6 h | +91 h / 100% | +6 h / 54% |
| 24 h | +73 h / 100% | -12 h / 32% |
| 72 h | +25 h / 81% | -60 h / 4% |

Reading: **reserve on the forecast ranking**. The observed-rain ranking (AUC 0.86 instead of 0.78) arrives too late for any survey lead beyond about 6 h to be booked before the claims clock, in the median event; it is still useful to
choose *where* the booked capacity points (top block groups) once flying or tasking becomes possible, and to size the preliminary estimate while only a quarter of the claims have been opened (median 25% by S+2).
The ranking is a relative-risk list (section 2); the expected claimed block groups per event were too low for the largest events (stage 55), so it should not be used as the reserve amount.

## 10. T-0 on a laptop, updated
At the hazard T-0 the model has no new information of its own: the last forecast update is more than a day old (6-hourly) or hours old (hourly), the rain is falling, and the next skill step is the observed-rain ranking at about
T0+54 h with MRMS or a daily QPE product (not validated for those products). What a laptop running continuously produces is therefore an **update lane** (6-hourly: 6 updates, hourly: 37 before the hazard T-0), a **ranking that stops changing materially**
(+0.003 AUC between the two streams) and **two clocks**: the hazard clock reaches zero while the claims clock still has a median of 66 h (days for the insurer, none for the emergency manager). The clip shows both clocks, the lanes, the maps that follow the hourly stream, the
observed-rain switch, and red rings that appear as real claim open dates pass.

## 11. Limits of part 2
- The synthetic forecasts are built from the observed rain with an assumed error law (growth exponent, displacement 25 km, amplitude 0.5, blur 16 km per unit eps); only one point (60 h) is calibrated against a real run. At leads under 12 h the simulation is optimistic: nowcast rain is only partly observed at that time, and real rapid-refresh runs have their own error.
- In the v3 clip successive runs of a stream share only part of their error (AR(1), persistence 12 h, assumed): a run brings new information and the dispatch set changes by a median 2.7% (p90 12%) per hourly run. This changes the look of the clip, not the calibrated marginal error; the v2 clip and the tables use runs that share one error pattern (their dispatch set hardly changes between runs).
- The "hourly stream" is an idealised rain forecast of any lead; real rapid-refresh runs cover 18 h (48 h at four cycles) and the model that consumes them was not trained on them.
- Hazard T-0 is proxied by the first affected-county warning (or T0), claims T-0 by the 10%-opened day at day resolution; neither is an independent measurement of flood onset or of the insurer's information. The OpenFEMA open date is the NFIP claim open date, not the date a private insurer first hears of a loss.
- GEFS and QPE delays are assumptions; the NWS API polling propagation is assumed; WEA eligibility is from the tag policy and the tags of the archive (2020 onward), and the forecast-mode test years 2017-2019 predate them.
- Texas, S0, 2017-2019 for the synthetic part (210 events), 2017-2023 for the claim clocks (394 events), no intervals for the medians; events within a storm are dependent.

## 12. Proof of sub-county ability: within-county ranking and a featured county

**What it means.** A county-level dispatch sends resources to a county and says nothing about its neighbourhoods (Harris County has 2,109 block groups). The sub-county claim is only worth something if the probability
orders the block groups *inside* a county better than exposure alone. Test (`scripts/106`): for every (event day, county) with >= 20 block groups and >= 3 claimed block groups, in the events with >= 10 claimed block groups,
the AUC of the score across the county's own block groups and the share of the county's claimed block groups that fall in its top 10% (random: 0.10); intervals by bootstrap over event days.

| | pairs / event days / counties | policies only | model | model minus policies only |
|---|---|---|---|---|
| forecast mode, real GEFS run, 2017-2019: AUC | 511 / 207 / 37 | 0.667 [0.654, 0.682] | **0.735** [0.724, 0.748] | **+0.068** [+0.060, +0.076] |
| forecast mode: share of claimed block groups in the county's top 10% | | 0.221 [0.204, 0.240] | **0.296** [0.274, 0.318] | **+0.075** [+0.062, +0.090] |
| the same rows with observed rain: AUC / top-10% | | | 0.768 / 0.358 | +0.100 / +0.137 |
| observed-rain CNN, 2017-2023: AUC / top-10% | 993 / 388 / 54 | 0.666 / 0.223 | 0.767 / 0.360 | +0.101 [+0.086, +0.116] / +0.137 [+0.120, +0.156] |

93% of the forecast-mode county-days have a within-county AUC above 0.5 (policies only: 90%, random: 50%). The sub-county information is real and the real forecast already carries about two thirds of the gain of observed rain.
It is modest in absolute terms: the model finds about 30% of a county's claimed block groups in 10% of its block groups, where exposure alone finds 22% and chance 10%.

**Featured county (in the clip).** Rule, not skill: the (event day, county) pair with the most claimed block groups among the forecast-mode events: **Harris County, 2019-05-07**, 166 of 2,109 block groups with claims. The clip uses the
real forecast-mode predictions (GEFS run ready at T-31 h), not the simulated stream. Budget: the top 10% of the county's block groups (211). Claimed block groups inside the budget: **forecast 39**, exposure only 22, random 17, observed rain 53
(within-county AUC 0.72 / 0.62 / 0.70; this county-day sits at the 45th percentile of the 511 county-days, so it is typical, not the best). Reading: sending to the county covers all 166 claimed block groups but also 1,943 block groups without claims;
a 211-block-group budget ranked by the forecast reaches 39 of them, 1.8 times what exposure alone reaches. Block-group positions are drawn at the 4 km pixel centres of the rain grid, so the dots sit on a grid.
