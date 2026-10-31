# Results log

## Stage 0: how strong is the equal-weight vote (2026-10-08)

**Setup** (`scripts/02_baseline_vote.py --train 2018 --test 2021 --quantile 0.99`)
- Hazard: 24 h accumulated precipitation > grid point × season ERA5 (1990–2017) p99, and ≥10 mm (median threshold 11.8 mm)
- Forecast: IFS ENS, 50 members, lead = 72 h, daily 00Z init, 64×32 global grid
- Train 2018 (365 inits), test 2021 (365 inits), event base rates 0.76% / 0.80%
- Thresholds, climatological rates and calibration curves are all fitted on the training / climatology period only

| Method | BS | BSS/B0 | reliability | resolution | AUC-PR | AUC-ROC |
| --- | --- | --- | --- | --- | --- | --- |
| B0 climatology | 8.059e-3 | 0.0000 | 1.11e-5 | 4.8e-8 | 0.0111 | 0.5950 |
| B1 equal-weight vote | 4.576e-3 | **0.4322** | 1.03e-5 | 3.334e-3 | 0.6072 | 0.9627 |
| B2 vote + global isotonic | 4.573e-3 | 0.4326 | 6.29e-6 | 3.333e-3 | 0.6022 | 0.9627 |
| B3 vote + per-latitude-band isotonic | 4.571e-3 | 0.4329 | 6.69e-6 | 3.349e-3 | 0.6148 | 0.9781 |
| B4 global logistic (vote + spread features) | 4.687e-3 | 0.4184 | 4.94e-5 | 3.252e-3 | 0.6030 | **0.9902** |

BS difference relative to B1, from a paired bootstrap by init day (2000 draws, resampling whole inits rather than grid points):

| Comparison | ΔBS | 95% CI | Verdict |
| --- | --- | --- | --- |
| B2 − B1 | −3.3e-6 | [−1.24e-5, +5.1e-6] | **tie** |
| B3 − B1 | −5.2e-6 | [−2.01e-5, +1.01e-5] | **tie** |
| B4 − B1 | +1.11e-4 | [+8.7e-5, +1.38e-4] | **worse** |
| B0 − B1 | +3.48e-3 | [+3.30e-3, +3.66e-3] | worse |

### Conclusions

1. **The equal-weight vote is already strong, and almost already calibrated.** B1's reliability term (1.03e-5) is two and a half orders of
   magnitude smaller than its resolution term (3.33e-3): a democratic vote of 50 members basically needs no calibration at the p99 precipitation scale.
   B2/B3 cut the reliability term by 40%, but on BS this is a **statistical tie**, not a win. Any later method that only "recalibrates the vote"
   has a ceiling of about ΔBS ≈ 1e-5 and is not worth doing.
2. **The spread features do carry extra information, but a global linear combiner cannot extract it.** B4's AUC-ROC jumps from 0.9627
   to **0.9902** (ranking ability clearly improves), but BS gets worse and reliability degrades 5×. This is an exact reproduction of the lesson of the previous line of research,
   "a globally shared linear mixing weight does worse on a heterogeneous population": it is not that the features are useless, it is that the
   **pooled global weights** are useless. → This is exactly the entry point for "per-block mode decision".
3. **The tropics are the top target.** B1's BSS by latitude band: 0.51–0.63 in the northern and southern mid-latitudes, but only
   **0.281 / 0.287** in the two bands on either side of the equator. Convective precipitation has the lowest predictability and the most headroom, and it is also where real hazards concentrate.

### Caveats to remember

- One training year / one test year and a single lead of 72 h; no cross-year robustness check was done (a download takes only 10 s per year, which is cheap, so this should be added)
- Thresholding ENS precipitation with the ERA5 climatological p99 introduces a cross-dataset model bias; a control version using the "ENS's own climatological p99" is still needed
- The 2022 test year **has not been looked at yet**, and is reserved for final confirmation once the method is frozen

### Next steps (by priority)

1. Extend training to 2018–2020 and validation to 2021, and add the BSS curve for lead = 24/72/120/240 h (very cheap)
2. The purely geometric test of stage 1: can displacements between member fields (motion vectors) align the 50 members onto the same
   feature? If the inter-member residual drops significantly after alignment, then "flow-aligned voting" has a physical basis
3. Stage 2: block-level RD mode decision, working only on the gap of "high AUC but poor BS": keep B1's calibration and
   adjust weights per block using only the spread features (rather than refitting one probability globally)

## Stage 1: are motion vectors real (2026-10-08)

The user's intuition: physical processes are continuous → motion vectors should carry time-evolution information → if so, the nanoTSFM
frequency-feature line can be brought back. Also: a fixed FFT window may not be able to dig it out, and wavelets do not have that limitation.

### 1.0 Two real bugs caught during implementation (both "run fine but give wrong results")

1. **A fixed Hann window kills phase correlation.** On a band-limited synthetic field with a true displacement of +3.00 cells, window=True estimates
   **0.23** and window=False estimates **3.00**. The fixed window is stationary in space while the content moves, and for a low-mode field the
   window's own spectrum dominates the cross-spectrum, collapsing the peak to zero lag. → `estimate_shift` now defaults to no window, and this is
   written into the docstring. This is a direct measurement of the cost of a "fixed window".
2. **Pure-phase normalization amplifies empty frequency bins.** Fully normalized (amplitude-blind) phase correlation on a band-limited field also lifts bins whose value is zero
   to unit weight, and a true (0,+2) is estimated as **(+6.16, …)**. → Added a joint amplitude-significance mask.
3. Sub-pixel accuracy: parabolic peak interpolation only reaches ~0.4 cells (2.5→2.08); after switching to a **weighted least squares on the cross-spectrum phase slope**,
   the error is ≤0.11 cells, and still stable at SNR=1 (2.48, -1.43).
4. **Wrong sign in the wavelet-phase LSQ**: b(x)=a(x−d) ⇒ Δφ=−k·d, and the right-hand side dropped the minus sign. The symptom was
   λ=32 estimating **(−3.04, −0.36)** against a true (+3.00, 0.00): the magnitude is right but the direction is flipped overall, and
   judging by the numbers alone it looks completely "usable".

### 1.1 Fair synthetic comparison (same fields, same metric, median EPE, only the estimator changes)

| Case | FFT whole field | FFT block=32 | Wavelet phase flow |
| --- | --- | --- | --- |
| S1 global rigid translation (broadband power-law spectrum) | **0.019** | 2.132 | 0.474 |
| S1b same but narrow-band field (aperture problem) | **0.000** | 1.850 | 0.057 |
| S2 translation + amplitude ×0.7 + SNR5 | — | 2.289 | **0.507** |
| S3 spatially varying flow (shear amp3) | — | 2.247 (block=12 best) | **0.826** |

S4: two scales moving in different directions (coarse scale (+3,0) + 0.8 × fine scale (0,+2.5)):

| Estimator | Estimate | EPE vs coarse | EPE vs fine |
| --- | --- | --- | --- |
| FFT block=32 | (+0.12, +1.39) | 3.210 | 1.112 |
| Wavelet λ=32 | (+2.14, −0.10) | **0.889** | 3.358 |
| Wavelet λ=8 | (+0.14, +1.19) | 3.109 | **1.329** |

**Conclusion**: in the matched case of "global rigid translation", FFT is still the most accurate (0.019), so it cannot be called useless; but once the
flow varies in space (S3, wavelet 2.7× better) or **different scales move in different directions** (S4, FFT can only give one blended vector,
which matches neither truth), the fixed window cannot dig it out. The user's judgement holds. The wavelet magnitude is systematically low
(2.14 vs 3.0), coming from ridge shrinkage + spatial pooling + inter-scale leakage, and needs calibration.

### 1.2 Real IFS ENS fields (Z500, 1.5°, dt=6h, 6 inits × 3 lead pairs)

**A physical trap first**: without removing the background, all estimators nearly fail (wavelet regression slope 0.022, i.e. the estimated
speed is 50 times smaller than the wind). The reason is that the Z500 amplitude is dominated by the **quasi-stationary pole-to-equator geopotential gradient**, which does not advect;
the amplitude-weighted phase estimate is pulled toward zero by a feature that does not move. After removing the zonal mean + high-pass filtering:

| Mode | Residual MAE | vs zero motion |
| --- | --- | --- |
| Zero motion (persistence) | 122.95 | 1.000 |
| **Warp by true 500 hPa wind advection** | 153.72 | **1.250 (worse than not moving)** |
| 0.5 × true wind | 109.18 | 0.888 |
| One vector for the whole field | 108.69 | 0.884 |
| FFT block=32 | 108.96 | 0.886 |
| Wavelet phase flow | **77.13** | **0.627** (wins 18/18) |

**This is the most informative result of the whole stage**: pure advection by the true wind field is worse than doing nothing. So "the estimated motion
vectors correlate with the wind field at only r≈0.33, slope 0.12" is **not a failure but expected**: what it measures is the phase propagation speed of the features
(mid-latitude median |dx|≈0.9 cells vs 2.4 cells of wind advection, about 1/3; Rossby waves retreat westward relative to the flow, with both sign and magnitude right).
Using the wind field as ground truth to "validate" the motion estimate itself is the wrong reference.

### 1.3 Held-out extrapolation: motion vectors carry time-evolution information (8 inits × 5 triples = 40 cases)

The flow is estimated on (l−1, l) and used to predict the **unseen** l+1, with fair degrees of freedom:

| Mode | MAE | vs persistence | Win rate |
| --- | --- | --- | --- |
| persistence (zero motion) | 171.52 | 1.000 | 0% |
| Linear field extrapolation (amplitude only, no motion) | 139.39 | 0.813 | 100% |
| 0.5 × held-out flow | 125.90 | 0.734 | 100% |
| **Held-out flow (estimated from l−1, l)** | **100.64** | **0.587** | **100%** |
| Oracle flow (estimated from l, l+1, not held out) | 79.61 | 0.464 | 100% |

- Held-out extrapolation gets **77%** of the oracle's gain, winning 40/40
- Pure amplitude extrapolation only reaches 0.813 → the dominant mode of 6 h evolution is **motion**, not intensification or decay
- **The motion field's own temporal autocorrelation is r(dx)=0.867±0.046, r(dy)=0.739±0.062**
  → this directly confirms the user's intuition: motion vectors are not frame-by-frame noise, and along lead they are smooth, low-dimensional and predictable

### 1.4 Caveats

- The residual self-fit trap: in script 05 the flow is fitted on the very pair being scored, so lowering `smooth` necessarily lowers the residual
  (0.627→0.486), which does not mean the method got better. Only the held-out extrapolation numbers of script 06 may be cited
- The sample is still small (≤12 inits, all in 2018), and only the ENS **ensemble-mean** field is used; member-level alignment has not been done
- The wavelet magnitude is low and uncalibrated; the confidence metric for the aperture problem (`condition`) has never been used as a gate
- The FFT coherence metric discriminates weakly (pure noise 4.48 vs true translation 5.56) and cannot be used as a confidence measure

### 1.5 Does the frequency representation actually buy anything (57600 trajectories × 20 intervals)

For each pixel-component MV(τ) trajectory take the FFT and compute the cumulative energy share of the first k non-DC coefficients.
**The control is AR(1) surrogate series matched in variance and lag-1 autocorrelation** (8 realizations);
otherwise "a smooth series can be represented by a few low-frequency coefficients" is a triviality and shows nothing.

Measured lag-1 autocorrelation of the trajectories: median 0.726, IQR [0.572, 0.827]

| k | Real trajectories | AR(1) surrogate | Difference |
| --- | --- | --- | --- |
| 1 | 0.412 | 0.328 | +0.084 |
| 2 | 0.699 | 0.566 | **+0.133** |
| 3 | 0.822 | 0.694 | +0.128 |
| 4 | 0.884 | 0.776 | +0.108 |
| 5 | 0.926 | 0.834 | +0.092 |
| 6 | 0.948 | 0.880 | +0.068 |

**Conclusion**: the difference is positive throughout and largest at k=2–3 (+0.13), i.e. the energy concentration of the MV trajectories **exceeds** what
their own autocorrelation implies: there is frequency structure beyond "mere smoothness". **3 coefficients cover 82% of the trajectory
energy**, so the nanoTSFM frequency head can indeed be brought back: not to encode the meteorological field itself, but to
encode the **trajectory of the motion field along lead**.

**Caveats**: AR(1) can serve only as a null model for "pure smoothness"; the low-frequency excess might also come from a monotonic drift along the lead direction
(a trend) rather than oscillatory structure, and the two cannot currently be told apart. Telling them apart needs an additional "recompute after detrending" control,
and a check of whether the excess energy is concentrated at particular periods (rather than all piled up at k=1).

---

# Stage 2: classical K-means baseline on discretised targets (CO/NM/AZ county-day, SCS3)

Scripts `08_build_labels.py` → `09_fetch_era5_county_env.py` → `10_kmeans_baseline.py`.
Target: wind + hail + tornado county-days, 4 nested heads (occurrence / ≥$10k / ≥$100k / ≥$1M, nominal dollars), May–Sep,
112 counties × 2000–2022. Train 2000–2016, K chosen on 2014–2016 (previously fitted on 2000–2013 only),
finally refitted on 2000–2016, test 2017–2022. Baseline **p0 = county rate × month rate / overall rate** (training period only).

## Three nested feature tiers

| Tier | Meaning | Features |
| --- | --- | --- |
| A | Beyond T-3D | Calendar sin/cos(doy) + county lat/lon |
| B | T-3D to T-3h | A + same-day 12Z ERA5 environment (CAPE, CIN, dewpoint depression, PW, 850q, 700-500 lapse rate, 500 hPa to 10 m shear, 700 wind speed, MSLP) |
| C | Below T-3h | B + the same set at 18Z + 12→18Z tendencies (CAPE/T2m/Td2m/PW) + two 6 h precipitation windows |

ERA5 0.25° nearest grid point; **observations stand in for the forecast (perfect-prog), so B/C are upper bounds**. Apart from calendar/location,
features first have the training-period "county × month" mean subtracted and are then standardised. ENSO/AO/MJO indices are **not included**, so tier A is only a refined climatology.

## Results (test 2017–2022, BSS relative to p0, year-resampled 95% CI in parentheses)

Read-out `lift`: p = p0 × cluster lift factor (shrunk with 20 pseudo-counts).

| Tier | K | Occurrence | ≥$10k | ≥$100k | ≥$1M |
| --- | --- | --- | --- | --- | --- |
| A | 128 | +0.0069 (0.0044, 0.0099) | +0.0041 (0.0018, 0.0064) | +0.0022 (0.0010, 0.0031) | +0.0002 |
| B | 32 | **+0.0265** (0.0202, 0.0338) | **+0.0176** (0.0127, 0.0222) | +0.0069 (0.0029, 0.0098) | +0.0004 |
| C | 64 | **+0.0431** (0.0362, 0.0518) | **+0.0188** (0.0137, 0.0233) | +0.0053 (0.0020, 0.0075) | +0.0003 |

Occurrence-head AUC-PR: p0 0.070 → A 0.083 → B 0.113 → C 0.139. Number of test positives: 2442 / 288 / 97 / 16.

- **Using the within-cluster occurrence rate directly (`rate`) loses to p0**: −0.013 for tier B ≥$10k, with a CI that excludes 0. K-means alone cannot serve as a probability;
  it must sit on top of climatology, with clusters only making a correction
- **The gain is mainly in the occurrence head and ≥$10k**; ≥$100k is small, and ≥$1M has only 16 positives in the test set, so it is unreadable
- **The +0.007 of tier A is not predictive skill**: calendar + location clustering with K=128 is equivalent to a climatology finer than "county × month"
  (on the order of county × week); this is a granularity issue of p0, not lead-time information
- **By state**: AZ gains the most (occurrence head tier C +0.063, ≥$10k +0.029), CO and NM about +0.04 (occurrence);
  CO/NM are basically 0 above ≥$10k
- The B→C gain (occurrence head +0.017) comes from the 18Z state, tendencies and the preceding 6 h precipitation, which within T-3h are **observed quantities**,
  not forecasts, so tier C is closer to "nowcasting" than to "forecasting"

## Caveats

- B/C use ERA5 observations; a real forecast would have errors and the actual gain would be smaller; this is a perfect-prog upper bound
- The day boundary uses the Storm Events local date while features are taken at UTC 12Z/18Z, so there is a small misalignment at the boundary
- The K selection curve is very flat (tier B differs by only 0.005 between K=16 and 128), so the chosen K is unstable; do not interpret K itself
- CIN is undefined when CAPE=0 and was filled with 0; the K index was not returned by ERA5 and was not used
- The 2017–2022 test period was used only once, and no parameter was tuned on the test results
- Features take the 0.25° nearest grid point to the county centroid (about 25 km), which has limited representativeness for within-county variation in large western counties

---

# Stage 3: how much can a wavelet scale encoder add on the existing data (2026-10-19)

Scripts `11_build_scale_features.py` (features), `12_scale_encoder.py` (experiment). Data, split and p0 are the same as stage 2:
train 2000–2016, test 2017–2022 (the test period is used only to produce numbers, with no tuning on it), **K fixed at 32, not tuned**.

## Encoder

12Z/18Z fields over a 0.25° box around the county (about 21°×14°), 9 environment channels. Each channel is split with `wavelet_motion.morlet_bank`
into 4 scales (wavelengths 32/16/8/5 cells, about 800/400/200/125 km); at each scale the county's **band-pass value** and **local amplitude** are taken,
plus a **low-pass term** with σ=16 cells. From 12Z→18Z, `scale_resolved_flow` gives **per-scale motion vectors** for PW and MSLP.

Three nested tiers: A3 = the state at 12Z of D-3 (truly using only information from 3 days earlier); B = 12Z of day D; C = 12Z+18Z+motion+tendency+6 h precipitation.
`Cnm` = C with the motion channels removed (ablation). B and C still use ERA5 observations and are upper bounds.

Methods (all report BSS vs p0; "paired" differences use a year-grouped bootstrap):

| Method | Meaning |
| --- | --- |
| KM-point | K-means on point features (stage 2 baseline) |
| KM-pooled | K-means on all scale features concatenated |
| Vote-LR | One K-means expert per scale (+ low-pass context), fused with learned weights |
| RandVote-LR | **Control**: the same columns randomly split into 5 groups, then voted |
| ChanVote-LR | **Control**: the same columns split into 10 groups by variable, then voted |
| HGB-point / HGB-scale | Gradient boosting, point features vs scale features, learner held fixed |

## Results: occurrence head y0, BSS vs p0

| Method | A3 | B | C | Cnm |
| --- | --- | --- | --- | --- |
| KM-point | 0.004 | 0.029 | 0.038 | 0.038 |
| KM-pooled | 0.002 | 0.034 | 0.051 | 0.050 |
| Vote-LR (per scale) | 0.005 | 0.040 | 0.060 | 0.061 |
| RandVote-LR (control) | 0.005 | 0.042 | 0.059 | 0.063 |
| ChanVote-LR (control) | 0.007 | **0.050** | **0.075** | 0.075 |
| HGB-point | 0.010 | 0.055 | 0.091 | 0.091 |
| HGB-scale | 0.008 | **0.068** | **0.108** | 0.105 |

Paired differences (ΔBSS, 95% CI), y0:

| Comparison | B | C |
| --- | --- | --- |
| Vote-LR − KM-point | +0.012 (0.007, 0.016) | +0.023 (0.018, 0.028) |
| Vote-LR − RandVote-LR | −0.002 (−0.006, 0.001) | +0.002 (−0.000, 0.004) |
| Vote-LR − ChanVote-LR | **−0.011 (−0.018, −0.006)** | **−0.016 (−0.021, −0.012)** |
| HGB-point − KM-point | +0.027 (0.021, 0.033) | +0.055 (0.046, 0.061) |
| HGB-scale − HGB-point | **+0.014 (0.009, 0.019)** | **+0.018 (0.015, 0.022)** |

## Conclusions

1. **Scale voting beats point-feature K-means, but this is not thanks to "scale structure"**. Vote-LR **ties** the random-grouping control
   (B: −0.002, CI contains 0), which shows the gain comes from "an ensemble of several K-means + learned weights". The control grouped by variable is actually significantly better
   (B higher by +0.011, C by +0.016). **The hypothesis "vote by scale" is not supported on the existing data**
2. **The learner matters more than the features**: with the same point features, GBM beats K-means by +0.027 (B) and +0.055 (C). The K-means
   cluster read-out is a weak learner, and the +0.027/+0.043 of stage 2 cannot represent the information content of these features
3. **Multi-scale features are indeed useful with a fixed learner (GBM)**: HGB-scale beats HGB-point by +0.014 (B) and +0.018 (C),
   with CIs that exclude 0. This is the only evidence in this stage that "scale decomposition has value"
4. **But the ≥$10k head does not show this advantage**: HGB-scale − HGB-point is B +0.006 (−0.003, 0.014) and
   C −0.008 (−0.017, 0.003), not significant. For the ≥$100k head all methods have CIs containing 0, so it is unreadable
5. **Motion features have no demonstrable contribution**: for C vs Cnm on y0, HGB-scale is 0.108 vs 0.105 (a difference of about 0.003, within noise),
   the voting methods are slightly lower at C than at Cnm, and on y10k removing motion is better. **The motion structure of stage 1 did not translate into skill on this target**
6. **A3 (beyond T-3D, using only the state of D-3) is close to 0 for all methods** (highest 0.010). The 3-day persistence of the large-scale state carries almost
   no information; this tier needs forecasts or climate signals (ENSO/MJO), not a better encoding

## Caveats

- **A key control is missing**: the gain of HGB-scale may come from "neighbourhood information around the county" rather than from the "wavelet scale decomposition" itself.
  There is not yet a "fixed-radius neighbourhood mean/variance" control to tell the two apart, so conclusion 3 can only say that "neighbourhood + multi-scale features are useful"
- The box is only about 21°×14° with a maximum scale of about 800 km, which cannot cover planetary waves or upstream troughs and ridges, so **large-scale conclusions cannot be drawn from this data**
- The motion estimate's magnitude is low by about half (synthetic test: truth 2 cells, estimate about 1 cell), with correct direction and sign; channels other than PW and MSLP were not tried
- B and C are perfect-prog upper bounds; A3 is the true beyond-T-3D tier and also the weakest
- The number of GBM iterations was chosen on 2014–2016; the hyperparameters of K-means, voting and HGB were not tuned on the test period
- The test-period data was read repeatedly in the smoke test and two full runs, but no method setting was modified based on it; the second run
  only added the paired comparisons and the Cnm ablation

---

# Stage 4: data audit: is the weak lift a normalisation problem, a label ceiling, or a learner problem (2026-10-19)

Script `13_data_audit.py`, with full output in `~/wxrisk_data/audit/report.txt`. Tier B (target day 12Z), train 2000–2016,
test 2017–2022, BSS vs p0. Several impressions from the small-sample smoke test (blank rate of early damage fields, first-half vs second-half ratio of y0) were **corrected** on the full data;
only full-data numbers are quoted below.

## 1. Normalisation: not a problem, but absolutely necessary

- The standard deviations of the raw features span **8×10⁵** (0.0006 to 509). K-means with **no normalisation at all** has a BSS of **0.0003 / 0.0012**
  (point/scale), i.e. it learned nothing
- Under four reasonable normalisations (county × month anomaly z, global z, anomaly + quantile → normal, anomaly z clipped at ±3) the K-means BSS values differ from one another by
  ≤0.003 (point 0.024–0.029, scale 0.033–0.034). **Changing the normalisation does not change the conclusion**
- Skew / heavy tails are concentrated in CIN amplitude, precipitation windows and motion vectors (kurtosis 8–44); 12 columns in total have kurtosis >20; clipping at ±3 brought no improvement
- Train→test drift is small: the median drift of the mean of anomaly features is 0.04σ, the maximum 0.23σ (cape_12_lp)
- Collinearity: of 20301 pairs, 103 have |r|>0.95 and 24 have >0.99, which is not a problem
- **Anomalising is not a pure win**: with GBM on scale features, y0 using "global z (keeping absolute level)" is **0.0771**, higher than "county × month anomaly"
  at 0.0676 (+0.010); for y10k it is the other way round (0.0418 vs 0.0439). Absolute level carries extra information for the occurrence head, so both are worth providing

## 2. Pipeline leakage: none

After shuffling features within "county × month", the test BSS is +0.0047 (scale) and +0.0056 (point), not 0. This **~0.005 is recalibration of static information**
(the combination of lat/lon/month/logit p0 is better than p0's "county rate × month rate" product form), showing that **p0 itself is not the optimal static baseline**:
all earlier BSS values contain about 0.005 of this, and the gain truly coming from the environment fields must have it subtracted

## 3. Learner: capacity is not the bottleneck

Scale features, y0, test BSS:

| Learner | Test BSS | Train BSS |
| --- | --- | --- |
| LogReg (C=0.01…1) | 0.064 | 0.053 |
| HGB depth 2 / 3 / 4 / 6 / 8 | 0.065 / 0.066 / 0.068 / 0.066 / 0.067 | 0.066 / 0.091 / 0.131 / 0.134 / 0.163 |
| MLP 128-64 | 0.060 | 0.067 |

- The linear model already gets 95% of HGB; going from depth 2 to 8, train BSS rises 2.5× while test does not move, so **the extra capacity is only overfitting**
- On point features it is different: LogReg 0.044, HGB 0.052–0.054, MLP 0.055. **The multi-scale representation "linearises" the problem**:
  with LogReg, scale features are higher than point features by +0.020, while with HGB only by +0.014
- K-means (0.034) is half the linear model (0.064); **K-means is the genuinely weak learner in this batch of experiments**

## 4. Data volume: limited

Learning curve (the training set is a random subset of years from 2000–2013, 3 repeats):

| Training years | 2 | 4 | 7 | 14 |
| --- | --- | --- | --- | --- |
| Point feature BSS | 0.038 | 0.041 | 0.045 | 0.054 |
| Scale feature BSS | 0.045 | 0.053 | 0.061 | 0.068 |

At 14 years it is **still rising** (7→14 years +0.007~0.009), and train BSS falls as the number of years grows (overfitting converging). The advantage of scale features over point features
widens with data volume (+0.007 → +0.014), and the 2-year scale-feature model (0.045) is already higher than the 4-year point-feature model (0.041)

## 5. Label ceiling: county-day randomness accounts for most of it

**oracle**: use "actual same-day reports in neighbouring counties" as a feature (not forecastable, only a reference for regional coherence), test BSS:

| Neighbourhood | y0 | y10k |
| --- | --- | --- |
| Same state | 0.104 | 0.033 |
| Within 150 km | **0.138** | 0.036 |
| Within 300 km | 0.121 | 0.035 |
| Whole box | 0.043 | 0.017 |

- y0: tier C HGB-scale (0.108) is already at **78%** of the 150 km oracle
- y10k: our 0.042–0.049 is **already higher than** the neighbouring-county oracle (0.036): county-day outcomes at damage magnitude are mostly individual randomness
- This is not a strict upper bound (the oracle and the environment fields are different information), but it shows that **the county-day variance a regional daily factor can explain is limited**

State of the labels themselves (full data):
- The y0 annual rate is 156–295 per 10,000 county-days, and the ratio of the second half to the first half (2000–2010 vs 2012–2022) is 1.04; for y10k it is 0.90. **Large year-to-year fluctuation, very small trend**
- The blank rate of damage fields falls from 85% up to 2005 to 27–38% later, but the actual effect on y10k is only a ratio of 0.90,
  smaller than feared in the small sample
- SCS3 records in CO/NM/AZ are **all county-level records** (no loss from regional records)
- The median time of the first report is 23Z, 25–75% within 21Z–24Z, and 38% fall on the next UTC day; only 1.8% are before 12Z and 3.5% before 18Z, so
  **the feature frames of tiers B/C are almost all before the event**, and contamination is negligible

## Attribution

1. **Not a normalisation problem**: no normalisation fails completely (0.0003), but all reasonable normalisations are equivalent to each other; there is no pipeline leakage
2. **Not a learner-capacity problem**: from the linear model through depth-8 GBM to MLP, all land at 0.060–0.068 on scale features;
   more capacity only adds overfitting. **K-means is the exception; it really is too weak**
3. **It is information / label limited**: (a) county-day Bernoulli randomness is large, and the neighbouring-county oracle only reaches 0.138; (b) the learning curve is still rising,
   showing that data volume is also a constraint; (c) y0 fluctuates strongly year to year across the 23 years 2000–2022
4. So whether "reasonable feature engineering helps" must be read in layers: **the (multi-scale) features do help** (with both linear and GBM, and more clearly with small data),
   but the reachable ceiling is capped by label randomness, and piling on complexity cannot break through it

## Caveats

- The audit was done on tier B, y0/y10k; the ≥$100k head has too few samples (97 test positives) and was not included
- The test period was used repeatedly by these diagnostics, but no setting was tuned on test results; the number of GBM iterations is still chosen on 2014–2016
- The oracle is a diagnostic quantity, not a forecast: same-day reports in neighbouring counties are not available at forecast time
- The learning curve only sampled 2000–2013, with a maximum of 14 years, and was not extrapolated to more years

---

# Stage 5: exploring other label sources, and how big the problem with the labels themselves is (2026-10-19)

Scripts: `15_source_agreement.py` (agreement of SPC/RMA with SE), `16_warning_labels.py` (IEM warning polygons → county-day),
`17_mesh_vs_reports.py` (MESH radar hail vs hail reports), `14_label_comparison.py` (same features and learner, different labels).
Data are in `~/wxrisk_data/labels_extra/` (SPC, RMA, IEM, Census county boundaries, single-day MESH files).

## Source availability

| Source | Status |
| --- | --- |
| SPC reports | Downloaded; 19,560 records with county codes for CO/NM/AZ May–Sep 2000–2022 |
| RMA cause of loss | Downloaded for 2000–2023; the payout amount is in column 29 (cross-checked by back-computing from loss ratios) |
| IEM warning polygons | Downloaded 2002–2023 storm-based (about 5 MB per year), mapped to 25,902 county-days |
| MRMS MESH | AWS `noaa-mrms-pds` starts **2020-10-14**; May–Sep 2021–2023 fetched, 456/459 days (one ~1 MB file per day). A spot check of IEM MTArchive for 2018-06-01 found **no MESH** |
| EAGLE-I | **Could not be obtained**: figshare and the ORNL page return 403 to the command line, and the DOI page only offers a Globus download. Needs a manual download |
| SHELDUS | The subscription contains only aggregated data (no exact dates); county-day needs raw data bought per record (education tier $0.10 per record) |

## Consistency checks (external corroboration of the labels)

- **SPC and SE are the same batch of data**: exact county-day dates coincide for 99.0% (9,054 / 9,149), and within ±1 day for 99.7%. SPC provides no independent information,
  only timeliness and magnitudes
- **RMA crop insurance payouts** (county-month, May–Sep, agricultural counties only):

| | Months with a payout that have an SE report | Months with an SE report that have a payout | Spearman (payout vs report count, all county-months) | Spearman (payout vs SE dollars, both >0) |
| --- | --- | --- | --- | --- |
| Hail | 58% | 43% | 0.40 | **0.07** |
| Wind (Wind/Excess Wind) | 36% | 22% | 0.16 | 0.17 |

  Among county-months with an actual insured hail payout, **42% have no hail report at all in SE**; and the dollar loss recorded in SE is almost uncorrelated with the actual payout
- **Within SE, dollars are decoupled from intensity**: the Spearman correlation of hail/wind intensity with loss is only 0.15 / 0.11. The share of hail loss ≥$10k
  by size is 0.2% (≤0.75 in), 0.5% (0.75–1), 2.5% (1–1.75), 7.2% (1.75–2.75), 18.9% (>2.75)
- **MESH vs reports** (2021–2023, 51,072 county-days, MESH taken as the county maximum):

| MESH threshold | County-days | Of which with ≥1 in report |
| --- | --- | --- |
| ≥10 mm | 11,630 | 4.8% |
| ≥25.4 mm (1 in) | 3,281 | **15.7%** |
| ≥38 mm | 1,337 | 30.3% |
| ≥50 mm (2 in) | 614 | 44.0% |

  In the other direction: among county-days with a ≥1 in report, 90–97% have MESH ≥25.4 mm. The share with MESH ≥1 in but no ≥1 in report:
  AZ 93%, NM 87%, CO 78%. **Radar sees almost all the hail that is reported, but reports cover only a small fraction of what radar shows**
- **Warnings and reports** (2017–2022): 80% of county-days with a report have an SVR/TOR warning; but of warned county-days only 31.5% have a report.
  Warnings are 610 per 10,000 county-days, reports 239

## Same features, different labels (test 2017–2022, GBM scale features)

`AUC lift` = model AUC − AUC of p0 alone (independent of base rate, comparable across labels); BSS varies with the base rate and **cannot be compared directly across labels**.

| Label | Per 10,000 county-days | BSS (tier B / C) | AUC lift (tier B / C) |
| --- | --- | --- | --- |
| Any report | 239 | 0.068 / 0.108 | 0.107 / 0.134 |
| NWS severe (hail ≥1″, wind ≥50 kt or wind damage with no gust value, tornado) | 216 | 0.065 / 0.102 | 0.106 / 0.134 |
| Significant severe (hail ≥2″, wind ≥65 kt, EF2+) | 31 | 0.016 / 0.027 | 0.079 / 0.104 |
| Hail ≥1″ | 107 | 0.061 / 0.093 | 0.111 / 0.130 |
| Wind ≥50 kt | 119 | 0.037 / 0.047 | 0.069 / 0.096 |
| Tornado | 20 | 0.015 / 0.017 | 0.114 / 0.141 |
| Loss ≥$10k | 28 | 0.043 / 0.042 | **0.064 / 0.075** |
| Loss ≥$100k | 9.6 | 0.018 / 0.009 | **0.052 / 0.073** |
| Any report within 150 km | 1,303 | 0.168 / 0.239 | 0.120 / 0.147 |
| Any report in the same state | 3,541 | 0.207 / 0.246 | 0.149 / 0.166 |
| Warning SVR/TOR (2002–2022 sample) | 610 | 0.149 / 0.222 | 0.129 / 0.157 |
| Warning TOR | 43 | 0.033 / 0.053 | 0.086 / 0.106 |
| Warning FF (flash flood) | 345 | 0.097 / 0.115 | 0.093 / 0.113 |

In the same sample (2002–2022), any report is BSS 0.067 / 0.107 and AUC lift 0.108 / 0.133.

## Conclusions

1. **Dollar loss is the weakest label**: its AUC lift is only 55–60% of that of the occurrence labels (0.064/0.075 vs 0.107/0.134),
   the neighbouring-county oracle is also below the model (no regional coherence), dollars are almost decoupled from intensity (Spearman 0.11–0.15), and almost uncorrelated with insurance payouts
   (hail 0.07). As a prediction target, a loss threshold is much worse than the occurrence / intensity labels
2. **The problem with the occurrence labels is that "what is measured is incomplete"**: of radar-indicated ≥1 in hail county-days, 84% have no report; of county-months
   with an insured hail payout, 42% have no report. A report looks more like the product of "hail × an observer being present"; p0's county × month rate already absorbs the average observer density,
   and what remains is the day-to-day variation
3. **Changing the label is not a shortcut to more skill**: replacing "any report" with an intensity threshold (NWS severe, hail ≥1″) leaves AUC lift unchanged
   (79% of reports are severe anyway); smoothing to 150 km or to state level more than doubles BSS, but AUC lift gains only 10–39% (150 km: +12%/+10%, state: +39%/+24%),
   and most of the growth in BSS comes from the higher base rate; the warning label doubles BSS (0.149/0.222) and gains about 20% in AUC lift (0.129/0.157)
4. So the answer to "is the label a big problem" is **layered**: for **loss** yes; for **occurrence**, it is mainly completeness (false negatives)
   rather than large random noise, and switching to a more objective or smoother label can only raise "the discrimination the environment can explain" by 10% to 40%
5. Consistent with the project slides' structure "hazard forecast q → impact model τ → county loss ψ": **the hazard layer should be trained with objective labels
   (MESH, warnings, intensity), and loss should be left to the impact layer**. Predicting directly with a loss threshold, as is done now, puts the weakest link at the very front

## Caveats

- **MESH is not ground truth**: it overestimates, and here the county maximum is taken, so large counties more easily have a ≥1 in grid cell, so "84% unreported"
  is an upper bound and not all of it is missing reports; only three seasons, 2021–2023
- RMA covers only agricultural counties and is monthly rather than daily; matching payouts to hail by month risks a lag
- Warnings are forecasters' judgements, reflecting hazards visible to radar/environment, which is not the same as occurrence; there are 2.6 times as many as reports
- AUC lift is not a skill score, only a difference in discrimination; cross-label comparison is still affected by the number of positives (tornado and significant severe have fewer than 320 positives)
- SE intensity thresholds: wind records with no gust value are treated as severe (the SPC convention); 99% of wind records in this region carry a gust value
- The same test years were read many times in this stage, but no setting was tuned on them
- **EAGLE-I was not explored** (it cannot be downloaded outside a browser), and remains the preferred candidate for the wind impact layer

---

# Stage 6: quality of flooding data sources (2026-10-19)

Scope as for SCS: CO/NM/AZ, 112 counties, May–Sep, 2000–2022, local standard-time dates. Scripts: `18_flood_sources.py`
(source agreement), `19_qpe_vs_flood.py` (objective rainfall), `14_label_comparison.py --labels f_ff …` (same features, different labels);
modules `labels_flood.py`, `mrms.py`. Data are in `~/wxrisk_data/labels_extra/flood/`.

## Source profile

| Source | County-day positives | Status |
| --- | --- | --- |
| SE Flash Flood (county-level) | 3,601 | **100%** of Flash Flood records are county-level, 93% of Flood; 40% have a loss, 23.5% ≥$10k, 6.4% ≥$100k |
| SE Flood (multi-day, expanded to begin..end) | 806 | Median duration 0 days, 90th percentile 4 days, 99th percentile 18 days |
| NWS FF warnings (IEM, from 2002) | 8,877 | 127→666 per 10,000 county-days from 2002 to 2022 (about 5×); issuing habits are changing |
| NFIP claims (OpenFEMA **v3**; v2 was retired on 2026-10-15) | 1,953 | Extremely low penetration, see below |
| USGS daily-mean discharge surge (≥3× the previous day and reaching the site's 95th percentile) | 7,616 (only the 92 counties with gauges) | Daily means miss short flash floods; the threshold was set by me and is unvalidated |
| FEMA county-level flood declarations | 1,558 | Event-level, covering only 4.9% of SE county-days |
| RMA crop "Flood" | Only 69 county-months | Basically unusable |
| NWM reanalysis (AWS) | — | Only confirmed to exist, **not explored** |

## Agreement (county-day)

| A → B | P(B\|A) | P(A\|B) | Notes |
| --- | --- | --- | --- |
| SE flash flood → FF warning | 0.81 (±1 day 0.86) | 0.32 (0.42) | Almost the same as SCS's 0.80 / 0.31 |
| SE flood → NFIP claim | 0.15 (0.22) | 0.32 (0.46) | |
| FF warning → NFIP claim | 0.07 (0.12) | 0.35 (0.51) | |
| SE flood → USGS surge (counties with gauges) | 0.36 (0.51) | 0.18 (0.32) | The surge criterion is on the loose side |
| SE flood → FEMA declaration | 0.05 | 0.14 | |

## NFIP: independent loss data barely exists in this region

- County residential penetration: median **0.48%**, with 65% of counties <1%; region-wide 40,100 policies / 5,448,744 residential structures = 0.74%
- CO 0.57%, AZ 0.66%, NM 1.4%; against 2.9% nationally, Louisiana 22%, Florida 13.7%, South Carolina 6.1%, Texas 5.8%;
  only 109 of 3,159 counties are ≥10%
- May–Sep claims over 23 years total 1,953 county-days; 2013 (the Colorado flood) alone has 2,426 records, and the top 5 dates account for 39% of all claims
- As with the loss labels of SCS, **NFIP is not a usable independent loss source in this region**

## Objective rainfall (MRMS MultiSensor QPE 24 h, 2021–2023, 51,072 county-days)

| QPE county maximum | County-days | SE flash flood | FF warning | NFIP claim | SE ≥$10k |
| --- | --- | --- | --- | --- | --- |
| ≥25 mm | 11,772 | 5.4% | 15.3% | 1.5% | 1.9% |
| ≥50 mm | 4,813 | **11.1%** | 27.2% | 2.8% | 3.8% |
| ≥75 mm | 1,614 | 19.7% | 38.8% | 4.1% | 7.2% |
| All | 51,072 | 1.3% | 3.8% | 0.4% | 0.4% |

- In the other direction: 98% of flash flood county-days have QPE ≥25 mm (90% ≥40 mm), with a median of 74 mm against 3 mm on other days; warnings 92% / 79%,
  NFIP 87% / 75%
- Ranking by QPE alone, AUC: flash flood **0.952**, warnings 0.928, ≥$10k 0.945, NFIP 0.867, stable across states (0.82–0.95)
- **The coupling between flood occurrence and rainfall is tighter than the coupling between SCS reports and any existing variable**; but reports are just as incomplete: only 11% of county-days with ≥50 mm
  have a flash flood report (SCS with MESH≥1″: 16%)

## Same features, different labels (test 2017–2022; 2002–2022 sample; GBM scale features; the features are SCS-oriented, not flood-oriented)

`AUC lift` = model AUC − AUC of p0 alone.

| Label | Per 10,000 county-days | BSS (B / C) | AUC lift (B / C) | Compared with neighbouring-county oracle |
| --- | --- | --- | --- | --- |
| SCS any report (same sample) | 239 | 0.067 / 0.107 | 0.108 / 0.134 | 49% / 77% |
| SE flash flood | 118 | 0.085 / 0.095 | 0.070 / 0.085 | **103% / 115%** |
| SE flood or flash flood | 125 | 0.080 / 0.092 | 0.085 / 0.100 | 94% / 109% |
| SE flood loss ≥$10k | 35 | 0.030 / 0.032 | 0.064 / 0.081 | oracle lower |
| SE flood loss ≥$100k (118 positives) | 12 | 0.004 / 0.009 | 0.090 / 0.103 | — |
| FF warning | 345 | 0.097 / 0.115 | 0.093 / 0.113 | 45% / 54% |
| NFIP claim | 44 | 0.029 / 0.027 | 0.035 / 0.039 | oracle≈0 |

- p0 (county × month) already has an AUC of 0.85 for floods (SCS 0.77); flood season and county location are highly predictable, so **the features can buy less extra discrimination**
- Model BSS matches or even exceeds the neighbouring-county oracle: the regional coherence of flood reports is weak; they are **local events**
- NFIP is the least learnable (AUC lift 0.035/0.039)

## Flood dollar loss is likewise decoupled from intensity

Among flash flood county-days, the Spearman correlation between the county maximum QPE and SE dollar loss is **0.034** (0.042 when loss >0); the share ≥$10k barely changes across QPE
bins (≤25 mm 45%, 25–50 38%, 50–75 30%, 75–100 35%, >100 mm 37%). As with SCS (intensity–loss 0.11–0.15),
**the dollar label does not vary with hazard intensity**.

## Stability

| | Ratio of later to earlier period, 2000–2010 / 2012–2022 | Largest single-year share |
| --- | --- | --- |
| SE flash flood county-days | 0.65 | 9% |
| SE flood ≥$10k | 0.51 | 11% |
| NFIP claims | 0.64 | 12% |
| FF warnings (from 2002) | about 0.2 (5× growth) | — |

SCS equivalents: any report 1.04, ≥$10k 0.90. **The reporting convention of flood labels drifts more severely**, and the many events of 2013, 2021 and 2022
(the Colorado flood, the monsoon and burn scars) raise the later period.

## Comparison with SCS

| | SCS | Flooding |
| --- | --- | --- |
| Under-reporting of the occurrence label | 16% of MESH≥1″ have a report | 11% of QPE≥50 mm have a report |
| Coupling of occurrence to objective driver | MESH: 90–97% of reported hail has MESH≥1″ | QPE: AUC 0.95, 98% of flash floods ≥25 mm |
| Dollars vs intensity | Spearman 0.11–0.15 | **0.034** |
| Independent loss data | EAGLE-I (wind, not obtained), RMA (sparse) | NFIP penetration 0.74%, RMA 69 county-months: **worse** |
| County-day positives (any) | 9,149 | 4,310 |
| Label stability | Fairly steady (1.04) | Marked drift (0.65) |
| Warnings agree with reports | 80% / 31% | 81% / 32% |

## Conclusions

1. **The "hazard layer" of floods is better than that of SCS**: occurrence is tightly coupled to a measurable driver (24 h rainfall), AUC 0.95, whereas the radar quantity corresponding to SCS reports
   is only MESH from 2020 onward and is looser
2. **The "impact layer" of floods is not better**: dollars are almost unrelated to rainfall amount (0.034), there is almost no independent loss data in this region (NFIP penetration 0.74%,
   RMA 69 county-months); labels are less stationary and positives fewer
3. With the same (SCS-oriented) features, the learnability of floods is comparable to SCS (BSS 0.085 / 0.095 vs 0.067 / 0.107), and AUC lift is actually lower,
   because p0 already explains a lot; this comparison is **unfair to floods**, since there are no flood-oriented features such as rainfall, soil moisture, terrain / burn scars
4. **Switching to flooding does not escape the problem of "weak loss labels"; it only changes its form**. If a switch is considered, **region** matters more than hazard type:
   NFIP has usable coverage only in Louisiana, Florida, South Carolina and Texas (10–40 times higher than CO/NM/AZ)

## Caveats

- QPE takes the county maximum, which is high for large counties; 2021–2023 gives only three seasons; Pass2 is radar plus gauges and mountains are still blocked, but the AUC of 0.95 here shows
  it is sufficient for these labels
- The USGS surge criterion (3×, 95th percentile) was set arbitrarily by me, with no sensitivity analysis and **no learnability test**
- The NWM reanalysis was only confirmed to exist; its time range and quality were not explored
- In the label comparison the flood features are SCS-oriented; the AUC-ceiling effect of p0 makes AUC lift low
- The fields of NFIP v3 may differ slightly from v2; I used only the date, county code and payout amounts

---

# Stage 7: mentor feedback and the granularity of flood outcomes (2026-10-20)

Mentor feedback (summary): the other three groups chose hurricanes, floods and wildfires, and the flood group moved to county-level analysis; if we choose floods we should work at **finer granularity**;
SHELDUS aggregated data is usually free of charge, but county-level outcomes are too coarse; they suggested looking at NFIP claims or remote-sensing indicators; the flood scope must explicitly state what is included and excluded,
but must not be so narrow that there is no data. Scripts: `20_fetch_nfip_claims.py`, `21_nfip_granularity.py`; data `~/wxrisk_data/labels_extra/flood/`.

## NFIP v3 claims (2000–2023, national)

1,564,325 claims, $69.5B paid, covering 2,855 counties, 95,652 block groups and 21,560 ZIP codes.

| Dimension | Precision and completeness |
| --- | --- |
| Space | `censusGeoid` is a **12-digit census block group**, 96–99.7% complete; ZIP 100%; **latitude/longitude only to 1 decimal place (about 11 km), unusable as point locations** |
| Time | `dateOfLoss` is exact to the day |
| Intensity | Building-level water depth (`waterDepth`) is present for 80–95% (dropping to about 80% from 2019); **interior water depth only 14.5%** |
| Duration | `floodWaterDuration` is almost always present before 2018, **52% in 2019 and 0.1–6% in 2020–2023**, unusable in recent years |
| Damage | `buildingDamageAmount` present for 72–90%; Spearman 0.968 with the payout |
| Event | The event name `floodEvent` is present for 75% overall, but varies greatly by year (17% in 2006, 94% in 2012); the EDN field is almost empty |
| Cause | 4 rainfall/snowmelt ponding 44.1%, 2 river/lake overflow 22.2%, 1 tidal overflow 19.5% (25.7% of payouts), 0 other 10.7%, 3 alluvial fan 0.3% |

## Defining the flood scope

- **Named tropical events account for 62.4% of claims and 78.7% of payout dollars** (Katrina, Sandy, Harvey, Ike, Irene, Ian, etc.;
  the corresponding figure in Storm Events is 55%)
- Suggested scope: inland flooding (cause codes 2/3/4), excluding tidal overflow (1), landslide/erosion (7/8/9) and named tropical events
  - Non-tropical inland scope: **462,684 claims, $12.7B**, covering 74,968 block groups, 2,792 counties, 55 states
  - Top ten states: TX 64,417, LA 55,589, FL 43,158, NJ 27,706, NY 23,947, IL 18,345, PA 18,016, MO 13,567, OH 11,633, CA 10,961
  - 246,902 block-days, an average of 1.87 claims each, 22% with ≥2 claims; median single payout $8,284, 90th percentile $74,558
- Event naming is incomplete: 25% of claims have no event name, so the "non-tropical" judgement mixes in some unnamed tropical claims;
  named non-tropical events include Nor'easter, "Mid-summer severe storms", "Torrential rain", etc.

## How many block groups per event are studyable

Clustered by gaps of ≤3 days between claims within a county (non-tropical inland scope): 70,663 event clusters, with a median of only 1 claim.

| Threshold | Events | Train 2000–2016 / test 2017–2023 | Share of claims |
| --- | --- | --- | --- |
| ≥10 claims, ≥5 block groups | 4,998 | 3,905 / 1,093 | 72% |
| ≥30 claims, ≥10 block groups | 1,775 | 1,442 / 333 | 59% |
| ≥100 claims, ≥20 block groups | 506 | 431 / 75 | 42% |

## Does the outcome track hazard intensity

| Test | n | Spearman |
| --- | --- | --- |
| Any water-depth field vs payout (raw dollars) | 242,103 | 0.132 |
| Interior water depth vs payout | 33,348 | 0.182 |
| Single-family, interior water depth vs **payout / coverage** | 11,274 | **0.319** |
| By cause: river/lake overflow / rainfall ponding (depth vs payout) | 108,808 / 132,599 | 0.101 / 0.134 |

For single-family homes the median payout / coverage rises with interior water depth (inches): 13% for ≤6, 24% for 6–12, 31% for 12–24, 38% for 24–48, 42% for >48.
**After normalising by coverage and controlling for confounders, the NFIP outcome tracks hazard intensity, whereas SHELDUS/SE dollars are almost uncorrelated with rainfall amount (0.034)**;
raw dollars fail because of heterogeneity in coverage, deductibles and water-depth measurement.

## Exposure denominator (policies)

`NfipPolicies` v3 has 74.70 million records, with effective/termination dates and block groups. **An "in force during a window" filter returns 503**; filtering by state plus effective date is fast
(LA: 1,121 records for a single day, 26,622 for a single month, about 0.3 s). So the policy denominator **can be fetched per event and region** (take policies effective in the year before the event,
then filter by termination date locally); a national panel needs the bulk file.

## Finer-granularity hazard-layer inputs (confirmed to exist, not downloaded)

- Stage IV 4 km hourly precipitation: NCAR RDA d507005, since 2002-01-01
- gridMET 4 km daily precipitation: 1979 to present, OPeNDAP/THREDDS
- MRMS MultiSensor QPE (about 1 km): AWS since 2020-10, also archived by IEM
- Sentinel-1 GFM flood extent: operational in 2021–2022, start not verified
- Global Flood Database: 250 m, 2000–2018, 913 global events (the number of US events was not checked)

## Caveats

- NFIP covers only insured buildings and is concentrated in floodplains and mandatory-purchase zones, so it is a conditional sample of "insured exposure"
- Tropical events are identified by name; 25% have no event name and were not cross-checked against HURDAT2
- Only 14.5% of claims have interior water depth; the depth unit may mix inches and feet
- Payouts are affected by coverage caps (building limit $250k, 0.7% hit the cap) and deductibles
- Event clusters were defined by me using a 3-day gap within a county; they are not DFO/GFD events
- The relationship between hazard-layer inputs and block-group outcomes has not been validated, and no modelling has been done

---

# Stage 8: Texas sub-county flood pilot (block group × rain episode, 2010–2023) (2026-10-20)

Scripts: `22` (data fetch), `23` (definition diagnostics), `24` (extract Texas policies), `25` (build table), `26` (ladder experiment), `27` (terrain),
`28` (wavelet scale features), `29` (diagnostics); modules `bg_crosswalk.py`, `flood_pilot.py`. Everything ran on a Mac CPU, with no GPU.

## Design

- **Unit**: 2010-vintage census block group × rain episode. An episode = consecutive days on which a county's daily maximum precipitation (gridMET 4 km) is ≥30 mm (gaps of ≤1 day merged),
  with tropical-cyclone windows excluded (HURDAT2 Atlantic + Eastern Pacific, 800 km, ±2 days)
- **Rows**: all block groups of the county with ≥1 NFIP policy in force on the episode's start day (policies from the OpenFEMA bulk file)
- **Outcome**: whether the block group has an inland claim not named as tropical (cause codes 2/3/4) in the window [start, end + 2 days]
- **Split**: train 2010–2016, test 2017–2023; plus a seven-fold cross-validation blocked by year (each of the 14 years held out once)
- **Size**: 2,996,576 rows, 44,212 events, 11,264 positives (0.38%); test 3,711 positives, 1,015 events for which within-event AUC can be computed
- **Main metric**: within-event AUC (pairwise comparisons of block groups within the same event; a county-level model is 0.5 by definition); overall AUC and AP are also reported

## Results (test 2017–2023; within-event AUC, event-cluster bootstrap 95% interval in parentheses)

| Scheme | Overall AUC | Within-event AUC | Seven-fold CV within-event (mean, worst–best) |
| --- | --- | --- | --- |
| County-level rainfall (L0) | 0.880 | 0.500 | 0.500 |
| Point vote: the block group's own rainfall grid point | 0.902 | 0.740 | — |
| Neighbourhood exceedance fraction (NEP) 7×7, ≥50 mm | 0.875 | 0.651 | — |
| L1 + exposure (policy count, flood-zone share, coverage) | 0.894 | 0.647 (0.633, 0.663) | 0.670 (0.633–0.698) |
| **L2 + block-group grid-point rainfall** | **0.926** | **0.786 (0.765, 0.805)** | **0.787 (0.751–0.814)** |
| L3 + neighbourhood rainfall | 0.927 | 0.776 | 0.785 |
| L4 + location, antecedent rainfall | 0.930 | 0.779 | 0.792 |
| L4b + terrain (elevation, relative height, slope) | 0.930 | 0.781 | 0.794 (0.758–0.820) |
| L5 + wavelet scale features (the earlier encoder) | 0.930 | 0.784 (0.761, 0.808) | 0.789 (0.758–0.824) |

Paired bootstrap (within-event AUC difference): exposure **+0.147** (0.133, 0.163); grid-point rainfall **+0.139** (0.118, 0.157);
neighbourhood −0.010 (−0.022, +0.001); location +0.003; terrain +0.002 (−0.001, +0.005); wavelet +0.003 (−0.000, +0.006).
Adding terrain on L2 alone gives −0.002, wavelet −0.002, both −0.003, all with intervals spanning 0.

## Diagnostics

| Test | Result |
| --- | --- |
| Shuffling labels within events | 0.781 → **0.501**, not a pipeline artifact |
| Learning curve (training 2/4/7 years) | 0.774 / 0.768 / 0.781, saturated |
| Learner | Boosted trees depth 3/5/8: 0.782/0.786/0.786; **logistic regression 0.781** |
| Dropping the flood-zone share | L2 0.783, L4b 0.781: terrain adding no gain is **not** because the flood zone already contains it |
| Trigger ≥50 mm | L2 0.802, L4b 0.800 |
| Tropical window 1200 km / ±4 days | L2 0.785, L4b 0.780 |
| Keeping only claims with a high-confidence block-group mapping | L2 0.786, L4b 0.783 |
| Strata | Block groups per county ≤20: 0.737; >150: 0.781. 1-day events 0.801; >4-day events 0.754 |

**Within-event AUC is stable at 0.78–0.80 across the three "definition choices": trigger threshold, tropical window, and claim confidence.**

## Dataset diagnostics (problems found and how they were handled)

1. **NFIP block-group codes mix the 2010 and 2020 vintages**: Texas claims match 90.2% against the 2010 vintage alone and 70.1% against the 2020 vintage alone, and in recent years the 2010 match rate drops to 66–82%.
   Mapping with the official Census 2020→2010 relationship file by largest overlap area gives **100% matched, 92–99% high-confidence (land area ≥80%)**
2. **The policy bulk file only starts in 2009** (in-force count is 0 for 2005), so the exposure denominator and the pilot can only start in 2010
3. Policy data: terms are almost all 365–366 days, and 99.65% map to 2010 block groups; statewide policies in force are 587,804 on 2017-08-24 and 774,767 in 2020-07,
   which is plausible; **0% of claims occur in a block group with no policy that day**
4. Single-family coding: it changed to `11` from 2021, so the original test `==1` would make the 2023 share 0; fixed to `{1, 11}`
5. Event coverage: the event windows contain 31,057 claims, 88.8% of the 34,989 Texas claims in 2010–2023
6. **Large label drift**: claims per million policy-events are 621 in 2015 and 710 in 2016, versus 17.5 in 2011, 15 in 2020 and 24 in 2022;
   the positive rate is 0.58% in training against 0.22% in test, so the Brier skill is about 0. **Ranking metrics are stable, but absolute probabilities cannot be used directly across eras**
7. The wavelet feature file corresponds only to the main table for T=30; the L5 row in the sensitivity table is invalid (all missing after the merge) and is no longer used
8. Texas has 64,395 inland claims not named as tropical, but they are very uneven across years (about 10 k in 2015 and 2016, under 500 in 2011, 2022 and 2023),
   and the test period holds only about 24% of the 2010–2023 claims

## Conclusions

1. **The sub-county skill is real and robust**: within-event AUC about 0.78–0.80, with even the worst of the seven CV folds ≥0.75, against 0.5 for the county-level scheme
2. **All the skill comes from two items**: exposure (policy count, +0.147) and the block group's own rainfall grid point (+0.139). Removing the flood-zone share has almost no effect
3. **Neighbourhood smoothing, terrain, location and the wavelet encoder have no significant gain**. Logistic regression is on par with boosted trees, and the learning curve is saturated:
   **it is limited by feature information, not by the learner or the data volume**
4. On "observed rainfall", taking the value at the point beats neighbourhood voting. **This cannot be extrapolated to forecasts**: NEP addresses position error in forecasts, and an observed field has no position error
5. The earlier wavelet encoder **brings no demonstrable benefit** on this task

## Caveats

- Rainfall is the gridMET **daily** value (4 km) at the grid point of the block-group centroid, losing sub-daily intensity and position; terrain is sampled from 150 m tiles at the centroid,
  not averaged over the block-group area, and carries no drainage or storm-sewer information
- The sample is conditional on "block groups with policies", so it reflects the exposure of the insured population and not all buildings
- A single state and a single region; events of 2015–2016 dominate the training set
- This is an **observed-rainfall** experiment, not a forecast experiment; whether neighbourhood probabilities, ensemble members and the encoder are useful in forecast mode needs a separate test
- Within-event AUC is weighted by the number of pairs, so large events carry more weight; the event-cluster bootstrap does not account for correlation across counties within the same weather system
- Payout amounts and claim counts were not evaluated (only occurrence probability was)

---

# Stage 9: data audit of building-structure / foundation priors (2026-10-23)

Scripts: `30` (audit, reproducible; the report is stored in `$WXRISK_DATA/audit/building_attrs.txt`), `31` (data manifest), `scripts/fetch/` (consolidated downloads).
Documents: `docs/FLOOD_PRIORS.md` (detailed results and to-do), `docs/DATA_SOURCES.md` (all data sources), `AGENTS.md` (for AI agents / new members).
This stage **trained no model**; it only tests the data and the interpretation hypotheses for "can building structure provide a prior".

## Decision

Building structure (foundation, elevation, basement) is used only as auxiliary labels / vulnerability priors, not as forecast inputs. Duration of inundation, seepage and equipment placement cannot be determined even by claims adjusters,
and the data has no reliable field for them, so they can only be treated as priors.

## Results (single-family claims, 2010 onward, loss ratio = building damage / coverage; records with exterior water depth 0 removed)

- **Fields that exist**: `foundationType`, `elevatedBuildingIndicator` (both policies and claims); `basementEnclosureCrawlspaceType`,
  `enclosureTypeCode`, `elevationDifference`, number of floors, `buildingDescriptionCode` (policies only).
  **No** wood/brick construction field exists (`construction` is only an "under construction" boolean, 0.25% true); `floodWaterDuration` is almost all 0 and unusable.
- **The unknown share of `foundationType` changes drastically by year** (Texas policies 86% in 2010, 58% in 2016, 32% in 2020, about 0% from 2022),
  which is a change in reporting rules; and the unknown group has a higher loss ratio (2017–2021: 0.39 vs 0.32 for known slab), so the known subset is not a random sample.
- **Elevated buildings lose less at the same water depth**: 0.27× slab after controlling for depth bin and year (Texas); 0.32 nationally with state-month fixed effects;
  monotone across depth bins. This is the most robust structural signal.
- **Basements are not more vulnerable (on this basis)**: only 62 Texas basement claims, untestable; nationally with state-month fixed effects they are 0.43× slab,
  and state-by-state estimates in 14 states are all below 1 (0.26–0.79). Basement losses are nearly flat (about $10–20 k), while slab rises to about $100 k with interior water depth.
  This is consistent with a threshold-type loss shape of "equipment sits in the basement, so it is flooded together as soon as water gets in", but it **cannot be distinguished** from "NFIP covers only limited basement items".
- The annual median loss ratio of slab buildings is 0.04–0.35; year/event effects are no smaller than structure effects and must be controlled.
- Flood-zone priors should come from FEMA NFHL (not USGS) and HAND; not yet downloaded.

## Conclusions

- Priors that can be built: elevated share, `elevationDifference`, number of floors, post-FIRM share; foundation-type shares may only use the known subset, together with an explicit unknown share.
- Model basements as "threshold event + slow increase with depth"; flag the "basement × small residential" interaction separately; "basement × non-residential" has no data, so do not fill it from the former.

## Caveats

- The `foundationType` codes were inferred from cross-tabs and are not from an official dictionary; the share of "foundation type = 2" in NC and KY is only 2–18%, so its meaning is doubtful.
- "Loss / coverage" is pushed down for basements by the scope of coverage, so "basements are lower" cannot be interpreted as "safer". The commercial vs residential comparison has no data (non-residential claims almost
  never have a foundation type).
- State-month fixed effects are only a rough proxy for events; `floodEvent` covers only about 81% of records and 72 events, and is unusable.
- The Texas and national ratios are computed on claims whose foundation is known; whether they hold for the unknown part is not known.
- These features were not attached to `flood_pilot_table`, so **it is not yet known whether they add anything to prediction**.

# Stage 10: compliance ladder and NFIP take-up selection (candidate features only) (2026-10-23)

Scripts: `scripts/33_compliance_ladder.py`, `scripts/34_takeup_rate.py`. Background: ASCE 24 (editions 98 / 05 / 14 / 24) is referenced by the IBC;
Texas has no statewide building code, flood standards are local ordinances above the NFIP floor (44 CFR 60.3), and no NFIP field records the adopted edition.
So code compliance is a **latent variable**; only coarse proxies are observable.

## Method

- Four ordered tiers from observable fields: T0 post-FIRM + elevated + SFHA (closest to compliant); T1 post-FIRM, not elevated, SFHA (NFIP minimum);
  T2 pre-FIRM SFHA; T3 non-SFHA (owner unaware, often insured at sale). Not observable: flood-damage-resistant materials, equipment elevation, flood openings, freeboard.
- Loss side: single-family claims 2010+, exterior depth > 0, damage ratio by exterior-depth bin and tier; event (state-month) clustered bootstrap.
- Frequency side (Texas): claims per 1000 policy-years, tier from policies without elevation.
- Take-up: in-force single-family policies in X/B/C/D zones on 2022-12-31 / ACS housing units, per 2020 block group, block groups with SFHA policy share < 5%.

## Results

- **T0 is far below T2**: median damage ratio 0.05–0.13× (Texas), 0.09–0.24× (national) at the same exterior depth; bootstrap intervals exclude 1.
- **The rest of the ladder is not resolved**: T1/T2 intervals include 1 in most bins; T3 is below T2 at 1–6 in and equal deeper, so T3 is not "least protected".
  Monotone T0 <= T1 <= T2 in 5/5 depth bins in Texas but only 2/5 nationally.
- **Residual-loss floor**: T0 q10 building damage about $650–4,600 (Texas), close to deductible level, so it is mostly a deductible floor, not a design residual.
- **Frequency (Texas, 2015+)**: post-FIRM SFHA rate is 0.14–0.75× pre-FIRM SFHA, non-SFHA 0.20–0.58×. 2010–2014 are empty (policy exposure denominator problem, cause not investigated).
- **Take-up is bimodal across block groups** (7,180 kept; modes near 0.2% and 6%; GMM BIC prefers 3 components over 1). But the high mode is concentrated in Gulf coast counties
  (Harris 1,270 of 2,408 high block groups; Fort Bend, Jefferson, Montgomery, Galveston, Brazoria next), and the median mandatory-purchase share there is 0, so it is voluntary.
- Texas policies whose rated zone differs from the current zone: X→SFHA 1.3%, SFHA→X 9.3% of rows; only 8% of X→SFHA rows are flagged mandatory purchase.


## Addendum: take-up after an incident (Hurricane Harvey event study, `scripts/35_takeup_after_flood.py`)

- 5,161 Texas block groups binned by Harvey claims per in-force policy; stock of in-force single-family non-SFHA voluntary policies, indexed to 2017-07-31.
- Pre-trend is parallel across bins (all about 1.04 a year earlier, stock drifting down 4%/yr). After Harvey the **no-claim control still grew 30-35%** by 2020, so most of the regional
  rise is spillover (news, neighbours, lenders) or direct experience; the control cannot separate them.
- Block groups with > 10% claims grew 45-53% (2018-07 to 2020-07), 12-20 points more than the control; the excess peaked at +19.5% (2018-06) and fell to +11.7% by 2020-12 (half-life about 2.5 years, memory fades).
  The 2-10% bin: +11.6% peak, +7.0% by 2020-12; the 0-2% bin: +8.4% peak, +3.0%.
- Reading: directly experienced take-up is 7-12% of the current stock (lower bound; the control may itself be Harvey-driven). The total event-driven share is between that and most of the post-event growth.
- **Corrected 2026-10-25 (stage 11 table, re-run on the clean tables)**: the figures above include a phantom null-key block group and cancelled policies. Now 5,088 block groups; the >10% bin's excess peaks at +18.6% (2018-06) and is +9.9% at 2020-12 (was +19.5% / +11.7%), the 2-10% bin +11.7% / +6.5% (was +11.6% / +7.0%), the 0-2% bin +7.4% / +1.8% (was +8.4% / +3.0%). Pre-trend and the 30-35% no-claim control growth are unchanged in shape (control 1.353 by 2020-07); the conclusion stands with a smaller excess.
- Caveats: `mandatoryPurchaseFlag == False` only approximates voluntary purchase; policy rows are terms, not households, so no per-household probability (the uninsured flooded households are not in the data);
  one event, one region; claim-rate bins correlate with urban growth.


## Addendum: event-known loss and post-event buyers (`scripts/36_event_features.py`)

- Event definition: FEMA declarations for Texas (flood, hurricane, severe/coastal/tropical storm; DR and EM) by county and incident period, not the NFIP `floodEvent` field.
  `scripts/24` now also extracts `originalNBDate`, `rolloverTransferCode`, `propertyPurchaseDate`, `nfipRatedCommunityNumber` (Texas policy file re-extracted, 11,612,805 rows).
- **Feature A (loss inside a declared event, ex-post)**: 88% of Texas single-family claims 2010+ fall inside a declared window (2017: 99%). Inside: mean building damage $74 k, median $58 k, 56% above $50 k;
  outside: mean $22 k, median $11 k, 12% above $50 k. Heavy-tail difference is large, but dominated by Harvey. It uses the loss's own date, so it is a label/stratifier only, never a forecast input.
- **Feature B (post-event buyer, ex-ante)**: policy NB date within 24 months after the county's last declared event that ended before it (no look-ahead). Among 342,995 voluntary non-SFHA single-family policies in force at Harvey,
  60% qualify (block-group 10-90%: 0.43-0.78), because county-wide declarations cover almost every county in 2015-2017, so the flag mostly encodes "bought in 2015-2017".
  Within-county Poisson of Harvey claim rate on the block-group share: coefficient -0.37 [-3.07, +0.51] (rate ratio 0.69x for 100% vs 0%): **no detectable effect**. The post-FIRM share coefficient is negative and clear (-0.90 [-1.13, -0.14]).
- The null does not mean post-event buyers behave the same: the feature has little variation and the interval is wide.

### Local-intensity version (`scripts/36_event_features.py --sections local`)

- Hit = tract with >= 3 single-family claims inside a county's declared window and claims / in-force policies >= 2% (seven declared events 2013-10 to 2017-08: 917 hit (tract, event) pairs in 727 tracts).
  Post-hit buyer = NB date within 24 months after the latest hit of the policy's own tract, no look-ahead. It is far more selective than the county flag: 9.1% of policy rows (block-group 90th percentile 0.37).
- Harvey claim rate on block-group post-hit buyer share, Poisson, county fixed effects, 18 counties with >= 50 Harvey claims, 3,356 block groups, county-clustered bootstrap (100 draws):
  - no hazard control: +1.80 [+1.26, +2.81] (a strong association, the same direction as "bought after being hit");
  - plus tract prior claim intensity and ever-hit: **+0.10 [-0.49, +2.80]**, i.e. the association is gone once hazard is controlled.
  - placebo (40 within-county reshuffles of which tracts were hit, true hazard controls kept): coefficient 2.5/50/97.5% = -0.66 / -0.22 / +0.26; real +0.10 sits inside it (share of placebo draws >= real: 0.10).
### Replication on other outcome events (`--disaster 4332 4269 4272 4466`; outcome = claims in the declared counties and window, policies of those counties only)

| Outcome event | Counties | Post-hit buyer share | Coef with hazard controls [95% CI] | Placebo 2.5-97.5% |
|---|---|---|---|---|
| Harvey 2017 (4332) | 18 | 9.5% | +0.14 [-0.45, +2.85] | [-0.77, +0.25] |
| Tax Day 2016 (4269) | 4 | 2.3% | unusable (CI [-4.7, +6.9]) | too few counties |
| Memorial/June 2016 (4272) | 4 | 3.2% | unusable (CI [-9.9, +3.6]) | too few counties |
| Imelda 2019 (4466) | 6 | 39.2% | **+1.72 [+0.10, +3.72]** | [-0.59, +0.78] |

- Imelda (2019) is the only replicate with enough exposure and a post-hit share that varies: after controlling for tract prior intensity and ever-hit, the post-hit buyer share is associated with a higher claim rate, above the placebo range, but the bootstrap
  lower bound is barely above 0 and rests on six counties. Harvey does not show it. So the evidence is mixed, not a replication.
- Geoid vintage check: 99% of 2019 claim block groups match the 2010-vintage policy block groups (96-98% in 2015-2017), so no crosswalk was needed up to 2019.
- **There was no 2019 surge in NFIP new business.** First-term purchases (policy effective date within 45 days of `originalNBDate`; only about 12% of policy rows are first-term, counted by `policyCount`), Texas single-family non-SFHA, by quarter:
  2016Q2/Q3 24.9k/28.3k voluntary; 2017Q4 60.5k (Harvey); 2018Q2/Q3 42.1k/39.6k; 2019Q2/Q3 23.7k/23.6k; 2020Q2/Q3 22.8k/31.5k; Q1 and Q4 are always 3-4x lower (seasonal). 2019 is back at the 2016 and 2020 level.
  The 39% post-hit share for Imelda is a **stock effect**: on 2019-09-16, 50% of the voluntary non-SFHA stock in the six Imelda counties (plus Harris, all Harvey counties) had its first term inside the previous 24 months, 27.6% in the 12 months after Harvey,
  against 35.5% before Harvey itself. The 24-month window simply still contained the Harvey wave. Pulse of new sales after an event lasts about four quarters; the stock excess decays slower through lapses.
- Consequence for the Imelda coefficient (+1.72): the "post-hit buyer" is mostly a Harvey-wave buyer in tracts Harvey flooded, and those tracts can flood again for hazard reasons the tract-level controls miss. It does not show a 2019 shock.
### Stock-flow model of take-up inertia (`scripts/37_stock_flow_inertia.py`, helper `src/wxrisk/panel.py`)

Texas single-family non-SFHA voluntary policies, county x quarter 2010Q1-2023Q4, 47 counties with mean stock >= 300. New purchases (first-term rows) follow
Poisson(Sref[c] * exp(county effect + season + 7 lags of log1p(100 * claims per in-force policy))); stock is rebuilt from purchase cohorts with a pooled survival curve. The model gives exposure *given* the realised local hazard.

- **Cohort survival is the same for every cohort**: pooled share still in force at age 1 / 2 / 3 years about 0.75 / 0.62 / 0.51 (train to 2019Q4). The Harvey cohort (2017Q4: 0.75, 0.63, 0.48) is not special: an event raises the stock through inflow, not through lower churn.
- **Pulse kernel** (train to 2019Q4, per unit of the transformed intensity): lag 0 +0.0, lag 1 +0.5, lags 2-4 +0.2 to +0.3, lag 5-6 about 0: a pulse of about four to five quarters on top of a strong season (Q2/Q3/Q4 vs Q1: +0.95 / +1.12 / +0.46 in log rate).
- **Held out, origin 2019Q4 (test 2020Q1-2023Q4)**: one-step skill of the event model over season-only +0.18 [+0.13, +0.25] (log1p N, county-quarter, county-clustered bootstrap); state-total absolute error 0.28 vs 0.47 (season-only) and 0.36 (last-year scaled).
  Stock: event model beats persistence only at h = 12 quarters (county wAPE 0.083 vs 0.156); at h = 4 and 8 persistence is better (0.050 / 0.028 vs 0.108 / 0.097) because the stock was nearly flat.
- **Held out, origin 2016Q4 (test 2017-2023, Harvey out of sample)**: event model skill over season-only +0.02 [-0.01, +0.05], no better than the interval: the pulse could not be learned from the small events before 2017, and the model missed the size of the 2017Q4 peak (12.8k predicted vs 63.4k observed).
  Stock with the event model still beats persistence at h = 8 / 12 (county wAPE 0.156 / 0.197 vs 0.246 / 0.271), partly because cohort decay is modelled.
- Overprediction in 2021Q4-2022Q1 (12.8k vs 5.0k, 8.0k vs 3.7k) coincides with the start of Risk Rating 2.0 (October 2021); not tested.
- **Economic covariates (new data: Freddie Mac mortgage rate, Zillow home values, Census building permits, NFIP premium; lagged one quarter, standardised on the training quarters).**
  All four together raise the county-quarter log1p skill (event+econ vs event, origin 2019Q4: +0.37 [+0.26, +0.45]) but make the state totals worse (absolute error 0.41 vs 0.28 for the event model; level underpredicted by 40-50% in 2020-2022, the covariates extrapolate far outside the training range: 2021-22 home-value boom, premium rise around Risk Rating 2.0).
  One feature at a time (origin 2019Q4, skill over the event model / state-total error): mortgage-rate change -0.11 [-0.13, -0.09] / 0.36; home-value change +0.11 [+0.02, +0.20] / 0.37; **previous-year permits +0.15 [+0.10, +0.22] / 0.25**; premium +0.06 [-0.18, +0.26] / 0.51.
  Permits is the only one that helps both metrics, **but its sign flips with the training window**: coefficient +0.60 per s.d. (train to 2019Q4) and -1.19 (train to 2016Q4, skill +0.03 [-0.25, +0.24]). Permits plus home value together: +0.005 [-0.10, +0.12].
  Reading: with annual or smooth county series, 40-ish quarters and two different regimes, the economic covariates do not give a stable gain; the 2019Q4 result is not a finding. What would help is a variable with real transaction content (county monthly home sales, mortgage originations).
- **Home sales (Redfin county monthly `homes_sold`, same quarter, training from 2012Q1)**, the feature closest to the mechanism "insurance bought at closing": coefficient +0.044 per s.d. (train to 2019Q4, skill over the event model -0.015 [-0.022, -0.009]) and -0.219 (train to 2016Q4, skill +0.052 [+0.017, +0.086]):
  tiny and sign-flipping. With permits: -0.088 [-0.122, -0.054] (2019Q4) and +0.054 [-0.064, +0.137] (2016Q4). Season dummies and county effects already absorb most of the sales variation, and voluntary non-SFHA NFIP purchases appear to follow the flood-event pulse far more than the housing market.
  Conclusion for the take-up dynamics: season + decaying event pulse + age-dependent lapse carry the signal; the economic cycle (rates, prices, permits, sales, premium) adds nothing stable at this resolution.
- Caveats: realised hazard is conditioned on, so this is not a forecast of events; unconstrained lags are noisy when training events are small; survival pools counties; no housing-market, rate or premium covariates; statewide counties with stock >= 300 only.
  A first version without the saturating intensity transform and with stock-proportional flow predicted 160k purchases in 2017Q3 and diverged: the exponential link must not be extrapolated.
- Candidate explanation (the user's, **untested**): word of mouth along a small-world-like social network with random links, from flooded owners to uninsured neighbours. This data has no social network, so it cannot be told apart from
  (a) finer hazard that tract-level controls miss, (b) community outreach (CRS, officials, lenders, insurers), (c) housing growth in new subdivisions. Not a conclusion.

- Reading: the raw association is mostly "places that flooded before flood again". No human-factor effect is detectable here; with 18 counties and one outcome event the interval is too wide to rule one out.
- Caveats: Harvey claim rate mixes hazard and insured-composition; tract-level hit and prior intensity are crude hazard controls; claims and policies of 2015-2017 share 2010 block-group codes, later years do not.
  A second outcome event (Imelda 2019, Beryl 2024) would need the 2020 block-group crosswalk (`wxrisk.bg_crosswalk`).

## Conclusions

- Elevation (a proxy for the engineered tier) has a large, robust effect; the finer steps of the ladder are not identified.
- Selection of who insures is a real feature of the data, but the observed bimodality is mostly geography and flood experience, not evidence of owner types.
- Everything here goes to the **candidate pool**: tier, take-up rate, and uptake-based weights may be tried later in the ablation; none is promoted without beating the year-grouped bootstrap interval.

## Caveats

- Exterior depth is above ground, so elevated buildings need more water to flood: part of the T0 gap is geometry, not a design effect. Depth relative to the floor is not in claims.
- Professional owners likely also maintain better, so the T0/T2 ratio mixes design and owner type; read it as an upper bound on the design effect.
- Claims under the deductible are not in the file (left truncation); tiers are insured, self-selected populations; no causal reading.
- Take-up denominators are all ACS housing units (including multifamily), numerators are single-family policies; 21% of Texas policy block groups (older vintage) have no ACS match; the
  `X→SFHA` map-revision sample is not a clean quasi-experiment (only 8% mandatory). Whether owner-type mixtures exist inside a block group cannot be tested with this data.

## Addendum: exposure dynamics in the Texas pilot table (`scripts/38_exposure_dynamics_features.py`, `scripts/26 --only L5 L6`)

- Features per (episode, block group), ex ante (day before the episode starts): share of in-force policies in their first 12 months (`new12_share`) and 12-36 months (`mid_share`) of NFIP coverage. `mand_share` was **not** used:
  it drifts from 0.7% (2010) to 11.6% (2021), a reporting change that would teach the model the year.
- Test 2017-2023 (within-episode AUC, 1015 episodes): L5 (wavelet scales) 0.7839 -> L6 (+ exposure dynamics) 0.7829; AUC 0.9298 -> 0.9301; AP 0.1139 -> 0.1120; BSS 0.0296 -> 0.0283. Blocked CV over year pairs (within-episode AUC mean / min / max):
  L5 0.7893 / 0.7576 / 0.824; L6 0.7905 / 0.7567 / 0.823, a +0.0012 difference against a fold spread of 0.07.
- **No gain.** The composition of the insured stock (new vs old buyers) adds nothing to sub-county skill beyond rain, exposure counts, location, terrain and wavelet scale features. Null result: it does not say these features are uninformative about who files claims, only that
  this table and this learner do not benefit.
- Also: with the economic covariates (stage 10 stock-flow model) this closes the "human factors" candidates for the pilot table: season, event pulse, lapse, economic cycle and exposure composition were tested; none improves the forecasts.
- Side note: `scripts/26` writes `flood_pilot_test_preds.npz` on every run (an `--only` run overwrote it with two groups; a full rerun restored it, now with the L6 predictions).

---

# Stage 11: preprocessing pipeline, cleaning register, and what it changed in stages 8-10 (2026-10-24)

Code: `src/wxrisk/prep/` (spec, clean, build, normalize, register), `scripts/40_prep.py`, `scripts/41_cleaning_evidence.py`,
`scripts/fetch/fetch_openfema_dictionary.py`; docs: `docs/DATA_PREP.md`, `docs/CLEANING_REGISTER.md` (generated; every rule with its origin and evidence).
Scripts 33-37 read the clean tables `policies_tx` and `claims_nfip`; script 30 audits raw data on purpose. Every migration was run before and after and diffed.

## Method: who defines a value, and what the data say

Provider sources: the OpenFEMA data dictionary of NfipClaims v3 and NfipPolicies v3 (84 and 87 fields) and the OpenFEMA FAQ page. A value the
provider documents is handled as documented. A value it does not define goes through a spike test (count at v >= 5x its neighbours, >= 20 rows;
run over all 24 numeric columns of both tables) and a profile test (does the outcome at v look like its neighbours?). Sporadic implausible values with
no signature are entry errors: the value (never the row) is nulled. Missingness diagnostics (standardised mean difference, Cramer's V) say whether dropping rows
would be safe. Imputation is not done in clean tables; it needs a masked-value test with RMSE gain >= 30% and a typical multiplicative error <= x1.25.

## What the provider documentation settled (and corrected in our own earlier assumptions)

| Item | Documentation / evidence | Consequence |
|---|---|---|
| Mid-term cancellation | FAQ: check whether the policy period was cancelled mid-term. 409,641 of 410,241 cancelled policies end before the nominal expiry, which is always effective date + 1 year | `coverage_end = min(expiry, cancellation)`; in-force counts were 2.6% too high (single-family, 2017-07-31: 527,075 -> 513,153) |
| `foundationType` | codes 1-6 only; **3 = crawlspace** (65% of its claims are flagged elevated, 93-95% for 4-6); 0 undocumented | earlier "3-6 elevated" was wrong (AGENTS.md trap 8, script 30); 0 -> null + flag |
| `numberOfFloorsInInsuredBuilding` | a code (1 one floor ... 5 manufactured home, 6 townhouse), not a count | handled as a code |
| Negative premium, negative building payment, negative depth | documented as refund, re-issued check, below the lowest adjacent grade | kept (an intermediate version nulled them) |
| 99XX placeholders | FAQ: "99XX is often used as placeholder or default value"; scan finds 999 in exterior/interior depth, 99 in `waterDepth`, 999 in `floodWaterDuration` | nulled; no 99XX spike in policies |
| `waterDepth` | dictionary: some measurements are in feet; 52% of rows agree with feet (x12), 8% with inches | column not carried |
| `elevationDifference` 9999 | documented "not reported" (0 rows in Texas); 26 values below -1000 ft are entry errors | nulled |
| Building coverage 0 in claims | **not** a placeholder: 94% have no building damage, 97% have contents damage (contents-only policies) | kept as 0; ratios filter `cov_building > 0` (an intermediate version nulled them; the diagnostic caught it) |
| First-policy date later than effective date (563,198 rows) | 4-6% of new-business/renewal rows, ~0 for transfers: a booking convention | no rule |
| Dates | construction date 1492-10-12 on 14,212 rows (undocumented placeholder), scattered years 0001-1100 and 9999; first-policy dates before 1968 (201) | nulled + flag |
| Exterior depth availability | null share 96-100% in every loss year 2000-2015, 61% in 2016, 18-28% from 2017 | depth analyses effectively start in 2016-2017 |

Not retrievable: the FAQ PDF link (HTTP 403 to our downloader; the browser extension was unavailable). The HTML FAQ was read through a web-fetch tool that summarises, so
its quotations are partial; the per-dataset "Data Fields" pages and legacy manuals were not read.

## Effect on earlier results (same script, before the pipeline -> after)

| Script | Result | Before | After | Cause |
|---|---|---|---|---|
| 35 Harvey take-up | blocks with >= 20 policies | 5,161 | 5,088 | phantom null-key block group (945 policies, 153 claims) and cancelled policies removed |
| 35 | excess vs control, peak / 2020-12, >10% bin | +19.5% / +11.7% | +18.6% / +9.9% | phantom group; cancellations |
| 35 | 0-2% bin, peak / 2020-12 | +8.4% / +3.0% | +7.4% / +1.8% | same |
| 37 stock-flow | stock check 2023Q4 | 403,008 | 398,058 | cancellations |
| 37 | cohort survival at age 0, 1, 2 quarters | 1.00 | 0.97, 0.97, 0.96 | cancellations are now visible |
| 37 | flow skill, event vs season (origin 2019Q4) | +0.254 [+0.174, +0.351] | +0.264 [+0.156, +0.378] | same conclusion |
| 37 | event+econ vs event (2019Q4) | -0.208 [-0.369, -0.106] | -0.242 [-0.412, -0.109] | same conclusion |
| 36 event features | cohort policies | 342,995 | 335,497 | phantom county `00000`; cancellations |
| 36 | share_post coefficient | -0.373 [-3.074, +0.509] | -0.443 [-3.290, +0.508] | interval still contains 0 |
| 36 | share_hitbuy with tract prior | +0.099 [-0.483, +2.988] | +0.218 [-0.343, +3.163] | interval still contains 0 |
| 36 | placebo 2.5/50/97.5% | -0.662 / -0.217 / +0.26 | -0.779 / -0.051 / +0.386 | 40 random reshuffles; the county set changed |
| 34 take-up rate | snapshot policy rows / blocks | 627,691 / 17,340 | 607,283 / 17,209 | phantom group; unlocatable and cancelled policies |
| 34 | median / p90 take-up | 0.0083 / 0.1718 | 0.0082 / 0.1691 | |
| 33 compliance ladder | claims | 63,925 | 63,164 | 753 Texas claims at depth 999 (unknown code) removed |
| 33 | ">48 in", T3: 10th percentile / median damage ratio | 0.059 / 0.708 | 0.342 / 0.838 | the 999 claims had a median ratio of 0.135 |
| 33 | ">48 in" T3/T2 median ratio | 0.85 [0.01, 0.94] | 0.98 [0.83, 1.21] | the earlier reading "T3 much lower at depth > 48 in" is not supported |
| 33 | ">48 in" T1/T2 | 0.92 [0.05, 0.95] | 0.94 [0.32, 1.11] | |
| 33 | 2017 claims per 1000 policy-years, T1 / T2 | 130.2 / 173.7 | 135.8 / 181.7 | policy-years shrink with cancellations; ratios T1/T2 = 0.75 unchanged |
| 30 (raw audit) | national fixed effect vs slab, elevated / basement | 0.32 / 0.43 | 0.27 / 0.45 | code 3 (crawlspace) split out of "elevated"; 109,906 claims instead of 122,415 |

Conclusions in stages 8-10 survive qualitatively; magnitudes move by 1-5% for the stock-based numbers and substantially only for the deepest bin of stage 10's compliance ladder.

## Caveats

- Exterior depth is almost entirely missing before 2016 (null share 96-100% in 2000-2015, 61% in 2016, 18-28% from 2017): the compliance ladder (starting 2010) has depth bins from 2016-2017 on, i.e. mostly the Harvey year and later.
- The claims frequency table in script 33 counts only claims with a usable depth, coverage and damage amount, a selection in the numerator that predates this stage.
- Imputation of `buildingReplacementCost` was tested (gradient boosting, RMSE -30.6% vs a per-year median, but typical multiplicative error x1.69): not adopted.
- The spike test cannot see a placeholder that is also a common true value (the 0 depth is a mixture of both; null + flag keeps the fact).
- Normalised columns were tested for reproducibility on macOS/arm64 only.
- Stage 8's pilot-table construction (scripts 25-29) still reads raw files and was not migrated.

---

# Stage 12: scaling audit (which columns, group-aware or not, discretise or not, which targets need an inverse) (2026-10-24)

Code: `scripts/42_scaling_audit.py` (Texas flood pilot, logistic regression and boosting), `scripts/43_scaling_kmeans.py` (SCS K-means),
`src/wxrisk/prep/scaling.py` (feature plan, target transform, inverse guard), `src/wxrisk/prep/scaling_plan.py` (decisions with basis), `docs/SCALING_PLAN.md` (generated).
Metric: within-event AUC on 2017-2023 (fit 2010-2016), 1,015 test episodes, paired event-clustered bootstrap; K-means: BSS against the climatology, year-clustered paired bootstrap.

## Results

| Question | Learner | Result (difference, 95% interval) | Decision |
|---|---|---|---|
| Is scaling needed? | K-means, SCS tier B / C | no scaling (column std ratio 2e5): y0 BSS 0.0009 vs 0.0253; -0.0244 [-0.0296, -0.0200] / -0.0184 [-0.0233, -0.0138] | standardise |
| | logistic regression | unscaled AUC 0.7817 vs standardised 0.7806, but 400 iterations (not converged) at every C vs 15-23 | standardise; non-convergence counts as a scaling failure |
| | boosting | transformed + cyclic vs raw: -0.0014 [-0.0042, +0.0011] | no scaling for trees |
| Which columns to transform? | logistic regression | log1p/asinh: +0.0032 [-0.0024, +0.0078]; + cyclic month: +0.0048 [-0.0009, +0.0096] | transform by role (rain, counts, depth: log1p; signed heavy tail: asinh; angles, shares, existing logs: standardise; month: sin/cos), kept for conditioning, not for a proven gain |
| Group-aware? | logistic regression, within-episode z and percentile | +0.0040 [-0.0001, +0.0087] (without them: +0.0048) | not adopted |
| | boosting | +0.0028 [-0.0013, +0.0064] | not adopted |
| | K-means, global z vs county x month anomaly, tier B / C (y0) | +0.0013 [-0.0018, +0.0045] / -0.0016 [-0.0055, +0.0022] | no demonstrable gain; anomalies kept to leave stages 2-5 unchanged |
| | year-wise normalisation | between-year variance share <= 0.09 for every feature; would use test-year statistics | forbidden |
| Discretise? | logistic regression, supervised bins added | all 12 columns +0.0084 [+0.0021, +0.0142]; exposure columns +0.0041 [+0.0009, +0.0072]; rain -0.0003 [-0.0034, +0.0025]; geography -0.0014 [-0.0045, +0.0021] | candidate for the exposure columns (sfha_share +0.0017 [+0.0005, +0.0030]); never for trees, never instead of the continuous column |
| | single-feature shape | bg_f1 within-event AUC 0.7398 continuous vs 0.7082 binned | bins tie the rows inside a bin |
| Does the plan work? | logistic regression on `PILOT_PLAN` | 0.7858 (audit reference 0.7854), 23 iterations | |
| Inverse transform | log-OLS vs log-link GLM, damage ratio vs slab | elevated 0.208 (geometric means) vs 0.327 (means); crawlspace 0.467 vs 0.569; basement 0.478 vs 0.502 | stage 9's "0.27x slab" is a geometric-mean ratio |

Variance structure (share between episodes): block-group and neighbourhood rain 0.83-0.95, log policy count 0.60, mean coverage 0.01; so only 5-17% of the
rain variance is within an episode, which is what the within-event metric sees. Retransformation: the Duan factor for the damage ratio differs by foundation
(slab 1.96, basement 2.05, crawlspace 2.21, elevated 2.86): one global factor would be wrong.

## Inverse-transform inventory (no missing inverse found in the existing scripts)

Flood and SCS targets are probabilities (nothing to invert). Stock-flow (37) and event features (36) are Poisson with a log link: `exp(eta)` is applied inside the
prediction and the score uses the original-scale prediction (37 then scores `log1p` of it, a loss different from the training likelihood). Script 33 reports medians, which commute with
monotone transforms. The one case needing action is the damage-ratio regression of scripts 30 / stage 9 (geometric-mean ratio, see above). New code that transforms a target must use `TargetTransform`;
`Scaled` outputs are refused by the metrics until inverted.

## Caveats

- The pilot results are one test period (2017-2023). 15 bin comparisons and ~10 other comparisons were made without multiplicity correction; the bin result is therefore labelled a candidate.
- Within-event AUC is invariant to a monotone transform of any single feature, so the transform results above are about the multi-feature linear fit, not about single features.
- The SCS comparison used the K chosen for the current normalisation, which favours it slightly.
- `FeaturePlan` is not yet wired into the SCS scripts (10-14), which keep `anomalise`; the group-aware `fitted` mode with shrinkage is implemented and tested but unused there.
- Stage 9's foundation estimate has not been recomputed with the GLM in script 30 itself; the comparison above is a separate fit with different controls (depth bin and loss year).

---

# Stage 13: raw-data diagnostics, cluster and covariate reports (2026-10-24)

Report: `RAW_STATICS_DA_REPORT.md` (generated; commands, figures in `docs/figures/raw_report/`). Code: `scripts/44_raw_report.py` (+ `44b`, `44c`), `src/wxrisk/diag.py`, `tests/test_diag.py`.
It states every claim of the project that a dataset can be clustered or regressed, tests it, and gives the verdict computed from the data.

## What it found

- **No natural clusters in the SCS environment.** K-means on the anomaly-scaled inputs: silhouette 0.091 (tier B, K=32) and 0.050 (tier C, K=64), against 0.049 / 0.015 for a column-permuted null; the best K is the smallest tried (2);
  ARI between fits about 0.5; Hopkins statistic 0.86 / 0.88 against 0.79 / 0.77 for a Gaussian cloud with the same covariance. K-means is a **quantiser**, not a discovery of groups; it is justified by out-of-sample information:
  cluster labels carry mutual information with every head (permutation p = 0.005, the floor of 200 permutations) and the cluster event rates agree between 2000-2016 and 2017-2022 (Spearman 0.65-0.92), explaining 0.6-4.7% of the label variance.
- **Take-up is bimodal** (bimodality coefficient 0.594 > 0.555; two components cut the BIC by 1,927) but the **number of components is not identifiable** (BIC keeps falling to four); stage 10's "three components" is one point on a flat curve. No Gaussian-process model is used.
- **Flood occurrence is regressable.** Within-event AUC, established in both periods for 17 of 27 cell-level features (strongest: cell rain 0.725, neighbourhood 3x3 mean 0.723, policy count 0.68); no evidence for lat, lon, elevation, rel1, pos3, slope. Events per variable 222. The rain features are
  nearly one piece of information (20 pairs with |r| >= 0.9, variance inflation up to 184).
- **SCS events are regressable** (21 of 28 inputs established for y0; CAPE at 18Z 0.76, 850 hPa humidity 0.73), but **tier C has exact linear dependencies by construction** (`dd_18 - dd_12 = d_t2m - d_d2m`, `d_pw = pw_18 - pw_12`) plus a near one (`msl_12`, `msl_18`): harmless for trees, unidentifiable
  coefficients for the logistic regression, and a double count of those quantities in the K-means distance.
- **Purchases vs claim intensity: present, not stable.** rho +0.13 / +0.17 at lags 0 / 1 over all quarters, +0.23 / +0.28 in 2012-2017 (Harvey) and about 0 in 2018-2023; none of the five economic covariates holds in both halves.
- **Damage ratio is regressable** on depth (within-state-month rho +0.30), foundation group (eta squared 0.092) and elevated indicator (0.068); the SFHA adds nothing alone (0.0008).

## Findings that change earlier statements

1. **Foundation code in the claims file is a function of the occupancy code.** In the raw data every claim with a legacy occupancy code (1, 2, 3, 4, 6) has foundationType '0' and every claim with a two-digit code (11, 12, 15, 16, 18) has a known foundation (year by year for codes 1 and 11, 2000-2023).
   So the "unknown vs known foundation" contrasts of stages 9-10 (for example "the unknown group has a higher loss ratio") are contrasts between occupancy-code families, and the foundation-group effects (stage 9) are estimated on two-digit-coded claims only.
   In the policies file, by contrast, '0' falls within code 1 (86% in 2010, 22% in 2021) and is absent for code 11. The register entries P05 / C10 now say so (the earlier text called it a reporting-year marker in both files).
   Follow-up (`scripts/30 --sections occ`, national single-family claims 2010+, state-month and depth-bin fixed effects): occupancy 11 claims have **0.74×** the loss ratio of occupancy 1 claims with no foundation effect involved, so most of the "unknown is higher" gap (Texas 2017-2021: 0.39 vs 0.32) is a code-family offset.
   The elevated indicator, which both families carry, replicates in each (0.42× in occupancy 1, 0.34× in occupancy 11), so the elevated finding of stage 9 stands. The slab / basement / elevated ratios (Texas 0.27× elevated, national 0.45× basement and 0.27× elevated after the crawlspace fix) are occupancy-11 estimates and are not known to hold for occupancy-1 buildings.
   Consequence for features: in claims-based models `foundation_missing` is an occupancy-family flag (do not offer it as a building property); in the policy-based exposure shares the unknown share is still a legitimate regime/filling-in feature.
2. **Exact duplicate rows in the Texas policies**: 157,230 rows (1.35%) beyond the first occurrence. The redacted file has no policy number, the share depends on block-group size (0.6% to 1.9%) but not monotonically; not removed, effect on counts at most 1.35%.
3. **gridMET**: 27.85% of the cells are a fixed NaN mask (not US land), and cell-days above 300 mm fall on 38 days (84% on 8 days, including 2017-08-26 to 29); no calendar gap, no negative value.
4. The ERA5 "outside physical range" counts of an early draft were my wrong unit assumptions (`lapse75` is a temperature difference in K, `w700` a wind speed); with the correct bounds there are none.

## Caveats

- Within-year AUC and within-episode AUC measure discrimination, not calibrated skill; univariate AUCs up to 0.76 sit next to Brier skill of 0.03-0.04.
- The stock-flow statistics rest on 45 counties and the first half contains one dominant event; intervals understate the uncertainty.
- Hopkins and silhouette depend on the scaling of the inputs; clustering raw physical values would give a different partition.
- The OpenFEMA FAQ was read through a summarising fetch tool, so its quotations are partial.

# Stage 14: scaling plan wired into the models, pilot inputs from the clean tables, open items closed (2026-10-25)

Scripts: `10`, `12`, `14` (SCS, via `scs_eval.anomalise`), `25`, `26`, `29` (flood pilot), `30 --sections glm occ`, `42 bins-cv`, `45` (input parity). No stored result of stages 0-10 was overwritten; every re-run wrote a side file.

## What changed in the code

- `scs_eval.anomalise` is now a `FeaturePlan` (county x month centre shrunk toward the global mean for sparse cells, global fallback for a cell not seen in the fit rows, unit variance on the fit rows). The old function is `anomalise_plain` (it returned NaN for an unseen county-month).
- Script 29's logistic regression runs on `PILOT_PLAN` (log1p / asinh / cyclic month, fitted on 2010-2016 only) next to the plain z-score version.
- `flood_pilot.exposure_matrices` and `claims_tx` read the clean tables (`policies.frame`, `claims_nfip` with the new `flood_event` column); the raw versions stay as `*_raw` for the parity check. The stored `expo_2010_2023.npz` / `claims_cum.npy` had been written by an ad hoc call (no script); script 25 now builds them (`--inputs clean`, default) or reads the stored ones (`--inputs stored`).

## Results

- **SCS K-means (script 10), lift readout, BSS against p0**: tier C y0 0.0431 -> 0.0446, y10k 0.0188 -> 0.0193; tier B y0 0.0265 -> 0.0321 (K chosen 32 -> 128, a flat curve). Tier A uses only static columns, so its change (0.0069 -> 0.0047, K 128 -> 64) is K selection on a flat curve, not the normalisation. No change exceeds the bootstrap interval.
- **Script 14 (20 tier x label rows)**: logistic regression BSS changes by at most 0.006 (median below 0.001); boosting by at most 0.006. **Script 12 (96 rows)**: median |change| 0.0007, maximum 0.0078. The SCS conclusions of stages 3-5 do not move; their stored numbers were produced with `anomalise_plain`.
- **Pilot logistic regression (script 29, within-event AUC, test 2017-2023)**: z-score only 0.7806, scaling plan 0.7858, boosting depth 5 0.7859. The plan closes the gap between the linear and the tree learner.
- **Pilot inputs from the clean tables (script 45)**: claims in the cumulative array identical (0 differing cells). Exposure differs in 7% of block-group-days (totals -0.19% to +0.06%): the raw code ended every policy 366 days after its effective date, the provider term is 365 days (366 across a leap day), so the clean arrays are the faithful ones. The rebuilt table (`flood_pilot_table_clean.parquet`) has 2,996,630 rows against 2,996,576, identical labels, feature correlations >= 0.9993. Script 26 on it: within-event AUC L2 0.7857 -> 0.7864, L4b 0.7809 -> 0.7861, L5 0.7839 -> 0.7878 (the stored table remains the one quoted in stage 8).
- **Exposure bins, blocked-by-year CV (E9, 7 folds of two test years, cut points refitted per fold)**: pooled within-event AUC 0.7892 -> 0.7919, difference +0.0028 [+0.0003, +0.0051], positive in 6 of 7 folds. Confirmed, small; for linear / distance learners only.
- **Foundation effect restated with the log-link GLM (script 30 `glm`, occupancy 11 claims, depth bin + fixed effects)**: ratio of geometric means (log-OLS) against ratio of means (GLM): Texas elevated 0.27 / **0.47**, crawlspace 0.29 / 0.50, basement 0.74 / 0.96 (62 claims); national with state-month fixed effects elevated 0.26 / **0.39**, crawlspace 0.55 / 0.67, basement 0.46 / 0.55. In dollars the elevated reduction is about 0.4-0.5x, not 0.27x. The stage 9 text quotes the geometric-mean figure.
- Stage 10's Harvey take-up figures carry a dated correction pointing at the stage 11 table.

## Caveats

- The SCS re-runs show equivalence within noise, not an improvement; K selection on flat validation curves moves by itself between runs.
- The clean-input pilot table was not adopted as the default table of stages 8-10; adopting it needs the wavelet and terrain features regenerated for the 376 changed (episode, block group) rows.
- E9 uses folds that train on later years to predict earlier ones; it checks the feature-engineering decision, not forecast skill.
- GLM ratios have no bootstrap interval here (single fit); Texas basement is untestable.

# Stage 15: sparse-to-dense test on hourly MRMS, motion-compensated prior vs observations alone (2026-10-25)

Scripts: `39` (8 Texas events 2021-2023, ~450 MB transferred, 42 MB kept as crops, MRMS MultiSensor QPE Pass2 01H from `noaa-mrms-pds`, ~4 km), `46` (the test; output `audit/mrms_sparse.{txt,json}`).
Question (the idea of the first request): does an old dense field, advected by the motion measured between the two frames before it, add information to a few current observations?

## Design

Truth = MRMS at hour t. Observations = a random 0.5 / 2 / 8% of cells (perfect point values, an upper bound on a gauge network). Methods: S (inverse-distance interpolation of the observations), P (persistence of F[t-L]), M (F[t-L] shifted by L times the phase-correlation motion of F[t-L-1] -> F[t-L]), PO / MO (P / M plus the interpolated residual at the observed cells). L = 1 and 3 h. Rainy hours only (>= 2% of cells at >= 1 mm/h), scored on the cells that were NOT observed; interval = bootstrap over the 8 events.

## Results (mean over events; RMSE in mm/h; CSI at >= 5 mm/h)

| L, observed cells | S RMSE / CSI | M | MO | MO - S in RMSE skill (1 - RMSE/RMSE_S) |
|---|---|---|---|---|
| 1 h, 0.5% | 2.12 / 0.205 | 2.20 / 0.280 | 2.15 / 0.308 | -0.003 [-0.049, +0.043] |
| 1 h, 2% | 1.72 / 0.348 | 2.19 / 0.280 | 1.87 / 0.374 | -0.077 [-0.118, -0.028] |
| 1 h, 8% | 1.27 / 0.499 | 2.19 / 0.280 | 1.45 / 0.477 | -0.136 [-0.168, -0.095] |
| 3 h, 2% | 1.73 / 0.340 | 3.03 / 0.078 | 2.15 / 0.290 | -0.242 [-0.279, -0.202] |

- **Motion compensation works as a motion estimate**: the advected field beats persistence by 0.42 mm/h RMSE at L = 1 [-0.66, -0.27] and 0.22 at L = 3 [-0.33, -0.11]; CSI 0.280 against 0.203. Mean estimated shift 4.8 pixels (about 19 km per hour), coherence 17.
- **It does not beat the observations alone** once there are a few percent of cells observed: with the residual correction (MO) it is worse than S in RMSE at 2% and 8% (and at 3 h at every density) and ties at 0.5% (skill -0.003, interval contains 0). At 0.5% and 2% it is better in CSI of heavy rain (0.308 against 0.205; 0.374 against 0.348), so the prior helps place the heavy cells when the network is very sparse, at no gain in squared error.
- The prior's advantage decays quickly with its age (L = 3 h loses most of it), as expected for convective rain.

## Conclusions

- The hypothesis "an advected dense prior plus sparse observations is a better dense field than the observations alone" is **not supported** on RMSE for networks denser than about 0.5% of 4 km cells (about one point per 3,000 km^2); it is weakly supported on heavy-rain CSI at 0.5-2%.
- As a source of data augmentation the idea needs a dense prior at t-1, which this setting assumes (radar). Where only sparse gauges exist at t-1 as well, the prior is not available, so this test is an upper bound for that use.

## Caveats

- 8 events, 3 days at most each, Texas only; intervals are over events and the hours inside an event are not independent.
- The prior and the truth come from the same product, and the observations are perfect: both favour the method, which still lost.
- Kriging variant (below) tried the obvious fix of the naive residual correction; it did not close the gap. Not tried: a non-stationary error model (error variance growing with the prior's value), a different motion model than one global shift, an ensemble of priors.
- Edge effects of the shifted field (zero fill) and the 4x4 block average are not examined.
- The first half of the original idea (extrapolating the RISK of a neighbouring cell, as opposed to its weather) is not tested here.

## Addendum: the same test with kriging (optimal interpolation) instead of inverse-distance weighting

Simple kriging of the residual (exponential covariance, 5% relative nugget, 16 nearest observations, range chosen per hour among 2 / 4 / 8 / 16 pixels by holding out 20% of the observations), so the correction fades to zero away from the observations and the prior takes over there. SK = kriged observations alone (prior 0), POK / MOK = kriged residual added to persistence / to the advected prior. Same hours, same observation sets; RMSE in mm/h, difference MOK - SK with the event bootstrap interval (positive = the prior makes it worse):

| L, observed cells | SK RMSE | MOK RMSE | MOK - SK |
|---|---|---|---|
| 1 h, 0.5% | 2.146 | 2.106 | -0.040 [-0.138, +0.060] |
| 1 h, 2% | 1.722 | 1.868 | +0.146 [+0.061, +0.233] |
| 1 h, 8% | 1.210 | 1.421 | +0.210 [+0.136, +0.290] |
| 3 h, 0.5% | 2.130 | 2.659 | +0.530 [+0.364, +0.690] |
| 3 h, 8% | 1.217 | 1.507 | +0.290 [+0.208, +0.372] |

- Kriging helps every variant a little at 8% (SK against inverse-distance S: RMSE skill +0.04 [+0.03, +0.06]) and nothing at 0.5-2%, but it **does not change the ordering**: observations alone are at least as good as observations plus the advected prior from 2% of the cells upward, and clearly better at a 3 h prior age. At 0.5% (1 h) the prior ties (skill +0.026 against inverse-distance S, interval [-0.014, +0.069]; difference to SK above includes 0).
- The motion still pays against persistence (MOK - POK = -0.37 [-0.52, -0.26] at 0.5%, 1 h): the advected prior is a better prior than the old field, it is just not a better one than the observations once there are a few of them.
- The prior's heavy-rain CSI advantage at very low density survives only for inverse-distance weighting (0.306 against 0.203 for S); kriged observations alone have 0.167 at 0.5% / 1 h (the kriged zero-mean field smooths peaks away), where MOK has 0.307.

**Conclusion of the stage**: with a global advection of a dense hourly field and the observation densities tested, the motion-compensated prior is not worth adding to point observations for the squared error; its value is in placing heavy-rain cells when almost nothing is observed (0.5% of cells).

# Stage 16: forecast-mode flood pilot, observed rain replaced by GEFSv12 reforecast rain (2026-10-26)

Scripts: `47` (GEFSv12 reforecast APCP, 00Z runs of the day before each episode, control + 4 members, leads 36-84 h, Texas crop; 1,640 days, 410 MB kept of ~21 GB read), `48` (features and models; output `audit/flood_forecast_pilot.{txt,json}`).
Question: stage 8 used observed rain, so its 0.78 within-event AUC is a perfect-prognosis number. How much of the rain skill survives when the rain is a forecast?

## Design

- Same unit, label and learner as stage 8 (block group x rain episode, any inland non-tropical claim, boosting depth 5), restricted to episodes starting 2010-2019 because the reforecast ends in 2019: 30,362 episodes, 2.05 M rows; train 2010-2016 (7,553 positives), test 2017-2019 (2,075 positives, 497 episodes with a within-event AUC).
- Rain enters through the two gridMET days the run can see (day S and S+1), F1 = largest daily rain, S = sum, truncated at the episode end. A gridMET day was matched to 12Z..12Z by pooled correlation with the ensemble mean over 24 prototype days (r 0.70 at that window against 0.55-0.61 for the neighbouring 6 h shifts).
- Sources, same features in every row of the table: obs4k (gridMET at the block group's 4 km cell), obsC (gridMET averaged over the 0.25 deg GEFS cell), fcmean (5-member mean at the cell), fcens (fcmean + share of members >= 25 / 50 mm, spread, 3x3 neighbourhood). County max / mean of the same quantity added in each. Exposure columns and duration / month identical.

## Results (test 2017-2019, within-event AUC, pair weighted)

| Rain input | within-event AUC | paired difference, event-clustered bootstrap |
|---|---|---|
| exposure and calendar only | 0.622 | |
| obs4k | 0.736 | +0.114 [+0.071, +0.171] over exposure only |
| obsC (resolution only) | 0.712 | -0.024 [-0.047, -0.004] vs obs4k |
| fcmean | 0.667 | -0.045 [-0.074, -0.008] vs obsC; +0.046 [+0.020, +0.083] over exposure only |
| fcens | 0.669 | +0.002 [-0.017, +0.016] vs fcmean |
| trained on obsC, scored with fcmean | 0.667 | same as training on the forecast (+0.0004 [-0.009, +0.013]) |

Untrained scores: point value obs4k 0.647, obsC 0.604, fcmean 0.539; vote >= 25 mm 0.519 (overall AUC 0.60).

## Conclusions

- The rain skill drops from +0.114 to +0.046 over exposure, so about 40% of the observed-rain gain survives a 1.5-3.5 day forecast. Of the 0.068 lost, about 0.024 is resolution (4 km to a 0.25 deg cell) and about 0.045 is forecast error.
- Ensemble information (vote fractions, spread, neighbourhood) adds nothing on top of the mean (+0.002 [-0.017, +0.016]), and training on forecast rain instead of observed rain does not help (+0.0004). With five members and one run the position error is not what limits the score.
- Exposure alone carries 0.62 of the 0.67 forecast score, so in forecast mode exposure is the larger part of the skill. The ranking inside a county would improve mostly with finer rain forecasts, not with a smarter use of five members.
- Decision effect: neighbourhood / encoder machinery is still not supported (it was not even needed at 0.25 deg); the open question moves to forecasts of finer resolution and more members.

## Caveats

- Test = 3 years (2017-2019), 497 episodes; stage 8's 0.786 is on 2017-2023 and all rain days, so absolute levels here (obs4k 0.736) are not comparable with it, only the rows of this table are.
- One run per episode (00Z the day before, leads 36-84 h) and episodes truncated at two days; later days and longer leads are not tested. Control + 4 members only (11 on Wednesdays were cut to 5).
- GEFSv12 reforecast is a fixed 2000-2019 re-run of one model, not an operational forecast archive; its skill is not IFS's. 0.25 deg is about 25 km, still much coarser than a block group (the resolution cost above is measured on gridMET).
- The day alignment came from 24 days; rain gain depends on it (a 6 h shift lowered the pooled correlation from 0.70 to 0.55).
- No calibration of the level was tried (positive rate changes by era), and the model is the stage 8 learner with default settings.

# Stage 17: information audit of the flood models, which data would lift the bottleneck (2026-10-26)

Script `49` (output `audit/information_audit.txt`). Within-event AUC, event-clustered paired bootstrap. Base model = stage 8 L2 features (county rain, exposure, cell rain, boosting depth 5); observed-rain arm trained 2010-2016, tested 2017-2023 (3,711 positives); forecast arm = stage 16 subset with ensemble-mean rain, tested 2017-2019 (2,075 positives).

## 1-2. Where the stage 8 model (L5 predictions) is weak

- By stratum: many-positive episodes are easiest (>= 10 positives 0.798, 1 positive 0.703); the lowest-rain quarter of episodes 0.68-0.70 (62-101 episodes, wide), the highest 0.799; episodes of 1 day 0.807, of >= 5 days 0.757; county size 21-60 block groups 0.725 against > 150 at 0.784; by year 0.738 (2018) to 0.838 (2020).
- Positives ranked in the bottom quarter of their episode are 4.9% of positives (61% are in the top quarter). They are **small-exposure block groups**: median 16 policies against 76 for top-quarter positives, 0 flood-zone share against 0.24, area 1.3 against 2.6 km2. Only 3.9% of them have cell rain below 10 mm and 3.7% of all positives have cell rain below a quarter of the county peak: the missed claims are not mostly rain placed in the wrong cell.

## 4. Label-randomness ceiling

If the model's own probabilities (re-centred to the test prevalence) were the truth, redrawn labels would give within-event AUC 0.7909 (sd 0.0045 over 10 redraws) against the observed 0.7839. The model is at the level its own probabilities imply; remaining misranking is consistent with Bernoulli noise given those probabilities. It is not a bound on the achievable skill (new information changes the probabilities, see below, and 0.812 exceeds it).

## 3. Information added to the base model

| Added information | observed rain, 2017-2023 | forecast rain (fcmean), 2017-2019 |
|---|---|---|
| base | 0.7857 | 0.6674 |
| + claims 2000-2009 in the block group (`hist00`) | +0.016 [+0.011, +0.021] | +0.037 [+0.013, +0.057] |
| + claims in the 3 years before the episode (`prior3`) | +0.018 [+0.013, +0.024] | +0.030 [+0.017, +0.042] |
| + both | **+0.026** [+0.020, +0.033] | **+0.051** [+0.023, +0.075] |
| + terrain (elevation, relative height, slope) | -0.002 [-0.010, +0.007] | -0.031 [-0.070, +0.009] |
| + history + terrain + location / area / 14-day antecedent rain | +0.025 [+0.018, +0.031] | +0.044 [+0.018, +0.067] |
| oracle: outcome of the 5 nearest other block groups (not known ahead) | +0.022 [+0.015, +0.029] | +0.085 [+0.045, +0.116] |

(A first oracle, the leave-one-out share of the whole episode, was anti-correlated with a block group's own label by construction and gave -0.03; it was dropped.)

## Conclusions

- **Flood history is the largest available gain not yet in the pilot**: +0.026 with observed rain and +0.051 with forecast rain, known ahead and from data already in the repository. The gain doubles in forecast mode, so static susceptibility replaces some of the local rain detail that a 25 km forecast loses.
- **Centroid terrain stays null, also in forecast mode**, and history absorbs location / area / antecedent rain. Susceptibility needs a different representation (below), not more centroid features.
- **Spatial coherence of the outcome is modest in observed mode** (+0.022 even knowing the 5 nearest outcomes) and large in forecast mode (+0.085): the forecast leaves about that much local information on the table. History recovers about 60% of it.
- Observed-rain misses are exposure-noise (small block groups) rather than rain placement; the data that would change the ranking is therefore building / susceptibility detail, not finer daily rain, in observed mode. Finer rain matters for forecast mode (stage 16: 0.024 resolution + 0.045 forecast error).

## Caveats

- Same tests as before (one split, 2017-2023 / 2017-2019); intervals from event-clustered bootstrap, no multiplicity correction over the 6 added-information rows.
- History counts claims of the same peril in the same block group: it is partly a proxy for flood zone, building stock and take-up, so it is not a clean "susceptibility" measure and its gain may shrink once better static layers are added.
- `prior3` counts claims before the episode start only; episodes inside a continued wet spell share history with the previous one.
- The ceiling is a self-consistency check, not an information bound.


# Stage 18: USGS streamgage history as a susceptibility prior (script 50)

Question: does the flood record of the nearest small-basin gauge (annual peak discharge before 2010, so known ahead) add to the pilot model, alone or on top of claim history?
Data: api.waterdata.usgs.gov through the `dataretrieval` package with an API key (896 Texas stream locations with a daily-discharge series; peaks at 845 of them, 31,861 annual peaks). Eligible gauges: at least 10 pre-2010 peaks and drainage area up to 1000 mi2 (374 gauges); 75.8% of block groups have one within 25 km. Features of the nearest eligible gauge: distance, log area, log specific peak, sd of log peaks, log(max/median).

| Added information (within-event AUC difference to base, event-clustered 95% interval) | observed rain, 2017-2023 | forecast rain (fcmean), 2017-2019 |
|---|---|---|
| base | 0.7857 | 0.6674 |
| + gauge history | +0.0049 [+0.0007, +0.0101] | -0.0376 [-0.0857, +0.0071] |
| + claim history (`hist00`, `prior3`) | +0.0262 [+0.0198, +0.0325] | +0.0510 [+0.0225, +0.0752] |
| + claim history + gauge history | +0.0278 [+0.0218, +0.0341] | +0.0482 [+0.0244, +0.0682] |

## Conclusions

- **Gauge history adds almost nothing.** Alone it is +0.005 with observed rain (the interval barely excludes 0) and unstable in forecast mode (-0.038, interval spans 0). On top of claim history the extra is +0.0016 (observed) and -0.0028 (forecast), well inside the noise.
- Claim history stays the best static prior (+0.026 / +0.051); a gauge is a coarser proxy for the same thing (how flood-prone the neighbourhood is).
- Not worth adding to the model; do not rebuild without a better link than "nearest gauge".

## Caveats

- The gauge is the nearest eligible one within 25 km, not the stream reach of the block group; 24% of block groups have none (NaN handled by the tree learner), which is mostly arid west Texas and small-exposure areas.
- Gauge peaks describe large rivers; NFIP claims in small urban basins (flash floods) are often unrelated to them. This does not say hydrology is uninformative, only that this link is.
- One split, one learner (HistGradientBoosting depth 5), no multiplicity correction.


# Stage 19: claim history in the pilot model, full evaluation (script 53)

Stage 17 found +0.026 from `hist00` (claims 2000-2009 in the block group) and `prior3` (claims in the 3 years before the episode). Here the gain is checked against the evaluation rules: two learners, overall AUC and AP next to within-event AUC, and blocked cross-validation.

| Main split (train 2010-2016, test 2017-2023) | within-event AUC | overall AUC | AP | difference to the same learner's base [95% interval] |
|---|---|---|---|---|
| boosting, base (L2 features) | 0.7857 | 0.9258 | 0.1169 | |
| boosting + history | **0.8120** | 0.9307 | 0.1315 | +0.0262 [+0.0198, +0.0325] |
| logistic on the scaling plan, base | 0.7856 | 0.9224 | 0.1354 | |
| logistic + history | 0.8076 | 0.9257 | 0.1429 | +0.0219 [+0.0168, +0.0265] |

Blocked cross-validation (boosting, 7 year-pair folds, training on the other 12 years), within-event AUC:

| Held-out years | 2010-11 | 2012-13 | 2014-15 | 2016-17 | 2018-19 | 2020-21 | 2022-23 | pooled |
|---|---|---|---|---|---|---|---|---|
| base | 0.7908 | 0.7912 | 0.7572 | 0.8142 | 0.7505 | 0.7916 | 0.8113 | 0.7827 |
| + history | 0.8087 | 0.8205 | 0.7800 | 0.8344 | 0.7878 | 0.8092 | 0.8291 | 0.8067 |
| difference | +0.018 | +0.029 | +0.023 | +0.020 | +0.037 | +0.018 | +0.018 | **+0.0240** [+0.0172, +0.0322] |

## Conclusions

- **The history gain is robust**: positive in all 7 folds (+0.018 to +0.037), pooled +0.024 with an interval above 0, and +0.022 / +0.026 for the logistic and boosting learners on the main split. It is also positive on overall AUC and AP.
- Boosting and the linear model are tied again on within-event AUC with history (0.8076 vs 0.8120 is within about twice the interval half-width of the difference to the base); the linear model has the better AP in both rows. Keep S12 (linear default, boosting as a check).
- The gain does not fade in the latest folds (2022-23: +0.018), although `hist00` covers 2000-2009 only; `prior3` keeps the history current.
- **Decision**: `hist00` and `prior3` enter the pilot feature set (add them to `PILOT_PLAN` as standardise-only columns when the pilot table is next rebuilt).

## Caveats

- Blocked CV trains on years after the held-out pair too; history features are ex ante for every row, but the fitted relation uses later data, as in every blocked-by-year CV here.
- The history comes from the same NFIP claim file as the label: block groups with many claims 2000-2009 are partly the ones with many policies, so part of the gain is exposure and take-up history, not only physical flood-proneness (the model already has current policy counts).
- Observed-rain arm only; the forecast-rain arm is stage 17 (+0.051, one split). Texas only.
- Fold differences are not independent (shared episodes' neighbours across years are not an issue, but weather regimes cluster by year); the pooled interval is event-clustered, not year-clustered.

# Stage 20: learner convergence, Bayesian search per family (scripts 54, 55; run on Sol)

Question: is classical ML exhausted on the pilot table (base + `hist00`, `prior3`) before deep learning is tried? One optuna study (TPE, multivariate) per learner family, 60 trials each, objective = mean within-event AUC over three rolling-origin folds (train ≤2011 → 2012-13, ≤2013 → 2014-15, ≤2015 → 2016); the best parameters are refitted on 2010-2016 and scored once on 2017-2023. Rows 2,996,576 (negatives subsampled to 5% in the search), 3,711 test positives. Output: `audit/learner_convergence.txt`.

| Family | validation (best of 60) | test within-event AUC | overall AUC | AP | vs logistic [95%] | vs boosting [95%] |
|---|---|---|---|---|---|---|
| logistic | 0.8123 | 0.8076 | 0.9256 | 0.1424 | | -0.0070 [-0.0122, -0.0021] |
| logistic, degree 2 | 0.8112 | 0.8120 | 0.9260 | 0.1308 | +0.0044 [-0.0001, +0.0089] | -0.0026 [-0.0074, +0.0020] |
| boosting (HGB) | 0.8136 | 0.8146 | 0.9306 | 0.1281 | +0.0070 [+0.0021, +0.0122] | |
| random forest | 0.8103 | 0.8080 | 0.9316 | 0.1374 | +0.0004 [-0.0048, +0.0059] | -0.0067 [-0.0105, -0.0020] |
| extra trees | 0.8090 | 0.8112 | 0.9299 | 0.1491 | +0.0035 [-0.0021, +0.0091] | -0.0035 [-0.0067, +0.0005] |
| MLP (sklearn) | 0.8177 | 0.8131 | 0.9281 | 0.1393 | +0.0055 [-0.0000, +0.0112] | -0.0015 [-0.0053, +0.0029] |
| average of logistic, boosting, MLP | | **0.8162** | 0.9313 | 0.1524 | +0.0086 [+0.0050, +0.0123] | +0.0016 [-0.0008, +0.0045] |

Convergence of the searches (best validation value after 10 / 20 / 40 / 60 trials): logistic 0.8123 flat from trial 10; logistic2 0.8111 → 0.8112; boosting 0.8085 → 0.8128 → 0.8128 → 0.8136; rf 0.8037 → 0.8080 → 0.8086 → 0.8103; extra trees 0.8054 → 0.8057 → 0.8090 → 0.8090; MLP 0.8114 → 0.8123 → 0.8165 → 0.8177. Learning curve (share of training episodes, test): logistic 0.8022 / 0.8060 / 0.8074 / 0.8075, boosting 0.8065 / 0.8119 / 0.8137 / 0.8146 at 10 / 25 / 50 / 100%.

## Conclusions

- **Every family lands in a band of 0.808-0.815 on test** (0.809-0.818 on validation). Tuning moved boosting only from 0.8120 (stage 19, default) to 0.8146; the logistic model is unchanged at 0.8076. The best single family is boosting, +0.007 over logistic with an interval above 0; the other families are not separable from it or from logistic at the 95% level.
- The searches have flattened: logistic and extra trees stopped improving by trial 25-53 with a top-5 spread under 0.001; boosting gained 0.0008 in the last 40 trials. Random forest and the MLP were still creeping up at the end (+0.0017 and +0.0012 in trials 40-60), so their ranks are not final, but the size of the creep is below the test intervals.
- The learning curve is flat from 50% to 100% of the training episodes (+0.0001 logistic, +0.0009 boosting): more rows of the same kind will not help these learners.
- A plain average of three families gains +0.0016 over boosting [-0.0008, +0.0045]: no evidence that combining learners adds information.
- **Decision**: classical tabular learners are converged on this feature set at about 0.81-0.815 within-event AUC. A deep model is only worth trying if it brings new input (spatial neighbourhoods, rain fields over time), not as another learner on the same columns. Keep S12 (linear default, boosting as a check).

## Caveats

- The validation folds end at 2016 and the search subsamples negatives to 5%; validation and test values differ by up to 0.005 (MLP 0.8177 vs 0.8131, logistic 0.8123 vs 0.8076), partly selection on the validation folds, partly year effects. Rank on the validation column and on the test column disagree (MLP first on validation, boosting on test).
- 60 trials per family is a fixed budget, not a proof of the optimum; the search space was set by hand (e.g. HGB depth 2-10, MLP at most 128-64-32). MLP is sklearn's, with no early stopping or embeddings; this is not a test of deep learning.
- The test set is one split (2017-2023), scored once. Intervals are event-clustered, not year-clustered, and nine comparisons are made without multiplicity correction; "vs logistic" for logistic2, extra trees and MLP touch zero.
- The ceiling block printed by script 55 (0.8004 if the average's probabilities were true) is below the observed 0.8162, so the model's probabilities are under-confident or miscalibrated, and that number is not an upper bound on achievable AUC. It does not show how much signal is left.
- Observed-rain table, Texas only; absolute probabilities do not transfer across years (AGENTS.md).

# Stage 21: neighbour-smoothed claim history, year split and year + region split (script 56)

Question: does an ex ante spatial version of the stage 17 neighbour oracle add to the stage 19 model (base + `hist00`, `prior3`)? Features: `nbhist00` / `nbprior3` (log1p of the mean claim count of the 8 nearest block groups, 2000-2009 and the 3 years before the episode), `nb_dist` (mean distance to them), and `lat`/`lon` in the third arm. Boosting as in stage 19. Two evaluations: time split (train ≤2016, test later) and year + region split (train ≤2016 outside the test region, test later inside it; 5 k-means regions of county centroids held out in turn, predictions pooled). Output: `audit/spatial_features.txt`.

Within-event AUC (difference to the stage 19 base, event-clustered 95% interval):

| Rain arm / split | base | + neighbour history | + neighbour history + lat/lon |
|---|---|---|---|
| observed, time split (2017-2023) | 0.8120 | -0.0014 [-0.0043, +0.0010] | -0.0024 [-0.0051, +0.0006] |
| observed, year + region split | 0.7841 | +0.0012 [-0.0042, +0.0063] | +0.0061 [-0.0029, +0.0162] |
| forecast fcmean, time split (2017-2019) | 0.7184 | -0.0023 [-0.0115, +0.0051] | -0.0589 [-0.1367, +0.0061] |
| forecast fcmean, year + region split | 0.6468 | **+0.0310 [+0.0068, +0.0709]** | +0.0151 [-0.0024, +0.0377] |

## Conclusions

- **Under observed rain, neighbour history adds nothing** in either split (intervals contain 0), consistent with the oracle bound of +0.022 and with the own-block-group history already carrying most of it.
- **Holding out a region costs the base model 0.028 (observed) and 0.072 (forecast)**: part of the stage 19 history gain relies on having seen the same region in training. The time split overstates how the model transfers to a new area.
- **The only interval above zero is forecast rain on unseen regions: +0.031 from neighbour history**, recovering about 40% of the region-split loss. Where the model has no training data for the region, the neighbours' claim record is the information it lacks. Under the time split the same feature is flat.
- **Raw `lat`/`lon` hurt** with forecast rain (-0.059 in the time split, wide interval) and are no better than zero elsewhere: with 3 test years and 7 training years the trees memorise locations. Do not add coordinates.
- **Decision**: keep `nbhist00`, `nbprior3`, `nb_dist` as an optional feature set for the forecast, new-region setting; do not add them to the observed-rain model. Adjacency / upstream graphs and a GNN are not supported by this result: the neighbourhood information is worth at most about +0.03 here, and only in the transfer setting.

## Caveats

- One significant cell out of eight comparisons, no multiplicity correction; the forecast arm has 3 test years (2,075 positives) and a wide interval ([+0.007, +0.071]). Treat it as a lead, to be repeated with more years (GEFS reforecast ends 2019).
- Region-split predictions come from 5 different models and are pooled before the within-episode AUC; episodes crossing regions compare scores of different models, which can shift the level of both arms.
- Regions are k-means on county centroids, not watersheds; flood-relevant neighbours (upstream) may lie in another region. K = 8 nearest by centroid distance, not adjacency or flow.
- Neighbour features use all block groups' claim records, including those in held-out regions (ex ante, no label of the test episodes): that is the setting of a new region with a claim record, not one without.
- Boosting only, one parameter set (stage 19); Texas only.

# Stage 22: FEMA NFHL flood-zone area shares and computed HAND (scripts 57, 58, 59; fetch_nfhl_tx)

Question: does physical hazard geometry that the pilot table lacks (stage 8 terrain was a single centroid at 150 m) add to the stage 19 model? Features per 2010 block group: `nfhl_sfha`, `nfhl_x02` (area shares of the SFHA and of shaded X from FEMA NFHL polygons, county by county from FEMA's ArcGIS service), `nfhl_mapped` (county has NFHL polygons), and HAND statistics (`h1_*` streams ≥ 1 km², `h10_*` streams ≥ 10 km²) as zonal statistics over the block-group polygon of a HAND raster computed from AWS Terrarium z11 (about 70 m; pyflwdir D8, priority flood). The official Texas 3 m HAND (ORNL/TACC) was not reachable and would be hundreds of GB. The NFHL area share is only moderately related to the policy-based `sfha_share` (Spearman 0.54 across block groups), HAND median to `nfhl_sfha` -0.11.

Within-event AUC, difference to the stage 19 base (event-clustered 95% interval):

| Rain arm / split | base | + NFHL shares | + HAND | + NFHL + HAND |
|---|---|---|---|---|
| observed, time split | 0.8120 | -0.0011 [-0.0033, +0.0008] | -0.0023 [-0.0051, +0.0002] | -0.0034 [-0.0068, -0.0004] |
| observed, year + region split | 0.7841 | +0.0005 [-0.0060, +0.0069] | +0.0017 [-0.0048, +0.0082] | -0.0021 [-0.0080, +0.0038] |
| forecast, time split | 0.7184 | -0.0005 [-0.0046, +0.0025] | -0.0006 [-0.0036, +0.0033] | -0.0018 [-0.0072, +0.0039] |
| forecast, year + region split | 0.6468 | -0.0014 [-0.0040, +0.0001] | **+0.0274 [+0.0039, +0.0625]** | -0.0032 [-0.0084, +0.0021] |

## Conclusions

- **Neither NFHL area shares nor HAND add information** in 11 of the 12 cells; the combined arm is slightly negative under observed rain with the time split (-0.0034 [-0.0068, -0.0004]).
- The single positive cell (HAND, forecast rain, unseen regions, +0.027) is the same noisy cell in which neighbour history gave +0.031 (stage 21), and it disappears when NFHL is added next to HAND (-0.003). With the base at 0.647 in that cell, a 0.03 swing is within what one more feature can do by shifting the fit; it is not evidence of a HAND effect.
- This extends the stage 8 and stage 18 pattern: static susceptibility, whether from a centroid, a flood-zone polygon, a streamgauge or a drainage-relative height, adds nothing once exposure, the block group's own rain and its claim history are known. Claim history (`hist00`, `prior3`) is the only static information that has helped (+0.026).
- **Decision**: do not add NFHL or HAND columns to the pilot feature set. The features and scripts stay for later use (building-level work, other states).

## Caveats

- HAND at ~70 m cannot see storm drains, culverts, levees or road embankments, and block-group zonal statistics blur the contrast between a floodplain and the terrace beside it. The result does not show that high-resolution HAND, or HAND at building footprints, adds nothing.
- NFHL maps are not complete (12% of block groups lie in counties with no NFHL polygons, where zero can mean unmapped), and maps date from different years; the `nfhl_mapped` column only partly absorbs this. Area shares by block group dilute small mapped floodplains inside large rural block groups.
- The same-cell caveats of stage 21 apply: forecast arm has 3 test years, region-split predictions come from five models, no multiplicity correction (one of twelve intervals above zero, one below).
- Boosting with the stage 19 parameter set only; Texas only; NFHL polygons were generalised on the server to about 30 m.


# Stage 23: noise in the flood label, cleaner / noise-aware targets, and other information (scripts 60, 61)

Question: how noisy is the stage 8 label (a block group has an inland non-tropical NFIP claim in [start, end + 2 d] of a rain episode), does a cleaner or exposure-aware label help, and which other data could add information about the label. Same split as stage 8 (train 2010-2016, test 2017-2023), L2 features, boosting depth 5, within-event AUC, event-clustered paired bootstrap. Output `audit/label_noise_audit.txt`.

## A. What the label is

- **Coverage**: 88.8% of the 34,989 Texas claims fall in a pilot episode window (2010 0.47, 2011 0.55, 2017 0.65, 2020 0.66, other years 0.82-0.95). The uncovered claims sit mostly in tropical-cyclone windows that the pilot drops on purpose (their county rain is often 150-260 mm); only 48% have a county daily maximum >= 30 mm in the 4 days before.
- **Payment**: 16.2% of claims have no building payment (12.3% missing, the rest 0): closed without payment, deductible, or not covered.
- **Exposure**: the positive rate rises from 0.07% (<= 2 policies) to 0.74% (> 99). Rows with <= 2 policies are 19% of rows and 3.6% of positives, and have the lowest claims per policy at the top (0.0005 per policy-episode at 1 policy, 0.0001 above 99): a block group label is a Bernoulli draw whose probability depends on how many policies are there.
- **Concentration**: 2,425 of 44,212 episodes have any positive; the top 1% of episodes hold 74% of positives and the top 5% 98%. 2015-2016 hold 46% of positives. The label is effectively about a few hundred events.
- **Independent report** (script 61, NWS Storm Events county Flash Flood / Flood records overlapping the episode): P(report | episode has an NFIP positive) = 0.43, P(NFIP positive | report) = 0.38. Both rise together with county peak rain (positive share 1.7% below 50 mm to 44% above 150 mm; report share 1.4% to 54%). Of episodes with rain >= 75 mm and a report, 54% (965 of 1,773) have no NFIP positive. Stable across years (0.24-0.46). Each source misses roughly half of what the other sees: neither is the truth, both are incomplete detectors.

## B. Label variants (trained on the variant, scored on the original label | on a strict paid > 0 label)

| Variant | vs original label: within-event AUC, diff to orig [95%] | vs strict label |
|---|---|---|
| orig | 0.7857 | 0.8007 |
| paid > 0 only | -0.0074 [-0.0166, +0.0016] | -0.0087 |
| cause 4 only | -0.0006 [-0.0041, +0.0032] | -0.0001 |
| window [start, end] | -0.0034 [-0.0072, +0.0007] | -0.0030 |
| window [start, end + 5 d] | -0.0078 [-0.0167, +0.0015] | -0.0084 |
| >= 2 claims positive | -0.0108 [-0.0173, -0.0050] | -0.0096 |
| train on >= 3 policies only | -0.0019 [-0.0071, +0.0030] | -0.0025 |
| event weights 1/sqrt(positives) | -0.0114 [-0.0246, +0.0009] | -0.0104 |
| Poisson on claims / policies, weight policies | -0.0205 [-0.0401, -0.0037] | -0.0226 |

## Conclusions

- **No cleaning or reweighting of the label helps; the original is best or tied.** Even the strict label does not prefer its own training variant. Rule-based cleaning of this label is closed.
- The exposure-aware Poisson target is worse (-0.02): the exposure effect is already in the features (log policies), and the occurrence indicator ranks better than a rate. Dropping the many small block groups from training loses nothing but gains nothing.
- The noise is mainly **random under-detection** (insurance take-up and claim filing), not systematic misdating or mislabelled cause: cause, payment and window changes move the score by under 0.01. This agrees with stage 17 (the model is at the level its own probabilities imply, 0.791 against 0.784).
- Where other information could matter is not the label construction but (a) **a second detector for the same flood** (episode level: Storm Events agrees on only ~40%), and (b) **sub-county ground truth** that does not depend on insurance. Within-episode ranking is what the main metric measures, and only sub-county sources inform it (see below).

## Other sources (assessment, not run)

| Source | Resolution | What it would do | Verdict |
|---|---|---|---|
| NWS flash flood warning polygons (IEM, storm-based, already used for CO/NM/AZ in stage 5) | polygon, minutes | forecaster's judgement of local hazard; a second label or a prior; ex ante in real time | cheap, TX download of the same product; subjective, issue habits drift (5x growth 2002-2022) |
| Storm Events flood / flash flood (have 2000-2023) | county mostly, some points | episode-level second detector | agreement 0.4 above; usable only to flag episodes, not to rank block groups |
| FEMA Individual Assistance housing registrations (OpenFEMA, ZIP) | ZIP, declared disasters only | take-up independent flood-loss indicator for the big events | covers the events that carry 74% of positives; does not need flood insurance; ZIP not block group |
| FEMA Public / Hazard Mitigation (PA, HMGP) | project level | none for labels | skip |
| USGS high-water marks and peak stages | points | validates large events | few points; gauge history was already null (stage 18) |
| MRMS / Stage IV hourly QPE | 1-4 km, hourly | better hazard feature, not a label; stage 15 found no gain from sparse reconstruction | only for forecast mode |
| Sentinel-1 GFM / Landsat flood extent | 20-30 m, 2021+ | physical inundation label independent of insurance | clouds and revisit gaps miss flash floods; only the last 3 years, too few events to train |
| City 311 flood calls (Houston, Austin, San Antonio, Dallas) and Harris County FWS gauges | address / gauge | dense urban local ground truth, no insurance dependence | strongest sub-county candidate, but 4 cities, API work, and the reporting depends on population engagement |
| Texas DOT road closures (DriveTexas) | road segment | flash flooding on roads | ephemeral, no archive found by me |
| NWM reanalysis (v2.1, 1979-2020) | reach | modelled streamflow percentile as hazard feature | hazard side; a candidate for the susceptibility representation stage 22 asked for |

Recommended order if the label is to be improved: (1) run script 61's join at block-group level using Storm Events point records and TX FF warning polygons to see whether positives located by an independent sub-county source are ranked better or worse than NFIP positives (test only, as extra evaluation labels, not training); (2) FEMA IA registrations for the largest events as a check that the NFIP positives of those events are representative; (3) 311 for Houston only as a pilot of a non-insurance label. Do not spend more effort on rule-based cleaning of NFIP.

## Caveats

- One split and one learner (boosting depth 5, L2 features), differences between variants are mostly inside the intervals; the only significant ones are negative (>= 2 claims, Poisson). No multiplicity correction over 8 variants.
- Variants were scored on the original label, which is the noisy one; the strict label shares most positives with it, so the second column is not independent evidence.
- "Random under-detection" is inferred from small effects of cleaning and from Storm Events agreement; take-up and filing were not modelled separately (see scripts 34-35 for take-up).
- Episode-level agreement with Storm Events uses county records and a +-days window chosen by me; Storm Events itself under-reports and drifts (stage 6).
- The table in "Other sources" is an assessment from documentation and earlier stages; nothing in it was downloaded or tested in this stage.


# Stage 24: NFIP coverage by state (script 62)

Question: is NFIP coverage (policies against insurable structures) uneven between states, so that the claim label means something different in different places? Penetration is FEMA's county table (contracts in force / residential structures, as of 2026-08-03, 3,159 counties); policy flow is the bulk policy file by effective year; claims are the national claim files. Output `audit/nfip_state_coverage.txt`.

| State | penetration | in SFHA | share of counties < 1% | policies written 2023 (k) |
|---|---|---|---|---|
| Louisiana | 22.0% | 73% | 13% | 461 |
| Florida | 13.7% | 67% | 7% | 1,751 |
| South Carolina | 6.1% | 85% | 68% | 207 |
| Texas | 5.8% | 48% | 55% | 677 |
| New Jersey | 5.0% | 63% | 19% | 208 |
| New York / North Carolina | 2.5% / 2.6% | 76% / 53% | 57% / 57% | 172 / 135 |
| California | 1.3% | 31% | 45% | 197 |
| Colorado / Arizona / New Mexico | 0.6% / 0.7% / 1.4% | 24% / 21% / 16% | 71% / 60% / 55% | |
| Midwest (OH, IN, IL, MI, WI, MN, MO, IA, KS, OK) | 0.2-0.5% | 12-26% | 78-95% | |
| national | 2.9% | | | |

## Conclusions

- **The between-state difference is very large.** Among the 51 states with at least 100k structures, 24 have penetration under 1% and 5 above 5%; the 90th to 10th percentile ratio is 10.5. Five states (FL, TX, LA, NJ, SC) hold 68% of policies and 21% of structures.
- **It is the state, not only the county.** 30% of the variance of county log-penetration is between states; inside a state the spread is still large (Texas: 55% of counties under 1%, Louisiana: 13%).
- **Even inside the flood zone coverage is incomplete outside the Gulf and Atlantic coast**: SFHA penetration is 73% (LA), 67% (FL), 48% (TX), 16% (NM), 12-19% (Midwest). Where a house is in the mapped floodplain, a claim can exist only if it was insured, so a zero label there is mostly "uninsured".
- **The state ranking is stable in time** (Spearman of state policy counts 2010 vs 2023: 0.96; Florida 37% and Texas 12-14% of national policies throughout), and the policy counts have been falling in most states (CA 277k to 197k, MS 78k to 56k) while Texas rose to 810k in 2020.
- **Inland claims per 1,000 policy-years differ by about 20x** (HI 1.2, CA 1.7, FL 2.5 ... TX 10.8, LA 9.6, NY 14.2, NJ 21.5, PA 22.6): they mix hazard and policy mix (Florida's policies are mostly coastal), so this is not a hazard map.
- For the project: Texas is one of the few states where the label has enough coverage (stage 6 reached the same conclusion for CO/NM/AZ), but it is also a state with 55% of counties under 1% penetration, so the label's noise differs between counties of the same state. Models and metrics from Texas should not be carried to another state without its own coverage check; and a regional pooled model would learn penetration as much as hazard.

## Caveats

- The penetration table counts residential contracts and Census structures (as of 2026-08); the policy file has all occupancies and all effective years. Absolute levels do not agree (Florida 1.04 M in force against 1.7-1.8 M written per year), so use each table for rankings, not for combining.
- Claims in part 3 are all inland-cause claims including named tropical events (Texas 105 k, against 64 k in the pilot label, which drops Harvey and similar), and the per-policy rate is claims per policy written, not per exposure-day.
- Penetration is a ratio to all residential structures, not to structures in a flood hazard area outside the SFHA; low rates in dry states are partly low risk, not under-insurance.
- No test of how coverage relates to the label noise inside a state (stage 34/35 have county take-up only for Texas block groups).


# Stage 25: why some states have high NFIP coverage, and which states to study as a group (script 63)

The project will not run on all of the US. This stage asks what explains high coverage (so that the choice is not arbitrary) and screens states on the properties the flood pilot needs: label coverage, inland non-tropical claim volume, and enough usable events in both the training and test periods. Output `audit/state_selection.txt`, `audit/state_screen.csv`.

## 1. What makes penetration high (counterfactual decomposition: penetration = SFHA share x SFHA take-up + rest x non-SFHA take-up)

| State | penetration | SFHA share of structures | SFHA take-up | non-SFHA take-up | gap to national: exposure share / SFHA take-up / non-SFHA take-up (points) |
|---|---|---|---|---|---|
| LA | 22.0% | 15.3% | 73% | 12.8% | +5.5 / +0.9 / **+11.1** |
| FL | 13.7% | 13.3% | 67% | 5.6% | +4.6 / +0.7 / +4.2 |
| SC | 6.1% | 3.9% | 85% | 2.9% | +0.2 / +1.3 / +1.6 |
| TX | 5.8% | 4.1% | 48% | 4.0% | +0.3 / 0.0 / **+2.6** |
| NJ | 5.0% | 6.7% | 63% | 0.8% | +1.5 / +0.5 / -0.4 |
| NY | 2.5% | 2.0% | 76% | 1.0% | -0.7 / +1.0 / -0.3 |
| WV | 1.0% | 8.1% | 10% | 0.2% | +2.1 / -1.3 / -1.0 |

- Averaged over the six states above 4% (LA, FL, SC, TX, NJ, DE), the gap is **+3.2 points from voluntary take-up outside the mapped floodplain, +2.2 from a larger mapped floodplain, only +0.6 from take-up inside it**. Take-up inside the SFHA is near mandatory-purchase levels almost everywhere with a mapped floodplain (WV and the Midwest are the exceptions); what separates states is whether people outside it buy.
- Large exposure alone is not enough: West Virginia has 8% of structures in the SFHA but 1.0% penetration; Mississippi 7.6% and 3.5%.

## 2. Candidate drivers (county log-penetration, 2,900 counties with >= 1,000 structures)

| Model | R2 |
|---|---|
| state effects only | 0.30 |
| SFHA share of structures | 0.24 |
| + earlier claims per structure (2000-2009) | 0.49 |
| + share of policies in a CRS community | 0.55 |
| state effects + SFHA share + earlier claims | 0.64 |
| + CRS | 0.67 |

Standardised coefficients (no state effects): earlier claims +0.67, CRS share +0.35, SFHA share +0.33.

- **Earlier flood experience is the strongest single driver**: Louisiana 144 claims per 1,000 structures in 2000-2009 (Katrina, Rita), Mississippi 23, Florida and Texas 13, New Jersey 7, most others under 5.
- **Community floodplain management matters** (CRS community share of policies: FL 97%, SC 94%, LA 81%, TX 63%, NJ 69%, against NY 10%, PA 20%, OH 8%), but it is not necessary for coverage (NY has high SFHA take-up with low CRS) and CRS discounts and take-up reinforce each other.
- **Mandatory purchase is not what separates states**: the flagged share is 8% (TX) to 24% (KY) of 2022-23 policies, lowest in the high-coverage states, which fits "voluntary buying" as the difference.
- A third of the county variance remains unexplained even with state effects, and state effects remain after the covariates (R2 rises from 0.30 to 0.67 only by adding both), so state-level factors outside this data (income, home value, mortgage share, state law and agencies, insurer and agent practice) are likely to matter.

## 3. State screen (inland non-tropical claims 2010-2023; events = claim clusters with gaps <= 3 days in a county, >= 10 claims in >= 5 block groups)

| State | penetration | SFHA take-up | inland non-tropical claims | share of all claims | usable events (train <= 2016 / test >= 2017) | claims in those events | counties >= 1% |
|---|---|---|---|---|---|---|---|
| LA | 22.0% | 73% | 42,617 | 48% | 225 (138 / 87) | 89% | 88% |
| TX | 5.8% | 48% | 35,092 | 23% | 302 (194 / 108) | 83% | 34% |
| FL | 13.7% | 67% | 17,563 | 13% | 143 (87 / 56) | 73% | 91% |
| NJ | 5.0% | 63% | 13,073 | 11% | 117 (79 / 38) | 81% | 81% |
| NY | 2.5% | 76% | 11,643 | 12% | 125 (66 / 59) | 71% | 41% |
| SC | 6.1% | 85% | 6,465 | 31% | 50 (32 / 18) | 79% | 30% |
| MS | 3.5% | 33% | 5,214 | 52% | 74 (36 / 38) | 51% | 37% |
| PA | 0.7% | 23% | 8,013 | 37% | 153 (79 / 74) | 63% | 28% |
| IL / MO / TN / KY | 0.4-0.8% | 17-25% | 6.7-11.8 k | 86-92% | 79-117 | 59-79% | 9-28% |

## Recommendation

- **Core group: Texas + Louisiana.** Highest combined label coverage with the most inland non-tropical events (527 usable events, 195 in the test period), the same Gulf climate and flood culture, adjacent, and Texas data and rain pipelines (gridMET, GEFS) already built. Louisiana's inland share is 48% of claims; its penetration is four times Texas', which makes it the natural test of whether a model transfers between a mid-coverage and a high-coverage state, and of what happens to the label noise when coverage changes.
- **Optional third: Mississippi** (adjacent, 74 events, 38 in test) with lower coverage (3.5%) and a half-tropical claim mix.
- **Not as training states for the inland label**: Florida (inland claims are 13% of the total; coastal and tropical regime, so a rain-episode label does not describe it), South Carolina and Delaware (too few events), New Jersey / New York (usable volume, but a nor'easter / snowmelt regime and no existing pipeline; consider a later northeast block).
- **Low-coverage states with many claims (IL, MO, TN, KY, OH, PA)** have enough events but penetration under 1% and SFHA take-up under 25%: their zeros are mostly "uninsured". Use them as an out-of-region stress test of how the model degrades with coverage, not as a place to train.

## Caveats

- The drivers are county-level correlations from NFIP and FEMA fields only; no income, home value, mortgage share, state policy or insurer data was added, and FEMA disaster declarations are available locally only for CO/NM/AZ. Causal reading ("flood experience causes take-up") is not supported.
- `exp0009` is claims in 2000-2009 per structure, including named tropical events, and policies and claims are not independent (claims need a policy); the CRS share is policy-weighted and CRS discounts feed back on take-up.
- Penetration is a 2026 snapshot; policy flow is stable in time (stage 24), but a state that changed regime after the Risk Rating 2.0 transition (2021-22) would show it only in later years.
- Event counts use my claim clustering (3-day gaps), not rain episodes; the usable events in a state also depend on the rain trigger (>= 30 mm) and need to be checked with each state's gridMET series.
- Louisiana's block-group boundaries and 2010/2020 crosswalk, and its tropical-window exclusion, were not checked.


# Stage 26: relaxing the filters that isolate storm-triggered floods (script 64)

Question: nothing proves that the stage 8 filters (cause 2/3/4, no tropical event name, no episode inside a tropical-cyclone window) separate storm-triggered floods from the rest. How much does the label and the model move when they are loosened one at a time? Texas, 2010-2023; table rebuilt without the tropical exclusion (`table_notc.parquet`, 3.17 M rows, 46,659 episodes of which 3,270 in a TC window; built with `scripts/25 --tc-file tc_flag_none.npy`). Output `audit/label_scope_relaxation.txt`.

## A. Do the filters separate rain-triggered claims? (share of claims with county 1-day maximum >= 30 mm in the 4 days up to the loss date)

| Class | claims | rain >= 30 mm | rain >= 50 mm |
|---|---|---|---|
| inland, named tropical | 70,300 | 0.987 | 0.984 |
| **inland, not tropical (stage 8 label)** | 33,364 | **0.944** | **0.899** |
| inland, unnamed, in TC window | 1,625 | 0.890 | 0.859 |
| other / missing cause, named tropical | 30,929 | 0.985 | 0.981 |
| other / missing cause, not tropical | 10,500 | 0.823 | 0.754 |
| tidal, not tropical | 475 | 0.686 | 0.600 |

- The cause filter does what it should: claims dropped as tidal or "other" have clearly less rain support (0.69-0.82) than the kept class (0.94). But the separation is a gradient, not a switch: 5.6% of the kept claims have no 30 mm day nearby, and "other" non-tropical claims are still 82% rain-supported.
- **The tropical filter is not removing non-storm floods; it removes the biggest storm floods.** Named tropical claims are as rain-supported as any (0.99). The stage 8 class holds only 22% of Texas claims 2010-2023 (Harvey alone is most of the rest).

## B. Scope ladder (trained 2010-2016 on its own scope, tested 2017-2023)

| Scope | episodes | label | positives train / test | share of positives in TC episodes | top-10 episodes' share |
|---|---|---|---|---|---|
| S0 (stage 8) | non-TC | inland, no tropical name | 7,519 / 3,673 | 0 | 0.22 |
| S1 | all | inland, no tropical name | 8,223 / 3,837 | 0.07 | 0.21 |
| S2 | all | inland, incl. named tropical | 8,361 / 8,237 | 0.33 | 0.28 |
| S3 | all | every cause | 10,312 / 10,348 | 0.31 | 0.26 |

Within-event AUC of each model on fixed evaluation sets (difference to the S0 model, event-clustered 95% interval):

| Evaluation set | S0 model | S1 | S2 | S3 |
|---|---|---|---|---|
| E0: non-TC test episodes, stage 8 label (995 episodes) | 0.7858 | +0.0016 [-0.0025, +0.0054] | +0.0010 [-0.0033, +0.0045] | +0.0036 [-0.0007, +0.0077] |
| E1: TC test episodes, label incl. named tropical (152 episodes, 4,561 positives) | 0.7652 | -0.0019 [-0.0060, +0.0063] | -0.0031 [-0.0080, +0.0055] | +0.0010 [-0.0025, +0.0093] |
| E2: all test episodes, same label (1,149 episodes) | 0.7768 | +0.0001 [-0.0029, +0.0036] | -0.0008 [-0.0040, +0.0025] | +0.0025 [-0.0003, +0.0061] |

Own scope (train and test on the same scope): S0 0.7858, S1 0.7865, S2 0.7760, S3 0.7666.

## Conclusions

- **The model's within-event ranking does not depend on these filters.** A model trained with the tropical episodes and the named tropical and tidal/other claims added ranks the stage 8 test set the same (+0.001 to +0.004, all intervals contain 0), and the stage 8 model ranks tropical-window episodes at 0.765, only 0.02 below its score outside them. Rain at the cell plus exposure ranks block groups within a tropical storm almost as well as within an ordinary storm.
- **The label composition does depend on them.** Relaxing the tropical filter doubles the test positives (3,673 to 8,237), and 55% of the test positives (4,561 of 8,237) come from TC episodes, mostly one system (Harvey, 2017). The own-scope scores fall 0.01 (S2) and 0.02 (S3) because the test set then consists of those large systems, not because training was hurt.
- So the unproven separation matters little for **ranking inside an event**, which is what the pilot measures, and a lot for **how many positives and which events dominate**, hence for absolute probabilities, event-level skill and any pooled score. The filter choice should be fixed by what the downstream use is (rain-only forecast vs a model with wind / surge), not by a test of ranking skill.
- The cause code is the only filter with observed separating power (rain support 0.94 against 0.69-0.82); a hard test of "storm-triggered" would need an independent label (stage 23: warning polygons, Storm Events, FEMA IA).

## Caveats

- The tropical distinction is by the claim's event name and by an 800 km, +-2 day best-track window; 25% of claims have no event name, so "not tropical" is still not certain. Rain support uses the county daily maximum, which is high for big counties, and a rain-supported claim is not necessarily rain-caused.
- E1 is 152 episodes dominated by few systems (Harvey 2017, Imelda 2019); the event-clustered bootstrap ignores correlation across counties of the same storm, so its intervals are too narrow.
- Features contain only gridMET daily rain, antecedent rain and exposure; nothing about wind, surge, river routing or tides, so the model ranks tropical floods well only to the extent they are rain-driven. Tidal cause (475 non-tropical claims) is too small to test.
- One split, one learner, no multiplicity correction over the 3 x 3 differences; Texas only (Louisiana, with more tropical flooding and a higher tidal share, may behave differently).


# Stage 27: state group when tropical / extreme storm floods stay in the label (scripts 65, 66)

Decision (user): model storm and tropical floods together as "extreme weather" floods, and choose the data group on that basis; the feature that separates ordinary from extreme weather may also be useful for extreme weather in general. Stage 26 showed that keeping them changes little in within-event ranking, so the screen of stage 25 is redone for the wider label. Outputs `audit/state_group_extreme.txt`, `audit/state_screen_extreme.csv`.

## A. Screen (all inland-cause claims 2010-2023 incl. named tropical; events = county claim clusters with gaps <= 3 days, >= 10 claims in >= 5 block groups; regime `extreme` = at least half of the claims named tropical or inside a best-track window of 800 km, +-2 days, all 3,108 counties; train <= 2016 / test >= 2017)

| State | penetration | claims | extreme share of claims | ordinary events train / test | extreme events train / test |
|---|---|---|---|---|---|
| TX | 5.8% | 105,410 | 0.68 | 177 / 96 | 24 / 58 |
| LA | 22.0% | 67,028 | 0.39 | 131 / 68 | 24 / 56 |
| FL | 13.7% | 66,659 | 0.79 | 66 / 32 | 63 / 117 |
| PA | 0.7% | 19,566 | 0.61 | 80 / 69 | 39 / 36 |
| NY | 2.5% | 36,010 | 0.69 | 62 / 51 | 54 / 19 |
| NC | 2.6% | 24,509 | 0.85 | 39 / 12 | 64 / 53 |
| NJ | 5.0% | 66,681 | 0.82 | 64 / 30 | 42 / 26 |
| VA | 2.7% | 8,108 | 0.60 | 33 / 19 | 41 / 12 |
| MS | 3.5% | 7,653 | 0.38 | 36 / 31 | 11 / 16 |
| SC | 6.1% | 11,926 | 0.50 | 30 / 11 | 15 / 25 |
| IL / MO / KY / OH / TN / IN / IA | 0.4-0.8% | 3-12 k | 0.01-0.05 | 42-69 / 15-48 | 0-5 / 0-6 |

Pooled candidate groups:

| Group | states | events | ordinary train / test | extreme train / test | extreme share of claims | penetration (weighted; min-max) |
|---|---|---|---|---|---|---|
| TX alone | 1 | 355 | 177 / 96 | 24 / 58 | 0.68 | 5.8% |
| TX + LA | 2 | 634 | 308 / 164 | 48 / 114 | 0.57 | 8.3% (5.8-22) |
| **TX + LA + FL** | 3 | **912** | 374 / 196 | 111 / 231 | 0.63 | 10.6% (5.8-22) |
| TX + LA + MS + FL | 4 | 1,006 | 410 / 227 | 122 / 247 | 0.62 | 10.2% (3.5-22) |
| TX LA MS AL FL GA SC NC | 8 | 1,359 | 517 / 278 | 210 / 354 | 0.63 | 7.3% (1.3-22) |
| above + VA NJ NY | 11 | 1,812 | 676 / 378 | 347 / 411 | 0.67 | 6.3% (1.3-22) |

## B. Conclusions on the group

- With tropical floods in scope **Florida becomes a core state** (stage 25 rejected it for an inland-only label): 13.7% penetration, the most extreme events of any state (63 / 117) and enough ordinary ones (66 / 32).
- **Recommended core: Texas + Louisiana + Florida.** 912 events, both regimes present in both periods (>= 24 events in each cell, >= 32 for Florida), the strongest and most varied coverage (5.8%, 13.7%, 22%), one broad Gulf climate, and the Gulf coast is where extreme events recur. TX + LA alone has only 48 extreme events in training.
- **Second ring, as transfer tests**: New Jersey (5.0%, 64 / 30 ordinary, 42 / 26 extreme; Irene, Sandy, Ida, nor'easters) and New York as a different climate with both regimes; Mississippi as a cheap adjacent addition (11-16 extreme events); North Carolina (extreme-rich, only 12 ordinary events in test) and South Carolina (small).
- **Low-coverage states with many events (PA, IL, MO, KY, OH)**: events are plentiful and Pennsylvania has both regimes, but penetration is under 1%; use as a coverage stress test, not for training. The Midwest has almost no extreme events in this definition (tropical remnants only), so it cannot teach the regime contrast.
- The group choice reduces to **two axes**: coverage (label reliability) and the presence of both regimes. TX + LA + FL is the smallest set that has both on all four cells (train / test x ordinary / extreme) with meaningful counts.

## C. Does a regime feature help? (script 66, Texas, episode level)

Within-event AUC cannot show it, because a regime feature is constant inside an event; the test is between events: does a county rain episode have any claim (all inland claims), and how many (train <= 2016, test >= 2017, 20,248 / 26,411 episodes; 7.0% of episodes are in a tropical window and they hold 33% of positives; P(any positive) is 9.5% in TC episodes and 5.5% otherwise).

| Features | AUC, any positive | Spearman, log positives | AUC in TC test episodes |
|---|---|---|---|
| county rain, duration, month, block groups | 0.9176 | 0.303 | 0.919 |
| + TC-window flag | 0.9175 | 0.303 | 0.918 |
| + extremeness percentile (county peak and 3-day sum against the county's own training distribution) | 0.9178 | 0.303 | 0.922 |
| + both | 0.9175 | 0.303 | 0.920 |

- **No gain from the TC flag or from the percentile feature once raw county rain is there** (differences of 0.0003 against a year-to-year spread of 0.906-0.928). Rain already carries the extremeness: the share of episodes with a claim rises smoothly with the county-relative percentile of the episode peak (1.9% below the median, 4.9%, 12%, 27.5%, 47% in the top 1%), and the top 1% of episodes (395) hold roughly three quarters of claims. Ordinary versus extreme is **a gradient, not two regimes** (the TC flag is only a rough marker of its upper end: tropical share 6% at the bottom, 20% at the top).
- The size of an event (number of claimed block groups) is poorly predicted at the event level (Spearman 0.30) with this information; the within-event ranking (0.79) and the event-level occurrence (0.92) are the strong parts.

## Caveats

- "Extreme" here is tropical (name or best-track window). Non-tropical extremes (the 2015 Texas floods, Louisiana in August 2016) are counted as ordinary in part A; a rain-based extremeness (percentile of the county's own history) would cover both, but needs the rain series of every candidate state, which only Texas has.
- The percentile feature is built from the county's own 2010-2016 episodes, so it is a county-normalised rain. Its value should appear when states are pooled (a millimetre of rain does not mean the same in Texas and Louisiana) and in forecast mode (where the rain is uncertain); neither is tested here, so the null above is Texas, observed rain, only.
- Event counts use my 3-day clusters, penetration is a 2026 snapshot, and claims in overlapping episode windows can be counted twice in the bin table; 25% of claims have no event name, so the `extreme` regime relies on the best-track window for those.
- Florida, with its coastal and tropical claims, needs its own check of the tidal cause (cause 1 is excluded here) and of block-group boundaries before it is added to a pipeline built for Texas.


# Stage 28: NOAA regional definitions of the candidate states (literature lookup, nothing computed)

Question: which climate / weather region does NOAA assign the candidate states to? Looked up on 2026-10-28 in NOAA/NCEI, NWS and the Fifth National Climate Assessment (NCA5, NOAA-led). NOAA has no scheme of "weather-phenomenon zones" for flooding; the available definitions are climate regions (NCEI), service regions (NWS, Regional Climate Centers) and assessment regions (NCA5).

| State | NCEI climate region (9, Karl & Koss) | NCA5 region | NWS region | Regional Climate Center |
|---|---|---|---|---|
| TX | South | Southern Great Plains (KS, OK, TX) | Southern | Southern (TX A&M) |
| LA | South | Southeast | Southern | Southern |
| MS | South | Southeast | Southern | Southern |
| FL | Southeast | Southeast | Southern | Southeast |
| NC, SC | Southeast | Southeast | Eastern | Southeast |
| NJ, NY, PA | Northeast | Northeast | Eastern | Northeast |

- NCEI South = AR, KS, LA, MS, OK, TX; Southeast = AL, FL, GA, NC, SC, VA; Northeast = CT, DE, ME, MD, MA, NH, NJ, NY, PA, RI, VT (the dynamic NCEI page did not load; these lists come from search snippets of the archived NCEI page and a NOAA figure, not from a page read in full).
- NCA5 Southeast (11 states): AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA. NCA5 Northeast (13 incl. DC and WV). Texas is in the Southern Great Plains chapter, not with Louisiana and Florida.
- NWS Southern Region (Fort Worth) holds NM, TX, OK, AR, LA, MS, TN, AL, GA, FL; Eastern Region holds NJ, NY, PA and the Carolinas.

## What the sources say about the flood hazard

- **Southern Great Plains (TX)**: rainfall falls from about 10-15 in/yr in the west to over 50 in at the eastern edge; days with >= 2 in have become more frequent, especially in the east; Houston flooded in 2016, 2017, 2019 and 2020; Texas had 19 named tropical storms 2000-2021, 8 hurricanes (Harvey, Ike, Rita). The chapter text read here did not address severe convective storms.
- **Southeast (LA, MS, FL, ...)**: extreme rainfall is increasing; hurricanes are more likely to slow or stall near the coast, which increases rainfall flooding (link to climate change stated as uncertain); compound flooding (rain with tide or surge) already affects low-lying coastal plain; sea level rose about 6 in 1970-2020.
- **Northeast (NJ, NY, ...)**: the wettest 1% of days increased by about 60%, the largest of any region; short intense urban downpours (Henri and Ida in 2021 set one-hour records in New York City); tropical systems and nor'easters raise surge; a tropical contribution reaches inland to West Virginia.

## Implications for the group

- TX + LA + FL do not share one NOAA region under any single scheme, but all three lie in the NWS Southern Region, and the NCEI South / Southeast regions are adjacent on the same humid-subtropical Gulf coast. The common flood mechanism in the NCA5 text is heavy rain from convective systems and tropical cyclones (including stalled ones) on a low coastal plain, with compound rain-surge flooding. That supports treating them as one "extreme weather" group (stage 27).
- **Texas is internally heterogeneous** (an arid west, a humid east): its flood episodes come from the eastern half, which is the part that resembles Louisiana. A Texas-wide pilot mixes an arid and a humid climate; a precipitation-gradient stratum (or NCEI climate division as a grouping factor) is the natural way to control for it.
- NJ and NY are a different regime (NCA5 Northeast): the second-ring transfer test of stage 27 is also a test across NOAA regions.
- Louisiana and Mississippi are in the NCEI South with Texas but in the NCA5 Southeast with Florida, so the pairing TX + LA is not a NOAA unit by the NCA5 definition.

## Caveats

- NCEI's nine-region state lists were confirmed from search results, not from a full page read; the NCA5 chapters were read only in their first 100,000 characters (of about 230-290 k), so hazard statements are from the opening sections.
- Regional schemes differ by purpose (climate monitoring, forecasting services, assessment), so no single "region of the candidate group" exists; the table shows all of them.
- Nothing here was tested against the data; whether a region or climate-division factor explains label differences between states is open.


# Stage 29: USGS and university sources for prior features of the candidate states (literature lookup + script 67)

A first sweep of USGS reports and state-university studies (TX, LA, FL) for priors that could enter the model, plus one check against our own claims. Nothing was downloaded. Searches covered USGS regression-equation reports, USGS post-event data, NOAA Atlas 14 and university flood studies; the Florida university search found little (see caveats).

## What was found

| Source | Content | Prior feature it suggests |
|---|---|---|
| NOAA Atlas 14, Vol. 11 (Texas, 2018) and Vol. 9 (southeastern states, 2013; the state list of Vol. 9 was not confirmed in the search) | gridded rainfall depth by duration and return period; Texas 100-year 24-hour values changed from the older TP-40 by -40% to +152% (up to 13 in near Austin); Atlas 15 (trend-aware, national) is still in development | **rainfall as a return period at the block-group cell**: the same extremeness scale in every state, which the county-percentile feature of stage 27 cannot give when pooling states |
| USGS SIR 2024-5031 (Louisiana Coastal Plain peak-flow equations, 211 gauges to 2016) | the retained predictors are only drainage area and channel slope (10-85% points of the longest flow path); precipitation, soil, land use were tested and dropped; basins above 10% impervious, and sites affected by backwater, surge or tide, are excluded; regional skew -0.17 | stream-network descriptors (drainage area, channel slope) rather than centroid terrain, which stage 22 found null; and an explicit statement that urban and tidal flooding is outside what peak-flow regressions describe |
| USGS Texas equations (SIR 2009-5087, 559 stations in an older report) | undeveloped watersheds only | same: nothing for urban flooding |
| USGS Harvey products (SIR 2018-5070; data release DOI 10.5066/F7VH5N3N) | 2,123 high-water marks in 22 Texas counties and 3 Louisiana parishes, 19 inundation maps | **independent sub-county flood label for the largest event**, not tied to insurance (the "second detector" asked for in stage 23). Louisiana August 2016 and Florida Ian marks were not found by this search |
| Rice University / Texas A&M-Galveston (Armand Bayou, 1999-2009, 5 events, none reaching a 100-year rainfall) | about 75% of damage outside the mapped 100-year floodplain; reasons given: one-dimensional flow assumptions, neighbourhood-wide soil and surface classes, short intense storms below the 100-year threshold | prior on the **share of loss outside the mapped floodplain** and on imperviousness / flat terrain as reasons |
| Rice Kinder Institute (modelled Harvey damage, Harris County); Nature 2022 (parcel-level Harvey) | about 73% of expected damaged homes outside the 100-year zone; 76.1% of flooded parcels outside the SFHA; 47% of Greater Houston claims outside it over three decades | same prior, three methods agreeing at 47-76% |
| Louisiana recovery reports | August 2016: NWS called the rainfall a 1,000-year event; 91,628 households flooded; 45% of affected renter households outside the SFHA | LA 2016 as a non-tropical extreme case; state recovery data (FEMA IA, housing) as an independent label |
| SWFWMD (Florida) | preliminary flood-prone-area maps not in the FIRMs | local layers, case by case |

## Our own claims against that prior (script 67; ratedFloodZone of inland claims 2010-2023, share outside the SFHA)

| State | named tropical | not tropical |
|---|---|---|
| TX | 54% | 52% |
| NY | 33% | 46% |
| NC | 21% | 38% |
| MS | 36% | 35% |
| SC | 24% | 33% |
| FL | 16% | 32% |
| LA | 26% | 28% |
| NJ | 11% | 16% |
| all states | 29% | 35% |

- **The literature prior is reproduced by the claims**: about half of Texas claims (52-54%, 42-70% by year) sit in rated zones outside the SFHA (Kinder: 47% of Greater Houston claims), against 28% in Louisiana and 11-16% in New Jersey. National share outside is 32%.
- **This ties to stage 25**: the states with high take-up outside the floodplain (Louisiana 12.8%) show fewer claims outside it than Texas (4.0%) because Louisiana's insured base is mostly inside it; a state's label reflects where insurance is, not only where water goes.
- **Tropical vs not does not move Texas** (54% vs 52%) but does move Florida (16% vs 32%) and North Carolina (21% vs 38%), where tropical claims are mostly coastal zones. The "extreme" regime therefore has a different spatial footprint in different states, which a pooled model should not assume to be constant.

## Candidate priors, in order of expected value for the pilot

1. **Rainfall return period from Atlas 14** at the cell (24 h, 3 day), replacing or complementing raw mm and the county percentile; the cross-state scale for pooling and the extremeness gradient of stage 27.
2. **Share of the block group inside the SFHA / outside it** (already in the table as `sfha_share`, from policies) with a state-level prior on the expected outside share (TX about half, LA about a quarter, FL and NJ lower).
3. **USGS high-water marks (Harvey, then others)** as an independent test label for sub-county ranking.
4. **Drainage area and channel slope of the nearest stream reach** (USGS StreamStats / NHDPlus), against the failed centroid terrain.
5. Imperviousness / flat low-relief indicator (named in the Rice study; NLCD, not yet in the repository).

## Caveats

- Search-level reading: the Atlas 14 volume coverage for Louisiana, Mississippi and Florida was not confirmed; the Rice and Kinder figures are secondary descriptions; no university study from Louisiana or Florida on this topic was found by two searches (a narrower search of LSU, UL Lafayette, Tulane, UF, FIU is open). Not a systematic review.
- `ratedFloodZone` is the zone used to rate the policy, not necessarily today's mapped hazard at the building; shaded-X and unmapped X are both "outside", and the share of claims outside is conditional on being insured.
- None of the priors was tested against the pilot yet; the previous ablations (terrain, NFHL, HAND, gauges) were null, so the expectation for any new static layer should be low unless it carries information the claim history does not (stages 17-22).


# Stage 30: what meteorological event reports add (NWS / NHC / NOAA reports, script 68)

Question: can the meteorological analyses of the big events in the candidate states supply information the model can use? Sources: NHC tropical cyclone reports, NWS Service Assessments and office event summaries, NOAA-hosted papers (searches, partial reading of snippets; the full NOAA paper on the 2016 Louisiana rarity was not readable as a PDF here).

## What the reports say (facts to use as event descriptors)

| Event | Mechanism (from the reports) | Rain | Remarks |
|---|---|---|---|
| Harvey 2017 (TX, LA) | stalled and reversing hurricane, centre near the coast for four days | storm total 60.58 in (Nederland TX), a US record; NWS Service Assessment: about 57% of Harris County gauges at record flooding | WPC and models underpredicted the upper Texas coast and SW Louisiana; the heavy-rain centroid of the QPF was displaced about 110 miles |
| Imelda 2019 (TX) | slow, looping tropical storm; a feeder band kept storms training over the same area | peak 44.29 in near Fannett, 31 in in 12 h | short, localized, training |
| Louisiana August 2016 | slow upper-level low over deep tropical moisture (precipitable water roughly 2.0-2.9 in), continuous Gulf inflow, a non-tropical system | 20-30 in in 2-3 days; 48-hour total 31.39 in, a Louisiana record; Lafayette 22.89 in in two days | called a 1,000-year rainfall; 91,628 households flooded |
| Fort Lauderdale April 2023 (FL) | stalled front plus an upper-level disturbance, non-tropical | over 25 in in under 24 h, about 20 in in 6 h | NWS: 1,000-year rainfall; the previous one-day record was 14.59 in |
| Ian 2022 (FL) | landfall hurricane; worst flooding on rivers north of the track | peak 26.95 in (Grove City); second maximum 21.49 in at Daytona Beach; a dry slot kept totals lower south of the track | record river floods (Peace, Myakka, Alafia) |
| Memorial Day 2015, Tax Day 2016 (TX) | mesoscale convective systems; Tax Day with extreme-rain cells and tornadoes embedded | Blanco headwaters 12-13 in in 4-6 h; Houston Tax Day cells | peer-reviewed analyses exist (Nielsen and Schumacher for Tax Day; a NOAA paper on the Blanco); return periods are in their figures, not retrieved |

The reports agree on three ingredients: anomalous moisture (precipitable water), slow or stalled motion, and organised repeated rain over the same area (training). Four of the six events are not tropical or are only partly so; the difference between "tropical" and "other" is smaller than the difference in how organised and how long-lasting the rain was.

## Check in the pilot table (Texas; within-event AUC of two single scores, pooled over the event's county episodes)

| Event window | county episodes with an AUC | claimed share of block groups | county peak mm (p50 / p90 / max) | AUC cell rain | AUC policies |
|---|---|---|---|---|---|
| Memorial Day 2015 | 58 | 5.9% | 73 / 142 / 257 | 0.636 | 0.731 |
| October 2015 | 27 | 1.2% | 132 / 213 / 645 | 0.654 | 0.598 |
| Tax Day 2016 | 29 | 9.2% | 76 / 192 / 375 | 0.791 | 0.724 |
| Harvey 2017 | 28 | 29.5% | 63 / 341 / 672 | **0.571** | **0.784** |
| Imelda 2019 | 10 | 4.3% | 60 / 123 / 536 | **0.823** | 0.473 |

- **Where the rain falls inside a county decides which information ranks block groups.** In Harvey the whole county was wet, so cell rain ranks almost nothing (0.57) and exposure ranks well (0.78); in the localized, training events (Imelda, Tax Day) cell rain ranks well (0.82, 0.79) and exposure little or modestly.
- **Over all 2,570 episodes** the same holds as a gradient of the spread of cell rain inside the county (std / mean): cell-rain AUC rises from 0.559 in the quarter of episodes with the most uniform rain to 0.755 in the quarter with the most spread, while the exposure score stays at 0.67-0.70.
- **The stage 8 model does not gain from being given the spread** (std, CV and range of cell rain in the episode: within-event AUC 0.7849 against 0.7857, difference -0.0008 [-0.0045, +0.0029]): the boosting model already sees the county maximum, the county mean and the cell's deviation from it, which carry the same information. The mechanism shows up in the table above but needs no new feature.

## Conclusions

- The meteorological reports give a **mechanism-based description of extreme events** (moisture anomaly, motion, organisation / training) that explains why "ordinary vs extreme" is not what separates skill: the spatial structure of the rain is. Within-event ranking depends on it; the existing features already encode it.
- For the candidate group, the reports show the same three ingredients in Texas, Louisiana and Florida events, tropical or not, which supports one group model (stage 27, 28) and argues against a hard tropical / non-tropical switch (stage 26: no loss of skill when relaxed).
- What the reports add that the data cannot: event-level **forecast guidance on missed placements** (Harvey: QPF centroid 110 miles off), the moisture and motion ingredients available before the event (precipitable water, storm speed, stall time), and return-period statements (1,000-year) that can be checked against Atlas 14 once the grids are in the repository.
- Event-level ingredients (precipitable water, storm motion, duration of the stall) are constant inside an event, so they can improve between-event and forecast-mode discrimination, not within-event ranking; stage 27 found no gain from a TC flag or a county percentile at the event level, so the expectation for them should also be low unless they carry information beyond the rain amount.

## Caveats

- The reports were read through search results and a few pages; numbers are as reported (the NOAA site also quotes 51.88 in for Harvey in one place, against 60.58 in the NHC and NWS reports; I used the latter). The return-period figures of the Tax Day, Memorial Day and Louisiana papers were not retrieved.
- Five Texas events, two of them tropical windows with few county episodes (Imelda has 10 episodes with an AUC); the event table is descriptive, with no interval.
- Cell rain is a gridMET daily value at the block-group centroid; the stage 15 hourly MRMS test showed that sparse reconstruction does not improve on observations alone, so intra-day structure (training, short bursts) is not in the table.
- Louisiana and Florida events were not placed in a table because only Texas has the pilot pipeline.


# Stage 31: observation products that measure the three ingredients (lookup + script 69)

The three ingredients of the event reports (stage 30) are anomalous moisture, slow or stalled motion, and organised repeated rain. Which operational products describe each, from which sensor (radiosonde, satellite, radar), and what does the repository already hold? Product descriptions come from searches (NOAA / NSSL / NCEI / NWS training pages); items I could not confirm are marked.

## Product map

| Ingredient | Sensor | Product | Resolution / period | Use here |
|---|---|---|---|---|
| Moisture (precipitable water, PW) | radiosonde | IGRA v2 derived parameters: PW from the surface to 500 hPa, from over 2,700 stations | station, twice a day (00 and 12 UTC); long record | few sites per state; a check, not a grid |
| | GPS | NOAA GPS-Met integrated PW, hundreds of sites (mostly CONUS), 30 min, 2002-05 to 2016-11 at NCEI; a daily NOAA GPS-IPW series to 2025 at the NCAR RDA (d735000) | station | station anomaly series; coverage of TX / LA / FL not checked |
| | satellite | MIMIC-TPW2 (CIMSS): hourly global composite of microwave retrievals, moved to the target time with GFS 1000-600 hPa winds; NESDIS blended TPW with a percent-of-normal anomaly against the NVAP 1988-1999 climatology | hourly, global, coarse | anomaly field over the Gulf, but the archive access and the start date were not confirmed |
| | model / reanalysis | ERA5 total column water vapour; HRRR PWAT (archive on AWS `noaa-hrrr-bdp-pds` since 2014, Zarr copy at the University of Utah) | 0.25 deg hourly; 3 km hourly | the practical gridded choice; WPC / NAEFS standardised PW anomalies were not found in the searches |
| Motion / stall | model winds / soundings | slow Corfidi up / downshear vectors (< 15 kt) listed as favourable in NWS flash-flood training; slow storms flood alone, fast storms need training | derived from sounding or model winds | a derived index, needs the wind profile |
| | best track | HURDAT2 position and wind every 6 h (in the repository): dwell near a county, closest approach, translation speed | tropical cyclones only | script 69 below |
| | radar / satellite | storm-motion tracking from MRMS or GOES cloud motion: no product description found in the searches | | unconfirmed |
| Organisation, rain rate, training | radar | MRMS dual-pol QPE every 2 min; FLASH: 1 km, 5 min, QPE against flash-flood guidance and against average recurrence interval (ARI); CREST maximum unit streamflow (0.01 deg, 15 min in a 2015 draft description) | 1 km; MRMS on AWS since 2020-10 (stage 5) and at IEM; an archived MRMS-FLASH set was used by a lab for 2018-06 to 2019-05; no official archive portal found | the direct measure of rate x duration at a cell; short archive |
| | satellite | GOES-R ABI Rainfall Rate / QPE: per-pixel rate (0-100 mm/h) from bands 8, 10, 11, 14, 15, every 10-15 min; IMERG: 0.1 deg, 30 min, about two decades | 2 km / 10 km | IMERG for the earlier years (Final Run start not confirmed) |
| | framework | Doswell et al. (1996): heavy rain = sustained rate (ascent, moisture, precipitation efficiency) x duration (storm speed and size along the motion) | | how the three ingredients combine |

**In the repository**: ERA5 pressure-level fields (t, q, u, v at 850, 700, 500 hPa) and single-level fields, 2000-2022 May-September, but only for the box 29-43 N, 120-99 W (the CO / NM / AZ line), so not the Gulf; GEFS reforecast rain (APCP only); hourly MRMS crops of 8 Texas events (stage 15); gridMET daily rain; HURDAT2. No moisture or wind field exists for Texas, Louisiana or Florida.

## Test of the motion ingredient from the best track (script 69, Texas)

3,270 county episodes inside a tropical-cyclone window, 1,566 with a track point within 500 km; 9.5% have a claim. Spearman with the number of claimed block groups: county peak rain +0.35, mean rain +0.34, hours within 500 km +0.26, closest approach -0.27, translation speed +0.15, maximum wind +0.23. Leave-one-year-out logistic models, any claim:

| Features | AUC | Spearman with claimed block groups |
|---|---|---|
| county rain, duration, block groups | 0.9228 | 0.432 |
| + dwell, closest approach, speed, wind | 0.9274 | 0.437 |
| track features alone | 0.7226 | 0.230 |

- The track features carry information on their own (0.72) but add only +0.005 AUC and +0.005 Spearman over observed county rain: the rain already reflects the stall. They are available days ahead from the NHC forecast track, so their value should be larger in forecast mode, where the rain is uncertain; this is not tested.
- A speed of zero or a missing track within 500 km is coded as no dwell, which mixes "far away" and "fast"; the sign of speed is therefore not interpretable.

## Conclusions

- All three ingredients have operational products: PW (radiosonde, GPS, microwave satellite, reanalysis / HRRR), rain rate and organisation (MRMS radar QPE and FLASH, GOES RRQPE, IMERG), motion (derived from winds, and from the best track for cyclones). The gridded, long-record, multi-state choices are ERA5 (PW, winds) and IMERG / Stage IV (rain rate); radar products have short archives.
- The cheapest new information is not in the repository for the Gulf: ERA5 total column water vapour and low-level winds over TX / LA / FL (a CDS request, as for the existing box).
- Given stages 17, 22 and 27-30, expectations should be modest for within-event ranking (event-level ingredients are constant inside an event) and larger for between-event and forecast-mode use.

## Caveats

- Product descriptions are from search results; the archive start dates and access routes of MIMIC-TPW2, the NESDIS blended TPW, IMERG Final Run and GOES RRQPE were not confirmed; WPC / NAEFS standardised PW anomalies, SPC mesoanalysis, MRMS PrecipRate and satellite motion products were not found by the searches (not shown to be absent).
- Script 69 is one state, 3,270 correlated episodes (a few storms), no interval, a linear model; track features at the county scale, not the cell.


# Stage 32: ERA5 moisture and wind ingredients for the Gulf box (scripts 70, 71)

Data: ERA5 total column water vapour and specific humidity, u, v at 500 / 700 / 850 hPa, 24-37 N, 107-79 W, 0.25 deg, 00/06/12/18 UTC, 2009-2023 (`cds_gulf/`, 75 files, 2.8 GB; registered in `docs/DATA_SOURCES.md`, manifest and lock updated). A whole year of pressure levels exceeds the CDS cost limit (HTTP 403 "cost limits exceeded"), so the pressure-level jobs were queued by quarter. All 75 files were read through without errors; no missing values.

Question: do the moisture and wind ingredients of the event reports (stages 30, 31) explain differences between Texas rain episodes beyond the observed county rain? Per county episode (46,659; 5.8% with a claim; all inland claims including tropical): maximum tcwv and its z-score against the cell's own same-month climatology (2009-2016), mean 850 hPa humidity, speed and moisture flux q x |V|, mean 500 hPa speed (steering-flow proxy).

## Results

- **The ingredients describe the named events as the reports did.** Median tcwv and z-score of the county episodes: Harvey 58.4 kg/m2 (z +1.85, 90th percentile +3.10), Imelda 54.3 (+1.63), Memorial Day 2015 47.8 (+1.58), Tax Day 2016 42.3 (+1.56). The 500 hPa wind is weakest in the two stalled / slow tropical systems (Harvey 8.2 m/s, Imelda 7.6 m/s) and stronger in the convective events (Memorial Day 14.6, Tax Day 12.1 m/s), the stall signature the reports gave.
- **Each ingredient correlates with the number of claimed block groups** (Spearman, all episodes | tropical-window episodes): county peak rain +0.24 | +0.35; tcwv +0.14 | +0.31; tcwv z-score +0.12 | +0.25; 850 hPa flux +0.07 | +0.23; 850 hPa wind +0.05 | +0.21; 500 hPa wind +0.02 | +0.11; 850 hPa humidity +0.03 | +0.08. The association is stronger inside tropical windows.
- **None adds to the observed county rain** (leave-one-year-out boosting, AUC of any claim; all episodes | tropical-window episodes | the 500 episodes with the largest rain):

| Features | all | tropical | top-500 rain |
|---|---|---|---|
| county rain, duration, month, block groups | 0.9145 | 0.9271 | 0.8137 |
| + tcwv, z-score | 0.9140 | 0.9266 | 0.8079 |
| + 850 hPa humidity, wind, flux | 0.9135 | 0.9259 | 0.8210 |
| + 500 hPa wind | 0.9141 | 0.9263 | 0.8133 |
| + all ERA5 ingredients | 0.9135 | 0.9264 | 0.8156 |

Spearman of the predictions with the number of claimed block groups: 0.336 for the base, 0.335-0.336 with any addition.

## Conclusions

- The ingredients are real (they separate Harvey and Imelda from the convective events in the expected direction) but **the observed rain already contains them**: once the county rain is known, tcwv, low-level transport and steering wind add nothing at the episode level (differences of -0.001 in AUC, within the year-to-year spread; the top-500 column varies by +-0.007 on 500 episodes and is noise).
- This repeats the pattern of stages 27 and 31 (TC flag, county percentile, best-track motion): event-level descriptors do not improve on observed rain for deciding **whether** an episode floods or how widely. Their expected use is **forecast mode**, where the rain is uncertain and the ingredients (available from the forecast model days ahead) carry the information that the rain forecast lacks; that test needs forecast-time ingredients (GEFS reforecast PWAT and winds, or the ERA5 fields shifted by a lead) and is open.
- For the group (Texas, Louisiana, Florida) the same files cover all three states, so the features can be built for Louisiana and Florida once their pilot tables exist.

## Caveats

- One state, one learner, observed rain as the comparison (the strongest possible baseline); a county-scale value at the centroid, 6-hourly snapshots (the daily tcwv maximum can miss the peak between snapshots), no interval beyond the spread across held-out years.
- tcwv z-scores use the 2009-2016 climatology of each cell and calendar month, with all episodes being rainy days, so the median z-score is +1.3 even for ordinary episodes.
- Specific humidity at 850 hPa is a weak moisture proxy compared with the column; the flux uses the mean wind speed, not the component along the inflow.
- CAPE (instability) and 700 hPa fields were not used; the 700 hPa data are in the files.


# Stage 33: moisture and wind ingredients in forecast mode (script 72)

Question (stage 32 left it open): when the rain is a forecast instead of an observation, do the ERA5 ingredients (tcwv and its z-score, 850 hPa moisture flux, 500 hPa wind) add what the rain forecast lacks? Setting of stage 16 (block group x rain episode, 2010-2019, train 2010-2016, test 2017-2019, 2,075 test positives, boosting depth 5, rain = GEFSv12 reforecast ensemble mean at 36-84 h). The ingredients are taken at two timings: **perfect-prog** (ERA5 over the episode window, as if a moisture / wind forecast were perfect; an upper bound) and **run init** (the 00Z ERA5 analysis of the day before the episode, the state the reforecast starts from; nothing from after the run time). Output `audit/forecast_mode_ingredients.txt`.

## Within-event ranking (block groups inside an episode)

| Rain input | within-event AUC | difference to the same rain without ingredients [95%] |
|---|---|---|
| fcmean | 0.6674 | |
| fcmean + ERA5 perfect-prog | 0.6695 | +0.0021 [-0.0064, +0.0083] |
| fcmean + ERA5 at run init | 0.6583 | -0.0091 [-0.0155, -0.0013] |
| obsC (observed rain at the GEFS cell) | 0.7120 | |
| obsC + ERA5 perfect-prog | 0.7085 | -0.0035 [-0.0099, +0.0011] |

## Between episodes (does a county episode have a claim? leave-one-year-out, 30,362 episodes; AUC | Spearman with log claimed block groups)

| Features | AUC | Spearman |
|---|---|---|
| forecast county rain (max, mean, spread, member vote), month, duration, block groups | 0.8627 | 0.305 |
| + perfect-prog ingredients | 0.8702 | 0.311 |
| + ingredients at run init | 0.8617 | 0.304 |
| observed rain (GEFS cell) | 0.9039 | 0.339 |
| observed rain + perfect-prog ingredients | 0.9031 | 0.338 |

## Conclusions

- **Inside an episode the ingredients do nothing**, even perfect ones (+0.002, interval across 0), and the run-init version loses 0.009 (more noise features for the same rows). This is expected: they are constant inside an episode.
- **Between episodes there is a small gain, and only for perfect ingredients**: +0.0075 AUC over forecast rain (0.8627 to 0.8702), about 18% of the 0.041 gap between forecast and observed rain (0.8627 vs 0.9039); with observed rain the same ingredients add nothing (-0.0008). The initial moisture state alone does not help (-0.001).
- So the ceiling for a forecast of moisture and wind is a gain of about 0.0075 in episode-level AUC; a real forecast of tcwv, flux and steering wind has errors and would deliver less. **A GEFS PWAT and wind download would be unlikely to repay its cost on this evidence** (the reforecast archive is tens of GB per variable). The larger lever in forecast mode remains the rain forecast itself (stage 16: 0.024 resolution, 0.045 forecast error) and claim history (stage 19).

## Caveats

- Texas, non-tropical episodes only (the stage 16 subset excludes the tropical windows); 3 test years, one learner; the episode-level differences have no interval (leave-one-year-out pooled AUC; +0.0075 is read against the year-to-year spread of earlier stages of about 0.01).
- ERA5 stands in for the forecast model's own analysis; the perfect-prog run uses ERA5 values after the run time, so it is an upper bound, not a forecast skill.
- The run-init ingredients use the 00Z step of the day before the episode start, whereas the episode window may begin on a later calendar day than the gridMET day alignment of stage 16 (12Z to 12Z); a one-day shift in either direction was not tested.
- Tropical episodes, where stage 32 saw the strongest association of the ingredients with the outcome, are absent from this subset; a forecast-mode test for them needs the reforecast of tropical windows (not fetched).


# Stage 34: audit of the flood pilot's model assumptions against the task (script 73, `docs/MODEL_AUDIT.md`)

Question: before measuring the task's own metric, do the pilot's output definition (thresholds, sampling), input features and metric match the task in `DESIGN.md` (a probability issued from the forecast, judged by Brier skill against climatology, reliability, AUC-PR)? The full table of assumptions, the mismatches and the proposed metric specification are in `docs/MODEL_AUDIT.md`; output `audit/model_assumption_audit.txt`.

## Results

- **The pilot's metric is not the task's metric.** The stage 8 L2 model has Brier skill +0.005 against the training prevalence and -0.001 against the test prevalence; mean prediction 0.40% against a test rate of 0.22%, calibration intercept -0.97, every reliability decile about twice the observed rate. Rescaling only the mean to the test prevalence gives **+0.065**: the loss is the level (era drift), the ranking is good (overall AUC 0.926, AP 0.117 = 53x the base rate).
- **The headline within-event AUC depends on how it is aggregated**: 0.786 pair-weighted, 0.736 mean of episodes, 0.765 weighted by positives. Only 1,015 of 24,963 test episodes (4.1%) enter (they hold 98.7% of positives); the 10 largest carry 41% of the pairs; episodes with >= 100 block groups 0.786, smaller ones 0.735.
- **The episode is defined by the observed rain** (county 1-day maximum >= 30 mm): 5.3% of county-days 2010-2019. Forecast-mode rows exist only for county x observed episode, so false alarms and quiet days are not scored; 5.5% of episodes have any positive, a rate conditional on the weather.
- **The trigger moves the metric**: within-event AUC 0.766 at 20 mm, 0.786 at 30 mm, 0.802 at 50 mm; 20 mm adds 40% more episodes but only 2.8% more positives and 1.5 points of claim coverage (88.8% to 90.3%).
- **`dur`, the observed episode duration, is in the forecast pilot's feature set** and is not known at issue time. Without it the stage 16 forecast model loses 0.0097 within-event AUC ([-0.0257, +0.0021]) and 0.023 episode-level AUC (0.8627 to 0.8398). The episode-level forecast numbers of stage 33 contain the same leak, which is larger than the +0.0075 gain from perfect ingredients found there.
- The label scope (tropical windows removed) contradicts the decision of 2026-10-28 to model storm and tropical floods together.

## Conclusions

- The validated part is the **shape** (ranking inside an episode). The **level** and the **unit** are not what the task asks for; the plan is an all-days block-group x forecast-day table, issue-time features only, and Brier skill against a training-period climatology with the level reported both as predicted and recentred (`docs/MODEL_AUDIT.md` section 5).
- Stages 16, 17, 19 and 33 forecast-mode numbers should be read with the `dur` leak (about 0.01 within-event, 0.02 between episodes).

## Caveats

- One model (stage 8 L2, boosting) for the metric re-aggregation; the `dur` and trigger checks refit it once each. No interval on the trigger comparison.
- Statements about the SCS line were not audited; the DESIGN p99-precipitation hazard class belongs to the coarse 64x32 ENS testbed, not to the flood pilot.
- The 5.3% and the rates conditional on the weather are Texas gridMET statistics at county level.


# Stage 35: the ranking product and its probability companion, against Impact Forecasting.pdf (script 74, `docs/MODEL_AUDIT.md` rewritten)

**Correction to stage 34.** Stage 34 took `DESIGN.md` (a probability issued every day, judged by Brier skill) as the task. The slides `Impact Forecasting.pdf` define the task: rank units by forecast impact inside an event so that a limited number of resources go to the highest; the probability is kept as an **auxiliary product** (owner's decision, 2026-10-29) to show the level behind the ranking. Under that task the "unit conditions on observed rain" finding of stage 34 is weaker (the event is given), the stage 8 within-county metric is the wrong primary metric (it discards the between-county ranking), and the findings on `dur`, the trigger dependence, the aggregation of the AUC and the level of the probability stand. `docs/MODEL_AUDIT.md` is rewritten accordingly. Output `audit/ranking_and_probability_product.txt`.

Setting: stage 8 L2 model, test 2017-2023; event = county episodes starting the same day; 1,331 events, 1,203 with >= 2 counties; 408 with a claim for the block-group metrics, 386 for the county metrics.

## Ranking across counties inside an event

| Score | block-group AUC across counties (pair weighted) | share of claimed block groups in the top 5% / 20% of block groups |
|---|---|---|
| model L2 | 0.865 | 0.41 / 0.69 |
| cell rain only | 0.834 | 0.34 / 0.64 |
| county peak rain only | 0.747 | 0.20 / 0.47 |
| policy count only | 0.715 | 0.16 / 0.42 |

Counties (score = expected claimed block groups, sum of p): AUC 0.922; the top 10% of counties hold 74% and the top 20% hold 89% of the event's claimed block groups; policies alone 0.855 / 58% / 75%, county peak rain alone 0.782 / 40% / 56%. The within-county metric of stage 8 is 0.786, so the product ranks better across counties than the metric reported so far.

## The probability companion (146 events with >= 5 claimed block groups, medians)

- The highest block group has p = 24%; the top 5% have mean p 4.8% and an observed rate of 3.2%. **The other 95%: mean p 0.31%, observed 0.22%; 40% sit below 0.1%.** The others are small, not zero. **64% of an event's claimed block groups lie outside its top 5%.**
- Expected claimed block groups (sum of p) against observed: ratio 1.50 per event (pooled 1.58; 2.0 for the smallest events, 1.5 for the largest); Spearman of the event's expected with its observed count 0.66.
- Level: as predicted, Brier skill +0.002 against the train prevalence; re-centred with the **trailing three-year prevalence (deployable) -0.018**; with the same-year prevalence (oracle) +0.057. The level follows the size of the events, which the calendar does not give (yearly prevalence 1.16% in 2015, 0.19% in 2017, 0.10% in 2020).

## Conclusions

- Ranking is the validated product, and on the task's own units it is stronger than the stage 8 figure: cross-county block-group AUC 0.865, county AUC 0.922. Exposure carries a large part of the county ranking (policies alone 0.855); the model adds 0.067 over exposure and 0.14 over county rain.
- The answer to the emergency manager's doubt is quantitative: the others are not 0% (mean 0.3%), but a short list misses most of the event's claims (64% outside the top 5%). Show the probabilities as relative risk and expected number of affected units, with the 1.5x overprediction stated, until an event-level model of the level exists.

## Caveats

- One model, one split, Texas, observed-rain mode; no interval on these figures (events are the units for the bootstrap to come).
- Events by same start day (a heuristic for "one system"); counties with few block groups and exposure confound the county ranking (policy count is partly the adaptation A of the slides).
- The label is claim occurrence, not loss; graded relevance (claim count, paid amount) is not measured yet. The train prevalence of script 74 is the mean of the yearly rates, so the +0.002 differs slightly from the +0.005 of stage 34 (pooled rate).
- The slides' evaluation also asks for a check of the impact model on simulated losses with a known answer; not done.


# Stage 36: the task metrics measured (script 75, specification in `docs/MODEL_AUDIT.md` section 5)

Decisions applied (2026-10-29): k = 5 / 10 / 20% of units (1% shown) and the top 3 / 5 / 10 counties; relevance = claim count and paid building amount, both reported; event = county episodes starting the same day; GEFS all-days data postponed. Units: block groups, and counties (score = expected claimed block groups, sum of p). Intervals: 500 bootstrap draws over events. Output `audit/task_metrics.txt`.

## A. Observed rain, stage 8 L2 model, scope S0, test 2017-2023 (408 events scored at block-group level, 386 at county level)

| Score | block-group AUC across counties [95%] | top 5% of block groups: share of claim count (paid) | county AUC [95%] | top 10% of counties: share of claim count (paid) | top 3 / 5 / 10 counties |
|---|---|---|---|---|---|
| model | 0.865 [0.852, 0.878] | 0.425 (0.471) | 0.922 [0.910, 0.934] | 0.740 (0.756) | 0.70 / 0.82 / 0.93 |
| cell rain only | 0.834 | 0.353 (0.404) | 0.837 | 0.509 (0.575) | 0.47 / 0.62 / 0.80 |
| county peak rain only | 0.747 | 0.206 (0.232) | 0.782 | 0.396 (0.448) | 0.36 / 0.50 / 0.73 |
| policies only | 0.715 [0.688, 0.738] | 0.169 (0.134) | 0.855 | 0.580 (0.537) | 0.54 / 0.66 / 0.83 |

Model minus policies only (95% over events): block-group AUC +0.151 [+0.126, +0.182]; top-5% capture of claim count +0.256 [+0.218, +0.294]; NDCG at 5% with paid amount +0.127 [+0.104, +0.150]; county AUC +0.066 [+0.052, +0.083]; top-10% county capture +0.160 [+0.124, +0.197]. By test year, block-group AUC 0.842-0.878, county AUC 0.910-0.943, top-5% capture 0.34-0.54. Paid-amount capture follows claim-count capture closely.

## B. Forecast rain (GEFSv12 ensemble mean, 36-84 h, stage 16 setting) without `dur`, test 2017-2019 (187 / 174 events)

| Score | block-group AUC across counties [95%] | top 5% of block groups: share of claim count | county AUC [95%] | top 10% of counties: share of claim count |
|---|---|---|---|---|
| forecast model | 0.650 [0.596, 0.690] | 0.165 | 0.852 [0.828, 0.874] | 0.593 |
| observed-rain model (same days, reference) | 0.797 [0.751, 0.839] | 0.299 | 0.906 [0.890, 0.922] | 0.707 |
| forecast cell rain only | 0.534 | 0.099 | 0.597 | 0.154 |
| forecast county peak only | 0.525 | 0.100 | 0.591 | 0.157 |
| policies only | 0.715 [0.667, 0.749] | 0.165 | 0.847 [0.815, 0.876] | 0.629 |

Forecast model minus policies only: block-group AUC **-0.066 [-0.115, -0.014]**; top-5% capture -0.000 [-0.050, +0.051]; county AUC +0.005 [-0.013, +0.019]; county top-10% capture -0.037 [-0.086, +0.021].

## C. Observed rain, scope S2 (tropical windows kept, named tropical claims included; model retrained), test 2017-2023 (447 / 421 events)

Model: block-group AUC 0.886 [0.844, 0.912], top-5% claim-count capture 0.444, county AUC 0.924, top-10% county capture 0.764 (paid 0.775), top 3 / 5 / 10 counties 0.71 / 0.83 / 0.93. Policies only: 0.750 / 0.181 / 0.856 / 0.592. Model minus policies only: block-group AUC +0.137 [+0.096, +0.202], county AUC +0.068 [+0.055, +0.082].

## Auxiliary probability product

| Run | expected / observed claimed block groups (median per event) | others (outside the top 5%): mean p / observed | claimed block groups outside the top 5% | calibration slope | Brier skill vs the trailing 3-year prevalence: as predicted / re-centred on trailing 3 years / same year (oracle) |
|---|---|---|---|---|---|
| A (S0, observed) | 1.50 | 0.31% / 0.22% | 64% | 0.92 | +0.002 / -0.018 / +0.057 |
| B (forecast) | 1.06 | 0.37% / 0.34% | **91%** | 0.69 | -0.033 / -0.070 / -0.005 |
| C (S2, observed) | 1.37 | 0.32% / 0.28% | 60% | 0.98 | +0.220 / +0.188 / +0.286 |

## Conclusions

- **In observed-rain mode the ranking product meets its definition**: across counties inside an event the model beats the exposure-only ranking with intervals clear of zero; the top 10% of counties hold 74% of the claims (76% of the paid amount), the top 3 counties 70% in events with at least 10 counties, stable by year. Scope S2 (tropical kept) ranks as well or slightly better (0.886 / 0.924), so the extreme-weather scope costs nothing in ranking.
- **In forecast mode the ranking has no skill beyond exposure across counties.** The GEFS ensemble-mean rain on its own is close to uninformative between counties (county AUC 0.597, block-group 0.534), and the trained forecast model equals the policies-only ranking at county level (+0.005 [-0.013, +0.019]) and is worse at block-group level (-0.066). The observed-rain model on the same days reaches 0.906 / 0.797: **the whole skill of the product is the hazard forecast**, which is the slides' warning about chained errors. Stage 16's gain (0.667 against 0.622 exposure only) was within a county; across counties it disappears.
- **The probability companion matters most where the ranking is weakest**: in forecast mode 91% of an event's claimed block groups lie outside the top 5%, and the others carry a mean p of 0.37% against an observed 0.34% (calibrated at the event level, ratio 1.06, but over-high overall: slope 0.69, mean 0.50% against 0.28%). A bare short list would be nearly uninformative here; the probabilities say how flat the risk is.
- **The level behaves differently by scope**: with the tropical systems kept (S2) the probabilities are close to calibrated (slope 0.98, ratio 1.37) and the Brier skill against the trailing prevalence is +0.22 (oracle +0.29); with S0 it is about 0. Part of this is that a few large tropical events dominate the Brier numerator.

## Caveats

- Forecast mode: 3 test years, 187 events, 5 of the 11 reforecast members, lead 36-84 h, the first two days of each episode only; the sample is conditional on an observed episode in the county; intervals are wide (the block-group AUC interval spans 0.60-0.69). The reference model shows the same days are rankable (0.797).
- Events by same start day split multi-day systems; the bootstrap resamples events and ignores correlation between days of one storm, so intervals are too narrow, most for S2 where a few systems carry many claims.
- Relevance is NFIP claims in [start, end + 2 d] on the block groups with a policy; paid amounts are building payments; scope S0 relevance for A and B, S2 for C. County ranking is partly exposure (policies alone 0.855 in A).
- Brier skill is computed against a constant forecast at the trailing 3-year prevalence (deployable); an oracle re-centring uses the test year's own rate.


# Stage 37: why the forecast rain ranks counties so weakly in the pilot (script 76)

Stage 36 (run B): across counties inside an event the GEFS forecast county peak ranks counties at AUC 0.59, and the forecast model adds nothing to the exposure ranking (county AUC 0.852 against 0.847), while the observed-rain model reaches 0.906. Diagnosis on the 1,640 stored forecast days and the pilot's 30,362 county episodes (2010-2019). Output `audit/forecast_rain_diagnosis.txt`.

## Results

| Check | Result |
|---|---|
| **Day alignment** (ensemble mean against gridMET aggregated to the GEFS cells, pooled Pearson r over Texas cells) | forecast day S with gridMET day S: **0.594**; shifted by -1 / +1 day: 0.261 / 0.237. Forecast day S+1 with day S+1: **0.555**; shifted: 0.240 / 0.257. The alignment of stage 16 is right |
| **Cut by observed duration** (`nd`: one forecast day if the observed episode lasted one day) | no effect on the agreement (Spearman 0.47 for both) or on the county ranking (0.559 / 0.567), but it is a use of the observed duration in a forecast feature |
| Forecast vs observed county peak, pilot episodes | Spearman 0.50 pooled (0.41 against the 4 km value); **0.23 inside events**; median ratio forecast / observed 0.37; forecast peak below 10 mm in 42% of the episodes (observed: 4%) |
| By longitude (west / middle / east) | ratio 0.34 / 0.37 / 0.40, Spearman 0.43 / 0.49 / 0.52: the bias is spatially homogeneous |
| **Sample effect**, all 737,362 county-days of the stored days | Spearman forecast vs observed county peak **0.742** over all county-days, **0.286** among county-days whose observed peak reached 30 mm (the pilot's sample), 0.223 at 50 mm |
| Forecast peak for "the observed peak reached 30 mm" | AUC **0.915** pooled, **0.795** across counties inside a day |
| Bias correction, county AUC for any claim inside events (2017-2019) | raw 0.591; county scale factor 0.598; percentile within the county's own forecasts 0.571 |
| Same, with exposure | forecast peak + log policies 0.852; observed peak + log policies 0.900; policies alone 0.847; observed peak alone 0.760 (4 km 0.804) |
| **All counties inside a day** (dry ones included; label any claim in [S, S+3]; 132,826 test county-days) | forecast peak 0.662; observed peak 0.719; **policies only 0.881**; forecast peak + policies **0.889**; observed peak + policies **0.898** |
| The same restricted to counties with an observed peak >= 30 mm | forecast 0.568; policies 0.818; observed 0.739 |

## Conclusions

- **The forecast has real skill where it matters for the weather, and the pilot's sample removes it.** Over all county-days the forecast county peak tracks the observed one at Spearman 0.74 and separates the county-days that reach 30 mm from the rest at AUC 0.92 (0.80 across counties inside a day). Among the counties that did reach 30 mm, it ranks them at 0.29 (0.22 for 50 mm): the sample was selected on the outcome the forecast predicts. The stage 36 forecast result (forecast = exposure across counties) is largely an effect of defining the event's county set from observed rain.
- **Not the cause**: day alignment (right), the one-day cut by `nd` (no effect), and bias (homogeneous in space, ratio 0.34-0.40; county scale factors and percentiles do not help: 0.598 and 0.571 against 0.591).
- **What the forecast loses is mostly the ensemble mean's damping and its misses**: median forecast / observed county peak 0.37, and 42% of the episodes forecast below 10 mm (the 5-member mean at 0.25 degrees; stage 16: 0.045 of within-county AUC lost to forecast error).
- **With all counties of a day included, exposure dominates the county ranking whatever the rain**: policies alone 0.881; the forecast adds +0.008 (0.889) and the observed rain +0.017 (0.898). So at county level for "any claim in 4 days" the gap between forecast and observed mode is 0.009, not the 0.054 seen in stage 36 (0.852 against 0.906); the forecast recovers about half of what observed rain adds. The county label of an insured claim is mostly "where the insurance is"; rain information is worth more at finer scale (within a county, stage 16) and for the size of events.
- Consequence for the specification: the **event's set of counties must be defined before the event** (a forecast footprint or a fixed region per day), not from observed rain; the forecast-mode ranking should be re-measured on that set.

## Caveats

- Stored days are the days before some county's episode (45% of days), so the all-county sample is not all days of the year, and the AUC of the forecast for 30 mm (0.915) is on those days.
- Label for the all-county check: any claim in [S, S+3] in a county with a policy (1.7% of county-days), not the pilot's episode window; policies on day S from the clean exposure arrays. County rain is the peak of the ensemble mean; no block-group metric was repeated on the all-county set.
- Five of 11 members, 0.25 degrees; the correction fits are per county from 2010-2016 episodes (counties with fewer than 8 episodes use the global factor). Single state, single split.


# Stage 38: forecast-mode ranking on an ex-ante county set (script 77)

Stage 37 showed that the pilot's county set (counties with an observed episode) hides the forecast's between-county skill. Re-measured with every unit of every forecast day as a row, whatever the rain turned out to be. The GEFS reforecast is now complete on disk (3,652 initialisation days, 913 MB). Output `audit/forecast_ranking_all_counties.txt`.

## Design

- **Rows**: every block group with >= 1 policy on day S, for each initialisation day (00Z of S-1, leads 36-84 h) 2010-2019, except county-days inside a tropical-cyclone window (scope S0 as before): 41.7 M rows on 3,486 forecast days (about 11,960 block groups and 221 counties per day). Test 2017-2019: 11.0 M rows, 938 days, 9,814 claimed block groups. Training 2010-2016 with negatives subsampled 1 in 10 (weight 10); every test row scored.
- **Features at issue time only**: forecast cell rain (peak and sum over S, S+1, ensemble mean), forecast county peak and mean, exposure (log policies, SFHA share, single-family share, mean coverage), block groups in the county, month. No `dur`, no observed rain. Reference model: the same with the observed rain on the GEFS cells.
- **Label / relevance**: claim count and paid building amount in [S, S+3]. **Event** = the forecast day, all counties ranked together; second view = the counties with a forecast county peak >= 10 mm (an ex-ante footprint: 52 counties a day, 19% of the rows, 57% of the claim count).

## Results (point [95% over days])

| All counties of a day (705 days scored) | block-group AUC across counties | top 5% of block groups: share of claim count | county AUC | top 10% of counties: share of claim count (paid) |
|---|---|---|---|---|
| forecast model | 0.713 [0.682, 0.742] | 0.196 | **0.891** [0.883, 0.900] | **0.796** (0.771) |
| observed-rain model (reference) | 0.794 [0.769, 0.817] | 0.234 | 0.902 [0.894, 0.909] | 0.819 (0.806) |
| policies only | 0.747 [0.728, 0.764] | 0.164 | 0.881 [0.872, 0.889] | 0.767 (0.731) |
| forecast cell rain only | 0.587 | 0.101 | 0.664 | 0.265 |
| forecast county peak only | 0.595 | 0.099 | 0.658 | 0.260 |

Forecast model minus policies only: block-group AUC **-0.033** [-0.057, -0.008]; top-5% claim-count capture **+0.032** [+0.010, +0.058]; NDCG at 5% with paid amount **+0.009** [+0.003, +0.015]; county AUC **+0.010** [+0.005, +0.014]; top-10% county capture **+0.029** [+0.014, +0.044]. Top 3 / 5 / 10 counties: 0.37 / 0.45 / 0.61 (forecast), 0.34 / 0.42 / 0.62 (policies), 0.40 / 0.47 / 0.60 (observed rain).

Forecast footprint (322 days): forecast model county AUC 0.847 [0.827, 0.865] against policies 0.838 (+0.009 [+0.001, +0.017]); block-group AUC 0.700 against 0.737 (-0.036 [-0.065, -0.008]); top-5% and top-10%-of-counties capture equal to policies (-0.000); observed-rain reference 0.886 / 0.802.

| Probability companion | expected / observed claimed block groups (median per event) | others (outside the top 5%): mean p / observed | claimed block groups outside the top 5% | share of units with p < 0.1% | calibration slope | Brier skill vs the trailing 3-year prevalence: as predicted / trailing re-centred / same-year |
|---|---|---|---|---|---|---|
| all counties | 0.91 | 0.09% / 0.10% | 86% | 67% | 0.78 | -0.013 / -0.039 / -0.000 |
| footprint | 1.02 | 0.26% / 0.26% | 88% | 17% | 0.71 | -0.026 / -0.000 / -0.001 |

## Conclusions

- **The stage 36 verdict on forecast mode is revised.** On an ex-ante county set the forecast model beats the exposure ranking at county level with intervals clear of zero (AUC +0.010, top-10% capture +0.029) and recovers about half of the gain the observed-rain model gets over exposure (county AUC 0.881 to 0.891 to 0.902). Stage 36's gap to the observed-rain model (0.852 against 0.906) was mostly the selected sample.
- **The increment is small and sits in the head of the list**: at block-group level the pooled AUC across counties stays below the exposure ranking (-0.033), while the top-5% capture and the paid-amount NDCG improve slightly (+0.032, +0.009). Exposure dominates the county ranking for an insured claim in 4 days (policies alone 0.881).
- **Probabilities are calibrated in aggregate when trained on the unit of use** (events: expected / observed 0.91 for all counties, 1.02 for the footprint; others 0.09% vs 0.10%, 0.26% vs 0.26%), unlike the 1.5x of run A in stage 36, which was trained on observed episodes. The calibration slope is still below 1 (0.71-0.78) and the Brier skill against the trailing prevalence is about 0 (-0.013, -0.026); trailing re-centring does not help. The short list is thin: 86-88% of an event's claimed block groups lie outside its top 5%.
- Whether the forecast's hazard increment is larger where it matters (big events) is not shown here; this table pools 705 days, most of them with few claims.

## Caveats

- Daily windows [S, S+3] overlap, so consecutive days share claims and the bootstrap over days is too narrow; totals of claims or paid amounts over all rows count each claim up to four times (the raw totals printed by the script, 121,535 claims and $4.5 B, are about four times the real figures).
- The observed-rain reference stopped at 15 iterations (validation Brier on weighted, subsampled rows) and may be under-trained against the forecast model (83 iterations); its gap to the forecast model is a lower bound.
- Five of the 11 reforecast members, 0.25 degrees, ensemble mean at the cell; tropical-window county-days removed (scope S0); Texas only; test = 3 years. The scoring days contain many dry days (705 of 938 scored), so averages are dominated by small events.
- Not comparable one to one with runs A-C of stage 36: different unit (a forecast day, all counties) and label window.


# Stage 39: the forecast model's gain over exposure, by size of the event (script 78; script 77 rerun on the complete archive)

Question left by stage 38: is the forecast model's small gain over the exposure ranking larger in big events? Script 77 was rerun after the GEFS archive was completed (3,639 forecast days, 43.3 M rows; test 2017-2019: 1,091 days, 12.7 M rows, 10,075 claimed block groups, 778 days scored) and now saves its test predictions (`pilot/ranking_test_rows.parquet`); script 78 stratifies the test days. **The overall numbers of stage 38 are superseded with shifts of at most 0.005**: forecast model county AUC 0.890 [0.881, 0.897] against policies only 0.880 (+0.010 [+0.005, +0.014]) and the observed-rain model 0.901; block-group AUC 0.711 against 0.742 (-0.032 [-0.057, -0.008]); top-5% claim-count capture +0.034 [+0.011, +0.059]; NDCG at 5% with paid amount +0.009 [+0.003, +0.015]; top-10% county capture +0.030 [+0.018, +0.044]; probabilities: expected / observed 0.92, calibration slope 0.78, others 0.08% against 0.10%. Output `audit/ranking_by_event_size.txt`.

## A. By observed size of the day (claimed block groups; the outcome is used, so this is descriptive)

| Stratum | days (scored) | share of the test claim count | county AUC: forecast model minus policies [95%] | same, observed-rain model | top-10% county capture: forecast | observed | block-group top-5% capture: forecast | observed |
|---|---|---|---|---|---|---|---|---|
| 1-2 claimed block groups | 323 | 2.2% | +0.009 [+0.003, +0.015] | +0.007 | +0.027 [+0.002, +0.051] | +0.033 | +0.068 [+0.022, +0.112] | +0.057 |
| 3-9 | 245 | 6.1% | +0.011 [+0.005, +0.017] | +0.013 | +0.044 [+0.024, +0.065] | +0.052 | +0.006 | +0.054 |
| 10-29 | 134 | 14.2% | +0.012 [+0.003, +0.021] | +0.024 | +0.037 [+0.009, +0.067] | +0.057 | -0.003 | +0.064 |
| >= 30 | 76 | **77.6%** | **+0.006 [-0.003, +0.015]** | **+0.030 [+0.017, +0.044]** | -0.016 [-0.067, +0.027] | **+0.073 [+0.026, +0.126]** | +0.047 [-0.005, +0.103] | **+0.159 [+0.076, +0.247]** |

(313 days with no claimed block group are not scored.) Block-group AUC across counties, forecast model minus policies only: +0.003, -0.012, +0.026, **-0.058 [-0.094, -0.023]** for the four strata; observed-rain model +0.019, +0.016, +0.052, +0.052.

## B. By the day's largest forecast county peak (known at issue time)

| Stratum | days (scored) | share of claim count | county AUC: forecast minus policies [95%] | observed-rain model | NDCG at 5% with paid amount: forecast | observed |
|---|---|---|---|---|---|---|
| < 10 mm | 426 (243) | 6.1% | +0.000 [-0.004, +0.005] | -0.006 | +0.001 | +0.003 |
| 10-25 mm | 362 (264) | 16.4% | +0.004 [-0.001, +0.010] | +0.004 | +0.004 | +0.022 |
| 25-50 mm | 216 (185) | 37.7% | **+0.014 [+0.006, +0.021]** | +0.028 | **+0.022 [+0.010, +0.038]** | +0.056 |
| >= 50 mm | 87 (86) | 39.9% | **+0.018 [+0.006, +0.031]** | +0.052 | **+0.017 [+0.001, +0.032]** | +0.064 |

## Conclusions

- **The forecast model's gain does not grow with the size of the event; the potential gain does.** With observed rain the improvement over exposure rises with event size (county AUC +0.007, +0.013, +0.024, +0.030; top-5% block-group capture +0.057 to +0.159). The forecast model recovers it fully in the smallest events and less and less in the bigger ones: share of the observed-rain county-AUC gain recovered about 100%, 85%, 50%, **20%**. In the 76 largest days, which hold 78% of the claims, it does not beat the exposure ranking at county level (+0.006 [-0.003, +0.015]) and is worse at block-group level (-0.058).
- **Ex ante, the model helps where the forecast is large**: on days with a forecast county peak of 25 mm or more it improves county AUC by +0.014 and +0.018 and paid-amount NDCG by +0.022 and +0.017 (intervals above zero); below 25 mm it equals exposure. These days hold 78% of the claim count. The observed-rain model gains +0.028 and +0.052 there, so the forecast recovers half to a third.
- So the shortfall of the ranking product in forecast mode sits in the extreme events the forecast underestimates (stage 37: ensemble mean about 0.37 of the observed county peak, 42% of episodes forecast below 10 mm), not in the small ones.

## Caveats

- Stratum A uses the outcome (claimed block groups of the day); more positives also make each day's AUC easier, but the comparison is paired within a stratum. Stratum B is ex ante.
- 76 and 87 days in the top strata, overlapping four-day windows (consecutive days share claims), so the intervals over days are narrow; the biggest strata contain few distinct storms.
- The observed-rain reference stopped at 15 iterations and may be under-trained (stage 38); the observed gain in the large strata is a lower bound.
- Texas, scope S0 (tropical-window county-days removed), five of 11 members, 0.25 degrees, three test years.


# Stage 40: ensemble extremes in the all-days forecast model (script 79)

Hypothesis from stage 39: the five-member mean damps the rain in the biggest events, so ensemble extremes should lift the forecast ranking there. Test: the all-days model of script 77 plus the member maximum, the share of members >= 25 mm and >= 50 mm, the member spread and the 3x3 mean of the 25 mm share at the cell, and the county maximum of the member maximum, the county maximum and mean of the 25 mm share and the county maximum of the 50 mm share (19 features against 10). Same rows (43.3 M, 3,639 forecast days), split, subsampling and metrics; the extremes model needed 214 iterations against 83. Output `audit/ensemble_extremes.txt`, predictions in `pilot/ranking_test_rows_ext.parquet`.

## Results (extremes minus mean model [95% over days])

| Stratum | county AUC | county top-10% capture | block-group top-5% capture | block-group NDCG at 5% (paid) |
|---|---|---|---|---|
| all test days (778 scored) | **-0.003 [-0.004, -0.001]** | -0.010 [-0.018, -0.002] | **-0.026 [-0.044, -0.009]** | -0.004 [-0.008, -0.002] |
| 1-2 claimed block groups | -0.004 [-0.006, -0.002] | -0.019 [-0.036, -0.003] | -0.051 [-0.084, -0.019] | -0.005 |
| 3-9 | -0.001 | -0.014 [-0.025, -0.001] | -0.001 | -0.003 |
| 10-29 | -0.002 | +0.004 | -0.022 [-0.044, -0.003] | -0.006 |
| >= 30 (77.6% of the claims) | -0.005 [-0.008, -0.001] | +0.015 [-0.009, +0.043] | -0.019 [-0.040, -0.001] | -0.006 |
| forecast peak < 10 mm | -0.000 | -0.018 [-0.041, -0.003] | -0.019 | -0.005 |
| 10-25 mm | -0.001 | -0.008 | -0.025 | +0.000 |
| 25-50 mm | -0.003 [-0.006, -0.001] | -0.008 | -0.029 [-0.057, -0.002] | -0.005 |
| >= 50 mm | -0.006 [-0.010, -0.002] | +0.006 | **-0.055 [-0.098, -0.021]** | **-0.014 [-0.026, -0.003]** |

Against the exposure ranking, the extremes model gives county AUC +0.007 [+0.003, +0.010] overall (the mean model: +0.010) and +0.002 [-0.007, +0.012] in the days with >= 30 claimed block groups (the mean model +0.006); the observed-rain reference stays at +0.030 there. Block-group AUC across counties: extremes 0.704, mean 0.711, policies 0.742, observed 0.791.

## Conclusions

- **The hypothesis is refuted: ensemble extremes do not recover the big-event loss, and they slightly hurt.** Overall county AUC -0.003 and top-5% block-group capture -0.026 (intervals below zero); in the strata where the mean model fails (>= 30 claimed block groups, forecast peak >= 50 mm) the differences are zero or negative (county AUC -0.005 and -0.006; top-5% capture -0.019 and -0.055). The only positive signs (top-10% county capture +0.015, +0.006 in the largest strata) have intervals spanning zero.
- This matches stage 16 (member votes, spread and neighbourhood tied the ensemble mean in the episode sample): with five members at 0.25 degrees and leads of 36-84 h the extreme members add noise about where the heavy rain falls, not information. The missing information is not in the spread of this ensemble.
- The extremes model also fitted longer (214 against 83 iterations), a sign of fitting noise in the extra features.

## Caveats

- Five of the 11 reforecast members (the 11-member Wednesday runs are cut to 5); with 11 members the exceedance shares and the member maximum would be better defined, but only for about one day in seven.
- One lead range (the 00Z run of the day before, 36-84 h), 0.25 degrees, ensemble members of one model (GEFSv12), Texas, scope S0, three test years; the observed-rain reference stopped at 20 iterations.
- Overlapping four-day windows and the few distinct storms in the largest strata make the intervals over days narrow.


# Stage 41: E0 of the CNN hybrid, day tensors and registration checks (script 80)

E0 of `docs/CNN_HYBRID_DESIGN.md`: the two-dimensional inputs, the block-group indices and the rows of the all-days unit, built and checked. Output `$WXRISK_DATA/cnn/` (3.5 GB, registered in `docs/DATA_SOURCES.md`), checks in `audit/e0_day_tensors.txt`, maps in `cnn/e0_maps.png`. Build time about three minutes.

## What was built

- **3,647 forecast days** (2010-01-02 .. 2019-12-27): `gefs_day_tensors.npy`, float16 [day, 18, 53, 61] (424 MB): forecast mean peak and sum, the five member peaks sorted, log policies, SFHA share and `hist00` per cell, observed gridMET peak and sum on the GEFS cells, ERA5 tcwv, q850, u850, v850, u500, v500 (daily means over S and S+1). `obs4k_day_maps.npy`, float16 [day, 2, 260, 320] (1.2 GB): the gridMET 4 km peak and sum.
- **Rows in CSR form**: 43,337,315 block-group-day rows (the same number as script 77; the 8 days that script 77 does not count have no rows), tabular features [log policies, SFHA share, single-family share, mean coverage, block groups in the county, `hist00`, `prior3`, month, policies], claim count and paid amount in [S, S+3]; `static.npz` holds the grids, the masks, the block-group cell indices at 0.25 degrees and at 4 km, and the county index.

## Checks

| Check | Result |
|---|---|
| ERA5 registration | ERA5 coordinates equal the GEFS coordinates shifted by two rows and two columns (assertion); the outer two rows (north) and columns (west) of the GEFS grid lie outside the ERA5 box and are masked. 320 block-group comparisons of the tensor value against a direct nearest-point read of the ERA5 file: max difference 0.016 kg/m2, median 0.004, all within 0.1 |
| Consistency with script 77 | 12 random test days: the same block groups on 12 of 12, the same claim counts on 12 of 12, forecast peak at the cell equal within 0.014 mm (float16 storage) |
| Channels | no NaN over the valid Texas cells; fc mean peak mean 3.3 mm (maximum 243), observed peak 3.6 mm (440), ERA5 tcwv 24 kg/m2 (2.6-70.5); the member channels are ordered (member 1 mean 6.5 mm, member 5 mean 1.1 mm) |
| Geometry | Texas occupies 789 of 3,233 cells of the grid (24%), ERA5 covers 3,009 cells; every block group has a cell index inside the grid |
| **Observed-rain coverage** | gridMET covers only 1,637 of the 3,233 cells fully (72% of the 4 km pixels); 27 Texas cells (coast, border) are partly covered. The observed channels are zero outside the mask, and the cell averages of the partly covered cells are biased low (the same holds for `oc_f1` of scripts 77-79). The masks `obs_cover` and `obs4_valid` are saved; E1 must use them |

The maps (four days: Memorial Day 2015, Tax Day 2016, Harvey, Imelda) look right: the forecast mean peak and the observed peak show the same systems, the tcwv field has its maximum under the storms, and the policy map follows the cities. The Harvey panel shows the position error that motivates H3: the forecast spiral is centred near 28.8 N 97 W while the observed maximum lies near 30 N 95 W.

## Conclusions

- The inputs of E0 are ready and reproduce the rows and labels of script 77; E1 (the gate: two-dimensional reading with complete information) can start from `cnn/`.
- For E1 the observed channels need the coverage masks; cells outside gridMET (mostly outside Texas: the northern and eastern edges of the box, Mexico and the Gulf, as the maps show) carry no observed rain.

## Caveats

- Float16 storage rounds values (error up to about 0.01 mm at the checked cells, more at 200 mm and above); the 4 km maps are zero-filled where gridMET is missing.
- The rows exclude tropical-cyclone county-days (scope S0) while the tensors contain all days, so a model fed a tropical day sees the field but has no rows for the counties in the window.
- ERA5 maps are valid-time fields (E3 ceiling only); the GEFS fields are the forecast. No model has been trained.


# Stage 42: E1 of the CNN hybrid, two-dimensional reading with the observed rain field (scripts 81, `src/wxrisk/cnn.py`)

E1 is the gate of `docs/CNN_HYBRID_DESIGN.md`: with complete (observed) information, does a network that reads the field beat boosting on point and county summaries of the same field? Same rows, split and metrics as stages 38-40 (all block groups of every day of the day tensors of stage 41, scope S0, train 2010-2016, test 2017-2019: 1,091 days, 778 scored, 10,075 claimed block groups); events are days; differences are paired, bootstrapped over days. Output `audit/e1_cnn_observed.txt`; predictions and test frame in `cnn/e1_*`.

## Models

- **Boosting baseline** (strongest summaries): the 9 tabular rows (log policies, SFHA share, single-family share, mean coverage, block groups in the county, `hist00`, `prior3`, month, policies) + the observed rain at the 0.25 degree cell and at the 4 km pixel, each with the county peak and mean (17 features). Negatives sampled 1 in 10 with weight 10. Iterations by weighted log loss on blocked validation months (**150**; the Brier score picks 2, a straw man) and a version with 200: the two agree to 0.001-0.012 on every metric.
- **CNN hybrid** (46 k parameters at 0.25 degrees, 55 k at 4 km): fully convolutional encoder (stem, four or five residual dilated 3x3 blocks, 16-channel embedding), read at the block group's pixel, over its county (mean and maximum) and with the tabular features; one day per batch (all claimed rows and 2,000 others, weights restoring the normaliser), listwise loss + 0.3 x binary cross entropy, AdamW with one-cycle schedule, best epoch on blocked validation months (about 1 in 7; the model trains on the other 2,191 days). Inputs: observed peak and sum (gridMET averaged to the GEFS cells and masked where gridMET covers < 50% of the cell; or the 4 km maps) + log policies, SFHA share, `hist00` as image channels + coordinates + masks. 5 seeds at 0.25 degrees (20 epochs), 3 seeds at 4 km (8 epochs); the seed mean of p is the CNN's score.
- Tests of the module (`tests/test_cnn.py`): shapes, determinism, gradients reach the image, padding does not change the county pooling, loss ordering.

## Results (CNN minus boosting with validation iterations; point value [95% over days])

| Metric (all test days) | policies only | boosting | CNN 0.25 deg | CNN 4 km |
|---|---|---|---|---|
| block-group AUC across counties | 0.742 | 0.821 | +0.015 [+0.009, +0.021] | **+0.025 [+0.019, +0.031]** |
| block-group top-5% capture of claim count | 0.160 | 0.296 | -0.011 [-0.028, +0.006] | **+0.023 [+0.005, +0.040]** |
| block-group NDCG at 5%, paid amount | 0.026 | 0.079 | -0.006 [-0.011, -0.000] | **+0.017 [+0.011, +0.023]** |
| county AUC | 0.880 | 0.909 | +0.000 [-0.002, +0.002] | **+0.004 [+0.002, +0.006]** |
| county top-10% capture | 0.770 | 0.824 | +0.014 [+0.004, +0.024] | +0.008 [-0.001, +0.018] |

In the days with >= 30 claimed block groups (77.6% of the claims) the 4 km network gains **+0.049 [+0.015, +0.080]** in block-group top-5% capture and **+0.054 [+0.031, +0.077]** in NDCG with paid amount (block-group AUC +0.029), and nothing at county level (county AUC +0.004 [-0.000, +0.008], top-10% capture +0.015 [-0.004, +0.045]). By the day's largest observed county peak, the gain concentrates in the strata >= 50 mm (block-group top-5% capture +0.031 and +0.037; NDCG +0.023 and +0.051). Seed spread: 0.25 degrees county AUC 0.904-0.909 (sd 0.0016), top-5% capture 0.276-0.289; 4 km county AUC 0.908-0.914 (sd 0.0027), top-5% capture 0.300-0.318 (every seed above the boosting 0.296).

## Conclusions

- **Gate: passed at 4 km, at block-group level; not passed at 0.25 degrees.** With the observed field at 4 km the network beats the boosting baseline with intervals above zero in the block-group metrics (AUC, top-5% capture, NDCG with paid amount) and, small, in county AUC; the gain is largest in the big events. At 0.25 degrees, the resolution of the forecast fields, it ties boosting (county AUC +0.000) and is worse at the head of the list (top-5% capture -0.011, NDCG -0.006).
- So two-dimensional structure at fine scale carries information that the pixel value and the county peak and mean do not: the gain does not come from the coarse field. It is a gain in **where inside the county** the claims fall, not in which county.
- **What this implies for the forecast mode (E2):** the forecast fields are 0.25 degrees; the 0.25 degree network found nothing over boosting, so a gain from the GEFS fields is not expected unless the fine-scale structure is recovered some other way (the learning-stage route E4: a model trained on 4 km observed fields applied to the forecast, which needs a downscaling of the forecast, or a finer forecast).

## Caveats

- The CNN score is the mean of 3-5 seeds and the baseline a single boosting model; an ensemble gains by itself. Single seeds at 4 km give county AUC 0.908-0.914 against boosting 0.909 (no gain beyond the spread at county level) and top-5% capture 0.300-0.318 against 0.296 (above in every seed). A boosting ensemble was not run.
- Early stopping on blocked validation months picked epochs 1-8 (validation loss 10.48-10.59): the networks overfit after a few epochs (2,191 training days, few distinct storms); learning rate, regularisation and size were not tuned.
- Seeds: 5 at 0.25 degrees, 3 at 4 km (the design asks for 5); intervals over days, windows overlapping, few distinct storms in the large strata.
- Observed mode with gridMET (which also underlies the labels' era only through the same weather); the 4 km and 0.25 degree runs use different encoders' dilations; the exposure channels are in the image for H1 but H1 was not isolated (no run without them).
- Texas, scope S0, three test years; the validation months are blocked by month, not by year.


# Stage 43: V0/V1 of the latent route and data-sufficiency diagnostics (script 82, `scripts/81 --frac / --z`)

Question of `docs/CNN_HYBRID_DESIGN.md` section 10: is a pretrained latent code of the observed 4 km field feasible, does it carry what the labelled sample cannot teach, and how far is the labelled sample from sufficient? Setup: convolutional VAE (609 k parameters, stride 6 = one GEFS cell, 8 latent channels per cell, KL weight 1e-4, log rain, loss weighted towards heavy rain) on gridMET two-day windows (peak and sum, identical to `obs4k_day_maps`, checked on 20 days), pretrained on 2000-2016 (5,325 windows) or 2010-2016 (2,192), validation on blocked months, test years 2017-2019 never used. Trained on Colab (T4, 168 s per epoch, slower than the Mac MPS at 112 s) with checkpoint and resume; probes, V1 and the CNN runs on the Mac. Outputs `audit/latent_v0_*.txt`.

## Results

**V0, reconstruction on the test windows (1,095 windows)**

| | pretrained 2000-2016 | pretrained 2010-2016 |
|---|---|---|
| reconstructed / observed, pixels >= 25 mm | 0.990 | 0.994 |
| same, pixels >= 100 mm (mean absolute error) | 0.976 (9.0 mm) | 0.983 (13.1 mm) |
| reconstructed / observed county peak, >= 100 mm (correlation) | 0.936 (0.967) | 0.945 (0.957) |

County peaks are reconstructed 5-6% low (smoothing). Scale probe (R2 for the sub-cell maximum and standard deviation of log rain per 0.25 degree cell, test windows split in half): the cell mean alone gives 0.957 and 0.574 (boosting); mean + latent gives 0.985 and 0.898 / 0.901. Linear read-out differs by pretraining set (sub-cell standard deviation 0.428 with 2000-2016, 0.237 with 2010-2016). **The latent holds sub-cell structure beyond the cell mean.**

**V1, frozen latent (own cell + county mean, 16 features) added to the 17 observed summaries in the boosting model**: block-group top-5% capture -0.022 [-0.037, -0.008] (2010 version -0.020 [-0.033, -0.006]), NDCG with paid amount -0.009 [-0.014, -0.005], block-group AUC -0.002 [-0.010, +0.006], county AUC -0.001 [-0.003, +0.000]. **Gate failed as written**: no gain, and worse at the head of the list.

**V1c, the 8 latent channels as extra inputs of the 0.25 degree CNN** (3 seeds, against the same CNN without them; 2 seeds with shuffled latents of another day of the month as negative control):

| metric | CNN 0.25 deg | + latent | + shuffled latent | CNN 4 km |
|---|---|---|---|---|
| block-group AUC | 0.834 | -0.004 [-0.009, -0.000] | -0.007 [-0.010, -0.004] | +0.012 [+0.008, +0.017] |
| BG top-5% capture | 0.287 | +0.004 [-0.009, +0.017] | -0.002 [-0.015, +0.011] | +0.032 [+0.019, +0.046] |
| BG NDCG5 paid | 0.074 | +0.004 [-0.000, +0.008] | -0.005 [-0.009, -0.001] | +0.022 [+0.018, +0.027] |
| county AUC | 0.909 | -0.001 [-0.002, +0.001] | -0.004 [-0.005, -0.003] | +0.004 [+0.003, +0.005] |

The real latent beats the shuffled one (so it carries information) but adds nothing to the model without it, and recovers about 12% of the 4 km gain in top-5% capture (+0.004 of +0.032), inside the noise.

**Data sufficiency** (training sample 2010-2016 minus validation months: 2,191 days, 1,612 with a claim, about 110 claim episodes with gaps >= 3 days, 26,561 claimed block-group days in overlapping windows, 174 days with >= 30 claimed block groups). Supervised 4 km CNN trained on 25 / 50 / 100% of the training months (2 seeds, mean): top-5% capture 0.280 / 0.301 / 0.313, NDCG with paid amount 0.085 / 0.089 / 0.096, block-group AUC 0.831 / 0.833 / 0.845, county AUC 0.907 / 0.910 / 0.913: **still rising at 100%**, about +0.012 to +0.021 of top-5% capture per doubling. With a quarter of the months the CNN already exceeds the full-sample boosting model in NDCG (0.085 vs 0.079) and block-group AUC (0.831 vs 0.821). The boosting models are flat in the number of months (0.275 / 0.269 / 0.296, not monotone): they are limited by features, not by data. Training of the 0.25 degree network with latent channels overfits strongly (training loss 7.2, best validation loss 10.49 at an early epoch, validation 12.6 at epoch 20).

## Conclusions

- **The autoencoder is feasible but its code is not useful as built.** V0 passes (heavy-rain pixels within 1-2.4%, county peaks 5-6% low, sub-cell structure present in the latent), but the frozen code added to boosting hurts the head of the list (V1), and as channels of the 0.25 degree CNN it recovers none of the 4 km gain beyond noise (V1c). By the gate of section 10.6, V1 failed: **no diffusion model and no Transformer on this route for now.**
- **Likely reason (not tested):** the 4 km gain is position-specific: the block group reads its own pixel and neighbours. A latent that describes a 0.25 degree cell tells the model what fine structure exists in the cell, not where. For the forecast mode this is worse: where inside the cell the heavy rain falls is the part a 0.25 degree forecast cannot know, so a generator conditioned on the forecast would spread it, and the expected gain in the forecast mode stays small (consistent with stage 42 and E2 expectations).
- **Data are binding for the supervised network**: about 110 training episodes, the learning curve still rising and strong overfitting. More states (the group TX + LA + FL of stage 27) is the supported next step; the expected size of the gain is roughly +0.01 to +0.02 of top-5% capture per doubling of the sample, judged by two seeds and an extrapolation.
- Pretraining on more years (2000-2016) helps the reconstruction of the extremes (mean absolute error at >= 100 mm 9.0 vs 13.1 mm), with no effect on V1.

## Caveats

- The latent is the observed field's code (a ceiling); a forecast-side encoder was not trained. The link from latent cells to GEFS cells is approximate (the 6 x 6 gridMET blocks are offset from the GEFS cells by up to 0.08 degrees in longitude; each cell takes the latent cell holding its centre).
- Not tested: initialising the 4 km CNN's encoder from the VAE and fine-tuning end to end, a larger latent, a hierarchical latent with a 12 km level, other KL weights. The negative result is for a frozen 8-channel code at the forecast-cell scale.
- Learning curve: 2 seeds per fraction, no intervals, month subsets chosen by a hash; the CNN rows are the mean of two seeds (the 100% row uses seeds 0 and 1, so 0.313 differs from the 3-seed 0.319 of stage 42).
- The V1c baseline is 3 seeds of the 0.25 degree CNN (stage 42 used 5); validation losses of the two VAEs are not comparable (different validation months); Texas, scope S0, three test years.

# Stage 44: candidate-state scale-up, step 1 (inventory and inputs for LA and FL; scripts 22 `--box`, 24 `LA FL`, 83, `fetch_state_geography`)

Decision (user): go to the data scale-up of docs/CNN_HYBRID_DESIGN.md 10.7 (TX + LA + FL, stage 27) after the latent route failed (stage 43). This step builds the inputs; no model was trained.

## A. What was added

- `wxrisk.states` (state FIPS, box, file names); Texas files keep their old names, so every Texas output is unchanged.
- Block-group geography: TIGER 2010 and 2020 block groups and the Census 2020-to-2010 relationship file for LA and FL (`scripts/fetch/fetch_state_geography.py`); `bg_crosswalk.build(path)` takes a state's relationship file.
- Policies: `scripts/24 LA FL` extracted 8.46 M (LA) and 22.38 M (FL) policy records from the bulk file (Texas: 11.6 M) into `policies_la.parquet`, `policies_fl.parquet`.
- gridMET 4 km rain 2000-2023 for one LA + FL box (24.3-33.2 N, -94.2 to -79.8 E), `pr_lafl_<year>.nc`, 1.2 GB (`scripts/22 --box ... --tag lafl`). Verified for 2017: 365 x 196 x 346, 52% of the values are missing (ocean outside the gridMET land mask).
- `scripts/83_state_inventory.py` (output `audit/state_inventory.txt`).

## B. Inventory (inland claims = cause 2/3/4, any flood event name)

| | TX | LA | FL |
|---|---|---|---|
| 2010 block groups | 15,503 | 3,407 | 11,210 |
| policy records | 11.6 M | 8.46 M | 22.38 M |
| policy records with a block group; on a known crosswalk code | 99.7%; 100% | 99.7%; 100% | 99.7%; 100% |
| inland claims with a block group | 193,216 | 230,449 | 129,018 |
| claims, largest years | 2017: 62.3 k | 2016: 27.5 k, 2021: 14.8 k, 2012: 11.4 k | 2022: 23.9 k, 2017: 12.8 k, 2020: 6.6 k |

## Conclusions

- The block-group codes of LA and FL map through the Census relationship file as completely as Texas' (100% of coded policies and claims), so the label side has no new geography problem.
- **Claims concentrate in a few tropical years** (LA 2016 / 2012 / 2021; FL 2017 Irma / 2022 Ian). The GEFSv12 reforecast ends in 2019: the forecast-mode test years 2017-2019 hold the LA flood of 2016 only as training and, for FL, only Irma; Ida (2021) and Ian (2022) are outside the reforecast. Hence the forecast-mode data of the new states is thin in exactly the events that dominate their claims, while the observed-rain line (gridMET, to 2023) is not limited.
- Florida holds 22.4 M policy records (more than Texas): exposure tables need the same column-wise streaming as Texas, not a pandas read.

## Caveats

- Counts are before the tropical-name and tropical-window filters (scope S0 vs S2 is not applied yet), so the claim totals overstate the S0 labels.
- gridMET starts at 25.07 N: Florida Keys block groups south of it have no rain, they must be dropped or flagged.
- No clean-table specs exist yet for `policies_la` / `policies_fl` (`prep` is built for `policies_tx`), so exposure matrices for the new states are not built.
- Not done: GEFS reforecast crops for the new boxes, ERA5 for the Gulf box beyond what script 70 fetched, county masks, tropical-cyclone flags, day tensors, a state-held-out evaluation.

# Stage 45: observed-rain day products for LA and FL, boosting baseline pooled over TX + LA + FL, state-held-out check (scripts 84, 85)

Decision (user): start the scale-up with the observed-rain line. Scope S0 (rows inside a tropical-cyclone window dropped, label without tropical-named claims; S2 not yet run). Train 2010-2016, test 2017-2023 (gridMET reaches 2023). No CNN was trained.

## A. Inputs built

- Clean policy tables `policies_la`, `policies_fl` (same cleaning rules as Texas; `prep/specs.py`); `flood_pilot.exposure_matrices(table, crosswalk)`, `bg_geometry(zip)`, `load_precip(tag)`, `claims_state(abbr)` take the state as argument, Texas defaults unchanged.
- Script 84 writes per state `cnn_<st>/`: 5,109 days (2010-01-01 .. 2023-12-27) of 4 km peak / 2-day-sum maps, rows (block groups with >= 1 policy on the day: LA 16.7 M, FL 56.2 M, TX 63.8 M) with tabular features, a tropical-cyclone flag per row, claim counts in [S, S+3] for scope S0 and with tropical-named claims, and paid amount. In the rows' windows: LA 168.8 k claims in S0 (266.5 k with tropical-named), FL 70.0 k (266.1 k), TX 140.0 k (421.2 k); rows inside a cyclone window: LA 8.5%, FL 13.4%, TX 6.0%.
- Florida: 3 of 128,967 claims lie on block groups that the cartographic 2010 file lacks and were dropped.

## B. Boosting baseline (script 85; test rows of days with >= 1 claimed block group; days: TX 1,738, LA 1,487, FL 1,735)

Fit "own" = the test state alone, "tx" = Texas only, "pool" = TX + LA + FL, "loso" = the two other states. Differences from "own" with a bootstrap over days.

| Test state | metric | policies only | own | pool - own | loso - own |
|---|---|---|---|---|---|
| TX | block-group AUC | 0.732 | 0.822 | -0.036 [-0.056, -0.017] | -0.036 [-0.049, -0.024] |
| TX | BG top-5% capture | 0.183 | 0.295 | -0.002 [-0.013, +0.010] | -0.022 [-0.035, -0.010] |
| TX | BG NDCG5 paid | 0.027 | 0.082 | -0.002 [-0.006, +0.002] | -0.011 [-0.015, -0.006] |
| TX | county AUC | 0.885 | 0.911 | +0.002 [+0.001, +0.003] | -0.001 [-0.002, +0.001] |
| LA | block-group AUC | 0.646 | 0.748 | -0.046 [-0.082, -0.016] | +0.015 [+0.003, +0.026] |
| LA | BG top-5% capture | 0.123 | 0.210 | +0.018 [+0.007, +0.030] | +0.021 [+0.006, +0.038] |
| LA | BG NDCG5 paid | 0.027 | 0.074 | -0.008 [-0.015, -0.001] | -0.005 [-0.012, +0.003] |
| LA | county AUC | 0.783 | 0.801 | +0.006 [+0.003, +0.009] | +0.006 [+0.003, +0.009] |
| FL | block-group AUC | 0.746 | 0.830 | -0.028 [-0.046, -0.012] | -0.019 [-0.029, -0.009] |
| FL | BG top-5% capture | 0.158 | 0.283 | -0.002 [-0.015, +0.010] | -0.022 [-0.035, -0.009] |
| FL | BG NDCG5 paid | 0.020 | 0.074 | +0.005 [+0.001, +0.010] | -0.000 [-0.005, +0.005] |
| FL | county AUC | 0.831 | 0.852 | -0.003 [-0.005, -0.002] | -0.007 [-0.009, -0.005] |

(LA "tx - own": BG AUC +0.012, top-5% -0.004, NDCG -0.009, county AUC +0.004; FL "tx - own": top-5% -0.022, county AUC -0.007. Full table: `audit/state_group_baseline.txt`.)

## Conclusions

- **Pooling more states does not lift the boosting baseline.** Against the state's own fit the pooled fit is inside the noise or worse on most metrics (the block-group AUC drops by 0.03-0.05 in all three states); the only clear gains are LA top-5% capture (+0.018) and county AUC (+0.006), and one FL NDCG (+0.005). This agrees with stage 43: boosting on summary features is feature-limited, not data-limited, so more rows do not help it. The data hunger sits in the CNN, which has not been trained on the pooled rows yet.
- **The summaries transfer between states.** A model trained on the other two states is within a few hundredths of the own fit (LA even above it on AUC and top-5% capture), and every state is far above the policies-only ranking (top-5% capture 0.12-0.18 to 0.21-0.30). What the boosting learns is a property of the rain-exposure relation, not of Texas.
- Louisiana has the weakest signal (policies-only county AUC 0.78, top-5% capture 0.21): its claims sit in a few large floods (2016, 2012) over a flat landscape.

## Caveats

- Boosting only, one seed, observed rain, scope S0; S2 (tropical windows kept) is where Florida's claims are, and was not run. The S0 label removes 74% of Florida's claims (266 k to 70 k).
- Test days keep only days with a claimed block group, as before; the metrics are conditional on an event having happened.
- The block-group AUC falls with pooling because the pooled model fits a different base rate per state; it is a cross-county AUC inside a day of a single state, so a calibration shift between states should not matter, but the boosting does not know the state either.
- LA's own fit stops at 25 iterations (about 800 k training rows, few events), so "own" is a weak reference there.
- No forecast rain: the test years run to 2023 only because gridMET does; the GEFS reforecast (2019) and ERA5 for the new boxes are not used.

# Stage 46: why the 4 km CNN beats boosting with so little data (script 86; script 81 `--variant`)

Question (user): a 55 k-parameter CNN with a best validation epoch of 2-3 and a quarter of the months already above boosting looks like it reads sub-pixel structure; what does it actually do? Setting as stage 42: Texas, observed rain, scope S0, train 2010-2016, test 2017-2019 (778 test days with >= 1 claimed block group), the 4 km model retrained for 5 epochs, 2 seeds per variant (`full` = the stage-42 model). Metrics as scripts 75 / 81; differences against `full` with a bootstrap over days. `audit/cnn_mechanism*.txt`.

## A. One-change retrainings (seed means)

| Variant | BG AUC | BG top-5% | BG NDCG5 | county AUC | AUC inside a county-day |
|---|---|---|---|---|---|
| policies only | 0.742 | 0.159 | 0.026 | 0.880 | 0.668 |
| **full** | 0.843 | 0.322 | 0.098 | 0.912 | 0.763 |
| nocoord (lat / lon channels zeroed) | 0.842 | 0.318 | 0.093 | 0.911 | 0.759 |
| nostatic (exposure maps zeroed; tabular branch kept) | 0.849 | 0.323 | 0.101 | 0.915 | 0.762 |
| bce (no listwise term) | 0.848 | 0.312 | 0.095 | 0.913 | 0.762 |
| coarse_rain (rain = 6 x 6 block means, train and test) | 0.841 | 0.305 | 0.086 | 0.911 | 0.757 |
| shuffle_rain (rain of another day, same month) | 0.779 | 0.204 | 0.044 | 0.885 | 0.732 |

Differences from `full` with intervals: nocoord top-5% -0.004 [-0.015, +0.008]; nostatic +0.002 [-0.010, +0.013] (NDCG +0.003 [+0.000, +0.005]); bce -0.010 [-0.019, -0.001] top-5%, +0.005 [+0.002, +0.009] BG AUC; coarse_rain -0.017 [-0.027, -0.007] top-5%, -0.012 [-0.016, -0.008] NDCG; shuffle_rain -0.118 [-0.140, -0.097]. Seed spread of `full`: top-5% 0.320 / 0.323.

## B. Boosting given what the CNN has (feature ladder; same rows as stage 42; iterations on blocked validation months, cap 600)

| Boosting | BG AUC | BG top-5% | BG NDCG5 | county AUC | inside a county-day | iterations |
|---|---|---|---|---|---|---|
| stage-42 summaries (base) | 0.813 | 0.272 | 0.076 | 0.905 | 0.742 | 600 (cap) |
| + pixel coordinates | 0.814 | 0.276 | 0.080 | 0.911 | 0.733 | 310 |
| + multi-scale neighbourhood mean / max (3-33 px) | 0.793 | 0.282 | 0.072 | 0.907 | 0.720 | 241 |
| + day context (domain max, rank among the day's rows, peak minus county mean, rows > 25 mm) | 0.797 | 0.288 | 0.079 | 0.907 | 0.736 | 154 |
| + all three | 0.822 | 0.284 | 0.083 | 0.908 | 0.727 | 56 |
| + all three, day-normalised weights | 0.832 | 0.289 | 0.084 | 0.912 | 0.749 | 109 |
| no rain, tabular rows | 0.748 | 0.204 | 0.041 | 0.882 | 0.718 | |
| no rain, tabular rows + coordinates | 0.717 | 0.202 | 0.041 | 0.884 | 0.700 | |

All the enrichment together closes about 34% of the top-5% gap to the CNN (0.272 to 0.289 against 0.322), 36% of the NDCG gap, 63% of the BG AUC gap, 33% of the within-county AUC gap.

## C. Test-time probes of the trained full CNN (seed 0)

- **Blur is free, shift is graded.** Gaussian blur of the rain channels with sigma 1-2 px (4-8 km) changes nothing (top-5% 0.322 / 0.315 against 0.320); sigma 4 px -0.012. Shifting the rain field by 1 / 2 / 4 / 8 / 16 px moves top-5% by -0.009 / -0.022 / -0.051 / -0.067 / -0.092: a smooth decay over 10-60 km, no collapse at one pixel. Block mean 6 x 6 at test time -0.014. Rain removed: top-5% 0.207, BG AUC 0.774.
- **Where the head reads.** Zeroing the head inputs: own-pixel embedding -0.101 top-5% (BG AUC -0.058), the tabular branch -0.108 (BG AUC -0.160); the county mean and maximum of the embedding change nothing (dropping the county maximum even helps the county top-10%, +0.015 [+0.008, +0.023]).
- **Gradient footprint** of a claimed block group's logit on the two rain channels (278 block groups, share of |gradient| by distance, 1 px = 4 km): < 1 px 0.03, 1-2 px 0.18, 2-4 px 0.14, 4-8 px 0.15, 8-16 px 0.17, 16-32 px 0.13, 32-64 px 0.03, beyond 0.18. The signed share is positive inside 4 px (+0.17 of 0.35) and almost cancels from 4 px outwards (+0.03 of 0.47).

## Conclusions

1. **It does not read sub-pixel structure.** gridMET is itself a 4 km product, and a 4-8 km blur of the input does not move the score. The fine detail that matters lies between 4 km and the 0.25 degree cell: training on 6 x 6 block means costs 0.017 of top-5% capture (5%), and that coarse-rain CNN (0.305) is still above every boosting variant (best 0.289). Sub-0.25 degree structure is worth about a third of the CNN's lead over boosting; the rest is how the rain is read.
2. **It is not memorised geography.** Dropping the coordinate or the exposure channels does not hurt, and the CNN with shuffled rain (static knowledge only, top-5% 0.204) equals boosting without rain (0.204): the static part is the same for both. The whole difference is in the use of rain (+0.118 for the CNN, +0.07-0.09 for boosting).
3. **It is not the loss and not the county pooling.** A plain binary loss gives the same ranking quality (-0.010 top-5%, +0.005 BG AUC); the county mean / max of the embedding carry no skill.
4. *(Corrected by stage 47: the reach is mostly 2-3 px; the "tens of kilometres" below came from summing small gradients over a wide area.)* **What it does is a learned neighbourhood read at the block group**: one embedding per pixel from a receptive field of tens of kilometres, sensitive to rain 10-60 km away in a graded way, positive close by and of mixed sign further out (a local-contrast signature: rain here relative to the surroundings), plus the tabular branch. The hand-built version (multi-scale means / maxima, day context, coordinates, day-normalised weights) recovers about a third of the gap at the head of the list and most of the BG AUC gap, but not the within-county ordering.
5. **Why it is data-efficient (partly inferred).** The 55 k parameters are shared over all positions, so the 2,191 training days supply about 180 million pixel-days to the same filters while the labels act only at claimed block groups; with 5 epochs it stops early. Boosting with more features gets worse on the test years while its validation loss improves (+multi-scale: validation loss 0.0141 to 0.0115, test BG AUC 0.813 to 0.793; +all three stops at 56 iterations): its extra freedom fits year-specific structure that the convolution prior does not.

## Caveats

- Texas, S0, observed rain, three test years (778 days), two seeds per variant (seed spread of the top-5% is about 0.003-0.017, comparable to the smaller differences above), one architecture; nothing here concerns forecast rain.
- The test-time probes feed the trained network inputs it never saw (shifted or blurred fields); part of their degradation is out-of-distribution. The coarse-rain retraining is the clean version of that test.
- The gradient footprint is a local linearisation of a nonlinear network on 278 samples; the mixed-sign reading (contrast detection) is an interpretation, not tested by constructing a contrast feature.
- Boosting was not tuned beyond the iteration count (depth 5, leaf 300, base hit the cap of 600), and the hand-built multi-scale features are one choice of filters; "boosting cannot" is not shown, only "these features recover a third". Whether boosting on the CNN's own embedding reproduces its skill (representation versus learning procedure) was not run.
- The explanation of data efficiency in conclusion 5 is an inference from parameter sharing, early stopping and the boosting overfitting pattern, not a controlled test (the learning curve of stage 43 shows the CNN still rising and boosting flat).


# Stage 47: which neighbourhood patterns the 4 km CNN learned: filter groups and activation diagnostics (script 87, `87_cnn_filters_plot.py`)

Model: the stage-46 `full` CNN, seeds 0 and 1 (stem 8 -> 32, five dilated residual blocks, 1 x 1 to a 16-channel embedding). Figures in `docs/figures/cnn_filters/*_s0.png` / `_s1.png`; text output `$WXRISK_DATA/audit/cnn_filters_s<seed>.txt`.
Seven diagnostics: stem kernels split into DC / gradient / centre-surround; activation statistics of every layer on 100 training days (dead channels, correlation with rain at 1-33 px and with centre-minus-surround); mean input-gradient footprint of each embedding channel at 240 claimed block groups (test days); 21 synthetic rain patterns (point, blobs sigma 1-8 px, ring, hole, bands, edges, uniform +- spike) put at two busy block groups on a dry and a 15 mm background; response map of the logit to a 50 mm blob (sigma 1.5 px) moved over +-40 px; mean rain crop around the 200 most / least active pixels of each channel; replacing each channel (and each k-means group of channels) of the embedding map by its training mean before the county pooling, scored on the 778 test days (bootstrap over days).

## Results

- **The embedding is essentially one direction.** The first principal component of the own-pixel embedding holds 85% (seed 0) / 91% (seed 1) of the variance, three hold 97% / 96%. Every embedding channel is, up to sign, a rain-at-the-pixel channel with a different gain: 10 of 16 channels (14 of 16 in seed 1) correlate 0.5-0.86 with the local rain peak, the same at 1, 3, 9 and 33 px scale (gridMET day fields are smooth, so correlation cannot tell the scales apart), and only 0.0-0.2 with centre-minus-surround. No channel is dead in any layer.
- **Filter groups are not distinct.** K-means on the stem kernels gives k = 2 (silhouette 0.19) for seed 0 and k = 4 (0.2-0.3) for seed 1; on the embedding channels k = 2 for both seeds (silhouette 0.24 / 0.37). The groups do not match between seeds (stem), and the PCA projections show a continuum; I treat them as a descriptive partition. The only stable split is in the embedding: a "rain" group (10 channels in seed 0, 14 in seed 1, centre-weighted positive footprint) and a small remainder (6 / 2 channels) with sign-mixed footprints that carries nothing.
- **Skill sits in that rain group and in no single channel.** Replacing the rain group by its mean costs top-5% -0.099 [-0.123, -0.075] (seed 0) / -0.114 [-0.136, -0.089] (seed 1), BG AUC -0.064 / -0.071, NDCG5 -0.053 / -0.052; the remainder group changes nothing (+0.001 / +0.002, inside the interval). Replacing any single channel costs at most 0.003 top-5% (all intervals contain 0) and even raises BG AUC by 0.001-0.004 for several channels: the channels are redundant copies.
- **The stem kernels do not separate "contrast" from "smoothing" filters.** Rain takes only 17-36% of each filter's kernel norm (exposure 33-40%, latitude / longitude 22%, valid mask 12-15%); 9 of 32 peak-rain kernels have a net weight below 30% of their L1 norm (mixed sign), the rest are mostly positive or mostly negative. Both signs of the DC term exist (seed 0: 9 filters +0.30, 23 filters -0.33); with GroupNorm after the stem a negative DC is a sign flip, not a different feature.
- **The effective footprint is small.** The mean input gradient of the embedding at a claimed block group lies within 2-3 px (8-12 km) in every channel; the signed share beyond 8 px is below 3%. The response of the logit to a 50 mm blob moved around the block group (dry background): +3.2 (seed 0) / +3.4 (seed 1) at the pixel, mean +1.2 / +1.7 within 3 px, +0.15 / +0.16 at 3-8 px, +0.07 / +0.04 at 8-16 px, none beyond 16 px (both sites, both backgrounds; on the wet background the effects are 0.6-0.8 of those). Stage 46's gradient shares at 8-64 px were a sum of tiny per-pixel values over a wide area; the nonlinear map shows that rain 8+ px away moves the logit by less than 0.1 against 3 for rain at the pixel.
- **Part of the weak outer response is directional: rain to the west counts, rain to the east does not.** Mean logit change of a blob 4-20 px away, west / east / north / south sectors: +0.17 / +0.00 / +0.03 / +0.01 (site 0 dry, seed 0), +0.16 / +0.02 / -0.02 / +0.08 (site 1); seed 1 +0.11 / +0.01 / +0.04 / 0.00 and +0.08 / +0.01 / -0.01 / +0.04; wet backgrounds the same sign. The mean input gradient west of the block-group column is 3.3 times that east of it (seed 0). Registration is not the reason (the block-group centroid lies within 0.5 px of its pixel centre, mean offset 0.00 px).
- **A lone wet pixel and a wet pixel with dry neighbours do not count.** One 50 mm pixel at the block group: logit change -0.09 / -0.17 (seed 0), +0.25 / +0.22 (seed 1) against +3.6 for a blob of sigma 2 px. A wet ring 6-8 px away with a dry centre: +0.2; wet disc of 12 px radius with a dry 1.5 px centre: -0.4 / -0.7 (a dry centre inside rain is ranked below nothing at all). Blob sizes sigma 2 / 4 / 8 px give 3.6 / 3.8 / 3.6: saturation by 8 km. Bands through the block group (any of 0, 45, 90 degrees) give 3.2-3.5, bands 8 px beside it 0.0-0.5. Uniform 25 mm over the whole domain: +2.6 (dry background, seed 0) / +1.4 (seed 1); the edge probes (half-plane wet to the west / east / north / south) are seed-dependent (seed 0 +2.5 / +0.35 / +1.8 / +1.8; seed 1 +1.6 / +1.1 / +1.4 / +1.9) and are not read here.
- **Extreme-activation crops are rain at the centre, not structure.** The mean rain around the 200 most active pixels of a rain channel is 1.7 at the centre (log1p(mm)/3, about 160 mm) against 0.5 around claimed block groups on average and 0.8-1.1 at 8-16 px: the extremes are heavy-rain pixels in heavy-rain neighbourhoods, and the least active pixels of the sign-flipped channels are the same crops.

## Conclusions

1. **What the CNN learned is a local rain-intensity read with a spatial-smoothness requirement, not a bank of different neighbourhood patterns.** The 16-dimensional embedding is one direction (85-91% variance) of pixel rain scaled by about 8-12 km of support: one isolated pixel is ignored, a 2-px blob counts fully, more support adds nothing.
2. **The two reliable extras beyond the pixel are a weak west-sided reach of 4-16 px (0.1-0.2 logit against 3 for the pixel) and a penalty for a dry centre inside rain.** Both replicate in direction in both seeds, but their size is 1-5% of the pixel response; they do not explain the CNN's lead over boosting by themselves.
3. **Together with stage 46 this narrows the lead's source.** Block-group AUC / top-5% gains over boosting that survive a 4-8 km blur and a coarse-rain retraining (0.305) are not coming from long-range structure. A local, smooth, saturating, exposure-weighted response of the pixel (stem kernels put 60-80% of their norm on exposure, coordinates and the mask, only 17-36% on rain) is the most economical description; whether boosting with such a hand-built smoothed feature closes the gap is not tested.
4. **No single channel or filter is important; the rain direction is.** Single-channel ablations are within noise (largest +-0.003), the rain group -0.10 to -0.11.

## Caveats

- Texas, S0, observed rain, two seeds; the group partitions and the edge probes differ between seeds, the pixel / blob / ring / hole / west-bias results replicate.
- The synthetic probes are inputs the network never saw (a 50 mm Gaussian blob on an empty domain is not a storm); the numbers are logit responses of the trained model, not hydrologic response. A single block group row (median tabular features of two busy block groups) is used per site, so the response is conditional on that exposure.
- The west-sided reach is shown by a model response, not explained: it may be storm motion (rain to the west of a block group arrives at S+1 to S+3, inside the claim window while the rain channels cover S and S+1), drainage routing (the Texas rivers run towards the east and south-east), or a shortcut in the data; none of the three is tested.
- The mean gradient is a linearisation over 240 samples and per-pixel; its summed |gradient| over large areas (stage 46) overstates reach, as seen here. The "one direction" statement is about the own-pixel embedding at block-group rows of claim days; the county pooling inputs carry nothing (stage 46).
- Channel ablation replaces a channel by its training-day mean, an input the head may not have seen in combination; redundant channels would show no effect even if each carried skill when used alone, so single-channel nulls say "redundant", not "unimportant".


# Stage 48: coarse-to-fine multi-scale CNN with a learned vote (script 88, `cnn.MultiScale`)

Stage 47 found a local, smooth, saturating rain read. Question: do explicit coarse scales add anything, and how would a weighted vote across scales behave? `MultiScale`: one small CNN per scale s in {1, 2, 4, 8, 16} px (the day's stack average-pooled by s, embedding brought back bilinearly), a head per scale (own-pixel embedding + tabular branch), a gate that returns per-row softmax weights over the scales (input: tabular branch and all embeddings), output = weighted sum of the scale logits. Loss: listwise + binary loss of the vote + 0.5 x the mean of the same loss on each scale's own logit (every scale is also a stand-alone predictor). No county pooling. 54.7 k parameters (matched to the 55 k of stage 46). Controls: `single` (scale 1, same widths, 12 k parameters) and `single_wide` (scale 1, width 40, 50.7 k parameters). Same data, split, sampling, optimiser, 5 epochs; two seeds each; test: 778 days with claims, Texas, S0, observed rain. Validation losses of `ms` (15.8) contain the auxiliary terms and are not comparable with the others (10.5).

## Results (seed means; difference to the stage-46 `full` CNN with a bootstrap over days)

| model | BG AUC | BG top-5% | BG NDCG5 | county AUC | county top-10% |
|---|---|---|---|---|---|
| policies only | 0.742 | 0.160 | 0.026 | 0.880 | 0.770 |
| full (stage 46) | 0.843 | 0.322 | 0.098 | 0.912 | 0.824 |
| ms vote | 0.841 (-0.002 [-0.004,-0.001]) | 0.320 (-0.002 [-0.013,+0.007]) | 0.095 (-0.003 [-0.006,+0.001]) | 0.913 (+0.002 [+0.001,+0.003]) | 0.836 (+0.012 [+0.005,+0.021]) |
| single (12 k) | 0.841 | 0.310 (-0.011 [-0.018,-0.005]) | 0.094 | 0.912 | 0.831 (+0.007 [-0.001,+0.016]) |
| single_wide (51 k) | 0.842 | 0.316 (-0.005 [-0.013,+0.004]) | 0.095 | 0.913 | 0.831 (+0.007 [-0.002,+0.016]) |

- **The vote does not beat the full-resolution models at block-group level.** `ms` is within the interval of `full` and of `single_wide` on top-5% and NDCG; block-group AUC is 0.001-0.002 lower. Per seed, `ms` top-5% is 0.303 / 0.318 against 0.309 / 0.316 for `single` and 0.308 / 0.315 for `single_wide`: the seed spread is larger than any difference.
- **Each scale alone, from fine to coarse, degrades monotonically** (BG top-5% / NDCG5): 1 px 0.325 / 0.098, 2 px 0.310 / 0.092, 4 px 0.312 / 0.084, 8 px 0.306 / 0.071, 16 px 0.281 / 0.065. The 16 px branch alone (rain averaged over 64 km) still has top-5% 0.281, above boosting without neighbourhood features (0.272, stage 46), because the exposure and tabular inputs carry most of that.
- **Coarse-to-fine cumulative vote (learned weights renormalised over the coarsest k scales)** rises step by step: 16 px 0.282, +8 px 0.291, +4 px 0.308, +2 px 0.310, +1 px (the full vote) 0.320 in top-5%; NDCG5 0.066, 0.070, 0.078, 0.092, 0.095. Each finer scale adds; the finest adds the most. An equal-weight vote of the five scales is worse in NDCG5 (0.087) than the learned vote.
- **The learned gate drops the coarse scales.** Mean weights are 0.500 (1 px), 0.500 (2 px) and 0.000-0.001 for 4, 8, 16 px, in every rain class (< 1 mm to >= 50 mm) and for claimed rows; the gate output is nearly constant over rows, so the vote is a fixed average of the two finest branches.
- **At county level the coarse scales help a little:** county top-10% is 0.836 for the vote (+0.012 [+0.005,+0.021] over `full`, but +0.005 over `single_wide` with an interval that was not computed), 0.839 for the 4 px branch alone; county AUC differences are at most 0.003.

## Conclusions

1. **At 4 km observed rain, extra coarse scales give no gain at the block group.** The vote collapses onto the 1-2 px branches, and every coarser branch alone is worse and adds less. This agrees with stage 47 (reach 2-3 px) and with the stage-46 result that hand-built 3-33 px neighbourhood features recover only a third of the lead.
2. **The pooling question has a clean answer for this data: not needed.** No pooling-based branch, weighted by a gate or averaged equally, improved the ranking. The scale analysis itself is useful as a diagnostic: the cumulative coarse-to-fine ladder shows where the skill sits (the finest scale).
3. **Possible county-level value** of coarse scales (top-10% counties +0.005 to +0.012) is suggestive only: it is not separated from the single-scale controls and the metric is the one with the widest intervals.

## Caveats

- Texas, S0, observed rain, two seeds, one gate design. The gate output is nearly constant, so the result cannot distinguish "coarse scales carry no extra information" from "this gate and loss do not learn to use them" (e.g. a gate with per-scale supervision on the vote, or a coarse-to-fine residual design, were not tried).
- Average pooling mixes the valid mask and the rain channels at coastal and border pixels; a masked pooling was not used.
- `single` and `single_wide` are not the same architecture as `full` (no county pooling, 16 or 40 channels, three dilated blocks instead of five), so `ms` versus `full` also changes those; the controls are the fair comparison, and `ms` does not beat them outside the seed spread.
- The scale analysis concerns the 4 km observed field. Forecast-mode inputs at 0.25 degrees (or ensemble spread across members) have a different scale structure and were not tested.


# Stage 49: pooled TX + LA + FL 4 km CNN, own-state and state-held-out fits (script 89; `audit/cnn_pooled_S0.txt`)

Question (user): can a convolutional encoder replace the hand-built rain features and map the inputs of several states into one decision space? Setting: the stage-45 rows and boosting baseline (script 85, scope S0, train 2010-2016, test 2017-2023 days with >= 1 claimed block group: TX 1,738, LA 1,487, FL 1,735 days), the `cnn.Hybrid` of stage 42 (53.6 k parameters, listwise + 0.3 x BCE, 5 epochs, best validation epoch kept) on **rain only** (log peak, log 2-day sum, validity mask; coordinates and exposure maps dropped because stage 46 showed they add nothing) plus the 9 tabular columns. County pooling uses the block-group pixels of each county. Fits: `pool` (TX+LA+FL), `own` (test state only), `loso` (the other two states), `tx` (Texas model on LA / FL); 2 seeds each, seed-mean predictions; paired differences with a bootstrap over days. Boosting baselines are the stage-45 fits with the same names.

## Results (point value; top-5% = block-group top-5% capture of claim count, NDCG5 with paid amount)

| State | Metric | boosting own | boosting pool | boosting loso | CNN own | CNN pool | CNN loso |
|---|---|---|---|---|---|---|---|
| TX | BG AUC | 0.822 | 0.786 | 0.786 | 0.838 | 0.836 | 0.817 |
| | top-5% | 0.295 | 0.293 | 0.273 | 0.322 | 0.324 | 0.291 |
| | NDCG5 paid | 0.082 | 0.080 | 0.071 | 0.099 | 0.099 | 0.081 |
| | county top-10% | 0.796 | 0.794 | 0.794 | 0.804 | 0.797 | 0.786 |
| LA | BG AUC | 0.748 | 0.702 | 0.763 | 0.780 | 0.784 | 0.777 |
| | top-5% | 0.210 | 0.228 | 0.231 | 0.272 | 0.259 | 0.252 |
| | NDCG5 paid | 0.074 | 0.066 | 0.069 | 0.099 | 0.091 | 0.086 |
| | county top-10% | 0.498 | 0.489 | 0.503 | 0.513 | 0.515 | 0.517 |
| FL | BG AUC | 0.830 | 0.803 | 0.812 | 0.841 | 0.845 | 0.820 |
| | top-5% | 0.283 | 0.281 | 0.261 | 0.283 | 0.312 | 0.275 |
| | NDCG5 paid | 0.074 | 0.079 | 0.073 | 0.076 | 0.084 | 0.074 |
| | county top-10% | 0.564 | 0.552 | 0.554 | 0.530 | 0.553 | 0.553 |

Paired differences in top-5% capture [95% over days]:
- CNN pool - boosting pool: TX +0.031 [+0.018, +0.044], LA +0.031 [+0.019, +0.043], FL +0.031 [+0.018, +0.044] (BG AUC +0.04 to +0.08 in all; NDCG5 paid +0.019 TX, +0.025 LA, +0.005 FL [-0.000, +0.010]; county top-10% +0.003 TX [-0.002, +0.008], +0.026 LA, +0.001 FL [-0.008, +0.011]).
- CNN own - boosting own: TX +0.027 [+0.013, +0.039], LA +0.061 [+0.047, +0.077], FL -0.001 [-0.014, +0.011] (FL county top-10% -0.034 [-0.044, -0.024]).
- CNN loso - boosting loso: TX +0.018 [+0.004, +0.031], LA +0.021 [+0.007, +0.034], FL +0.013 [+0.001, +0.028].
- CNN pool - CNN own: TX +0.002 [-0.006, +0.011], LA -0.013 [-0.022, -0.004], FL +0.029 [+0.017, +0.042] (FL NDCG5 +0.008 [+0.005, +0.012], county top-10% +0.023 [+0.013, +0.032]).
- CNN loso - CNN own: TX -0.031 [-0.043, -0.019], LA -0.020 [-0.033, -0.006], FL -0.008 [-0.021, +0.006] (BG AUC -0.021 in TX and FL).
- CNN tx - CNN own: LA -0.021 [-0.035, -0.008], FL -0.030 [-0.043, -0.017].

## Conclusions

1. **For the rain part, the encoder replaces the hand-built features.** With identical training sets the rain-only CNN is above boosting on the five rain summaries at the head of the block-group list in all three states and in all three regimes (pool: +0.031 top-5% everywhere; own: TX, LA; loso: all three), and on block-group AUC. At county level it ties boosting (TX, FL) or is slightly better (LA). The tabular exposure / history columns were kept as hand features, so this is a statement about rain features only.
2. **One network for several grids works, but pooling does not reliably add skill.** The pooled model equals the own-state model in TX (already data-rich), is worse at the head in LA (-0.013) and clearly better in FL (+0.029 top-5%, +0.023 county top-10%). The extrapolation of stage 43 (+0.01-0.02 per doubling) is supported only in FL; the pooled gain is state-dependent. Boosting gets worse in block-group AUC when pooled (stage 45), the CNN does not.
3. **The shared space transfers only partly.** A model that never saw the test state loses 0.01-0.03 of top-5% capture (and 0.02 of BG AUC in TX and FL) against its own-state fit, yet stays above boosting trained the same way. Point values: the held-out-state CNN is above boosting's own fit in LA (0.252 vs 0.210), about level in TX (0.291 vs 0.295) and FL (0.275 vs 0.283); these three comparisons were not tested for significance. So the rain-to-risk mapping is largely shared across the Gulf states, and a state-specific part (exposure / reporting regime / drainage) remains.

## Caveats

- Scope S0, observed rain (gridMET), 2 seeds per fit (per-seed spread not tabulated here); test years 2017-2023. Nothing here concerns forecast rain, scope S2 or cross-state ranking (every event is a state-day).
- Probabilities are not comparable across fits (mean predicted rate differs by an order of magnitude between pool and own fits); only rankings are evaluated.
- The pooled and own fits differ in training days (TX 3x LA), hence in the number of steps and the schedule; one-cycle length follows the data. A second source of difference with stage 42: county pooling here uses block-group pixels, not county polygons (stage 46: pooling carries no skill).
- "Shared decision space" is judged only by transfer metrics; no probe checked whether the 16-dim embedding is state-invariant. FL's county top-10% is worse for the own CNN than for boosting (-0.034) so the CNN advantage is not uniform.
- The boosting baselines were not retuned for the pooled data and use hand-built rain summaries only (stage 46: richer hand features recover about a third of the gap).


# Stage 50: is the CNN embedding one shared decision space across states? (script 90; `audit/embedding_shared_pool_s0*.txt`)

Question (user, follow-up of stage 49): the pooled model beats the hand rain features, but does its 16-dim own-pixel embedding mean the same thing in TX, LA and FL? Frozen encoder of the pooled fit (seed 0, S0, tabular branch and head not used); 250 training-year and 250 test claimed days per state (train rows: all claimed block-group days + 5% of the others, weight 20; test rows: all rows of the days: TX 3.0 M, LA 0.8 M, FL 2.7 M). Reference = three hand summaries of the pixel (log peak, log 2-day sum, log 5 x 5 maximum).

## Results

**A. Linear read-out transfer** (row AUC | mean within-day AUC; fitted in the row state, scored in the column state)

| embedding | TX | LA | FL |
|---|---|---|---|
| fitted in TX | 0.782 \| 0.644 | 0.722 \| 0.592 | 0.781 \| 0.631 |
| fitted in LA | 0.771 \| 0.574 | 0.742 \| 0.587 | 0.735 \| 0.548 |
| fitted in FL | 0.796 \| 0.653 | 0.720 \| 0.595 | 0.790 \| 0.631 |

The hand summaries give 0.785 \| 0.614 (TX), 0.722 \| 0.592 (LA), 0.782 \| 0.600 (FL) whichever state they are fitted in (the read-out is practically monotone in the peak). TX- and FL-fitted embedding read-outs transfer with no loss (TX to FL 0.781 against FL's own 0.790; FL to TX 0.796 against 0.782); the LA-fitted one (smallest sample, 2,815 claimed rows) transfers worse (within-day 0.574 in TX, 0.548 in FL against 0.644 / 0.631 for the others). Within-day, the embedding read-out is +0.03 above the hand summaries in TX and FL and level in LA.

**B. State identifiability** (balanced accuracy of a boosted 3-class classifier, chance 0.333): **embedding 0.897, hand summaries 0.414**. Inside bins of the pixel's peak it is the same, 0.887-0.899 from < 1 mm to >= 100 mm (hand summaries 0.405-0.420). **Control**: with the rain channels zeroed at the encoder input (validity mask only), the state is identified at 1.000 and PC1 has no relation to rain (Spearman 0.02): the mask alone carries a location code.

**C. Geometry**: PC1 of the embedding holds 71% / 74% / 50% of the variance (TX / LA / FL) and orders with log peak (Spearman 0.78 / 0.85 / 0.68). The PC1 directions agree between states (|cosine| TX-LA 0.97, TX-FL 0.90, LA-FL 0.98). Median PC1 coordinate by peak bin (scaled by each state's 1st-99th percentile range): 0.08 / 0.17 / 0.27 / 0.40 / 0.62 / 0.96 in TX, 0.06 / 0.13 / 0.20 / 0.29 / 0.48 / 0.79 in LA and 0.17 / 0.24 / 0.31 / 0.42 / 0.59 / 0.89 in FL for bins 0-1 / 1-5 / 5-12.5 / 12.5-25 / 25-50 / 50-100 mm: the same monotone curve up to a state offset and slope.

## Conclusions

1. **The rain-related part of the code is shared.** The main embedding direction has the same orientation in all three states (cosine 0.90-0.98), maps rain intensity to a similar monotone curve, and a read-out fitted in one state predicts claims in another about as well as in its own state (TX and FL fitted read-outs).
2. **The embedding as a whole is not state-free.** A classifier recovers the state from it at 0.90 in every rain class, including dry pixels; the hand summaries recover it at 0.41. With rain removed the code is a pure location code, so the encoder keeps a geometry component (via the validity mask and the receptive field around the coast and domain edges) next to the rain component. The head can use it as a state/region intercept; this is a plausible reason why the state-held-out fit loses 0.01-0.03 (stage 49), but that link was not tested.
3. **So "mapping several inputs to one decision space" holds for the rain direction, and what it adds beyond three hand summaries is small in the linear read-out** (+0.03 within-day AUC in TX and FL, none in LA); the CNN's gain in stage 49 comes from the nonlinear head with the tabular branch and from the local contrast structure (stage 47), which a linear read-out of one pixel does not show.

## Caveats

- One model (pooled fit, seed 0), 250 + 250 days per state, one random draw; a first run with 120 days gave visibly different transfer numbers (TX-fitted read-out to FL 0.633 against 0.781), so differences of a few hundredths in A are not reliable. LA has few claimed rows.
- The norain control is an out-of-distribution input (the network never saw an all-zero rain field); it shows that geometry is present in the code, not how much the head uses it. Whether removing the geometry component (mask dropped, or the state direction projected out) changes skill or the held-out-state gap was not run.
- Probes use the own-pixel embedding only; the neighbourhood read and the tabular branch of the full model are not part of this test, and nothing here concerns forecast rain.


# Stage 51: multi-source neural encoder (src/wxrisk/encoders.py, script 91; `audit/multisource_encoder.txt`)

Question (user): replace the rule-based assembly of heterogeneous inputs by one learned encoder per source, fused into one embedding per block group. Implementation: `MultiSourceNet` (fine 4 km sources through a 1 x 1 projection + norm each and a shared dilated trunk; coarse 0.25 degree sources through their own trunk, put on the fine grid by a cell index and applied as FiLM, identity at initialisation; row sources through numeric + categorical-embedding MLP; `SetPool` for per-record sets; optional modality dropout; 8 unit tests). Setting: the Texas E1 data of stage 42 (all-days unit, S0, train 2010-2016, test 2017-2019, 1,091 test days), loss, schedule and validation of script 81, 5 epochs, 2 seeds per configuration, no county pooling, row branch always the 9 tabular columns. Sources: `rain` observed gridMET (3 channels), `expo` exposure maps (3), `fc` GEFS ensemble mean / member peaks (7 + mask, issue time), `era5` moisture and wind (6 + mask, valid time: ceiling only), `geo` NFHL shares + HAND statistics of the block group (9 row features, script 58). Differences against the stage-42 4 km CNN (3 seeds) with a bootstrap over days.

## Results (point value; BG = block group)

| Sources | BG AUC | BG top-5% | BG NDCG5 paid | county AUC | county top-10% |
|---|---|---|---|---|---|
| stage-42 CNN 4 km (hand-assembled input) | 0.846 | 0.319 | 0.097 | 0.913 | 0.832 |
| boosting, observed summaries | 0.821 | 0.296 | 0.079 | 0.909 | 0.824 |
| **rain** | 0.844 | 0.319 | 0.096 | 0.912 | 0.834 |
| rain + expo | 0.842 | 0.313 | 0.093 | 0.912 | 0.831 |
| rain + geo | 0.843 | 0.327 | 0.097 | 0.912 | 0.828 |
| rain + fc | 0.843 | 0.319 | 0.099 | 0.912 | 0.828 |
| rain + era5 | 0.842 | 0.308 | 0.092 | 0.911 | 0.836 |
| rain + expo + fc + era5 + geo | 0.841 | 0.310 | 0.091 | 0.910 | 0.837 |
| **fc + expo (forecast mode, no observed rain)** | 0.787 | 0.228 | 0.048 | 0.890 | 0.798 |
| fc + expo + geo (forecast mode) | 0.777 | 0.233 | 0.049 | 0.890 | 0.810 |
| policies only | 0.742 | 0.159 | 0.026 | 0.880 | 0.770 |

Paired differences in top-5% capture against the stage-42 CNN [95% over days]: rain -0.000 [-0.012, +0.011]; rain + expo -0.006 [-0.015, +0.002]; rain + geo +0.008 [-0.006, +0.021]; rain + fc -0.000 [-0.012, +0.012]; rain + era5 -0.011 [-0.022, -0.000]; all five -0.009 [-0.021, +0.005]; fc + expo -0.091 [-0.112, -0.071]; fc + expo + geo -0.086 [-0.106, -0.064]. Block-group AUC: rain -0.002 [-0.005, +0.000], the other observed-mode combinations -0.003 to -0.006 (intervals mostly just below zero), forecast mode -0.06 to -0.07.

## Conclusions

1. **The declarative encoder reproduces the hand-assembled pipeline.** With the rain source alone it equals the stage-42 CNN inside every interval (top-5% 0.319 vs 0.319, NDCG 0.096 vs 0.097), so declaring sources and letting per-source encoders do the assembly costs nothing in skill and is above the boosting baseline as before.
2. **With observed rain, no additional source helps.** NFHL / HAND (+0.008 top-5%, inside the interval), the GEFS forecast field (0), exposure maps (-0.006) and ERA5 (-0.011, borderline worse) all stay within noise of rain alone, and block-group AUC is slightly lower with every extra source. This agrees with the boosting results of stages 21-22 and 37-40: the observed rain already carries what the other sources would add, and extra input channels cost a little variance at this sample size.
3. **Forecast mode is far below observed mode but above the exposure ranking.** `fc + expo` reaches top-5% 0.228 against 0.159 for policies only (+0.069) and 0.319 with observed rain (-0.091 [-0.112, -0.071]); geometry adds +0.005 in point value. A forecast-mode boosting baseline was not run under this exact protocol, so whether the encoder beats hand-built ensemble summaries there is not answered here (stage 37-40 reported a small increment over exposure for boosting).
4. **What the encoder buys at this point is uniformity, not skill:** one code path takes grids at two resolutions, row features and (tested, not trained on) categorical codes and record sets, with missing-source handling, without per-source rules. Skill gains need a source that carries information the observed rain does not, which the forecast-mode setting is the natural place to look for.

## Caveats

- Texas only (the forecast, ERA5 and geometry sources exist only there), test years 2017-2019, 2 seeds per configuration (seed spread not tabulated), 5 epochs, no tuning of widths or learning rate.
- The coarse-source design (own trunk + FiLM from identity initialisation) is one choice: with 5 epochs and early stopping it may not have time to leave the identity, so a null for `fc` in observed mode does not show that a better fusion cannot use it.
- `geo` enters as row features of the block group, not as rasters; ERA5 is valid-time information (ceiling, not usable at issue time); modality dropout, categorical embeddings and `SetPool` were implemented and unit-tested but not used in these runs (no per-policy records or categorical fields were built).
- Several training epochs show elapsed times of 1,000-2,700 s in the logs because the Mac slept during the run; this does not change the results (best validation epoch is kept) but the timings are not benchmarks.
