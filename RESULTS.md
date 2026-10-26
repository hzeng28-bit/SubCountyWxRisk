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
