# Design: an ensemble-forecast risk index under a video-encoder analogy

## 0. Problem formulation

Given
- initial conditions $X_0 \in \mathbb{R}^{V \times H \times W}$ (the analysis field, V variables)
- ensemble members $\{M_m\}_{m=1}^{50}$, $M_m \in \mathbb{R}^{V \times L \times H \times W}$ (L lead times)

find
$$R_c(h, w, \tau) = \Pr\big[\text{hazard class } c \text{ occurs at grid point }(h,w)\text{ within window } [\tau-24\mathrm{h}, \tau] \mid X_0, \{M_m\}\big]$$

**Hazard classes (stage one covers only c=precip)**
- `precip`: 24 h accumulated precipitation > the grid point's climatological p99 (and ≥ 10 mm, to avoid spurious extremes at arid points)
- `wind`: 10 m wind speed within the window > the grid point's climatological p99 (stage two)
- `heat`: 2 m temperature > the grid point's calendar-day climatological p99 (stage two)

Climatological thresholds are estimated from **ERA5 1990–2017** (per grid point × calendar month, 30-day sliding window), which
does not overlap at all with the 2018–2022 evaluation period: the thresholds themselves must not leak information from the
evaluation period.

## 1. Mapping from encoder concepts to meteorology

The previous line of research (the nanoTSFM frequency head) used an audio-encoder analogy: sub-band splitting (Opus/CELT) plus a
RAW fallback (FLAC). This time we switch to a video encoder, because the problem goes from 1D to 2D plus time.

| Video-coding concept | Counterpart in this project | Notes |
| --- | --- | --- |
| I-frame (intra coding) | Initial conditions $X_0$ | Can only use the current spatial structure; pure 2D feature recognition |
| P-frame (inter prediction) | Member $M_m$'s field at lead $\tau$ | "Predicted" from the I-frame |
| Motion vector (MV) | Displacement / advection from initial-field features to a member's forecast field | The movement of weather systems is literally motion |
| Residual | The part of a member's field that advection cannot explain | Corresponds to genesis and decay, phase change, convective initiation |
| Multi-reference prediction | All 50 members are candidate references | **The entry point for the user's "conditional voting"** |
| Mode decision (RD-opt) | Which members / which prediction mode wins in each block | $J = D + \lambda R$ |
| Variable block size (macroblock partition) | Adaptive spatial scale of the risk features | Synoptic scale vs local convective scale |
| Intra-mode fallback | Climatology / persistence branch | Counterpart of the RAW fallback (FLAC) in the previous line |
| Entropy of the mode decision | Member disagreement → **width of the risk-index distribution** | Not a by-product; part of the target quantity |
| Deblocking filter | Spatial smoothing of the per-block risk index | Prevents block artifacts |

**The core bet**: the standard ensemble probability is a "democratic vote": 50 members, one vote each, equal weights, independent
at every grid point. A video encoder's mode decision is not a democratic vote; it is **per block and weighted by RD cost**. If
atmospheric predictability really is spatially non-uniform (it is), then "per block + flow-aligned + learned mode weights" should
beat the equal-weight vote.

## 2. Baselines that must be beaten (otherwise we are reinventing EPS probabilities)

Progressively stronger, **same labels / metrics / time split, only the method changes**:

| # | Baseline | Description |
| --- | --- | --- |
| B0 | Climatological frequency | Per grid point × calendar month event rate, ignoring the forecast |
| B1 | Equal-weight vote | $\hat p = \frac{1}{50}\sum_m \mathbb{1}[M_m > \text{thr}]$: this is what operations actually do |
| B2 | Globally calibrated vote | B1 passed through one global isotonic/logistic calibration |
| B3 | Per-grid-point calibrated vote | Each grid point has its own calibration curve (parameters shared within latitude bands) |
| B4 | Vote + member-spread features | Add scalar features such as ensemble spread, ensemble mean, ensemble max |
| C\* | Candidate methods | Block-level RD mode decision / flow-aligned member weighting |

B2/B3 are the truly hard ones: **an equal-weight vote is already strong once calibrated**, and the lesson of the previous line of
research (a globally linear mixing weight does worse on heterogeneous data) will reappear here as "global calibration vs
per-grid-point calibration".

## 3. Metrics

- **Brier score** and **BSS** (relative to the B0 climatological baseline): the primary metrics
- Reliability diagram plus decomposition (reliability / resolution / uncertainty): to see whether a gain comes from calibration or from resolution
- **AUC-PR** for extreme events (the p99 event base rate is ~1%, so AUC-ROC would be inflated)
- Stratified reporting: by latitude band, land/sea, season; a pooled global number alone is misleading

## 4. Split and leakage protection

- Time split: train 2018–2020, validate 2021, test 2022 (**strictly chronological, no random split**)
- All grid points of the same init time belong to the same group; any CV must group by **init time** (forecasts from adjacent
  inits are highly correlated, so splitting randomly by grid point would leak)
- Climatological thresholds and calibration curves are fitted on the training period only and merely applied in the evaluation period
- Adjacent init times (6 h apart) are strongly correlated → at evaluation take only the daily 00Z, to avoid an inflated sample size

## 5. Open risks (noted up front, no self-deception)

- The ENS `total_precipitation_24hr` itself has model bias; "exceeds the ERA5 climatological p99" is a cross-dataset threshold
  comparison, so a version that uses "exceeds the ENS's own climatological p99" must also be reported as a control
- "Extreme precipitation" has limited physical meaning on a 64×32 grid (~600 km); this is a methodological testbed, not an operational product
- 50 members × 2048 grid points × 1826 inits is not a large dataset, and it is easy to overfit by trying things repeatedly on the
  2022 test period: the test period may be looked at only once, after the method is frozen

## 6. Stage plan

1. **Stage 0 (current)**: data pipeline plus B0/B1/B2/B3 baseline numbers working, to establish how strong the equal-weight vote is
2. **Stage 1**: can motion vectors / optical flow really align the member fields (a purely geometric test, no learning)
3. **Stage 2**: block-level mode decision (first with a closed-form RD cost, no learning), compared against B3
4. **Stage 3**: only then bring in neural networks (video-encoder-style 3D conv / member attention)
