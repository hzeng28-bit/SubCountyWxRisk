# CNN hybrid model: design for two-dimensional field inputs (draft, 2026-10-29)

Status: proposal, nothing built. Judged by the metrics of `docs/MODEL_AUDIT.md` section 5 (ranking of units inside an event; the probability as an auxiliary product).

## 1. What the model has to prove, and what the evidence says so far

Every model so far reads the hazard through **point and county summaries** of the rain field: the value at the block-group cell, the county peak and mean, a 3x3 neighbourhood, member votes and spread.
The proposal is a model that reads the **field itself**. It is worth building only if two-dimensional structure carries information that those summaries lose. The repository has already tested pieces of that idea:

| Earlier test | Result | Reads as |
|---|---|---|
| neighbourhood rain (mean, max, exceedance share) and a wavelet encoder, observed rain (stage 8) | null on top of the cell value | with complete information the point value is enough |
| neighbourhood probabilities and member votes, forecast rain (stage 16) | tie with the ensemble mean | position tolerance did not help in the episode sample |
| ensemble extremes and spread, all-days unit (stage 40) | slightly worse than the mean | the spread of five members at 0.25 degrees is noise |
| ERA5 moisture and wind ingredients (stages 32, 33) | null on top of rain | event-level descriptors are already in the rain |
| forecast vs observed rain, big events (stage 39) | forecast recovers about a fifth of the observed-rain gain | the bottleneck is the forecast of the extremes |

So the prior for a gain from a CNN on the **same GEFS rain field** is low, and the design has to say beforehand how a gain would be recognised and how it would be refuted. The places where two-dimensional reading could still add information:

- **H1, footprint x exposure**: whether the rain footprint overlaps dense exposure (policies per cell), which no summary of the rain alone can express.
- **H2, flow-aligned structure**: rain organised along the low-level flow (training, stalled systems, stage 30), readable from the rain field together with the 850 hPa wind and moisture maps.
- **H3, position tolerance and learned regional bias**: a forecast of heavy rain displaced by a cell or two (the 110-mile miss of Harvey), and regional corrections (arid west, humid east) that a per-cell model can learn from the neighbours.
- **H4, learning stage**: a model of the impact tau(w; x, c) of the slides trained on **observed** fields at 4 km and applied member by member to the forecast fields (the expectation over q(w), stage 36 audit item A7).

## 2. Data for the two-dimensional input (all on disk)

| Field | Grid | Period | Role |
|---|---|---|---|
| GEFSv12 reforecast APCP: 5 members x 8 six-hour windows (leads 36-84 h) per initialisation day | 53 x 61 cells, 0.25 deg, 24.5-37.5 N, 107.5-92.5 W | all 3,652 days of 2010-2019 | the forecast field (issue-time information) |
| gridMET daily rain | 260 x 320 cells, 4 km | 2000-2023 | the observed field (learning stage, H4; coarsened to 0.25 deg for comparison) |
| ERA5 total column water vapour, q / u / v at 500, 700, 850 hPa, 6-hourly | 53 x 113 cells, 0.25 deg, 24-37 N, 107-79 W | 2009-2023 | moisture and flow maps; **valid-time values, not a forecast**: a perfect-prog ceiling only (stage 33) |
| exposure per block group and day (policies, SFHA share, single-family share, coverage) | 15,805 block groups; about 11,600 with a policy per day | 2010-2023 | rasterised to cells (policies per cell, SFHA share per cell) and kept as tabular features |
| claim history `hist00`, `prior3` (stage 19) | per block group | 2000-2023 | tabular and one raster channel |
| terrain, HAND, NFHL shares | per block group | static | optional (null in stages 8, 22); excluded from the first runs |

Registration: the ERA5 box starts at 37.0 N and 107.0 W while the GEFS crop starts at 37.5 N and 107.5 W, so the ERA5 maps are resampled to the GEFS grid (nearest cell, the two outer rows and columns padded and masked). Memory: 3,639 days x 20 channels x 53 x 61 cells in float16 is about 0.47 GB.

## 3. Architecture

**Day tensor** (one per forecast day, on the GEFS grid; the unit of a training batch is one day):

- forecast rain: ensemble mean of the 2-day peak and of the 2-day sum, and the per-member peaks sorted by value (5 channels, permutation invariant); the first run uses the mean channels only, the sorted members as a variant;
- exposure channels: log policies per cell, SFHA share per cell, `hist00` per cell;
- coordinates: latitude and longitude scaled to [-1, 1]; scalars (month as sine and cosine) added to the head;
- variant E3 only: ERA5 tcwv, q850, u850, v850, u500, v500 averaged over S and S+1.

**Encoder**: a small fully convolutional network, four residual blocks of 3x3 convolutions with dilations 1, 2, 4, 1, 32 channels, group normalisation, about 60 k parameters, receptive field of about 2 degrees, output of 16 channels per cell. No pooling to a vector: the output map keeps the cell resolution so that each block group can read its own cell.

**Read-out per block group**: the 16-channel embedding bilinearly sampled at the centroid; the same embedding averaged and maximised over the cells of the county; the tabular features (log policies, SFHA share, single-family share, mean coverage, block groups in the county, `hist00`, `prior3`, month) through a 32-unit layer; concatenated into a two-layer head with two outputs, a rate logit (for the probability product) and a ranking score.

**Two ways to use it** (both reported):

1. **End to end**: the head output is the model.
2. **Stacking (preferred if the network learns anything)**: the CNN's score and embedding enter the boosting model of script 77 as features, computed out of fold by year groups on the training years (3 folds), so that the boosting model never sees in-sample CNN outputs; test rows use the network trained on all training years.

## 4. Loss and sampling

- A batch is one day: the image and all of its block groups (about 11,600 rows), so the ranking loss sees the same set of units as the evaluation.
- **Listwise loss** over the block groups of a day: softmax cross entropy with the target distribution proportional to the claim count (matches top-k capture and NDCG); plus a **weighted binary cross entropy** on "any claim" with weight lambda (default 0.3) for the probability product.
- Days: 2,500 training days, all days with at least one claimed block group plus a random 25% of the days without; negatives inside a day kept (the day is the unit), the loss weights restoring the sampling.
- Positives available: 36,154 claimed block-group days in 2010-2016 (windows [S, S+3] overlap, so about a quarter of that many distinct claims): a small sample for a network, hence the size above, weight decay, early stopping on 2014-2016, and augmentation by translating the image by up to one cell and by dropping members.
- Scope S0 (tropical windows removed) for the first runs, to stay comparable with stages 36-40; scope S2 afterwards.

## 5. Evaluation (aligned with `docs/MODEL_AUDIT.md` section 5)

Rows, split and metrics are those of scripts 75, 77 and 78: all block groups of every forecast day, train 2010-2016, test 2017-2019; cross-county AUC, top-k capture of claim count and paid amount, NDCG with paid amount, county AUC, top 10% of counties; paired differences with the event bootstrap; strata by observed event size and by the day's largest forecast county peak; the probability companion (expected / observed per event, calibration slope, Brier skill as a separate measure).

**Comparison set**: policies only; the boosting mean model of script 77; the observed-rain reference; the CNN variants below.

**Success criteria (fixed before running)**, CNN hybrid minus the boosting mean model, 95% interval over days, 5 seeds:

1. county AUC and top-10% county capture above zero overall **and** in the days with >= 30 claimed block groups and in the days with a forecast peak >= 25 mm (the strata where the forecast loses, stage 39);
2. block-group top-5% capture and NDCG with paid amount above zero overall;
3. the gain larger than the spread between seeds;
4. calibration kept: expected / observed claimed block groups per event between 0.8 and 1.25.

**Negative controls** (they must collapse to the exposure-only result): the image of another day of the same month paired with the block groups of the day (tests whether the model reads the field of the day at all); the image translated by 2 cells (tests that it uses position); the image with the rain channels set to the day's mean (tests that the gain is spatial, not just intensity).

## 6. Experiments in order, with go or no-go

| Step | Content | Decision |
|---|---|---|
| E0 | `scripts/80_build_day_tensors.py`: day tensors, block-group-to-cell indices, tabular rows, registration check of the ERA5 and GEFS grids | data ready, shapes and masks checked |
| E1 | **Two-dimensional reading with complete information** (H1, H3): the same hybrid on observed gridMET fields coarsened to 0.25 deg (and at 4 km), observed mode, compared with the boosting observed-rain model | **gate**: if E1 does not beat the boosting model with an interval above zero, two-dimensional reading adds nothing even with perfect input (as stage 8 found for neighbourhoods) and E2-E4 stop |
| E2 | the hybrid on the GEFS fields (forecast mode), exposure channels on and off | criteria of section 5 |
| E3 | + ERA5 moisture and wind channels at valid time (H2), a ceiling for the ingredient maps | an upper bound; if it fails, H2 is closed |
| E4 | learning-stage model applied member by member (H4): train tau on observed fields, average the member impacts | against the direct forecast model |
| E5 | the negative controls on the best run | required before any claim |

Compute: day-sized batches of one small image and about 11,600 block-group rows make an epoch of 2,500 days a few minutes on the Mac (torch 2.14 with MPS is installed); the sweep over seeds and variants fits on Sol (`hpc/run.sbatch`). Code: `src/wxrisk/cnn.py` (model, with a shape and gradient test in `tests/`), scripts 80 (data), 81 (training), 82 (evaluation with the scripts 75 helpers).

## 7. Risks and limits

- **The network cannot create information that the fields do not contain.** Stage 37 and stage 39 locate the loss in the forecast of the extremes (ensemble mean about 0.37 of the observed county peak, 42% of the episodes forecast below 10 mm); a better reading of the same GEFS field may not change that. The most probable outcome of E2 is a small or null gain; E1 and the controls make a null result informative.
- Few independent storms: 2,500 training days, 36 k claimed block-group days in overlapping windows, a few hundred distinct storms; the network can memorise storms. Early stopping by year, weight decay, small size and translation augmentation are the safeguards; out-of-fold stacking prevents leakage into the boosting model.
- The exposure channel makes it easy for the network to rediscover "where the insurance is"; the comparison is against the boosting model with the same tabular features and against policies only for that reason.
- The ERA5 maps are valid-time fields; they cannot be used in a deployed forecast (GEFS precipitable water and winds would have to be fetched, stage 33 judged that unlikely to repay).
- Texas only; Louisiana and Florida pipelines do not exist yet (the day tensors on the same grid would extend to them for free once the block-group tables exist).

## 8. Decisions

Answered 2026-10-29:
1. **E1 is a hard gate**: if the hybrid does not beat the boosting observed-rain model with an interval above zero, E2-E4 do not run.
2. **First runs on scope S0**, then S2.

Not yet asked, proposed as defaults: the ERA5 valid-time maps only as a ceiling in E3, never in a headline forecast result; the Mac for E0 and E1, Sol for the seed sweeps.

## 9. Status

- E0 done (stage 41). E1 done (stage 42): at 4 km the hybrid beats the boosting baseline at block-group level (top-5% capture +0.023, NDCG with paid amount +0.017, largest in the big events) and by +0.004 in county AUC; at the 0.25 degree resolution of the forecast fields it ties boosting. Gate passed at 4 km; E2 expected to give little, because the forecast has no 4 km structure; E4 (learn on observed 4 km, apply member by member to a downscaled forecast) is the route that can use it.

## 10. Multi-scale latent model: autoencoder / diffusion route (proposal, 2026-10-29, nothing built)

Motivation (stage 42): the gain of two-dimensional reading sits at 4 km and inside counties; the forecast fields are 0.25 degrees. A model with separate scales and a latent code can be pretrained on unlabelled observed fields (labelled days: 2,191; unlabelled observed days <= 2016: about 6,200 in Texas, more if CONUS crops are used) and then asked to carry the forecast into the latent space of the observation.

### 10.1 Literature map (abstract-level search, papers not read in full, benchmark numbers not checked)

| Route | Work | Maps to |
|---|---|---|
| autoencoder + latent diffusion transformer | LaDCast (arXiv 2506.09193), PreDiff (2307.10422), VAE + DiT nowcasting (2601.20342) | forecast to observed field, conditional generation |
| deterministic mean + residual diffusion | CorrDiff (2309.15214; about 80 M parameters in the reference implementation; direct conditional diffusion reported unsuccessful) | coarse to 4 km downscaling |
| deterministic mesoscale + probabilistic small scale | CasCast (2402.04290) | multi-scale precedent |
| diffusion features for pixel classification | Baranchuk et al. (2112.03126), hybrid supervised/generative pretraining (2408.03433) | main task: unlabelled pretraining, few labels |
| foundation models | Aurora (1.3 B), Prithvi WxC (2.3 B), ClimaX | not feasible: need full atmospheric state, GEFS reforecast kept rain only; no 4 km structure |

Open concern in the literature: lossy latent compression can distort precipitation amounts; claims come from the tail, so fidelity on extreme pixels is the first gate.

### 10.2 Problem as CV tasks

- Main task = weakly supervised pixel classification: one pixel per block group (about 11,600 labelled points of 83,200 per day), listwise + weighted BCE loss as in section 4.
- Forecast side = generation of the **physical field** (observed rain given the forecast), where 24+ years of data exist. Claim probability is the impact head averaged over samples (audit item A7), not a diffusion target (a few hundred storms).

### 10.3 Grid fact (checked on `cnn/static.npz`)

gridMET spacing is 1/24 degree, GEFS 0.25 degrees: exactly 6 pixels per forecast cell. The 4 km crop (260 x 320) padded to 264 x 324 is 44 x 54 forecast cells. A stride-6 encoder (3 then 2, or patch size 6) puts one latent cell on each forecast cell, so the 36-pixel sub-cell structure the forecast does not have becomes a few latent channels that the forecast encoder must predict.

### 10.4 Complexity (hand estimates, not run)

| Tier | Content | Parameters | Tokens | Training cost |
|---|---|---|---|---|
| T0 | current Hybrid CNN | about 5e4 | none | minutes per epoch on the Mac (measured) |
| T1 | hierarchical conv autoencoder (6:1 then 2:1) | 1-5 M | none | Mac |
| T2 | small conv UNet diffusion (CorrDiff-like), attention at the bottleneck | 10-30 M | bottleneck 44 x 54 | Mac hours to a day or two; sampling 20-50 steps x members |
| T3 | DiT on 6 x 6 patches, no VAE (DiT-S/B: 33 M / 130 M) | 30-130 M | about 2,400 | about 70 GMAC forward for 12 layers at d = 384; about 2e17 FLOP for 6,000 samples x 200 epochs: Mac about a day, GPU hours |
| T3' | same, patch 3 | same | about 9,500 | 5-6 x T3 (attention term dominates) |
| T4 | foundation-model fine-tuning | 1 B+ | | not feasible (inputs, licence not checked) |

### 10.5 Decision: no Transformer first

Data (2,191 labelled days, about 6,200 unlabelled) is small for ViT/DiT; global attention buys long-range context, which stages 8, 32 and 33 found null; bottleneck attention in a UNet is nearly free. Escalate to T3 only if the conv model plateaus and the scale probes show missing long-range information.

### 10.6 Experiments and gates

| Step | Content | Gate |
|---|---|---|
| V0 | hierarchical conv autoencoder, stride 6 aligned to GEFS cells, log rain with tail weighting, pretrained on observed fields <= 2016 (test years never enter) | reconstruction fidelity on P95/P99 pixels; scale probes (fine latent predicts within-county deviation, coarse latent predicts county peak and mean) |
| V1 | frozen latent + boosting or small MLP, observed mode, against the E1 4 km CNN and boosting | not worse than the supervised 4 km CNN and better than boosting, interval above zero; stop if not |
| V1b | DDPM-seg style: small denoising UNet on the unlabelled fields, mid-noise features to a pixel classifier | parallel control to V1 |
| V2 | forecast mode: mean model + residual diffusion (CorrDiff style), samples through the impact head | beats the boosting mean model on section 5 metrics of MODEL_AUDIT |
| V3 | direct grid head (Poisson with policies as offset) and calibration | expected / observed 0.8-1.25 |
| T3 | DiT, only on the escalation condition of 10.5 | |

Risks: reconstruction dominated by light rain (tail weighting, or contrastive pretraining), posterior collapse (KL annealing, monitor), coordinates memorising counties (coordinate dropout and a no-coordinate control), pretraining scope (<= 2016 only). Prior stays low for the forecast end: every earlier attempt to add information to the 0.25 degree forecast was null; the realistic gain is at 4 km observed mode and within counties.

### 10.7 Next step if V0/V1 pass: candidate states for data scale

The labelled sample (Texas, 2,191 training days) limits every supervised model (best validation epochs 1-8). The group TX + LA + FL (stage 27) multiplies the storms and, with the same 0.25 degree grid, the day tensors extend for free once block-group tables exist. Needed: NFIP policy and claim extraction for LA and FL, block-group crosswalk, GEFS reforecast and gridMET crops over the Gulf box (ERA5 Gulf box already fetched, script 70), tropical-cyclone handling (scope S0 vs S2), and a state-held-out check so the gain is not a Texas artefact. Not started; it is conditional on the V0/V1 gates.
