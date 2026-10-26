# AGENTS.md — guide for AI agents (and humans) working in nanoWxRisk

Read this first. It tells you where things are, how to run them, and which traps have already cost us time.
Tool-agnostic: it applies to Claude Code, Codex, Cursor, or a person at a terminal.

## What this project is

Ensemble-forecast-conditioned **meteorological hazard risk index** (video-codec analogy, see `DESIGN.md`). Two lines:
1. **Severe convective storms (SCS3 = hail + thunderstorm wind + tornado)**, county-day, CO/NM/AZ — stages 0–5 in `RESULTS.md`.
2. **Flood**, sub-county (census block group × rain episode), Texas pilot, NFIP-based — stages 6–8 (+ priors, `docs/FLOOD_PRIORS.md`).

`RESULTS.md` is the research log: one section per stage, each ending with *Conclusions* and *Caveats*.
**Read the caveats before building on a result.**

## Map

| Where | What |
|---|---|
| `src/wxrisk/` | library: `wb2` (data dir, WeatherBench-2 access), `labels_*`, `scale_features`, `wavelet_motion`, `flood_pilot`, `mrms`, `bg_crosswalk`, … |
| `scripts/NN_*.py` | numbered pipeline steps; each has a docstring with inputs, outputs, and the command. Run in numeric order within a line |
| `src/wxrisk/fetch.py` | **the only download layer**: S3 (boto3), HTTPS with resume/atomic/space checks, batch preflight |
| `scripts/fetch/` | download helpers not tied to a numbered step (`fetch_raw.py`, `fetch_fema_conmaz.py`) |
| `docs/DATA_SOURCES.md` | **every data source, URL, script, local path, size, and known data trap** |
| `docs/data_manifest.json` | machine-readable inventory (paths, sizes, parquet columns/rows); regenerate with `scripts/31_data_manifest.py` |
| `docs/FLOOD_PRIORS.md` | building/foundation/flood-zone prior-feature audit and the hypotheses tested |
| `src/wxrisk/prep/`, `scripts/40_prep.py` | **preprocessing pipeline** raw -> clean -> normalized: declarative per-column rules for nulls/sentinels, sidecar with hashes, `verify` rebuilds and compares (`docs/DATA_PREP.md`). New analyses should read `clean/` tables, not re-clean raw ones |
| `RAW_STATICS_DA_REPORT.md` | generated: raw-dataset statistics, K-means/mixture diagnostics with PCA projections for every clustering claim, covariate reports for every regression claim (`scripts/44_raw_report.py all`; figures in `docs/figures/raw_report/`) |
| `docs/CLEANING_REGISTER.md`, `docs/SCALING_PLAN.md` | generated: why each cleaning rule exists (provider-documented / undocumented code / entry error) and the scaling decisions (which columns, group-aware or not, discretise or not, which targets need an inverse). `scripts/41`, `42`, `43` produce the evidence |
| `scripts/45_pilot_input_parity.py` | compares flood-pilot exposure/claim arrays built from the clean tables with the stored ones (`flood_pilot.exposure_matrices`, `claims_tx` read `clean/`; `scs_eval.anomalise` is a `FeaturePlan`, `anomalise_plain` the old one) |
| `scripts/39_mrms_hourly_events.py`, `scripts/46_mrms_sparse_reconstruction.py` | hourly MRMS crops of 8 Texas events and the sparse-to-dense test of a motion-compensated prior (RESULTS stage 15: not better than the observations alone) |
| `tests/` | `pytest` (motion estimation only so far) |

## Setup and conventions

```shell
uv sync                                  # Python 3.12–3.13, deps in pyproject.toml, lockfile uv.lock
uv run python scripts/<NN>_<name>.py     # or .venv/bin/python
export WXRISK_DATA=/path/to/data         # default ~/wxrisk_data; every script reads/writes here
uv run python -m pytest                  # tests/ (motion estimation, download layer); `uv run pytest` can pick another Python
uv run ruff check <files you touched>    # config says line length 100, but existing scripts run to ~110;
                                         # the repo already has ~200 pre-existing ruff findings, so it is not a gate
```

- **Compute/IO stack**: array kernels go through `wxrisk.compute` (PyTorch, `device()` picks cuda/xpu/mps/cpu, override with
  `WXRISK_DEVICE`; fp32 on device, each kernel has a numpy reference and a parity test; `scripts/probe_devices.py` shows what a
  machine can do). Tables: Polars/DuckDB scan parquet lazily instead of `read_parquet` into pandas (see `wxrisk.policies`).
  Big arrays: save as `.npy` (never `savez_compressed`) and open with `mmap_mode="r"`; copy the slice you need before sending it to a
  device. Rewriting a pandas step in Polars/SQL needs a parity check, **null keys behave differently** (pandas `astype(str)` turns
  them into a phantom group; Polars/DuckDB drop or group them explicitly).
- Scripts add `src/` to `sys.path`, import `from wxrisk import wb2`, and use `wb2.cache_path(...)`; do the same.
- **Downloads go through `wxrisk.fetch`, never curl/wget/shell** (must behave the same on macOS, Linux, Windows):
  vendor API → its library (CDS: `ecmwf-datastores-client`; WeatherBench-2: gcsfs/zarr); S3 → `fetch.s3`/`s3_bytes`;
  plain HTTPS file → `fetch.https` / `fetch.fetch_many` (resume, atomic `.part` rename, up-front space check and reservation);
  small payloads/JSON → `fetch.get_bytes`/`get_json`. Add a new source by adding a `fetch.Job`, not a new downloader.
- Downloads are idempotent (skip existing non-empty files). Delete a file to force a re-download. A leftover
  `*.part` / `*.part.json` is an interrupted transfer and will be resumed; do not delete it unless the source changed.
- If a download raises `InsufficientSpace`, free disk or point `WXRISK_DATA` elsewhere; do not lower the margin.
- Code comments are English, concise, "why" not "what". Match the surrounding style.
- **Never commit data** (`$WXRISK_DATA` is ~16 GB; `.gitignore` already excludes `*.npz`, `*.npy`, `data/`, `cache/`).
  Never commit credentials (`~/.cdsapirc`).
- `.venv/bin/python` has pandas/pyarrow; the system `python3` does not. Use `uv run`.
- Everything so far runs on a Mac CPU. No GPU assumed. `torch` is an optional extra.

## Evaluation rules (these are decisions, not suggestions)

- **Compare against climatology p0 and report skill (BSS/AUC lift) with year-grouped or episode-clustered bootstrap
  intervals.** Keep a feature or model step only if its gain exceeds the interval.
- **Split by year, never randomly** (train ≤ 2016 / test 2017–2023 in the flood pilot; blocked-by-year CV as a check).
  Rows within a rain event or a year are strongly dependent.
- Main flood metric is **within-event AUC** (county-level models are 0.5 by definition there). Also report overall AUC, AP.
- **Scaling is a pipeline step with rules** (`docs/SCALING_PLAN.md`): trees get raw features; logistic regression, K-means/GMM and BFGS fits get the plan's transforms + standardisation, fitted on training rows only; a model that fits a *transformed target* must go through `prep.scaling.TargetTransform` and be inverted (`Scaled`) before scoring; exp(coef) of a log-OLS is a ratio of geometric means, not of means.
- Absolute event probabilities do **not** transfer across years (Texas claim counts swing from <500 to ~10k per year).

## Known data traps (full list: `docs/DATA_SOURCES.md`, `docs/FLOOD_PRIORS.md`)

1. **NFIP `foundationType == 0` ("unknown") share collapses from 86% (2010) to ~0% (2022+)** in Texas policies. It is a
   reporting-regime change (in the CLAIMS file '0' is instead exactly the legacy occupancy codes 1, 2, 3, 4, 6, with a known foundation for the two-digit codes 11, 12, ...: RAW_STATICS_DA_REPORT.md). In claims, "unknown vs known foundation" is therefore a contrast between occupancy-code families (code 11 has 0.74× the loss ratio of code 1 even before any foundation effect); foundation ratios are code-11 estimates. A naïve feature leaks year. Use known-subset shares plus an explicit unknown-share feature,
   or post-2022 only. And the unknown group is *not* random (higher damage ratio).
2. **Depth = 0 means missing/default, not dry** (large damage at 0). `floodWaterDuration` is unusable (≈0 everywhere).
3. **No wood/brick/masonry field exists in NFIP.** `construction` is an "under construction" boolean. Do not use it as material.
4. NFIP policy bulk file starts at 2009 effective dates → exposure denominators only from 2010.
5. NFIP claim lat/lon are rounded to 1 decimal; locate by 12-digit `censusGeoid` (2010 vs 2020 block groups differ —
   use `wxrisk.bg_crosswalk`).
6. `floodEvent` covers ~81% of claims and only ~72 named events; do not use it as an event fixed effect.
7. NFIP covers only limited basement items: "damage / coverage" understates basement loss. Do not read "basement losses are
   lower" as "basements are safer".
8. `foundationType` codes (OpenFEMA dictionary): 1 slab, 2 basement, **3 crawlspace**, 4-6 elevated variants. **0 is not in the dictionary**
   (an unknown/default code; its share depends on the reporting year, see trap 1). Earlier versions of this note and of script 30 took 3
   for elevated; corrected 2026-10-24 (65% of code 3 claims are flagged elevated, against 93-95% for 4-6).
9. FEMA API: `NfipPolicies` with a "policy in force" filter returns HTTP 503; filter by state + effective date instead.
   The bulk-file URL is not recorded here; `HEAD` to fema.gov returns 403 (Akamai), so don't treat that as "missing".
10. The official floodplain layer is **FEMA NFHL**, not USGS. USGS FIM maps cover only a few gaged reaches.

## Working with the large data

- Do not `read_parquet` the 3.7 GB policy file whole. Use `pyarrow.dataset` with a `filter` and `columns` (see
  `scripts/24_extract_tx_policies.py`, `scripts/30_building_attribute_audit.py`).
- Check `docs/data_manifest.json` for columns/rows before opening anything big.
- Outputs of analysis scripts go to stdout (tee to `$WXRISK_DATA/audit/`) or to `$WXRISK_DATA`; put conclusions in
  `RESULTS.md` (new stage section) and, for data questions, in `docs/`.

## Integrity

- Upstream rarely publishes checksums. Use what exists: S3 ETag-MD5 and `expect_sha256` are enforced by `wxrisk.fetch`;
  OpenFEMA row counts are checked in scripts 20 and 24.
- **Raw bytes can differ for identical data** (NetCDF `History` timestamps, NWIS `# retrieved:` line, `np.savez` zip
  timestamps, parquet writer metadata). Compare with `wxrisk.integrity.content_hash`, not `sha256` of the bytes, and do
  not "fix" a mismatch by re-downloading before checking the content hash.
- `data.lock.json` fingerprints the data dir. After producing or changing shared data, run
  `python scripts/32_data_lock.py write`; teammates run `... verify`. A `WARN upstream changed` on a `mutable` source is expected.

## When you add something

- New data source → add a row to `docs/DATA_SOURCES.md` and a fetch step (numbered script, or `scripts/fetch/`); then
  `python scripts/31_data_manifest.py` and `python scripts/32_data_lock.py write`.
- New experiment → next free script number, docstring with inputs/outputs/command, a stage section in `RESULTS.md`
  with a *Caveats* subsection stating what the result does **not** show.
- Report negative and null results; several hypotheses in this repo were refuted by the data (see `docs/FLOOD_PRIORS.md`).
- Do not push, force-push, or rewrite history unless the user asked for it.

## Current state and next steps

Stage 8 (Texas flood pilot) is the live line. Next, in order (details in `docs/FLOOD_PRIORS.md`): block-group building
exposure summaries (elevated share, known-foundation shares + unknown share) with an ablation on the pilot table;
FEMA NFHL + HAND flood-zone features; basement analysis in basement-heavy states using absolute damage.
