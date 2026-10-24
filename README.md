# nanoWxRisk
## Proposed forcast option 1: Video codec like prediction
Take the "time-series / frequency-domain encoder" line of research and replace it with a **video-encoder analogy**,
applied to meteorological hazard risk indices.

- **Input**: initial conditions (the analysis field) plus a set of supercomputer ensemble members (IFS ENS, 50 members)
- **Output**: a chosen hazard risk index on a given geographic grid at a given lead time (event probability / index distribution)
- **Intuition**: atmospheric evolution = motion of 2D features over time → I-frame (initial field) + P-frames (each member's
  forecast field); disagreement between members = "mode competition" in the encoder, where an RD-style vote decides which
  feature wins, and the entropy / disagreement of the winning mode directly gives the distribution of the risk index.

See [DESIGN.md](DESIGN.md) for the design and experimental protocol. The research log is in [RESULTS.md](RESULTS.md).

## Data (public, no authentication)

The WeatherBench-2 public GCS bucket `gs://weatherbench2`:

| Use | Path | Shape |
| --- | --- | --- |
| Ensemble forecast | `datasets/ifs_ens/2018-2022-64x32_equiangular_conservative.zarr` | time 3652 × number 50 × lead 61 × lon 64 × lat 32 |
| Ground truth | `datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr` | time 93544 × lon 64 × lat 32 |

The 64×32 grid (5.625°) is a deliberately "nano" scale: 2048 grid points globally, so the full pipeline runs on one machine.
The resolution ladder 64x32 → 240x121 → 1440x721 is all in the same bucket; raise the resolution once the conclusions hold.

## Quick start

```shell
uv sync
.venv/bin/python scripts/01_fetch_subset.py --years 2018 2021 --lead-hours 72
.venv/bin/python scripts/02_baseline_vote.py
```

## Documentation index

- [AGENTS.md](AGENTS.md): entry point for AI agents and new team members (layout, how to run, evaluation rules, known data traps)
- [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md): every data source, how it is downloaded, local paths, and data traps
- [docs/data_manifest.json](docs/data_manifest.json): machine-readable data inventory
- [docs/FLOOD_PRIORS.md](docs/FLOOD_PRIORS.md): audit of flood prior features (building structure, foundation, flood zones)
- [RESULTS.md](RESULTS.md): research log, stages 0–9
