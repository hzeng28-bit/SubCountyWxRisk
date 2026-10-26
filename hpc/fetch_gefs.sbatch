#!/bin/bash
# GEFSv12 reforecast crops for the flood forecast-mode pilot, one array task per year (2010-2019).
#   sbatch hpc/fetch_gefs.sbatch
# Needs flood_pilot_table.parquet under $WXRISK_DATA (hpc/README.md, bundle A). Resumable: existing days are skipped, so a task that
# hit the time limit can simply be submitted again. Run hpc/probe.sbatch first; this job needs outbound HTTPS from the node.
#SBATCH --job-name=gefs
#SBATCH --partition=htc
#SBATCH --qos=public
#SBATCH --time=0-04:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=2010-2019%4
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source hpc/env.sh
uv sync --frozen
uv run python scripts/47_fetch_gefs_reforecast.py --years "$SLURM_ARRAY_TASK_ID" "$SLURM_ARRAY_TASK_ID" --workers "$SLURM_CPUS_PER_TASK"
