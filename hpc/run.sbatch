#!/bin/bash
# Generic wrapper: run one repository script as a batch job. Resources default to a mid-size public job and are overridden on
# the command line, which takes precedence over the #SBATCH lines:
#   sbatch hpc/run.sbatch scripts/49_information_audit.py
#   sbatch -c 32 --mem=128G -t 0-12:00 hpc/run.sbatch scripts/26_flood_pilot_experiment.py --cv
# Partitions (docs.rc.asu.edu/partitions-and-qos): htc (default, 4 h, never preempted), public (7 days), lightwork (<= 8 cores),
# highmem (1-2.2 TiB nodes). Scripts that finish in under 4 hours are best on htc.
#SBATCH --job-name=wxrisk
#SBATCH --partition=htc
#SBATCH --qos=class
#SBATCH --time=0-04:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f hpc/env.sh ] || { echo "hpc/env.sh not found in $PWD: run sbatch from the repository root (the folder that contains hpc/, src/, scripts/)"; exit 2; }
source hpc/env.sh
echo "job ${SLURM_JOB_ID:-local} on $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, data $WXRISK_DATA"
uv sync --frozen
/usr/bin/time -v uv run python "$@"
