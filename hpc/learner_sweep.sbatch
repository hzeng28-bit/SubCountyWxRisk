#!/bin/bash
# Bayesian hyper-parameter search, one Slurm array task per learner family (scripts/54_learner_search.py), then collect with script 55:
#   jid=$(sbatch --parsable hpc/learner_sweep.sbatch)
#   sbatch --dependency=afterok:$jid hpc/run.sbatch scripts/55_learner_convergence.py
# The class QOS allows 2 running jobs and 10 submitted ones, so the array runs two families at a time (%2) and the collector job
# makes the seventh submission. Limits per job: 32 cores, 320 GB, 24 h. Override on the command line, e.g. --export=ALL,TRIALS=10.
#SBATCH --job-name=sweep
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=0-20:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --array=0-5%2
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f hpc/env.sh ] || { echo "hpc/env.sh not found in $PWD: run sbatch from the repository root (the folder that contains hpc/, src/, scripts/)"; exit 2; }
source hpc/env.sh
FAMILIES=(logit logit2 hgb rf et mlp)
FAM=${FAMILIES[$SLURM_ARRAY_TASK_ID]}
echo "job ${SLURM_JOB_ID} family $FAM on $(hostname), ${SLURM_CPUS_PER_TASK} cpus, data $WXRISK_DATA"
uv sync --frozen
# the search stops by itself after 17 h so the refit and the test scoring still fit into the 20 h allocation
/usr/bin/time -v uv run python scripts/54_learner_search.py --family "$FAM" --trials "${TRIALS:-40}" --timeout-h 17
