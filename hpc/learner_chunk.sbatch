#!/bin/bash
# The learner search (scripts/54_learner_search.py) as a chain of one-hour jobs instead of two long ones. The search is resumable, so
# each element continues the journal where the previous one stopped, and a finished family ends in a second. Short jobs backfill into
# gaps of the queue more easily than a 4 h request; the price is one queue wait per element, because %1 holds an element back until
# the previous one has finished. Use it only if the long jobs of learner_sweep.sbatch sit in the queue (compare Submit and Start in
# `sacct -u $USER -X --format=JobID,Partition,QOS,Submit,Start,Elapsed,State`).
#   j1=$(sbatch --parsable --export=ALL,GROUP=hgb hpc/learner_chunk.sbatch)
#   j2=$(sbatch --parsable --export=ALL,GROUP=others hpc/learner_chunk.sbatch)
#   sbatch --dependency=afterok:$j1:$j2 hpc/run.sbatch scripts/55_learner_convergence.py
# Submitted jobs: 4 + 4 + 1 = 9 of the 10 the class QOS allows (the docs do not say whether array tasks count one by one).
#SBATCH --job-name=chunk
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=0-01:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --array=0-3%1
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f hpc/env.sh ] || { echo "hpc/env.sh not found in $PWD: run sbatch from the repository root"; exit 2; }
source hpc/env.sh
case "${GROUP:?set GROUP=hgb or GROUP=others with --export=ALL,GROUP=...}" in
  hgb) FAMILIES=(hgb) ;;
  others) FAMILIES=(logit logit2 mlp et rf) ;;
  *) echo "GROUP must be hgb or others"; exit 2 ;;
esac
BUDGET_H=${BUDGET_H:-0.8}  # search time of one element; the rest of the hour is start-up, the refit and the test scoring
echo "job ${SLURM_JOB_ID} element ${SLURM_ARRAY_TASK_ID} group $GROUP on $(hostname), ${SLURM_CPUS_PER_TASK} cpus"
uv sync --frozen
for FAM in "${FAMILIES[@]}"; do
  LEFT=$(python3 -c "print(max($BUDGET_H - $SECONDS / 3600, 0))")
  if python3 -c "import sys; sys.exit(0 if $LEFT > 0.05 else 1)"; then
    /usr/bin/time -v uv run python scripts/54_learner_search.py --family "$FAM" --trials "${TRIALS:-60}" --timeout-h "$LEFT"
  fi
done
