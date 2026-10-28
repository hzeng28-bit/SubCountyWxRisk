#!/bin/bash
# Bayesian hyper-parameter search (scripts/54_learner_search.py), two array tasks that use both running slots of the class QOS:
#   task 0: hgb (the expensive family)    task 1: logit, logit2, mlp, et, rf one after the other
#   jid=$(sbatch --parsable hpc/learner_sweep.sbatch)
#   sbatch --dependency=afterok:$jid hpc/run.sbatch scripts/55_learner_convergence.py
# Why this shape (docs.rc.asu.edu/fairshare, /partitions-and-qos): a large request waits long even with a good fairshare score, and the
# class account holds 2 running jobs and 10 submitted ones, so 6 cores / 48 GB per task: the three validation folds of a trial run in three processes with 2 threads each (boosting did not scale past 4 threads in hpc/bench_threads.py, processes did; debug job 66734646 kept only 8 of 16 threads busy) and two tasks in parallel, not one 32-core job
# per family in three rounds. The search is resumable: if a task runs out of time, submit it again (sbatch --array=0 hpc/learner_sweep.sbatch)
# and it continues from the journal. Override: --export=ALL,TRIALS=20.
#SBATCH --job-name=sweep
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --time=0-10:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --array=0-1
#SBATCH --output=%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f hpc/env.sh ] || { echo "hpc/env.sh not found in $PWD: run sbatch from the repository root (the folder that contains hpc/, src/, scripts/)"; exit 2; }
source hpc/env.sh
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then FAMILIES=(hgb); HOURS=8.5; else FAMILIES=(logit logit2 mlp et rf); HOURS=1.8; fi
echo "job ${SLURM_JOB_ID} families ${FAMILIES[*]} on $(hostname), ${SLURM_CPUS_PER_TASK} cpus, data $WXRISK_DATA"
uv sync --frozen
for FAM in "${FAMILIES[@]}"; do
  # --timeout-h caps the search of each family so the packed task stays inside its 12 h; the refit and test scoring come after it
  /usr/bin/time -v uv run python scripts/54_learner_search.py --family "$FAM" --trials "${TRIALS:-40}" --timeout-h "$HOURS"
done
