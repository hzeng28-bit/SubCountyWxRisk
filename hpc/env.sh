# Source on Sol (login node or inside a job):  source hpc/env.sh
# Everything lives on scratch; scratch is not backed up and old files are purged, so code stays in git, small results are
# copied back (see hpc/README.md) and the venv is rebuildable with `uv sync`.
export WXRISK_DATA="${WXRISK_DATA:-/scratch/$USER/wxrisk_data}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/$USER/.cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/scratch/$USER/venvs/wxrisk}"
# Fit thread pools to the allocation, not to the node (a node has far more cores than a job may use).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$WXRISK_DATA" "$UV_CACHE_DIR" "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$WXRISK_DATA/hpc"
