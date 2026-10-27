#!/bin/bash
# First job to run on Sol: records what a compute node actually offers, so later jobs are written against facts.
#   sbatch hpc/probe.sbatch        (debug QOS: 15 minutes max, one running job per user)
#SBATCH --job-name=wxrisk-probe
#SBATCH --partition=public
#SBATCH --qos=debug
#SBATCH --time=0-00:15
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
[ -f hpc/env.sh ] || { echo "hpc/env.sh not found in $PWD: run sbatch from the repository root (the folder that contains hpc/, src/, scripts/)"; exit 2; }
source hpc/env.sh
OUT="$WXRISK_DATA/hpc/probe_${SLURM_JOB_ID:-local}.txt"
{
  echo "== node"; hostname; date; echo "cpus allocated: ${SLURM_CPUS_PER_TASK:-?}  nproc: $(nproc)"; free -g | head -2
  echo "== scratch"; df -h "$WXRISK_DATA" | tail -1
  echo "== tools"; command -v uv || echo "uv: not on PATH"; command -v module >/dev/null && module list 2>&1 | head -5
  echo "== environment"
  (uv sync --frozen 2>&1 | tail -3) || echo "uv sync failed"
  uv run python - <<'PY' 2>&1 | grep -v Warn
import importlib, sys
print(sys.version.split()[0])
for m in ["numpy", "pandas", "polars", "sklearn", "pygrib", "eccodes", "boto3", "torch"]:
    try:
        importlib.import_module(m); print("  ok  ", m)
    except Exception as e:
        print("  FAIL", m, str(e)[:80])
PY
  echo "== outbound HTTPS from the compute node (wxrisk.fetch, the repo's download layer)"
  uv run python - <<'PY' 2>&1 | grep -v Warn
import sys; sys.path.insert(0, "src")
from wxrisk import fetch as dl
for name, f in {
    "S3 GEFS reforecast": lambda: dl.probe_s3("noaa-gefs-retrospective", "GEFSv12/reforecast/2015/2015080100/c00/Days:1-10/apcp_sfc_2015080100_c00.grib2"),
    "S3 MRMS": lambda: dl.s3_client().list_objects_v2(Bucket="noaa-mrms-pds", Prefix="CONUS/", Delimiter="/", MaxKeys=1)["KeyCount"],
    "USGS water data API": lambda: len(dl.get_bytes("https://api.waterdata.usgs.gov/ogcapi/v0/collections?f=json", headers={"User-Agent": "Mozilla/5.0"})),
    "OpenFEMA": lambda: len(dl.get_bytes("https://www.fema.gov/api/open/v2/OpenFemaDataSets?$top=1", headers={"User-Agent": "Mozilla/5.0"})),
}.items():
    try:
        print("  ok  ", name, f())
    except Exception as e:
        print("  FAIL", name, str(e)[:100])
PY
} 2>&1 | tee "$OUT"
echo "written to $OUT"
