# Running on ASU Sol (batch jobs)

Facts taken from the ASU Research Computing documentation (docs.rc.asu.edu/partitions-and-qos) and from the project owner; the
items marked *check* are not verified yet and are what `hpc/probe.sbatch` is for.

| Item | Value |
|---|---|
| Scheduler | Slurm; default with no flags is partition `htc`, QOS `public`, 4 h |
| Partitions we use | `htc` (4 h, never preempted), `public` (7 days), `lightwork` (1 day, <= 8 cores), `highmem` (1-2.2 TiB nodes) |
| QOS | `public` (default), `debug` (15 min, one running + two queued per user); `long` (14 days) is granted case by case |
| Scratch | `/scratch/hzeng28`, quota 100 TB, **not backed up, unused files are purged** |
| Data movement | Globus (the documentation's recommendation for large transfers) |
| Internet from a compute node | *check* (the login node answers ping; `probe.sbatch` tests HTTPS from a compute node) |
| uv / Python 3.12-3.13 / pygrib | *check* |

## Order of work

1. Put the code in `$HOME/nanoWxRisk` (backed up daily, 100 GiB); data, venv and caches stay on scratch (`hpc/env.sh`).
   - With a working git login on Sol: `git clone` the private repository.
   - Without one: `bash hpc/pack.sh` on the Mac builds `dist/wxrisk-code-<commit>.tar.gz` (tracked files at HEAD only) and a `.sha256`;
     upload both with Globus or the web file browser, then on Sol
     `cd ~ && sha256sum -c wxrisk-code-<commit>.tar.gz.sha256 && tar xzf wxrisk-code-<commit>.tar.gz`. A new version is a new archive;
     extract it over the old directory (code only, nothing under `$WXRISK_DATA` is touched).
   Then `cd ~/nanoWxRisk && sbatch hpc/probe.sbatch` and read `$WXRISK_DATA/hpc/probe_<jobid>.txt`.
2. Move bundle A (`hpc/transfer_manifest.txt`, about 380 MB) with Globus into `/scratch/hzeng28/wxrisk_data/` keeping the relative paths.
   Raw data is re-downloaded on Sol by the numbered fetch scripts when the probe shows outbound HTTPS works; the lock file
   (`python scripts/32_data_lock.py verify`) tells whether the copy matches.
3. `sbatch hpc/fetch_gefs.sbatch` (ten array tasks, resumable).
4. Analysis jobs: `sbatch hpc/run.sbatch scripts/49_information_audit.py`. Override resources on the command line.

## Rules for jobs in this project

- Scripts read `SLURM_CPUS_PER_TASK` (or `OMP_NUM_THREADS` through `hpc/env.sh`) instead of assuming the whole node.
- Outputs go to `$WXRISK_DATA` only; nothing is written next to the code.
- Every long step must be resumable (skip existing outputs) because `htc` stops at 4 hours; a job that needs more goes to `public`.
- **Scratch is a working area, not storage.** Copy results that matter (`$WXRISK_DATA/audit/*`, the small derived tables) back to the Mac with
  Globus after a run, and commit conclusions to `RESULTS.md`. Do not touch files only to dodge the purge rule; a data set that is
  needed again is either copied to storage that is kept or re-created from the fetch scripts.
- Data sharing inside the group goes through the group's own channel (the documentation asks groups to handle sharing), not by opening permissions on scratch.
- No credentials in the repository. An API key for USGS (`api.waterdata.usgs.gov`, 4000 requests an hour without one) is the owner's to request and
  would be read from an environment variable.
