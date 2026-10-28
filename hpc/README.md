# Running on ASU Sol (batch jobs)

Facts taken from the ASU Research Computing documentation (docs.rc.asu.edu/partitions-and-qos) and from the project owner; the
items marked *check* are not verified yet and are what `hpc/probe.sbatch` is for.

| Item | Value |
|---|---|
| Scheduler | Slurm; default with no flags is partition `htc`, QOS `public`, 4 h |
| Partitions we use | `htc` (4 h, never preempted), `public` (7 days), `lightwork` (1 day, <= 8 cores), `highmem` (1-2.2 TiB nodes) |
| QOS | `public` (default; rejected for this account: "Invalid qos specification"), `debug` (15 min, one running + two queued per user); `long` (14 days) is granted case by case |
| Scratch | `/scratch/hzeng28`, quota 100 TB, **not backed up, unused files are purged** |
| Data movement | Globus (the documentation's recommendation for large transfers) |
| Internet from a compute node | *check* (the login node answers ping; `probe.sbatch` tests HTTPS from a compute node) |
| uv / Python 3.12-3.13 / pygrib | *check* |

Account `class_cse47267488fall2026` (course account) is associated with QOS `class` only: MaxWall 1 day, 2 running jobs, 10 submitted, no per-user resource cap (sacctmgr, 2026-10-27). `run.sbatch` and `fetch_gefs.sbatch` request `--qos=class`; `probe.sbatch` uses `debug`.

## Order of work

1. Put the code in `$HOME/nanoWxRisk` (backed up daily, 100 GiB); data, venv and caches stay on scratch (`hpc/env.sh`).
   - With a working git login on Sol: `git clone` the private repository.
   - Without one: `bash hpc/pack.sh` on the Mac builds `dist/wxrisk-code-<commit>.tar.gz` (tracked files at HEAD only) and a `.sha256`;
     upload both with Globus or the web file browser, then on Sol
     `cd ~ && sha256sum -c wxrisk-code-<commit>.tar.gz.sha256 && tar xzf wxrisk-code-<commit>.tar.gz`. A new version is a new archive;
     extract it over the old directory (code only, nothing under `$WXRISK_DATA` is touched).
   - Globus sync is better than the archive: `bash hpc/pack.sh` also unpacks the same tracked files into `dist/stage/nanoWxRisk`;
     sync that folder to `~/nanoWxRisk` (sync level: checksum differs; leave "delete files on destination" off). Later versions move only
     the changed files. Never sync the repository working tree itself (it holds `.git`, caches and `dist/`).
   - Data bundle A: `bash hpc/stage_data.sh` builds `dist/data_A` (hard links, no extra disk) with the manifest paths; sync it to
     `/scratch/hzeng28/wxrisk_data`. Downloads happen on the Mac only (decision 2026-10-27); Sol computes. Thousands of small files (the GEFS crops) go as one tar, not as a folder.
   Then `cd ~/nanoWxRisk && sbatch hpc/probe.sbatch` and read `$WXRISK_DATA/hpc/probe_<jobid>.txt`.
1b. `uv` is not installed on Sol (probe 66728619). Install it once on the login node, into `$HOME/.local/bin` (`hpc/env.sh` puts that on PATH), and check `uv --version`; if the cluster blocks the installer, `pip install --user uv` after `module load` of a Python also works.
2. Move bundle A (`hpc/transfer_manifest.txt`, about 456 MB) with Globus into `/scratch/hzeng28/wxrisk_data/` keeping the relative paths.
   Raw data is fetched locally by the numbered fetch scripts and synced here (compute nodes do have outbound HTTPS, probe 66728864, but we do not use it); the lock file
   (`python scripts/32_data_lock.py verify`) tells whether the copy matches.
2b. Check that the bundle is complete: `sbatch hpc/run.sbatch scripts/49_information_audit.py`; its `$WXRISK_DATA/audit/information_audit.txt` must equal the Mac copy apart from the `fitted ... iterations` lines (verified on the Mac with a data directory holding bundle A only, 2026-10-27). Verified on Sol the same day: identical numbers and iteration counts (partition htc, QOS class).
3. (Not used: `hpc/fetch_gefs.sbatch` downloads on Sol; kept for the case that it becomes useful.)
3b. Learner search: two array tasks (hgb; the other five families in sequence), 6 cores / 48 GB each, 10 h; the 3 validation folds of a trial run as 3 processes, `public` + `class`, resumable from a journal file (resubmit `sbatch --array=<task> hpc/learner_sweep.sbatch` if a task runs out of time). `jid=$(sbatch --parsable hpc/learner_sweep.sbatch)` then `sbatch --dependency=afterok:$jid hpc/run.sbatch scripts/55_learner_convergence.py`. Queue advice from the Sol docs (fairshare page): a big request waits long even with a good score, split work into small jobs and use arrays, never a shell loop of `sbatch` calls; `htc` (4 h, never preempted, backfills easily) lists only `public` and `debug` QOS, so a class account cannot use it. `myfairshare` shows the score.
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
