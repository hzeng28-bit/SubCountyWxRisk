"""What does a Sol allocation really give a gradient-boosting fit? Prints the CPU layout and times one HGB fit per thread count,
then three folds side by side (the shape of one search trial).
  sbatch --qos=debug --time=0-00:15 --cpus-per-task=16 --mem=32G hpc/run.sbatch hpc/bench_threads.py
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

if "--worker" in sys.argv:  # one independent process: fit REPS models with THR threads and report when it started and stopped
    thr, reps = int(sys.argv[-2]), int(sys.argv[-1])
    spec = importlib.util.spec_from_file_location("m54", root / "scripts" / "54_learner_search.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    d = m.prepare("--full" not in sys.argv)
    trm, vam = d.folds[1]
    P = {"max_depth": 6, "min_samples_leaf": 300, "learning_rate": 0.05, "l2": 1.0, "max_iter": 150, "max_leaf_nodes": 31}
    with threadpool_limits(limits=thr):
        m.fit_score("hgb", P, d, trm, vam)  # warm-up, not timed
        a = time.time()
        for _ in range(reps):
            m.fit_score("hgb", P, d, trm, vam)
    print(f"{a} {time.time()}")
    sys.exit(0)

aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
print(f"cpu_count {os.cpu_count()}, usable by this process {aff}, SLURM_CPUS_PER_TASK {os.environ.get('SLURM_CPUS_PER_TASK')}, "
      f"OMP_NUM_THREADS {os.environ.get('OMP_NUM_THREADS')}")
try:
    print(subprocess.run("lscpu | grep -E 'Model name|Thread|Core|Socket|^CPU\\(s\\)|MHz'", shell=True, capture_output=True, text=True).stdout)
except Exception as e:  # noqa: BLE001
    print("lscpu unavailable", e)
print([(i["internal_api"], i["num_threads"]) for i in threadpool_info()])

t0 = time.time()
spec = importlib.util.spec_from_file_location("m54", root / "scripts" / "54_learner_search.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
d = m.prepare("--full" not in sys.argv)  # quick data (15% of the episodes) unless --full
print(f"prepare {time.time() - t0:.1f} s, rows {len(d.t):,}")
trm, vam = d.folds[1]
P = {"max_depth": 6, "min_samples_leaf": 300, "learning_rate": 0.05, "l2": 1.0, "max_iter": 150, "max_leaf_nodes": 31}


def fit():
    return m.fit_score("hgb", P, d, trm, vam)


for n in sorted({1, 2, 4, 8, 16, aff} & set(range(1, aff + 1))):
    with threadpool_limits(limits=n):
        t = time.time()
        fit()
        print(f"hgb, {n:2d} threads: {time.time() - t:6.1f} s", flush=True)

k = max(aff // 3, 1)
with threadpool_limits(limits=k), ThreadPoolExecutor(3) as ex:
    t = time.time()
    list(ex.map(lambda _: fit(), range(3)))
    print(f"3 fits at once, {k} threads each (= {3 * k} threads): {time.time() - t:6.1f} s")


def run_workers(procs, thr, reps=2):
    cmd = [sys.executable, str(Path(__file__).resolve()), *(["--full"] if "--full" in sys.argv else []), "--worker", str(thr), str(reps)]
    ps = [subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True) for _ in range(procs)]
    spans = [tuple(map(float, p.communicate()[0].split()[-2:])) for p in ps]
    wall = max(b for _, b in spans) - min(a for a, _ in spans)
    print(f"{procs:2d} processes x {thr} threads, {procs * reps} fits: {wall:6.1f} s  ->  {procs * reps / wall:5.2f} fits/s", flush=True)


for procs, thr in [(1, 4), (2, 2), (4, 2), (max(aff // 2, 1), 2), (aff, 1)]:
    run_workers(procs, thr)
