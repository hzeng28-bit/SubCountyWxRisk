"""Is file I/O what slows a Sol job down? Measures the shared scratch against node-local disk with this project's own files.
Prints: where scratch, /tmp, $TMPDIR and /dev/shm live and how big they are; sequential read speed of the two big pilot inputs (first
read, second read, and after a copy to local disk); eight processes reading one file at once; 2000 small files written and read back.
Only numpy and the standard library are used; the probe cleans up everything it writes.
  sbatch --qos=debug --time=0-00:15 --cpus-per-task=8 --mem=16G hpc/run.sbatch hpc/io_probe.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("WXRISK_DATA", Path.home() / "wxrisk_data"))
FILES = [DATA / "labels_extra/flood/pilot/flood_pilot_table.parquet", DATA / "clean/claims_nfip.parquet"]
CHUNK = 8 << 20


def read_all(p) -> int:
    n = 0
    with open(p, "rb", buffering=0) as f:
        while b := f.read(CHUNK):
            n += len(b)
    return n


def timed_read(p, label):
    t = time.time()
    n = read_all(p)
    dt = time.time() - t
    print(f"  {label:34s} {n / 1e6:8.0f} MB in {dt:6.2f} s  = {n / 1e6 / dt:8.0f} MB/s", flush=True)


def small_files(root: Path, n=2000, size=100_000):
    root.mkdir(parents=True, exist_ok=True)
    blob = np.random.default_rng(0).bytes(size)
    t = time.time()
    for i in range(n):
        (root / f"f{i}.bin").write_bytes(blob)
    w = time.time() - t
    t = time.time()
    for i in range(n):
        (root / f"f{i}.bin").read_bytes()
    r = time.time() - t
    print(f"  {n} files of {size // 1000} KB in {root.parent.name}: write {w:5.2f} s ({n / w:6.0f} files/s), read {r:5.2f} s ({n / r:6.0f} files/s)", flush=True)


if __name__ == "__main__":
    print("== where things live")
    for name, path in [("scratch (WXRISK_DATA)", DATA), ("/tmp", Path("/tmp")), ("$TMPDIR", Path(os.environ.get("TMPDIR", "/tmp"))), ("/dev/shm", Path("/dev/shm"))]:
        if path.exists():
            r = subprocess.run(["df", "-hT" if sys.platform.startswith("linux") else "-h", str(path)], capture_output=True, text=True)  # BSD df has no -T
            out = r.stdout.strip().splitlines()[-1]
            print(f"  {name:22s} {out}")
    print(f"  TMPDIR={os.environ.get('TMPDIR')}  SLURM_TMPDIR={os.environ.get('SLURM_TMPDIR')}  node {os.uname().nodename}")

    local = Path(tempfile.mkdtemp(prefix="wxrisk_io_"))
    try:
        print("\n== sequential read of the pilot inputs")
        for p in FILES:
            if not p.exists():
                print("  missing", p)
                continue
            timed_read(p, f"{p.name} first read (scratch)")
            timed_read(p, f"{p.name} second read (scratch)")
            t = time.time()
            shutil.copy(p, local / p.name)
            print(f"  {p.name} copy scratch -> local: {time.time() - t:6.2f} s")
            timed_read(local / p.name, f"{p.name} read (local copy)")

        if FILES[0].exists():
            print("\n== eight processes read the same table at once (scratch)")
            t = time.time()
            with ProcessPoolExecutor(8) as ex:
                n = sum(ex.map(read_all, [FILES[0]] * 8))
            dt = time.time() - t
            print(f"  total {n / 1e6:.0f} MB in {dt:.2f} s = {n / 1e6 / dt:.0f} MB/s")

        print("\n== many small files")
        scratch_dir = DATA / "hpc" / f"io_probe_{os.getpid()}" / "files"
        small_files(scratch_dir)
        small_files(local / "files")
        shutil.rmtree(scratch_dir.parent)
    finally:
        shutil.rmtree(local, ignore_errors=True)
    print("\n== python environment: where it lives and what importing costs")
    site = Path(sys.prefix)
    print(f"  interpreter prefix {site}  UV_PROJECT_ENVIRONMENT={os.environ.get('UV_PROJECT_ENVIRONMENT')}")
    t = time.time()
    n = sum(len(f) for _, _, f in os.walk(site))
    print(f"  walking the environment: {n:,} files in {time.time() - t:5.2f} s")
    for k in (1, 2, 3):
        t = time.time()
        subprocess.run([sys.executable, "-c", "import numpy, pandas, scipy, sklearn, pyarrow"], capture_output=True)
        print(f"  fresh interpreter, import numpy pandas scipy sklearn pyarrow (run {k}): {time.time() - t:5.2f} s")

    print("\n== the phases of prepare() in script 54")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from wxrisk import flood_pilot as fp

    import pandas as pd

    for label, fn in [("read pilot table", lambda: pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")),
                      ("claims_tx (clean claims)", lambda: fp.claims_tx()), ("bg_geometry", lambda: fp.bg_geometry())]:
        t = time.time()
        fn()
        print(f"  {label:28s} {time.time() - t:6.2f} s", flush=True)
    print("\ndone")
