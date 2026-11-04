"""Archive large .npy / .npz files in $WXRISK_DATA as zstd (`<name>.zst`), verified, then delete the original.

Inputs : $WXRISK_DATA/{cnn,cnn_tx,cnn_fl,cnn_la,cnn_pool,labels_extra/flood/pilot}/*.npy|*.npz of at least --min-mb
Outputs: `<file>.zst` next to each (zstd level 19); the original is removed after a byte-exact round-trip check
         (wxrisk.zstio.compress). Files that zstd shrinks by less than half (dense float32 model outputs) stay as they are.
Read   : scripts use `wxrisk.zstio.load(path)`, which finds either form (no mmap for compressed files).
Undo   : `wxrisk.zstio.restore(path)`.
Command: uv run python scripts/107_zstd_archive.py [--min-mb 100] [--level 19] [--keep] [--dry-run]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2, zstio  # noqa: E402

DIRS = ["cnn", "cnn_tx", "cnn_fl", "cnn_la", "cnn_pool", "labels_extra/flood/pilot"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mb", type=float, default=100)
    ap.add_argument("--level", type=int, default=zstio.LEVEL)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    root = Path(wb2.DATA_DIR)
    files = sorted(
        f
        for d in DIRS
        for f in (root / d).glob("*")
        if f.suffix in {".npy", ".npz"} and f.stat().st_size >= a.min_mb * 1e6
    )
    tot_in = tot_out = 0
    for f in files:
        rel = f.relative_to(root)
        if a.dry_run:
            print(f"would archive {rel} {f.stat().st_size / 1e6:.0f} MB")
            continue
        n_in, n_out = zstio.compress(f, level=a.level, keep=a.keep)
        tot_in += n_in
        tot_out += n_out
        tag = "kept as is" if n_out == n_in else f"{n_out / n_in:.3f}"
        print(f"{rel}: {n_in / 1e6:.0f} -> {n_out / 1e6:.0f} MB ({tag})", flush=True)
    print(f"total {tot_in / 1e9:.2f} -> {tot_out / 1e9:.2f} GB over {len(files)} files")


if __name__ == "__main__":
    main()
