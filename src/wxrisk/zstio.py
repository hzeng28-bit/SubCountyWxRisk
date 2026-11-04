"""Zstandard-compressed .npy / .npz files that load like plain ones.

Why: the large arrays (day maps, row tables, rain stacks) are mostly zeros or repeated values and shrink 10-60x with zstd.
Archived files are `<name>.npy.zst` / `<name>.npz.zst` (the frame holds the original file bytes); `zstio.load("<name>.npy")`
finds either form, so scripts do not care which one is on disk.

Choices, from the benchmark in RESULTS.md (stage 53): plain zstd level 19 beats byte-shuffle + zstd on rain maps and row tables
(0.035 vs 0.059 on pr_tx, 0.015 vs 0.018 on rows_tab), a long window and trained dictionaries gave nothing (dictionaries only
help files below ~50 KB), and scipy.sparse / COO are larger than dense + zstd at 20-40% density. Dense float32 model
outputs (p89 probabilities) only reach ~0.7 and are left uncompressed (MAX_RATIO).

Cost: no `mmap_mode` for compressed files (the array is decompressed into RAM; `mmap_mode` is ignored then), so keep the number
of simultaneously open big arrays in mind (~2-3 GB each for the largest). Producers that reopen an existing file in place
(`open_memmap(mode="r+")`) need `restore(path)` first.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import numpy as np
import zstandard

LEVEL = 19  # archive once, read often: decompression speed does not depend on the level
CHUNK = 1 << 24
MAX_RATIO = 0.5  # keep the original when zstd saves less than half (mmap is worth more than a small saving)


def zst_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".zst")


def exists(path: str | Path) -> bool:
    return Path(path).exists() or zst_path(path).exists()


def glob(directory: str | Path, pattern: str) -> list[Path]:
    """Sorted plain names matching `pattern`, counting archived `<name>.zst` files once (feed them to `load`)."""
    d = Path(directory)
    names = {p.name for p in d.glob(pattern)} | {p.name[:-4] for p in d.glob(pattern + ".zst")}
    return [d / n for n in sorted(names)]


def load(path, mmap_mode=None, **kw):
    """np.load that falls back to `<path>.zst`. For compressed files `mmap_mode` is ignored."""
    if not isinstance(path, (str, os.PathLike)):  # file object / buffer: plain np.load
        return np.load(path, mmap_mode=mmap_mode, **kw)
    p = Path(path)
    if p.exists() or not zst_path(p).exists():
        return np.load(p, mmap_mode=mmap_mode, **kw)
    with open(zst_path(p), "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        if p.suffix == ".npy":
            return np.lib.format.read_array(dctx.stream_reader(f), allow_pickle=kw.get("allow_pickle", False))
        buf = io.BytesIO()  # .npz is a zip: needs seek
        dctx.copy_stream(f, buf, read_size=CHUNK, write_size=CHUNK)
    buf.seek(0)
    return np.load(buf, **kw)


def compress(path: str | Path, level: int = LEVEL, keep: bool = False, max_ratio: float = MAX_RATIO) -> tuple[int, int]:
    """Compress `path` to `<path>.zst`, verify the round trip byte for byte, then delete the original unless `keep`.

    Atomic (.part then rename) and idempotent. Returns (original bytes, stored bytes); stored == original when the file
    was left as it is (already archived, or ratio above `max_ratio`).
    """
    p = Path(path)
    z = zst_path(p)
    if not p.exists():
        if z.exists():
            return z.stat().st_size, z.stat().st_size
        raise FileNotFoundError(p)
    with open(p, "rb") as f:  # cheap probe: three 4 MB windows at level 3 spare level 19 on dense data
        size, probe_in, probe_out = p.stat().st_size, 0, 0
        for off in (size // 4, size // 2, 3 * size // 4):
            f.seek(off)
            blk = f.read(1 << 22)
            probe_in += len(blk)
            probe_out += len(zstandard.ZstdCompressor(level=3).compress(blk))
    if probe_in and probe_out > (max_ratio + 0.2) * probe_in:
        return size, size
    part = z.with_name(z.name + ".part")
    src_hash = hashlib.sha256()
    cctx = zstandard.ZstdCompressor(level=level, threads=-1)
    with open(p, "rb") as fi, open(part, "wb") as fo, cctx.stream_writer(fo, closefd=False) as w:
        while b := fi.read(CHUNK):
            src_hash.update(b)
            w.write(b)
    n_in, n_out = p.stat().st_size, part.stat().st_size
    if n_out > max_ratio * n_in:
        part.unlink()
        return n_in, n_in
    back = hashlib.sha256()
    with open(part, "rb") as f:
        r = zstandard.ZstdDecompressor().stream_reader(f)
        while b := r.read(CHUNK):
            back.update(b)
    if back.hexdigest() != src_hash.hexdigest():
        part.unlink()
        raise RuntimeError(f"round-trip mismatch for {p}")
    part.replace(z)
    if not keep:
        p.unlink()
    return n_in, n_out


def restore(path: str | Path, keep: bool = False) -> Path:
    """Decompress `<path>.zst` back to `<path>` (for scripts that need mmap or in-place writes)."""
    p = Path(path)
    z = zst_path(p)
    if p.exists():
        return p
    part = p.with_name(p.name + ".part")
    with open(z, "rb") as fi, open(part, "wb") as fo:
        zstandard.ZstdDecompressor().copy_stream(fi, fo, read_size=CHUNK, write_size=CHUNK)
    part.replace(p)
    if not keep:
        z.unlink()
    return p
