"""File fingerprints that survive harmless metadata changes, and the data lock built from them.

Why two hashes. `sha256` is the hash of the bytes: exact, but several sources change bytes without changing data
(measured, see docs/DATA_SOURCES.md): THREDDS NetCDF subsets embed a translation timestamp in the `History`
attribute, NWIS `.rdb` files carry a `# retrieved:` line, and `np.savez` stamps zip members with the current time.
`content` hashes what the data IS, with those volatile parts left out:

  *.gz            the decompressed stream (the gzip header holds an mtime)
  *.zip, *.npz    sorted (member name, CRC-32, size) after CRC-checking every member: ignores member timestamps
                  and compression level, still catches damaged data
  *.nc            every variable's name, dtype, dims, attributes and values; global attributes are ignored
  *.parquet       schema + logical rows re-cut into fixed windows, so row-group layout, compression, writer
                  version and key-value metadata do not matter
  *.rdb           the text minus the `# retrieved:` line
  anything else   the bytes (kind 'bytes')

Verification order (`check`): identical bytes -> ok; else identical content -> ok-with-metadata-change; else the
file CHANGED upstream (or is corrupt: `corrupt` is reported when it cannot be opened at all).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

BLOCK = 1 << 20
WINDOW = 1 << 20  # parquet rows per hashed window
VOLATILE_LINE = {".rdb": re.compile(rb"^# retrieved:")}


# ---------------------------------------------------------------- hashes

def _blocks(f, n: int = BLOCK) -> Iterator[bytes]:
    while b := f.read(n):
        yield b


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in _blocks(f):
            h.update(b)
    return h.hexdigest()


def md5_file(path: str | Path) -> str:
    h = hashlib.md5()  # noqa: S324 - S3's own ETag is an MD5; used for comparison, not for security
    with open(path, "rb") as f:
        for b in _blocks(f):
            h.update(b)
    return h.hexdigest()


def _zip_members(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        if bad := z.testzip():  # the central directory is intact even when member data is damaged: read and CRC-check
            raise zipfile.BadZipFile(f"CRC mismatch in member {bad}")
        lines = [f"{i.filename}\0{i.CRC}\0{i.file_size}" for i in sorted(z.infolist(), key=lambda i: i.filename)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _gunzip(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as g:
        for b in _blocks(g):
            h.update(b)
    return h.hexdigest()


def _text_minus_volatile(path: Path, pat: re.Pattern) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for line in f:
            if not pat.match(line):
                h.update(line)
    return h.hexdigest()


def _netcdf(path: Path) -> str:
    import xarray as xr

    h = hashlib.sha256()
    with xr.open_dataset(path, decode_times=False, mask_and_scale=False, decode_cf=False) as ds:
        for name in sorted(ds.variables):
            v = ds.variables[name]
            h.update(f"{name}|{v.dtype}|{v.dims}|{v.shape}|{sorted((k, repr(a)) for k, a in v.attrs.items())}".encode())
            h.update(np.ascontiguousarray(v.values).tobytes())
    return h.hexdigest()


def feed_table(h, t) -> None:
    """Hash a table's logical content. Per column: the validity mask, then the valid values only. Arrow leaves the
    bytes under a null slot undefined (seen on real data: residue from the reader differed by row-group layout),
    so raw buffers must never be hashed. Nulls inside nested types are not canonicalised (none in our data)."""
    import pyarrow as pa
    import pyarrow.compute as pc

    for name in t.column_names:
        col = t.column(name).combine_chunks()
        h.update(f"{name}|{col.type}|{len(col)}".encode())
        h.update(np.packbits(np.asarray(pc.is_valid(col))).tobytes())
        sink = pa.BufferOutputStream()
        valid = pa.table({"v": col.drop_null()})
        with pa.ipc.new_stream(sink, valid.schema) as w:
            w.write_table(valid)
        h.update(sink.getvalue().to_pybytes())


def _parquet(path: Path) -> str:
    """Hash the logical table in fixed WINDOW-row slices, independent of row groups, compression and metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow.remove_metadata()
    h = hashlib.sha256(f"{schema}|{pf.metadata.num_rows}".encode())

    buf: list[pa.RecordBatch] = []
    n = 0
    for b in pf.iter_batches(batch_size=WINDOW):
        buf.append(b)
        n += b.num_rows
        while n >= WINDOW:
            t = pa.Table.from_batches(buf, schema=schema)
            feed_table(h, t.slice(0, WINDOW))
            rest = t.slice(WINDOW)
            buf, n = (rest.to_batches(), rest.num_rows)
    if n:
        feed_table(h, pa.Table.from_batches(buf, schema=schema))
    return h.hexdigest()


def content_hash(path: str | Path) -> tuple[str, str]:
    """(kind, hex) of the file's data with volatile metadata left out. See the module docstring."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".gz"):
        return "gunzip", _gunzip(p)
    if name.endswith((".zip", ".npz")):
        return "zip-members", _zip_members(p)
    if name.endswith((".nc", ".nc4")):
        return "netcdf-vars", _netcdf(p)
    if name.endswith(".parquet"):
        return "parquet-rows", _parquet(p)
    if p.suffix.lower() in VOLATILE_LINE:
        return "text-minus-volatile", _text_minus_volatile(p, VOLATILE_LINE[p.suffix.lower()])
    return "bytes", sha256_file(p)


def fingerprint(path: str | Path, *, content: bool = True) -> dict:
    p = Path(path)
    fp: dict = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    if content:
        kind, hx = content_hash(p)
        fp["content_kind"], fp["content_sha256"] = kind, hx
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq

        fp["rows"] = pq.ParquetFile(p).metadata.num_rows
    return fp


# ---------------------------------------------------------------- lock

def check(path: str | Path, entry: dict) -> str:
    """Compare a local file with its lock entry. One of: ok, ok-metadata-differs, missing, size-differs,
    changed, corrupt."""
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.stat().st_size != entry["bytes"] and entry.get("content_kind") == "bytes":
        return "size-differs"
    if p.stat().st_size == entry["bytes"] and sha256_file(p) == entry["sha256"]:
        return "ok"
    if "content_sha256" not in entry:
        return "changed"
    try:
        kind, hx = content_hash(p)
    except Exception:  # noqa: BLE001 - truncated zip, broken gzip, unreadable netcdf/parquet
        return "corrupt"
    if kind == entry["content_kind"] and hx == entry["content_sha256"]:
        return "ok-metadata-differs"
    return "changed"


def load_lock(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_lock(path: str | Path, lock: dict) -> None:
    Path(path).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
