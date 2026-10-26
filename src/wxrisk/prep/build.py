"""Build, record and verify clean tables.

The sidecar (`<name>.prep.json`) is what makes a table reproducible: it names the exact input (content hash), the exact
rules (spec hash + cleaning version) and what came out (content hash, row accounting). Nothing in it depends on the
time or the machine, so two people building from the same raw file get byte-identical sidecars.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import polars as pl

from .. import integrity, wb2
from .clean import clean, raw_files, raw_lazy
from .spec import TableSpec

CLEAN_DIR = wb2.DATA_DIR / "clean"
ROW_GROUP = 1 << 20


def paths(spec: TableSpec) -> tuple[list[Path], Path, Path]:
    return raw_files(spec.source, wb2.DATA_DIR), CLEAN_DIR / f"{spec.name}.parquet", CLEAN_DIR / f"{spec.name}.prep.json"


def input_hash(files: list[Path]) -> str:
    """One hash for the table's input: content hashes of its files, in name order (a single file: its own hash)."""
    hs = [integrity.content_hash(f)[1] for f in files]
    return hs[0] if len(hs) == 1 else hashlib.sha256("\n".join(f"{f.name}:{h}" for f, h in zip(files, hs, strict=True)).encode()).hexdigest()


def write_table(df: pl.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    df.write_parquet(part, compression="zstd", row_group_size=ROW_GROUP, statistics=True)
    part.replace(dest)  # atomic: a crash never leaves a truncated table that looks finished


def build(spec: TableSpec, force: bool = False) -> dict:
    src, dest, side = paths(spec)
    in_hash = input_hash(src)
    if side.exists() and dest.exists() and not force:
        old = json.loads(side.read_text())
        if old["spec_sha256"] == spec.digest() and old["input_content_sha256"] == in_hash:
            return {**old, "status": "up-to-date"}
    df, report = clean(raw_lazy(src), spec)
    write_table(df, dest)
    rec = {"name": spec.name, "source": spec.source, "spec_sha256": spec.digest(), "prep_version": spec.version,
           "input_content_sha256": in_hash, "output_content_sha256": integrity.content_hash(dest)[1],
           "report": report}
    side.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return {**rec, "status": "built"}


def verify(spec: TableSpec) -> dict:
    """Re-derive the table from the raw file and compare with the stored sidecar and file. Writes nothing to clean/."""
    src, dest, side = paths(spec)
    if not (side.exists() and dest.exists()):
        return {"status": "missing"}
    rec = json.loads(side.read_text())
    checks = {
        "spec_unchanged": rec["spec_sha256"] == spec.digest(),
        "input_unchanged": rec["input_content_sha256"] == input_hash(src),
        "stored_file_matches_sidecar": integrity.content_hash(dest)[1] == rec["output_content_sha256"],
    }
    with tempfile.TemporaryDirectory() as td:
        df, _ = clean(raw_lazy(src), spec)
        tmp = Path(td) / "again.parquet"
        write_table(df, tmp)
        checks["rebuild_identical"] = integrity.content_hash(tmp)[1] == rec["output_content_sha256"]
    return {"status": "ok" if all(checks.values()) else "MISMATCH", **checks}


def scan(name: str) -> pl.LazyFrame:
    """Lazy scan of a clean table: the one way analysis code should read cleaned data."""
    from .specs import SPECS

    dest = paths(SPECS[name])[1]
    if not dest.exists():
        raise FileNotFoundError(f"{dest} is missing: run  python scripts/40_prep.py build {name}")
    return pl.scan_parquet(dest)
