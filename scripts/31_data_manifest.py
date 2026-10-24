"""Write a machine-readable manifest of what sits under $WXRISK_DATA (sizes, parquet schemas, row counts).

For people and AI agents who need the shape of the data without opening 13 GB: parquet files get row count
and column dtypes (from the footer, no data read); every other file is listed with its size, and directories
of many similar files (daily grib2, zips, per-year parquet) are collapsed to a count and total size.
Paths are relative to $WXRISK_DATA. Output: docs/data_manifest.json (small, safe to commit).

    python scripts/31_data_manifest.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "data_manifest.json"
SKIP = {".DS_Store"}
COLLAPSE_AT = 8  # directories with more files of one family than this are summarised


def family(name: str) -> str:
    """Group per-year / per-day files (scalefeat_2003.parquet -> scalefeat_#.parquet), and per-tile files
    (terrain_tiles/10_232_413.png -> #_#_#.png), which carry short numbers."""
    return re.sub(r"\d+", "#", name) if re.fullmatch(r"[\d_]+\.\w+", name) else re.sub(r"\d{4,8}", "#", name)


def parquet_info(p: Path) -> dict:
    md = pq.ParquetFile(p).metadata
    return {"rows": md.num_rows, "columns": {md.schema.column(i).name: str(md.schema.column(i).physical_type)
                                              for i in range(md.num_columns)}}


def main() -> None:
    root = wb2.DATA_DIR
    files: dict[str, dict] = {}
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.name in SKIP:
            continue
        groups[(str(p.parent.relative_to(root)), family(p.name))].append(p)
    for (parent, fam), ps in groups.items():
        if len(ps) > COLLAPSE_AT:
            ex = ps[0]
            info = {"collapsed": True, "files": len(ps), "bytes": sum(q.stat().st_size for q in ps),
                    "example": ex.name}
            if ex.suffix == ".parquet":
                info["example_schema"] = parquet_info(ex)
            files[f"{parent}/{fam}".lstrip("./")] = info
            continue
        for p in ps:
            rel = str(p.relative_to(root))
            info = {"bytes": p.stat().st_size}
            if p.suffix == ".parquet":
                try:
                    info.update(parquet_info(p))
                except Exception as e:  # noqa: BLE001 - a corrupt/partial file should not stop the manifest
                    info["error"] = repr(e)
            files[rel] = info
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"root": "$WXRISK_DATA", "n_entries": len(files), "entries": files}, indent=1,
                              sort_keys=True))
    total = sum(v["bytes"] for v in files.values())
    print(f"{len(files)} entries, {total / 1e9:.1f} GB -> {OUT}")


if __name__ == "__main__":
    main()
