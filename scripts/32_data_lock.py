"""Data lock: fingerprints of the files under $WXRISK_DATA, so a teammate can check they hold the same data.

The upstream sources publish almost no checksums, and several change bytes without changing data (see
wxrisk/integrity.py), so each entry carries two hashes: `sha256` of the bytes and `content_sha256` of the data with
volatile metadata removed. Verification passes on identical bytes, or on identical content (reported as
'ok-metadata-differs'). Sources that legitimately change upstream are flagged `mutable`: a difference there is a
warning ('upstream changed'), not a failure.

    python scripts/32_data_lock.py write                  # (re)compute data.lock.json; unchanged files are not rehashed
    python scripts/32_data_lock.py verify                 # every locked file
    python scripts/32_data_lock.py verify --category raw --allow-missing   # a teammate with a partial data dir
    python scripts/32_data_lock.py verify --quick         # existence + size only

Categories: raw (downloaded as-is), api (assembled by us from an API), derived (produced by scripts from the others).
Skipped: logs, result CSVs, shell/html leftovers, the audit/ dir and the terrain tile cache.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import integrity, wb2  # noqa: E402

LOCK = Path(__file__).resolve().parents[1] / "data.lock.json"
CACHE = wb2.DATA_DIR / ".lock_cache.json"  # per-machine: (size, mtime) -> entry, so re-runs skip unchanged files
ALGO = 2  # bump when a content-hash definition changes

SKIP_DIRS = {"audit", "terrain_tiles"}
SKIP_SUFFIX = {".log", ".sh", ".csv", ".html", ".part", ".tmp"}
SKIP_NAMES = {".DS_Store", ".lock_cache.json", "manifest.json"}
DEEP_LIMIT = 1 << 30  # parquet / NetCDF above this get a bytes hash and a row count only (content pass is slow)

# first match wins
CATEGORY = [
    ("storm_events/*", "raw"), ("labels_extra/spc/*", "raw"), ("labels_extra/rma/*", "raw"),
    ("labels_extra/iem/*", "raw"), ("labels_extra/census/*", "raw"), ("labels_extra/mesh/*", "raw"),
    ("labels_extra/qpe/*", "raw"), ("labels_extra/flood/nwis/*", "raw"), ("labels_extra/flood/nwis_*", "raw"),
    ("labels_extra/flood/nfip_policies_v3.parquet", "raw"), ("labels_extra/flood/pilot/gridmet/*", "raw"),
    ("labels_extra/flood/pilot/bg20*_tx.zip", "raw"), ("labels_extra/flood/pilot/hurdat2*.txt", "raw"),
    ("cds/*", "raw"), ("idx/*", "raw"),
    ("labels_extra/flood/nfip_national/*", "api"), ("labels_extra/flood/nfip_claims_conmaz.parquet", "api"),
    ("labels_extra/flood/fema_flood_declarations.parquet", "api"), ("labels_extra/flood/nfip_penetration.parquet", "api"),
]
# upstream keeps changing these in place (provisional data, rolling releases), so a difference is a warning
MUTABLE = ["labels_extra/flood/nwis/*", "labels_extra/flood/nwis_*", "labels_extra/spc/*", "labels_extra/iem/*",
           "labels_extra/rma/*", "labels_extra/flood/nfip_policies_v3.parquet", "labels_extra/flood/nfip_*.parquet",
           "labels_extra/flood/nfip_national/*", "labels_extra/flood/fema_*.parquet", "labels_extra/flood/pilot/gridmet/*"]


def category(rel: str) -> str:
    return next((c for pat, c in CATEGORY if fnmatch.fnmatch(rel, pat)), "derived")


def included(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    return (p.is_file() and p.name not in SKIP_NAMES and p.suffix not in SKIP_SUFFIX
            and not (set(rel.parts[:-1]) & SKIP_DIRS) and ".part" not in p.name)


def files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if included(p, root))


def fingerprint(p: Path) -> dict:
    deep = p.suffix in {".parquet", ".nc", ".nc4"} and p.stat().st_size <= DEEP_LIMIT
    fp = integrity.fingerprint(p, content=deep or p.suffix not in {".parquet", ".nc", ".nc4"})
    if not deep and p.suffix in {".parquet", ".nc", ".nc4"}:
        fp["content_kind"] = "skipped-large"
    return fp


def cmd_write(a) -> None:
    root = wb2.DATA_DIR
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = files(root)

    def one(p: Path) -> tuple[str, dict]:
        rel, st = str(p.relative_to(root)), p.stat()
        c = cache.get(rel)
        if c and c["size"] == st.st_size and c["mtime_ns"] == st.st_mtime_ns and c["algo"] == ALGO:
            return rel, c["entry"]
        return rel, fingerprint(p)

    entries: dict[str, dict] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (rel, e) in enumerate(ex.map(one, todo), 1):
            e["category"] = category(rel)
            if any(fnmatch.fnmatch(rel, m) for m in MUTABLE):
                e["mutable"] = True
            entries[rel] = e
            if i % 200 == 0:
                print(f"  hashed {i}/{len(todo)}", flush=True)
    CACHE.write_text(json.dumps({r: {"size": (root / r).stat().st_size, "mtime_ns": (root / r).stat().st_mtime_ns,
                                     "algo": ALGO, "entry": {k: v for k, v in e.items() if k not in ("category", "mutable")}}
                                 for r, e in entries.items()}))
    integrity.write_lock(LOCK, {"algo": ALGO, "root": "$WXRISK_DATA", "entries": entries})
    n = Counter(e["category"] for e in entries.values())
    gb = sum(e["bytes"] for e in entries.values()) / 1e9
    print(f"locked {len(entries)} files, {gb:.1f} GB {dict(n)} -> {LOCK}")


def cmd_verify(a) -> None:
    root = wb2.DATA_DIR
    lock = integrity.load_lock(LOCK)
    if lock.get("algo") != ALGO:
        sys.exit(f"lock algo {lock.get('algo')} != {ALGO}: regenerate the lock")
    items = [(r, e) for r, e in lock["entries"].items() if a.category in (None, e["category"])
             and (not a.only or any(fnmatch.fnmatch(r, g) for g in a.only))]

    def one(item) -> tuple[str, str]:
        rel, e = item
        p = root / rel
        if a.quick:
            return rel, ("missing" if not p.exists() else "ok" if p.stat().st_size == e["bytes"] else "size-differs")
        return rel, integrity.check(p, e)

    res: dict[str, str] = {}
    with ThreadPoolExecutor(a.workers) as ex:
        for rel, status in ex.map(one, items):
            res[rel] = status
    mut = {r for r, e in items if e.get("mutable")}
    bad = {r: s for r, s in res.items() if s not in ("ok", "ok-metadata-differs")
           and not (s == "missing" and a.allow_missing) and not (r in mut and s in ("changed", "size-differs"))}
    warn = {r: s for r, s in res.items() if r in mut and s in ("changed", "size-differs")}
    print(dict(Counter(res.values())), f"| {len(items)} checked")
    for r, s in sorted(warn.items())[:20]:
        print(f"  WARN upstream changed (mutable source): {r} [{s}]")
    for r, s in sorted(bad.items())[:50]:
        print(f"  FAIL {s}: {r}")
    sys.exit(1 if bad else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--workers", type=int, default=4)
    w.set_defaults(fn=cmd_write)
    v = sub.add_parser("verify")
    v.add_argument("--category", choices=["raw", "api", "derived"])
    v.add_argument("--only", nargs="+", help="glob(s) on the relative path")
    v.add_argument("--quick", action="store_true", help="existence and size only")
    v.add_argument("--allow-missing", action="store_true")
    v.add_argument("--workers", type=int, default=4)
    v.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
