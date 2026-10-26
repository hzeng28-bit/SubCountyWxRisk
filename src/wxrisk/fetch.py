"""One place for every download, so runs are reproducible across operating systems.

Channel by source type (see docs/DATA_SOURCES.md):
  vendor API            use the vendor's library (cdsapi / ecmwf-datastores for CDS; gcsfs/zarr for WeatherBench-2)
  object storage (S3)   `s3` / `s3_bytes` / `s3_range`  (boto3, anonymous by default)
  plain HTTP(S) file    `https`            (urllib; no curl, so Windows works)
  small HTTP(S) payload `get_bytes` / `get_json`
  many files            `fetch_many`       (one space check for the whole batch, then parallel downloads)

What the file downloads guarantee:
  * atomic: data goes to `<dest>.part` and is renamed only when complete; a half file never looks finished
  * resumable when the size is known and the server honours Range: progress is kept in `<dest>.part.json`
    (source, ETag, size, bytes written), so a changed remote file restarts instead of being spliced
  * space is checked before the transfer (`InsufficientSpace` names the shortfall) and reserved up front where
    the OS can do it: posix_fallocate (Linux), F_PREALLOCATE (macOS), file extension via truncate (Windows).
    Not every filesystem honours it; the free-space check always runs, the reservation is best effort.
  * unknown size (chunked / generated responses): no resume, no reservation; pass `size_hint` to still get a check
  * verified when the source gives something to verify against: S3 objects against their ETag (an MD5 for
    single-part uploads; multipart ETags are not hashes and are skipped), any file against `expect_sha256`, OpenFEMA
    pulls against the server's own row count. A mismatch discards the partial file and retries from scratch.
    Most plain-HTTP sources publish no content hash (their ETag is size+mtime), see docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import errno
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

CHUNK = 1 << 20
SAVE_EVERY = 16 << 20  # flush + fsync + write progress every this many bytes
MARGIN = 256 << 20  # always keep this much free beyond what a transfer needs
UA = "Mozilla/5.0 (compatible; wxrisk-research)"  # some public servers reject unknown agents
S3_NO_RETRY = {"NoSuchKey", "404", "NotFound", "403", "AccessDenied", "NoSuchBucket"}
NO_RETRY_HTTP = set(range(400, 500)) - {408, 425, 429}


class InsufficientSpace(OSError):
    """Not enough free disk space for the transfer."""


class ChecksumMismatch(OSError):
    """Downloaded bytes do not match the hash the source (or the data lock) promised."""


def _verifier(md5: str | None, sha256: str | None) -> Callable[[Path], None] | None:
    if not (md5 or sha256):
        return None
    from . import integrity

    def verify(p: Path) -> None:
        if md5 and (got := integrity.md5_file(p)) != md5:
            raise ChecksumMismatch(f"md5 {got} != expected {md5}")
        if sha256 and (got := integrity.sha256_file(p)) != sha256:
            raise ChecksumMismatch(f"sha256 {got} != expected {sha256}")

    return verify


# ---------------------------------------------------------------- space

def free_bytes(path: str | Path) -> int:
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def require_space(path: str | Path, need: int, margin: int = MARGIN) -> None:
    """Raise InsufficientSpace unless `need` bytes plus `margin` are free on the filesystem of `path`."""
    free = free_bytes(path)
    if need > 0 and free < need + margin:
        raise InsufficientSpace(
            f"need {need / 1e9:.2f} GB (+{margin / 1e6:.0f} MB margin) under {path}, only {free / 1e9:.2f} GB free")


def _preallocate(f, size: int, written: int) -> str:
    """Reserve disk space for the rest of the file. Returns the method used ('none' if unsupported)."""
    fd = f.fileno()
    try:
        if hasattr(os, "posix_fallocate"):  # Linux, FreeBSD
            try:
                os.posix_fallocate(fd, 0, size)
                return "fallocate"
            except OSError as e:
                if e.errno in (errno.ENOSPC, errno.EDQUOT):
                    raise
                return "none"  # EOPNOTSUPP/EINVAL: filesystem cannot, keep going
        if sys.platform == "darwin":
            import fcntl
            import struct

            f_preallocate, contig, anyspace, peof = 42, 2, 4, 3  # <sys/fcntl.h>
            # the contiguous attempt also fails with ENOSPC when only fragmented space is left, so that error
            # is only final for the last (fragmented-allowed) attempt
            for flags in (contig | anyspace, anyspace):
                try:
                    fcntl.fcntl(fd, f_preallocate, struct.pack("IIqqq", flags, peof, 0, size - written, 0))
                    return "f_preallocate"
                except OSError as e:
                    if flags == anyspace and e.errno in (errno.ENOSPC, errno.EDQUOT):
                        raise
            return "none"
        if os.name == "nt":  # NTFS allocates clusters when the end-of-file is extended
            f.truncate(size)
            return "truncate"
    except OSError as e:
        if e.errno in (errno.ENOSPC, errno.EDQUOT):
            raise InsufficientSpace(f"could not reserve {size / 1e9:.2f} GB: {e}") from e
        raise
    return "none"


# ---------------------------------------------------------------- core transfer

@dataclass
class Info:
    size: int | None
    etag: str | None = None
    ranges: bool = False


def _meta_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part.json")


def _part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def _load_meta(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _save_meta(p: Path, meta: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(meta))
    os.replace(tmp, p)


def pending_bytes(dest: str | Path, info: Info, source: str) -> int:
    """Bytes still to fetch for `dest`, counting a valid partial file as progress."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return 0
    if info.size is None:
        return 0
    m = _load_meta(_meta_path(dest))
    done = int(m.get("written", 0)) if (info.ranges and m.get("source") == source and m.get("etag") == info.etag
                                       and m.get("size") == info.size) else 0
    return max(info.size - done, 0)


def _transfer(dest: Path, *, source: str, info: Info, opener: Callable[[int], tuple[Iterable[bytes], int]],
              preallocate: bool, size_hint: int | None,
              verify: Callable[[Path], None] | None = None) -> Path:
    """Stream `opener(offset)` into `<dest>.part` and rename on success. `opener` returns (chunks, actual_offset)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part, mp = _part_path(dest), _meta_path(dest)
    size, resumable = info.size, info.size is not None and info.ranges
    meta = _load_meta(mp)
    written = 0
    if resumable and part.exists() and (meta.get("source"), meta.get("etag"), meta.get("size")) == (
            source, info.etag, size):
        written = int(meta.get("written", 0))
        if part.stat().st_size < written:
            written = 0
    if written == 0:
        part.unlink(missing_ok=True)
        mp.unlink(missing_ok=True)

    chunks, start = opener(written)
    if start != written:  # server restarted from 0 (no Range support / file changed)
        written = start
    need = (size - written) if size is not None else (size_hint or 0)
    require_space(dest.parent, need)

    f = open(part, "r+b" if part.exists() else "wb")
    try:
        if preallocate and size is not None:
            _preallocate(f, size, written)
        f.seek(written)
        last = written
        for b in chunks:
            f.write(b)
            written += len(b)
            if size is not None and written > size:
                raise OSError(f"{source}: received more than the advertised {size} bytes")
            if resumable and written - last >= SAVE_EVERY:
                f.flush()
                os.fsync(f.fileno())
                _save_meta(mp, {"source": source, "etag": info.etag, "size": size, "written": written})
                last = written
        if size is not None and written != size:
            raise OSError(f"{source}: short transfer, {written} of {size} bytes")
        f.truncate(written)
        f.flush()
        os.fsync(f.fileno())
    except OSError as e:
        if e.errno in (errno.ENOSPC, errno.EDQUOT):
            e = InsufficientSpace(f"disk full while writing {dest.name}; partial file kept for resume")
        try:
            f.flush()
            if resumable:
                _save_meta(mp, {"source": source, "etag": info.etag, "size": size, "written": written})
        except OSError:
            pass
        raise e from None
    except BaseException:
        if resumable:
            try:
                f.flush()
                _save_meta(mp, {"source": source, "etag": info.etag, "size": size, "written": written})
            except OSError:
                pass
        raise
    finally:
        f.close()
    if verify is not None:
        try:
            verify(part)
        except ChecksumMismatch:
            part.unlink(missing_ok=True)
            mp.unlink(missing_ok=True)
            raise
    os.replace(part, dest)
    mp.unlink(missing_ok=True)
    return dest


def _retry(fn: Callable[[], Path], tries: int, what: str) -> Path:
    for k in range(tries):
        try:
            return fn()
        except InsufficientSpace:
            raise
        except urllib.error.HTTPError as e:
            if e.code in NO_RETRY_HTTP or k == tries - 1:
                raise
        except Exception as e:  # noqa: BLE001 - network errors are retried; the last one propagates
            if k == tries - 1 or _s3_code(e) in S3_NO_RETRY:
                raise
        time.sleep(min(60, 2 ** k * 2))
    raise RuntimeError(f"failed: {what}")  # pragma: no cover


def _s3_code(e: Exception) -> str | None:
    resp = getattr(e, "response", None)
    return resp.get("Error", {}).get("Code") if isinstance(resp, dict) else None


def _iter(read: Callable[[int], bytes]) -> Iterator[bytes]:
    while b := read(CHUNK):
        yield b


# ---------------------------------------------------------------- HTTP(S)

def _req(url: str, headers: dict | None = None, method: str = "GET") -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})}, method=method)


def probe_https(url: str, headers: dict | None = None, timeout: float = 60) -> Info:
    """Size / ETag / Range support without downloading the body (HEAD, then a 1-byte Range GET)."""
    try:
        with urllib.request.urlopen(_req(url, headers, "HEAD"), timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            return Info(int(n) if n else None, r.headers.get("ETag"),
                        (r.headers.get("Accept-Ranges") or "").lower() == "bytes")
    except Exception:  # noqa: BLE001 - many servers reject HEAD; try the Range probe
        pass
    try:
        with urllib.request.urlopen(_req(url, {**(headers or {}), "Range": "bytes=0-0"}), timeout=timeout) as r:
            m = re.match(r"bytes \d+-\d+/(\d+)", r.headers.get("Content-Range") or "")
            if r.status == 206 and m:
                return Info(int(m.group(1)), r.headers.get("ETag"), True)
    except Exception:  # noqa: BLE001
        pass
    return Info(None)


def https(url: str, dest: str | Path, *, headers: dict | None = None, probe: bool = True,
          size_hint: int | None = None, preallocate: bool = True, retries: int = 6, timeout: float = 60,
          info: Info | None = None, expect_sha256: str | None = None) -> Path:
    """Download one HTTP(S) file to `dest` (skipped if it already exists and is non-empty).

    `probe=False` for generated responses where a HEAD would trigger the work twice (e.g. THREDDS subsets).
    """
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    def once() -> Path:
        i = info if info is not None else (probe_https(url, headers, timeout) if probe else Info(None))

        def opener(offset: int):
            h = dict(headers or {})
            if offset > 0:
                h["Range"] = f"bytes={offset}-"
                if i.etag:
                    h["If-Range"] = i.etag
            r = urllib.request.urlopen(_req(url, h), timeout=timeout)
            start = 0
            if r.status == 206:
                m = re.match(r"bytes (\d+)-", r.headers.get("Content-Range") or "")
                start = int(m.group(1)) if m else 0
            return _closing(r, _iter(r.read)), start

        return _transfer(dest, source=url, info=i, opener=opener, preallocate=preallocate, size_hint=size_hint,
                         verify=_verifier(None, expect_sha256))

    return _retry(once, retries, url)


def _closing(resp, it: Iterator[bytes]) -> Iterator[bytes]:
    try:
        yield from it
    finally:
        resp.close()


def get_bytes(url: str, *, headers: dict | None = None, retries: int = 5, timeout: float = 120) -> bytes:
    """Small HTTP(S) payload into memory, with retries."""
    for k in range(retries):
        try:
            with urllib.request.urlopen(_req(url, headers), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in NO_RETRY_HTTP or k == retries - 1:
                raise
        except Exception:  # noqa: BLE001
            if k == retries - 1:
                raise
        time.sleep(3 * (k + 1))
    raise RuntimeError(f"failed: {url[:120]}")  # pragma: no cover


def get_json(url: str, **kw) -> dict:
    return json.loads(get_bytes(url, **kw))


# ---------------------------------------------------------------- OpenFEMA

OPENFEMA = "https://www.fema.gov/api/open"


def openfema_count(entity: str, version: str, filter: str | None = None) -> int:
    """Exact number of records the server holds for `entity` (optionally filtered), via `$count=true`."""
    q = {"$top": 1, "$count": "true"} | ({"$filter": filter} if filter else {})
    return int(get_json(f"{OPENFEMA}/{version}/{entity}?{urllib.parse.urlencode(q)}")["metadata"]["count"])


def openfema_record_count(name: str, version: int) -> int:
    """Total records the dataset catalog advertises, e.g. ('NfipPolicies', 3); matches the bulk file's row count."""
    flt = f"name eq '{name}' and version eq {version}"
    rows = get_json(f"{OPENFEMA}/v1/OpenFemaDataSets?{urllib.parse.urlencode({'$filter': flt})}")["OpenFemaDataSets"]
    if not rows:
        raise LookupError(f"no OpenFEMA dataset {name} v{version}")
    return int(rows[0]["recordCount"])


# ---------------------------------------------------------------- S3

_S3 = None


def s3_client(*, unsigned: bool = True, region: str | None = None):
    global _S3  # noqa: PLW0603 - one shared client per process (boto3 clients are thread-safe)
    if _S3 is None:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        cfg = Config(signature_version=UNSIGNED if unsigned else None, retries={"max_attempts": 5, "mode": "standard"},
                     max_pool_connections=32)
        _S3 = boto3.client("s3", region_name=region or "us-east-1", config=cfg)
    return _S3


def probe_s3(bucket: str, key: str) -> Info:
    h = s3_client().head_object(Bucket=bucket, Key=key)
    return Info(int(h["ContentLength"]), h.get("ETag"), True)


def s3(bucket: str, key: str, dest: str | Path, *, preallocate: bool = True, retries: int = 6,
       info: Info | None = None, expect_sha256: str | None = None) -> Path:
    """Download one S3 object (anonymous) with resume, space checks and ETag-MD5 verification.
    Missing keys raise ClientError."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    source = f"s3://{bucket}/{key}"

    def once() -> Path:
        i = info if info is not None else probe_s3(bucket, key)

        def opener(offset: int):
            kw = {"Bucket": bucket, "Key": key}
            if offset > 0:
                kw["Range"] = f"bytes={offset}-"
                if i.etag:
                    kw["IfMatch"] = i.etag
            body = s3_client().get_object(**kw)["Body"]
            return _closing(body, _iter(body.read)), offset

        etag = (i.etag or "").strip('"')
        md5 = etag if re.fullmatch(r"[0-9a-f]{32}", etag) else None  # multipart ETags contain '-' and are not hashes
        return _transfer(dest, source=source, info=i, opener=opener, preallocate=preallocate, size_hint=None,
                         verify=_verifier(md5, expect_sha256))

    return _retry(once, retries, source)


def s3_bytes(bucket: str, key: str, retries: int = 4) -> bytes:
    """Small S3 object into memory (e.g. map tiles)."""
    for k in range(retries):
        try:
            return s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:  # noqa: BLE001
            if _s3_code(e) in S3_NO_RETRY or k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed: s3://{bucket}/{key}")  # pragma: no cover


def s3_range(bucket: str, key: str, start: int, end: int, retries: int = 4) -> bytes:
    """Bytes [start, end] (inclusive) of an S3 object, e.g. single GRIB messages located through an .idx file."""
    for k in range(retries):
        try:
            return s3_client().get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")["Body"].read()
        except Exception as e:  # noqa: BLE001
            if _s3_code(e) in S3_NO_RETRY or k == retries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed: s3://{bucket}/{key}")  # pragma: no cover


# ---------------------------------------------------------------- batches

@dataclass
class Job:
    dest: Path
    url: str | None = None  # HTTP(S)
    bucket: str | None = None  # S3
    key: str | None = None
    size_hint: int | None = None  # for HTTP sources whose size cannot be probed
    probe: bool = True
    headers: dict | None = None
    expect_sha256: str | None = None

    @property
    def source(self) -> str:
        return f"s3://{self.bucket}/{self.key}" if self.bucket else str(self.url)


def fetch_many(jobs: list[Job], *, workers: int = 4, margin: int = MARGIN, preallocate: bool = True) -> dict[str, str]:
    """Download a batch. Sizes are probed first and the whole remainder is checked against free space ONCE,
    so a 60-file job fails in seconds with a clear message instead of half-way through.

    Returns {dest name: 'ok' | 'skip' | 'FAIL: reason'}. Raises InsufficientSpace before downloading anything.
    """
    todo = [j for j in jobs if not (j.dest.exists() and j.dest.stat().st_size > 0)]
    infos: dict[int, Info] = {}

    def probe(ix: int) -> tuple[int, Info]:
        j = todo[ix]
        try:
            if j.bucket:
                return ix, probe_s3(j.bucket, j.key)
            return ix, probe_https(j.url, j.headers) if j.probe else Info(None)
        except Exception:  # noqa: BLE001 - an unprobeable source is handled (and reported) by its own download
            return ix, Info(None)

    with ThreadPoolExecutor(workers) as ex:
        infos = dict(ex.map(probe, range(len(todo))))
    need = sum(pending_bytes(j.dest, infos[i], j.source) or (j.size_hint or 0) for i, j in enumerate(todo))
    if todo:
        require_space(todo[0].dest.parent, need, margin)
    print(f"{len(jobs) - len(todo)} present, {len(todo)} to fetch, ~{need / 1e9:.2f} GB, "
          f"{free_bytes(todo[0].dest.parent) / 1e9:.1f} GB free" if todo else f"{len(jobs)} present", flush=True)

    def run(ix: int) -> tuple[str, str]:
        j, i = todo[ix], infos[ix]
        try:
            if j.bucket:
                s3(j.bucket, j.key, j.dest, preallocate=preallocate, info=i if i.size is not None else None,
                   expect_sha256=j.expect_sha256)
            else:
                https(j.url, j.dest, headers=j.headers, probe=j.probe, size_hint=j.size_hint,
                      preallocate=preallocate, info=i if i.size is not None else None, expect_sha256=j.expect_sha256)
            return j.dest.name, "ok"
        except InsufficientSpace:
            raise
        except Exception as e:  # noqa: BLE001 - one bad file should not stop the batch
            return j.dest.name, f"FAIL: {type(e).__name__}: {str(e)[:120]}"

    out = {j.dest.name: "skip" for j in jobs}
    with ThreadPoolExecutor(workers) as ex:
        for name, status in ex.map(run, range(len(todo))):
            out[name] = status
            if status != "ok":
                print(f"{status}  {name}", flush=True)
    return out
