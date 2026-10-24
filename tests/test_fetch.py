"""wxrisk.fetch: resume, restart on a changed file, no-Range servers, unknown size, space checks."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from wxrisk import fetch

BODY = np.random.default_rng(0).integers(0, 256, 5 * (1 << 20) + 123, dtype=np.uint8).tobytes()


def make_handler(body: bytes, *, ranges: bool = True, chunked: bool = False, etag: str = '"v1"'):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _head(self, code, n, extra=None):
            self.send_response(code)
            self.send_header("ETag", etag)
            if ranges:
                self.send_header("Accept-Ranges", "bytes")
            if not chunked:
                self.send_header("Content-Length", str(n))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def do_HEAD(self):
            self._head(200, len(body))

        def do_GET(self):
            rng = self.headers.get("Range")
            if rng and ranges:
                start = int(rng.split("=")[1].split("-")[0])
                if self.headers.get("If-Range") not in (None, etag):
                    return self._full()
                self._head(206, len(body) - start, {"Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}"})
                self.wfile.write(body[start:])
                self.server.requested.append(rng)
            else:
                self._full()

        def _full(self):
            self._head(200, len(body))
            self.wfile.write(body)
            self.server.requested.append(None)

    return H


@pytest.fixture
def serve():
    servers = []

    def start(body=BODY, **kw):
        s = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(body, **kw))
        s.requested = []
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        return s, f"http://127.0.0.1:{s.server_port}/f.bin"

    yield start
    for s in servers:
        s.shutdown()


def plant_partial(dest, url, written, etag='"v1"', size=None):
    size = len(BODY) if size is None else size
    part = fetch._part_path(dest)
    part.write_bytes(BODY[:written])
    fetch._meta_path(dest).write_text(json.dumps({"source": url, "etag": etag, "size": size, "written": written}))


def test_full_download_is_atomic(serve, tmp_path):
    _, url = serve()
    dest = tmp_path / "a" / "f.bin"
    fetch.https(url, dest)
    assert dest.read_bytes() == BODY
    assert not fetch._part_path(dest).exists() and not fetch._meta_path(dest).exists()


def test_existing_file_is_skipped(serve, tmp_path):
    s, url = serve()
    dest = tmp_path / "f.bin"
    dest.write_bytes(b"x")
    fetch.https(url, dest)
    assert dest.read_bytes() == b"x" and s.requested == []


def test_resumes_from_partial(serve, tmp_path):
    s, url = serve()
    dest = tmp_path / "f.bin"
    plant_partial(dest, url, 2_000_000)
    fetch.https(url, dest)
    assert dest.read_bytes() == BODY
    assert s.requested == ["bytes=2000000-"]


def test_resume_survives_preallocated_full_length_part(serve, tmp_path):
    """After fallocate/truncate the .part has full length; progress must come from the sidecar, not the file size."""
    s, url = serve()
    dest = tmp_path / "f.bin"
    plant_partial(dest, url, 1_000_000)
    with open(fetch._part_path(dest), "r+b") as f:
        f.truncate(len(BODY))  # as if preallocated; bytes past 1_000_000 are zeros
    fetch.https(url, dest)
    assert dest.read_bytes() == BODY


def test_changed_remote_restarts(serve, tmp_path):
    s, url = serve(etag='"v2"')
    dest = tmp_path / "f.bin"
    plant_partial(dest, url, 2_000_000, etag='"v1"')
    fetch.https(url, dest)
    assert dest.read_bytes() == BODY
    assert s.requested == [None]  # no Range: partial discarded


def test_expect_sha256_accepts_match_and_rejects_mismatch(serve, tmp_path):
    _, url = serve()
    good = hashlib.sha256(BODY).hexdigest()
    ok = tmp_path / "ok.bin"
    fetch.https(url, ok, expect_sha256=good)
    assert ok.read_bytes() == BODY
    bad = tmp_path / "bad.bin"
    with pytest.raises(fetch.ChecksumMismatch):
        fetch.https(url, bad, expect_sha256="0" * 64, retries=1)
    assert not bad.exists() and not fetch._part_path(bad).exists() and not fetch._meta_path(bad).exists()


def test_checksum_failure_after_resume_discards_partial(serve, tmp_path):
    _, url = serve()
    dest = tmp_path / "f.bin"
    plant_partial(dest, url, 2_000_000)
    with fetch._part_path(dest).open("r+b") as f:  # corrupt the already-written prefix
        f.write(b"\xff" * 100)
    with pytest.raises(fetch.ChecksumMismatch):
        fetch.https(url, dest, expect_sha256=hashlib.sha256(BODY).hexdigest(), retries=1)
    assert not dest.exists() and not fetch._part_path(dest).exists()
    fetch.https(url, dest, expect_sha256=hashlib.sha256(BODY).hexdigest())  # a fresh attempt succeeds
    assert dest.read_bytes() == BODY


def test_server_without_range_restarts(serve, tmp_path):
    s, url = serve(ranges=False)
    dest = tmp_path / "f.bin"
    plant_partial(dest, url, 2_000_000)
    fetch.https(url, dest)
    assert dest.read_bytes() == BODY


def test_unknown_size_streams_without_resume(serve, tmp_path):
    _, url = serve(chunked=True)
    dest = tmp_path / "f.bin"
    fetch.https(url, dest, probe=False, size_hint=len(BODY))
    assert dest.read_bytes() == BODY


def test_insufficient_space_is_reported_before_transfer(serve, tmp_path, monkeypatch):
    s, url = serve()
    monkeypatch.setattr(fetch, "free_bytes", lambda p: 1_000)
    dest = tmp_path / "f.bin"
    with pytest.raises(fetch.InsufficientSpace, match="free"):
        fetch.https(url, dest)
    assert not dest.exists()


def test_fetch_many_checks_total_once_and_reports_failures(serve, tmp_path, monkeypatch):
    _, url = serve()
    jobs = [fetch.Job(tmp_path / f"{i}.bin", url=url) for i in range(3)]
    res = fetch.fetch_many(jobs[:3], workers=2)
    assert set(res.values()) == {"ok"}
    monkeypatch.setattr(fetch, "free_bytes", lambda p: 5 * len(BODY))  # room for 5 files' worth, margin kills it
    more = [fetch.Job(tmp_path / f"n{i}.bin", url=url) for i in range(4)]
    with pytest.raises(fetch.InsufficientSpace):
        fetch.fetch_many(more, workers=2, margin=len(BODY))
    assert not any((tmp_path / f"n{i}.bin").exists() for i in range(4))


@pytest.mark.parametrize("size", [4 << 20, 512 << 20])
def test_preallocate_reserves_or_degrades_quietly(tmp_path, size):
    """A modest reservation must never be reported as 'no space' on a disk with room (macOS contiguous-fail bug)."""
    if fetch.free_bytes(tmp_path) < size + fetch.MARGIN:
        pytest.skip("not enough free disk for this size")
    with open(tmp_path / "p", "wb") as f:
        method = fetch._preallocate(f, size, 0)
    assert method in {"fallocate", "f_preallocate", "truncate", "none"}


def test_preallocate_reports_real_shortage(tmp_path):
    with open(tmp_path / "p", "wb") as f, pytest.raises(fetch.InsufficientSpace):
        fetch._preallocate(f, fetch.free_bytes(tmp_path) * 4 + (1 << 40), 0)


@pytest.mark.skipif(not os.environ.get("WXRISK_NETWORK_TESTS"), reason="set WXRISK_NETWORK_TESTS=1")
def test_s3_anonymous_download_and_resume(tmp_path):
    bucket, key = "noaa-mrms-pds", "CONUS/MESH_Max_1440min_00.50/20210501/MRMS_MESH_Max_1440min_00.50_20210501-070000.grib2.gz"
    info = fetch.probe_s3(bucket, key)
    assert info.size and info.size > 1000
    dest = tmp_path / "m.gz"
    part = fetch._part_path(dest)
    full = fetch.s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    part.write_bytes(full[: info.size // 2])
    fetch._meta_path(dest).write_text(json.dumps(
        {"source": f"s3://{bucket}/{key}", "etag": info.etag, "size": info.size, "written": info.size // 2}))
    fetch.s3(bucket, key, dest)  # ETag-MD5 verified inside
    assert dest.read_bytes() == full
    with pytest.raises(fetch.ChecksumMismatch):
        fetch.s3(bucket, key, tmp_path / "m2.gz", expect_sha256="0" * 64, retries=1)
