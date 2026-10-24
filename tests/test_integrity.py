"""wxrisk.integrity: content hashes ignore volatile metadata (and only that), and `check` classifies correctly."""

from __future__ import annotations

import gzip
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xarray as xr

from wxrisk import integrity as ig

PAYLOAD = b"hello,world\n" * 5000


def same_content_diff_bytes(a, b):
    assert ig.sha256_file(a) != ig.sha256_file(b), "fixture should differ in bytes"
    assert ig.content_hash(a) == ig.content_hash(b)


def test_gzip_ignores_header_mtime(tmp_path):
    for n, m in (("a.csv.gz", 1), ("b.csv.gz", 99999)):
        with open(tmp_path / n, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=m, filename=n) as g:
            g.write(PAYLOAD)
    same_content_diff_bytes(tmp_path / "a.csv.gz", tmp_path / "b.csv.gz")
    with gzip.open(tmp_path / "c.csv.gz", "wb") as g:
        g.write(PAYLOAD + b"x")
    assert ig.content_hash(tmp_path / "a.csv.gz") != ig.content_hash(tmp_path / "c.csv.gz")


def make_zip(path, members, date, compression):
    with zipfile.ZipFile(path, "w", compression) as z:
        for name, data in members.items():
            z.writestr(zipfile.ZipInfo(name, date_time=date), data, compress_type=compression)


def test_zip_ignores_member_timestamps_and_compression(tmp_path):
    m = {"a.txt": b"A" * 1000, "b.txt": b"B" * 1000}
    make_zip(tmp_path / "a.zip", m, (2020, 1, 1, 0, 0, 0), zipfile.ZIP_DEFLATED)
    make_zip(tmp_path / "b.zip", m, (2024, 6, 6, 6, 6, 6), zipfile.ZIP_STORED)
    same_content_diff_bytes(tmp_path / "a.zip", tmp_path / "b.zip")
    make_zip(tmp_path / "c.zip", {**m, "b.txt": b"C" * 1000}, (2020, 1, 1, 0, 0, 0), zipfile.ZIP_DEFLATED)
    assert ig.content_hash(tmp_path / "a.zip") != ig.content_hash(tmp_path / "c.zip")


def test_zip_with_damaged_member_data_is_corrupt_not_metadata_change(tmp_path):
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    make_zip(a, {"a.txt": bytes(range(256)) * 100}, (2020, 1, 1, 0, 0, 0), zipfile.ZIP_STORED)
    entry = ig.fingerprint(a)
    raw = bytearray(a.read_bytes())
    raw[5000] ^= 0xFF  # inside the stored member data; the central directory (names, CRCs) is untouched
    b.write_bytes(bytes(raw))
    assert ig.check(b, entry) == "corrupt"


def test_npz_is_hashed_as_zip(tmp_path):
    np.savez(tmp_path / "x.npz", a=np.arange(10))
    assert ig.content_hash(tmp_path / "x.npz")[0] == "zip-members"


def dataset(history="h1", units="mm", scale=1.0):
    ds = xr.Dataset({"pr": (("day", "lat"), np.arange(6, dtype="f4").reshape(3, 2) * scale, {"units": units})},
                    coords={"day": [0, 1, 2], "lat": [30.0, 31.0]})
    ds.attrs["History"] = history
    return ds


def test_netcdf_ignores_global_attrs_but_not_data_or_variable_attrs(tmp_path):
    dataset("translated 03:06:03").to_netcdf(tmp_path / "a.nc")
    dataset("translated 03:06:07").to_netcdf(tmp_path / "b.nc")
    same_content_diff_bytes(tmp_path / "a.nc", tmp_path / "b.nc")
    dataset(scale=2.0).to_netcdf(tmp_path / "c.nc")
    dataset(units="in").to_netcdf(tmp_path / "d.nc")
    base = ig.content_hash(tmp_path / "a.nc")
    assert ig.content_hash(tmp_path / "c.nc") != base and ig.content_hash(tmp_path / "d.nc") != base


@pytest.fixture
def small_window(monkeypatch):
    monkeypatch.setattr(ig, "WINDOW", 1000)


def frame(n=5000, shift=0.0):
    r = np.random.default_rng(1)
    return pd.DataFrame({"id": np.arange(n), "x": r.normal(size=n) + shift,
                         "s": [f"k{i % 37}" if i % 11 else None for i in range(n)]})


def test_parquet_ignores_layout_compression_and_metadata(tmp_path, small_window):
    t = pa.Table.from_pandas(frame(), preserve_index=False)
    pq.write_table(t, tmp_path / "a.parquet", compression="snappy", row_group_size=700)
    pq.write_table(t.replace_schema_metadata({"writer": "other", "when": "now"}), tmp_path / "b.parquet",
                   compression="zstd", row_group_size=3000)
    same_content_diff_bytes(tmp_path / "a.parquet", tmp_path / "b.parquet")


def test_parquet_ignores_garbage_under_null_slots(tmp_path, small_window):
    """Arrow does not define the bytes under a null; layouts with all-null row groups must still hash alike.
    (Found on the real NFIP claims file: `floodWaterDuration` differed between row-group sizes.)"""
    n = 5000
    sparse = [None] * 1800 + [float(i) for i in range(n - 1800)]
    f = pd.DataFrame({"id": np.arange(n), "sparse": pd.array(sparse, dtype="Float64"),
                      "name": [None if i < 2600 else f"k{i % 9}" for i in range(n)]})
    t = pa.Table.from_pandas(f, preserve_index=False)
    for rg in (300, 1999, 5000):
        pq.write_table(t, tmp_path / f"rg{rg}.parquet", row_group_size=rg)
    hashes = {rg: ig.content_hash(tmp_path / f"rg{rg}.parquet")[1] for rg in (300, 1999, 5000)}
    assert len(set(hashes.values())) == 1, hashes


def test_feed_table_ignores_bytes_under_nulls():
    """Same logical column, different garbage under the null slots: must hash alike."""
    import hashlib

    def col(garbage: int):
        vals = np.array([1.5, garbage, 3.5, garbage], dtype="f8")
        return pa.Array.from_buffers(pa.float64(), 4, [pa.py_buffer(np.packbits([1, 0, 1, 0], bitorder="little")),
                                                      pa.py_buffer(vals.tobytes())])

    def h(a):
        m = hashlib.sha256()
        ig.feed_table(m, pa.table({"x": a}))
        return m.hexdigest()

    assert col(0.0).null_count == 2 and col(7.0).to_pylist() == col(0.0).to_pylist() == [1.5, None, 3.5, None]
    assert h(col(0.0)) == h(col(7.0))
    assert h(col(0.0)) != h(pa.array([1.5, None, 4.5, None]))


REAL = ig.Path("/Users/nickzeng/wxrisk_data/labels_extra/flood/nfip_national/nfip_2023.parquet")


@pytest.mark.skipif(not REAL.exists(), reason="real NFIP claims file not present")
def test_real_claims_file_hash_is_layout_independent(tmp_path):
    t = pq.read_table(REAL)
    base = ig.content_hash(REAL)
    for rg, comp in ((500, "zstd"), (7000, "snappy"), (len(t), "gzip")):
        pq.write_table(t, tmp_path / f"{rg}.parquet", row_group_size=rg, compression=comp)
        assert ig.content_hash(tmp_path / f"{rg}.parquet") == base, rg


def test_parquet_detects_value_order_and_row_changes(tmp_path, small_window):
    f = frame()
    pq.write_table(pa.Table.from_pandas(f, preserve_index=False), tmp_path / "a.parquet")
    base = ig.content_hash(tmp_path / "a.parquet")
    variants = {"value": frame(shift=1e-6), "order": f.iloc[::-1].reset_index(drop=True), "short": f.iloc[:-1],
                "null": f.assign(s=f.s.where(f.id != 1, None))}
    for name, v in variants.items():
        pq.write_table(pa.Table.from_pandas(v, preserve_index=False), tmp_path / f"{name}.parquet")
        assert ig.content_hash(tmp_path / f"{name}.parquet") != base, name


def test_rdb_ignores_retrieved_line_only(tmp_path):
    body = "agency_cd\tsite_no\tvalue\nUSGS\t01\t5.0\n"
    (tmp_path / "a.rdb").write_text("# retrieved: 2026-10-23 23:06:14\n" + body)
    (tmp_path / "b.rdb").write_text("# retrieved: 2026-10-20 01:01:06 (other server)\n" + body)
    (tmp_path / "c.rdb").write_text("# retrieved: 2026-10-23 23:06:14\n" + body.replace("5.0", "5.1"))
    same_content_diff_bytes(tmp_path / "a.rdb", tmp_path / "b.rdb")
    assert ig.content_hash(tmp_path / "a.rdb") != ig.content_hash(tmp_path / "c.rdb")


def test_check_classifies(tmp_path):
    a, b, c = (tmp_path / n for n in ("a.zip", "b.zip", "c.zip"))
    m = {"a.txt": b"A" * 1000}
    make_zip(a, m, (2020, 1, 1, 0, 0, 0), zipfile.ZIP_DEFLATED)
    make_zip(b, m, (2024, 6, 6, 6, 6, 6), zipfile.ZIP_STORED)
    make_zip(c, {"a.txt": b"Z" * 1000}, (2020, 1, 1, 0, 0, 0), zipfile.ZIP_DEFLATED)
    entry = ig.fingerprint(a)
    assert ig.check(a, entry) == "ok"
    assert ig.check(b, entry) == "ok-metadata-differs"
    assert ig.check(c, entry) == "changed"
    assert ig.check(tmp_path / "nope.zip", entry) == "missing"
    broken = tmp_path / "broken.zip"
    broken.write_bytes(a.read_bytes()[:40])
    assert ig.check(broken, entry) == "corrupt"
    plain = tmp_path / "p.txt"
    plain.write_text("abc")
    e = ig.fingerprint(plain)
    plain.write_text("abd")
    assert ig.check(plain, e) == "changed"  # same length, different bytes
