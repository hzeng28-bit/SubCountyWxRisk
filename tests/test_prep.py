"""The preprocessing stage: rules applied and counted, output independent of row order, params fitted on train only."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import integrity  # noqa: E402
from wxrisk.prep import Col, TableSpec, normalize  # noqa: E402
from wxrisk.prep import build as pb  # noqa: E402
from wxrisk.prep.clean import clean  # noqa: E402

D = dt.date


def raw_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "g": ["48001000100", "480010002001", None, "abc", " 480010002002 ", ""],
        "z": ["ae", "X ", None, "ve", "", "AE"],
        "f": ["0", "1", "4", "0", "1", "1"],
        "rc": [100, 0, 50, None, 70, -5],
        "eff": [D(2015, 1, 1), D(2016, 6, 1), D(2017, 1, 1), D(2018, 1, 1), D(2019, 1, 1), None],
        "cov": [100000, 250000, 300000, 120000, 90000, 500000],
    })


def spec(src: str) -> TableSpec:
    return TableSpec("t", src, (
        Col("g", "key", name="geoid", width=12, nulls="drop"),
        Col("z", "str", name="zone", upper=True, nulls="flag"),
        Col("f", "str", name="foundation", sentinel=("0",), nulls="flag"),
        Col("rc", "int", name="rc", lo=1, nulls="flag"),
        Col("eff", "date", nulls="drop"),
        Col("cov", "int"),
    ))


@pytest.fixture
def raw(tmp_path):
    p = tmp_path / "raw.parquet"
    raw_frame().write_parquet(p)
    return p


def test_rules_and_accounting(raw) -> None:
    df, rep = clean(pl.scan_parquet(raw), spec(str(raw)))
    # null, malformed ("abc") and blank keys are dropped (the blank one also lacks a date but is counted once, under geoid);
    # the padded key is trimmed and kept
    assert rep["rows_in"] == 6 and rep["rows_out"] == 3
    assert df["geoid"].to_list() == ["048001000100", "480010002001", "480010002002"]
    assert rep["dropped_null_or_malformed_key"] == {"geoid": 3, "eff": 0}
    by = {r["geoid"]: r for r in df.iter_rows(named=True)}
    assert by["048001000100"]["foundation"] is None and by["048001000100"]["foundation_missing"]  # sentinel '0' -> null + flag
    assert by["480010002001"]["foundation"] == "1" and not by["480010002001"]["foundation_missing"]
    assert by["480010002001"]["rc"] is None and by["480010002001"]["rc_missing"]  # 0 < lo -> null (a default, not a value)
    assert by["048001000100"]["zone"] == "AE" and by["480010002001"]["zone"] == "X"  # case and padding normalised
    assert rep["columns"]["rc"]["sentinel_or_out_of_range"] == 2  # 0 and -5 in the source


def test_output_is_independent_of_row_order(raw, tmp_path) -> None:
    df = raw_frame()
    a, _ = clean(df.lazy(), spec(str(raw)))
    b, _ = clean(df.sample(fraction=1.0, shuffle=True, seed=7).lazy(), spec(str(raw)))
    for n, x in (("a", a), ("b", b)):
        pb.write_table(x, tmp_path / f"{n}.parquet")
    assert integrity.content_hash(tmp_path / "a.parquet")[1] == integrity.content_hash(tmp_path / "b.parquet")[1]


def test_build_is_idempotent_and_verifiable(raw, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pb, "CLEAN_DIR", tmp_path / "clean")
    s = spec(str(raw))
    assert pb.build(s)["status"] == "built"
    assert pb.build(s)["status"] == "up-to-date"
    assert pb.verify(s)["status"] == "ok"
    raw_frame().with_columns(pl.col("cov") + 1).write_parquet(raw)  # upstream data changed -> must rebuild and fail verify
    assert pb.verify(s)["status"] == "MISMATCH"
    assert pb.build(s)["status"] == "built"
    assert spec(str(raw)).digest() != TableSpec("t", str(raw), spec(str(raw)).cols[:-1]).digest()  # rules are part of identity


def test_normalisation_fits_on_train_only() -> None:
    df = pl.DataFrame({"y": [2014, 2015, 2016, 2017, 2018], "x": [10.0, 20.0, 30.0, 1e6, 2e6]})
    p1 = normalize.fit(df, {"x": "standard"}, pl.col("y") <= 2016, "y<=2016")
    p2 = normalize.fit(df.with_columns(pl.when(pl.col("y") > 2016).then(-5.0).otherwise(pl.col("x")).alias("x")),
                       {"x": "standard"}, pl.col("y") <= 2016, "y<=2016")
    assert p1["cols"] == p2["cols"] and p1["fit_rows"] == 3  # test-period values cannot move the parameters
    z = normalize.apply(df.with_columns(pl.when(pl.col("y") == 2015).then(None).otherwise(pl.col("x")).alias("x")), p1)
    assert z["x_z"][1] is None  # a null stays null, no silent fill
    assert z["x_z"].dtype == pl.Float32
    p1["cols"]["x"]["center"] = 0.0
    with pytest.raises(AssertionError):
        normalize.apply(df, p1)  # edited after fitting
    with pytest.raises(AssertionError):
        normalize.fit(df, {"x": "standard"}, pl.col("y") == 2014, "single value")  # zero scale


def test_multi_file_glob_text_dates_and_unique_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pb, "CLEAN_DIR", tmp_path / "clean")
    monkeypatch.setattr(pb.wb2, "DATA_DIR", tmp_path)
    d = tmp_path / "raw"
    d.mkdir()
    # year files whose schemas drift: the second has an extra column and a float where the first has an int
    pl.DataFrame({"id": [1, 2], "d": ["2017-08-25T00:00:00.000Z", "2018-01-01T00:00:00.000Z"], "depth": [0.0, 12.0]}
                 ).write_parquet(d / "c_2017.parquet")
    pl.DataFrame({"id": [3], "d": ["2019-05-05T00:00:00.000Z"], "depth": [999.0], "extra": [1]}).write_parquet(d / "c_2019.parquet")
    s = TableSpec("c", "raw/c_*.parquet", (
        Col("id", "long"), Col("d", "date", name="loss_date", parse="%Y-%m-%d", nulls="drop"),
        Col("depth", "float", sentinel=(0.0, 999.0), nulls="flag")), expect_unique=("id",))
    r = pb.build(s)
    assert r["report"]["rows_in"] == 3 and r["report"]["columns"]["depth"]["sentinel_or_out_of_range"] == 2
    t = pl.read_parquet(pb.paths(s)[1])
    assert t["loss_date"].to_list() == [D(2017, 8, 25), D(2018, 1, 1), D(2019, 5, 5)]
    assert t["depth"].to_list() == [None, 12.0, None] and t["depth_missing"].to_list() == [True, False, True]
    assert pb.verify(s)["status"] == "ok"
    pl.DataFrame({"id": [3], "d": ["2019-05-06T00:00:00.000Z"], "depth": [1.0]}).write_parquet(d / "c_2019.parquet")
    assert pb.verify(s)["status"] == "MISMATCH"  # one of the several input files changed
    pl.DataFrame({"id": [2], "d": ["2019-05-06T00:00:00.000Z"], "depth": [1.0]}).write_parquet(d / "c_2019.parquet")
    with pytest.raises(AssertionError, match="duplicate"):
        pb.build(s, force=True)


def test_every_value_changing_rule_is_in_the_register() -> None:
    from wxrisk.prep.register import RULES
    from wxrisk.prep.specs import SPECS

    have = {(r.table, r.col) for r in RULES}
    for t, sp in SPECS.items():
        for c in sp.cols:
            if c.sentinel or c.lo is not None or c.hi is not None or c.nulls == "drop":
                assert (t, c.out) in have, f"{t}.{c.out} has a cleaning rule but no register entry (src/wxrisk/prep/register.py)"
        for d in sp.derived:
            assert (t, d) in have, f"{t}.{d} is derived but not in the register"
    for r in RULES:
        outs = {c.out for c in SPECS[r.table].cols} | set(SPECS[r.table].derived) | {"-"}
        assert r.col in outs or r.id in {"C07", "C11"}, f"{r.id} refers to unknown column {r.col}"
    assert len({r.id for r in RULES}) == len(RULES)
