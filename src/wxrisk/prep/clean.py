"""Apply a TableSpec to a raw parquet file: typed columns, rules for missing values, deterministic row order."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

from .spec import Col, TableSpec

DTYPE = {"int": pl.Int32, "long": pl.Int64, "float": pl.Float64, "bool": pl.Boolean, "date": pl.Date}


def typed(c: Col) -> pl.Expr:
    """The source column as its target type, before sentinel / range rules. Blank strings count as null."""
    e = pl.col(c.src)
    if c.kind in ("key", "str"):
        e = e.cast(pl.Utf8).str.strip_chars()
        e = pl.when(e == "").then(None).otherwise(e)
        if c.kind == "str" and c.upper:
            e = e.str.to_uppercase()
        if c.kind == "key":
            e = e.str.zfill(c.width)
            e = pl.when(e.str.contains(rf"^[0-9]{{{c.width}}}$")).then(e).otherwise(None)  # malformed -> null
        return e
    if c.kind == "date" and c.parse:
        return e.cast(pl.Utf8).str.slice(0, 10).str.to_date(c.parse, strict=False)
    return e.cast(DTYPE[c.kind], strict=False)


def unknown(c: Col) -> pl.Expr:
    """Rows where the typed value is a declared sentinel or outside the declared range."""
    t = typed(c)
    cond = pl.lit(False)
    if c.sentinel:
        cond = cond | t.is_in(list(c.sentinel))
    bound = (lambda v: dt.date.fromisoformat(v)) if c.kind == "date" else (lambda v: v)
    if c.lo is not None:
        cond = cond | (t < bound(c.lo))
    if c.hi is not None:
        cond = cond | (t > bound(c.hi))
    return cond.fill_null(False)


# Derived columns, computed from output columns. coverage_end: the NFIP policy file gives a nominal expiry
# (`policyTerminationDate`, always effective date + 1 year) and, separately, the date of a mid-term cancellation; the
# OpenFEMA FAQ says to check the latter when counting policies in force. min_horizontal ignores nulls.
DERIVED = {"coverage_end": pl.min_horizontal(pl.col("term"), pl.col("cancelled")).alias("coverage_end")}


def clean(raw: pl.LazyFrame, spec: TableSpec) -> tuple[pl.DataFrame, dict]:
    """Returns (clean frame, accounting report). The report is part of the sidecar and of the results write-up."""
    n_in = raw.select(pl.len()).collect().item()
    stats = raw.select(
        [pl.col(c.src).is_null().sum().alias(f"{c.out}|src_null") for c in spec.cols]
        + [(typed(c).is_null() & pl.col(c.src).is_not_null()).sum().alias(f"{c.out}|unparsable")
           for c in spec.cols]
        + [unknown(c).sum().alias(f"{c.out}|unknown") for c in spec.cols]
    ).collect().row(0, named=True)

    exprs, order = [], []
    for c in spec.cols:
        t = typed(c)
        val = pl.when(unknown(c)).then(None).otherwise(t).alias(c.out)
        exprs.append(val)
        order.append(c.out)
        if c.nulls == "flag":
            exprs.append(val.is_null().alias(f"{c.out}_missing"))
            order.append(f"{c.out}_missing")
    df = raw.select(exprs).collect()
    for d in spec.derived:
        df = df.with_columns(DERIVED[d])
        order.append(d)

    dropped = {}
    for c in spec.cols:
        if c.nulls == "drop":
            before = df.height
            df = df.filter(pl.col(c.out).is_not_null())
            dropped[c.out] = before - df.height
    if spec.expect_unique:
        dup = df.height - df.unique(subset=list(spec.expect_unique)).height
        assert dup == 0, f"{spec.name}: {dup} duplicate rows on {spec.expect_unique}"

    df = df.select(order).sort(order, nulls_last=True, maintain_order=False)  # total order -> identical bytes
    report = {
        "rows_in": n_in, "rows_out": df.height, "dropped_null_or_malformed_key": dropped,
        "columns": {c.out: {"src_null": stats[f"{c.out}|src_null"], "unparsable": stats[f"{c.out}|unparsable"],
                            "sentinel_or_out_of_range": stats[f"{c.out}|unknown"],
                            "null_out": int(df[c.out].null_count())} for c in spec.cols},
    }
    return df, report


def raw_files(source: str, root: Path) -> list[Path]:
    """The raw files of a table, sorted: a plain path or a glob under `root`."""
    if any(ch in source for ch in "*?["):
        return sorted(root.glob(source))
    return [root / source]


def raw_lazy(files: list[Path]) -> pl.LazyFrame:
    """Files of one table; columns are matched by name and types relaxed to a common one (schemas drift between years)."""
    assert files, "no raw files"
    if len(files) == 1:
        return pl.scan_parquet(files[0])
    return pl.concat([pl.scan_parquet(f) for f in files], how="diagonal_relaxed")
