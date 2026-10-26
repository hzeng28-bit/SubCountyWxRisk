"""Texas NFIP policy terms from the clean table (`prep` stage), with the "in force on day D" aggregation of stage 10.

Cleaning (types, flags; null / malformed block-group keys flagged) happens once in `scripts/40_prep.py build policies_tx`;
this module only selects and aggregates, so every analysis sees the same rows.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import polars as pl

from .prep import build as prep

OPEN_END = dt.date(2100, 1, 1)  # no end date at all means the policy is still running
SINGLE_FAMILY = (1, 11)  # coding changed to 11 from 2021


def frame(extra: tuple[str, ...] = (), occupancy: tuple[int, ...] = SINGLE_FAMILY, path=None) -> pl.DataFrame:
    """Policies with a block group: geoid (12-digit, 2010 vintage), eff, term (= coverage_end: the earlier of the nominal expiry and a mid-term
    cancellation, see prep.clean.DERIVED), w (policy count), sfha (rated zone A/V), vol (not mandatory purchase).
    `extra` names further clean-table columns to carry along."""
    q = pl.scan_parquet(path) if path else prep.scan("policies_tx")
    return (
        q.filter(pl.col("occupancy").is_in(occupancy) & pl.col("geoid").is_not_null())
        .select(
            "geoid", "eff",
            pl.col("coverage_end").fill_null(OPEN_END).alias("term"),
            pl.col("n_policies").cast(pl.Float64).alias("w"),
            pl.col("zone").str.slice(0, 1).is_in(["A", "V"]).fill_null(False).alias("sfha"),
            (~pl.col("mandatory")).alias("vol"),
            *extra,
        )
        .collect()
    )


def in_force(p: pl.DataFrame, day: pd.Timestamp | dt.date, where: pl.Expr | None = None) -> pd.Series:
    """Policy count in force on `day` per block group, as a pandas Series indexed by geoid."""
    d = day.date() if isinstance(day, pd.Timestamp) else day
    q = p.lazy().filter((pl.col("eff") <= d) & (pl.col("term") > d))
    if where is not None:
        q = q.filter(where)
    out = q.group_by("geoid").agg(pl.col("w").sum()).collect()
    return pd.Series(out["w"].to_numpy(), index=pd.Index(out["geoid"].to_list(), name="geoid"), name="w")
