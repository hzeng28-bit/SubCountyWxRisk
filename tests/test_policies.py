"""in_force: terms running on a day, open-ended terms, and optional zone / purchase filters."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("polars")

from wxrisk import policies  # noqa: E402

D = dt.date


def make(tmp_path):
    """A clean-format table (what prep.build writes): null-key handling is tested in test_prep, not here."""
    t = pd.DataFrame({
        "geoid": ["048001000100", "048001000100", "480010002001", "480010002001"],
        "zone": ["AE", "X", None, "VE"],
        "occupancy": [1, 11, 1, 2],
        "eff": [D(2017, 1, 1), D(2017, 9, 1), D(2016, 1, 1), D(2016, 1, 1)],
        "coverage_end": [D(2018, 1, 1), None, D(2017, 8, 1), D(2018, 1, 1)],
        "n_policies": [1, 2, 3, 7],
        "mandatory": [True, False, False, True],
    })
    f = tmp_path / "p.parquet"
    t.to_parquet(f)
    return f


def test_in_force_counts_running_terms(tmp_path) -> None:
    p = policies.frame(path=make(tmp_path))
    assert p["geoid"].to_list() == ["048001000100", "048001000100", "480010002001"]  # occupancy 2 dropped
    got = policies.in_force(p, pd.Timestamp("2017-07-31"))
    assert got.to_dict() == {"048001000100": 1.0, "480010002001": 3.0}  # row 2 starts in September; row 4 ends 08-01
    later = policies.in_force(p, pd.Timestamp("2017-09-15"))  # open-ended term counts as running; row 4 has ended
    assert later.to_dict() == {"048001000100": 3.0}
    sfha = policies.in_force(p, pd.Timestamp("2017-09-15"), policies.pl.col("sfha"))
    assert sfha.to_dict() == {"048001000100": 1.0}
