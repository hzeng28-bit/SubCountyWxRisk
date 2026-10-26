"""The device kernels must agree with the numpy reference in wxrisk.motion (on whatever device is selected)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("torch")

from test_motion import smooth_field  # noqa: E402

from wxrisk import compute  # noqa: E402
from wxrisk.motion import estimate_shift, warp  # noqa: E402


def test_estimate_shift_batch_matches_numpy() -> None:
    print("  " + compute.describe())
    shifts = [(3.0, 0.0), (0.0, 2.0), (2.5, -1.5), (-4.0, 3.25), (0.4, 0.6)]
    fields = [smooth_field() for _ in shifts]
    a = np.stack(fields)
    b = np.stack([warp(f, sx, sy) for f, (sx, sy) in zip(fields, shifts, strict=True)])
    got = compute.estimate_shift_batch(a, b)
    for g, f, bb, (sx, sy) in zip(got, fields, b, shifts, strict=True):
        ref = estimate_shift(f, bb)
        assert abs(g[0] - ref[0]) < 0.05 and abs(g[1] - ref[1]) < 0.05, (sx, sy, g, ref)
        assert abs(g[0] - sx) < 0.25 and abs(g[1] - sy) < 0.25
