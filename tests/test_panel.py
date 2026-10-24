import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import panel  # noqa: E402


def test_poisson_fe_recovers_beta():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(60), 40)
    x = rng.normal(size=(len(g), 2))
    a_true = rng.normal(size=60)
    alpha = a_true[g]
    expo = rng.uniform(50, 200, len(g))
    y = rng.poisson(expo * np.exp(alpha + x @ np.array([0.5, -0.3])))
    beta = panel.poisson_fe(y, x, expo, g)
    assert np.allclose(beta, [0.5, -0.3], atol=0.03)
    a = panel.group_effects(y, x, expo, g, beta)
    assert np.corrcoef(a.sort_index().values, a_true)[0, 1] > 0.95
