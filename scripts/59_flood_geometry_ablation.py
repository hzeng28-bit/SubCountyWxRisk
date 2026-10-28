"""Do NFHL flood-zone area shares and HAND add to the stage 19 pilot model? (stage 22)

Arms on top of base + hist00 + prior3: NFHL area shares (script 58), HAND statistics (scripts 57, 58), both.
Same evaluations as script 56: time split and year + region split, observed rain (2017-2023) and forecast rain (2017-2019).

Inputs : flood_pilot_table.parquet, flood_pilot_fc_features.parquet, bg_flood_geometry.parquet
Output : $WXRISK_DATA/audit/flood_geometry_ablation.txt
Run    : python scripts/59_flood_geometry_ablation.py [--obs-only] [--no-region]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

NFHL = ["nfhl_sfha", "nfhl_x02", "nfhl_mapped"]
HAND = ["h1_med", "h1_p10", "h1_le2", "h1_le5", "h10_med", "h10_le5"]


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-only", action="store_true")
    ap.add_argument("--no-region", action="store_true")
    a = ap.parse_args()
    e26, e48, e49, e56 = (_load("e26", "26_flood_pilot_experiment.py"), _load("e48", "48_flood_forecast_pilot.py"),
                          _load("e49", "49_information_audit.py"), _load("e56", "56_spatial_features.py"))
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t = t.join(e49.history(t)).merge(pd.read_parquet(fp.PILOT / "bg_flood_geometry.parquet"), on="bg10", how="left")
    arms = {"base (stage 19)": [], "+ NFHL area shares": NFHL, "+ HAND": HAND, "+ NFHL + HAND": NFHL + HAND}
    e56.say("== observed rain, test 2017-2023")
    e56.run(t, e26.COUNTY + e26.EXPO + e26.CELL + e56.HIST, arms, "observed rain", e26, e48, not a.no_region)
    if not a.obs_only:
        e56.say("\n== forecast rain (fcmean), test 2017-2019")
        u = t[t.year <= 2019].merge(pd.read_parquet(e48.FEATS), on=["ep", "bg"], how="inner")
        e56.run(u, e48.BASE + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"] + e56.HIST, arms, "forecast rain", e26, e48,
                not a.no_region)
    p = wb2.cache_path("audit") / "flood_geometry_ablation.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(e56.OUT) + "\n")


if __name__ == "__main__":
    main()
