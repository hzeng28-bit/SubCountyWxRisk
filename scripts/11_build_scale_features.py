"""Build scale-resolved county-day features for every year (see wxrisk.scale_features).

Output: $WXRISK_DATA/scalefeat_<year>.parquet, one per year. Re-running skips existing files.

    python scripts/11_build_scale_features.py --years 2000 2022 --workers 6
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import scale_features, wb2  # noqa: E402


def one(year: int) -> str:
    out = wb2.cache_path(f"scalefeat_{year}.parquet")
    if out.exists():
        return f"[skip] {out.name}"
    lab = pd.read_parquet(wb2.cache_path("labels_scs3_conm_az.parquet"), columns=["cty", "lat", "lon"])
    cty = lab.drop_duplicates("cty").reset_index(drop=True)
    t0 = time.time()
    scale_features.year_features(year, cty).to_parquet(out)
    return f"[{year}] {time.time() - t0:.0f}s -> {out.name}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2000, 2022])
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(one, range(args.years[0], args.years[1] + 1)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
