"""Radar hail (MRMS MESH) vs hail reports: how many radar-indicated hail county-days go unreported?

MESH_Max_1440min at 07:00 UTC of day D+1 is the 24 h maximum ending at 07Z, i.e. the local
standard-time day D (UTC-7), matching the Storm Events date convention. NOAA's AWS archive starts
in October 2020, so this covers the 2021-2023 May-Sep seasons.

Outputs: $WXRISK_DATA/mesh_county_day.parquet (cty, date, mesh_max_mm, frac_ge_25)
and a printed agreement table against Storm Events hail reports.

    python scripts/17_mesh_vs_reports.py --years 2021 2023
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import labels_alt, mrms, wb2  # noqa: E402

BASE = "https://noaa-mrms-pds.s3.amazonaws.com/CONUS/MESH_Max_1440min_00.50"
EXTRA = wb2.cache_path("labels_extra")
OUT = EXTRA / "mesh"
BOX = (-117.0, 30.5, -101.0, 42.5)
MM_1IN = 25.4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2021, 2023])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    days = [d for d in pd.date_range(f"{a.years[0]}-05-01", f"{a.years[1]}-09-30") if d.month in range(5, 10)]
    mesh = mrms.county_day_max("MESH_Max_1440min_00.50", days, OUT, a.workers, fracs=(MM_1IN,))
    mesh = mesh.rename(columns={"max_mm": "mesh_max_mm", "frac_ge_25.4": "frac_ge_25"})
    mesh.to_parquet(wb2.cache_path("mesh_county_day.parquet"))

    ev = labels_alt.storm_events(range(a.years[0], a.years[1] + 1))
    hail = ev[ev.EVENT_TYPE == "Hail"]
    rep = hail.groupby(["cty", "date"]).mag.max().rename("hail_in").reset_index()
    t = mesh.merge(rep, how="left", on=["cty", "date"])
    t["any_hail"] = t.hail_in.notna()
    t["hail_ge1"] = t.hail_in >= 1.0
    print(f"\ncounty-days {len(t)}; with a hail report {int(t.any_hail.sum())}; with a >=1 in report {int(t.hail_ge1.sum())}")
    print("\nagreement by MESH threshold (mm):")
    rows = []
    for thr in (10, 15, 25.4, 38, 50):
        hit = t.mesh_max_mm >= thr
        rows.append({"MESH>=mm": thr, "county_days": int(hit.sum()),
                     "P(report>=1in | MESH)": t.hail_ge1[hit].mean(), "P(any hail report | MESH)": t.any_hail[hit].mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    for lo, hi in ((0, 1), (1, 1.75), (1.75, 99)):
        s = t[(t.hail_in >= lo) & (t.hail_in < hi)] if lo else t[t.hail_in < hi]
        print(f"reported hail in [{lo}, {hi}) in: n={len(s)}, share with MESH>=25.4mm {(s.mesh_max_mm >= MM_1IN).mean():.3f}")
    print("\nby state, MESH>=25.4 mm county-days with no >=1 in report:")
    t["st"] = t.cty.str[:2].map({"08": "CO", "35": "NM", "04": "AZ"})
    hit = t[t.mesh_max_mm >= MM_1IN]
    print(hit.groupby("st").apply(lambda g: pd.Series({"n": len(g), "unreported_share": 1 - g.hail_ge1.mean()}),
                                  include_groups=False).round(3).to_string())


if __name__ == "__main__":
    main()
