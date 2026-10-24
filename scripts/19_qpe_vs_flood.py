"""Objective rainfall (MRMS MultiSensor QPE, 24 h, Pass 2) vs flood labels, 2021-2023 May-Sep.

For each county-day, the maximum 24 h gauge-corrected radar rainfall over the county (local
standard-time day) is compared with Storm Events flash floods, NWS flash-flood warnings and NFIP
claims: how often a heavy-rain day carries each label, and how well rainfall ranks the label.

    python scripts/19_qpe_vs_flood.py --years 2021 2023
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import mrms, wb2  # noqa: E402

PRODUCT = "MultiSensor_QPE_24H_Pass2_00.00"
THRESH = (15, 25, 40, 50, 75)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs=2, default=[2021, 2023])
    a = ap.parse_args()
    days = [d for d in pd.date_range(f"{a.years[0]}-05-01", f"{a.years[1]}-09-30") if d.month in range(5, 10)]
    q = mrms.county_day_max(PRODUCT, days, wb2.cache_path("labels_extra") / "qpe", 8)
    q.to_parquet(wb2.cache_path("qpe_county_day.parquet"))

    fl = pd.read_parquet(wb2.cache_path("flood_labels_county_day.parquet"),
                         columns=["cty", "date", "f_ff", "f_any", "f_10k", "w_ff", "nfip"])
    t = q.merge(fl, on=["cty", "date"], how="left").fillna(0)
    t["st"] = t.cty.str[:2].map({"08": "CO", "35": "NM", "04": "AZ"})
    print(f"\ncounty-days {len(t)}; per 1e4: " + ", ".join(
        f"{c} {t[c].mean() * 1e4:.0f}" for c in ("f_ff", "f_any", "f_10k", "w_ff", "nfip")))
    print("QPE max (mm) quantiles 50/90/99/99.9:", t.max_mm.quantile([.5, .9, .99, .999]).round(1).tolist())

    print("\nrainfall -> label: share of county-days above each QPE threshold that carry the label")
    rows = []
    for thr in THRESH:
        hit = t.max_mm >= thr
        rows.append({"QPE>=mm": thr, "county_days": int(hit.sum()), "P(SE flash flood)": t.f_ff[hit].mean(),
                     "P(FF warning)": t.w_ff[hit].mean(), "P(NFIP claim)": t.nfip[hit].mean(),
                     "P(SE flood >=$10k)": t.f_10k[hit].mean()})
    base = {"QPE>=mm": "all", "county_days": len(t), "P(SE flash flood)": t.f_ff.mean(),
            "P(FF warning)": t.w_ff.mean(), "P(NFIP claim)": t.nfip.mean(), "P(SE flood >=$10k)": t.f_10k.mean()}
    print(pd.DataFrame([*rows, base]).round(3).to_string(index=False))

    print("\nlabel -> rainfall: share of labelled county-days with QPE above each threshold")
    rows = []
    for c, name in (("f_ff", "SE flash flood"), ("w_ff", "FF warning"), ("nfip", "NFIP claim"), ("f_10k", "SE flood >=$10k")):
        s = t[t[c] > 0]
        rows.append({"label": name, "n": len(s), **{f">={thr}mm": (s.max_mm >= thr).mean() for thr in (15, 25, 40)},
                     "median_mm_on_label_days": s.max_mm.median(), "median_mm_other": t[t[c] == 0].max_mm.median()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nAUC of QPE max as a ranker (all county-days; by state)")
    rows = []
    for c, name in (("f_ff", "SE flash flood"), ("w_ff", "FF warning"), ("nfip", "NFIP claim"), ("f_10k", "SE flood >=$10k")):
        r = {"label": name, "ALL": roc_auc_score(t[c] > 0, t.max_mm)}
        for st in ("CO", "NM", "AZ"):
            s = t[t.st == st]
            r[st] = roc_auc_score(s[c] > 0, s.max_mm) if s[c].sum() > 0 else np.nan
        rows.append(r)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
