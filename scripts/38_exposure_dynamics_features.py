"""Exposure-composition features for the Texas flood pilot table: who is insured, and for how long.

CANDIDATE pool (see docs/FLOOD_PRIORS.md, RESULTS.md stage 10). For every (episode, block group) row of
`flood_pilot_table.parquet`, on the day before the episode starts (strictly ex ante):

  new12_share   share of in-force policies whose first NFIP term (originalNBDate) is less than 12 months old
  mid_share     share with first term 12-36 months old    (old_share = 1 - new12 - mid, left out as redundant)
  mand_share    share flagged mandatory purchase (lender requirement) rather than voluntary

These are the stock-flow model's exposure ages (scripts/37): after an incident the stock fills with new
buyers who have not yet been through a flood with a policy; whether this composition adds sub-county skill on top
of policy counts is tested by scripts/26 (group "L6 + exposure dynamics"), keeping it only if the gain
exceeds the year-grouped interval. Block groups are put on the 2010 geography with wxrisk.bg_crosswalk.

Output: $WXRISK_DATA/labels_extra/flood/pilot/flood_pilot_expodyn.parquet (keys: ep, bg)

    python scripts/38_exposure_dynamics_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import bg_crosswalk  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402


def interval_stock(bi, lo, hi, w, nb_bg, n):
    """float32[bg, day]: sum of w over policies whose interval [lo, hi) covers the day."""
    ok = hi > lo
    up = np.bincount(bi[ok] * (n + 1) + lo[ok], weights=w[ok], minlength=nb_bg * (n + 1))
    dn = np.bincount(bi[ok] * (n + 1) + hi[ok], weights=w[ok], minlength=nb_bg * (n + 1))
    return np.cumsum((up - dn).reshape(nb_bg, n + 1), axis=1, dtype=np.float64)[:, :n].astype(np.float32)


def main() -> None:
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "bg", "bg10", "start"])
    geo = fp.bg_geometry()
    assert (geo.bg10.to_numpy()[t.bg.to_numpy()] == t.bg10.to_numpy()).all(), "table bg index != bg_geometry order"
    bg_ids = geo.bg10.tolist()
    nb_bg, n = len(bg_ids), len(fp.expo_days())
    pol = pd.read_parquet(fp.PILOT / "policies_tx.parquet",
                          columns=["censusGeoid", "policyEffectiveDate", "cancellationDateOfFloodPolicy", "policyCount",
                                   "mandatoryPurchaseFlag", "originalNBDate"])
    cw = bg_crosswalk.build().set_index("code")
    bi = pol.censusGeoid.astype("string").str.zfill(12).map(cw.bg10).map({b: i for i, b in enumerate(bg_ids)})
    keep = bi.notna().to_numpy()
    pol, bi = pol[keep], bi[keep].astype(np.int64).to_numpy()
    d0 = fp.EXPO_START
    e = (pd.to_datetime(pol.policyEffectiveDate) - d0).dt.days.to_numpy()
    end = e + 366
    cend = (pd.to_datetime(pol.cancellationDateOfFloodPolicy) - d0).dt.days.to_numpy(dtype="float64")
    end = np.where(np.isnan(cend), end, np.minimum(end, cend)).astype(np.int64)
    nb = (pd.to_datetime(pol.originalNBDate, errors="coerce") - d0).dt.days.fillna(-10_000).to_numpy().astype(np.int64)
    w = pol.policyCount.to_numpy().astype(np.float64)
    mand = pol.mandatoryPurchaseFlag.astype(bool).to_numpy()
    c = lambda x: np.clip(x, 0, n)  # noqa: E731
    s0, s1 = c(e), c(end)
    tot = interval_stock(bi, s0, s1, w, nb_bg, n)
    new12 = interval_stock(bi, c(np.maximum(e, nb)), c(np.minimum(end, nb + 365)), w, nb_bg, n)
    mid = interval_stock(bi, c(np.maximum(e, nb + 365)), c(np.minimum(end, nb + 1095)), w, nb_bg, n)
    mnd = interval_stock(bi, s0, s1, w * mand, nb_bg, n)
    day = np.clip((t.start - d0).dt.days.to_numpy() - 1, 0, n - 1)
    b = t.bg.to_numpy()
    den = np.clip(tot[b, day], 1e-6, None)
    out = pd.DataFrame({"ep": t.ep.to_numpy(), "bg": b,
                        "new12_share": new12[b, day] / den, "mid_share": mid[b, day] / den,
                        "mand_share": mnd[b, day] / den}).astype({"new12_share": "float32", "mid_share": "float32",
                                                                    "mand_share": "float32"})
    out.to_parquet(fp.PILOT / "flood_pilot_expodyn.parquet")
    print(f"{len(out):,} rows -> {fp.PILOT / 'flood_pilot_expodyn.parquet'}")
    print(out[["new12_share", "mid_share", "mand_share"]].describe().round(3).to_string())
    by = out.assign(year=t.start.dt.year.to_numpy()).groupby("year")[["new12_share", "mid_share", "mand_share"]].mean().round(3)
    print("mean by year:\n", by.to_string())


if __name__ == "__main__":
    main()
