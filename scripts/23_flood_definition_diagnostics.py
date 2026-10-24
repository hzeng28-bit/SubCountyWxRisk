"""Diagnostics behind the flood definition (Texas pilot): what to exclude, how to trigger, what lag.

  1. tropical-cyclone exclusion   how many non-tropical-named claims still sit inside a tropical-cyclone
                                  window, for several radii and day windows (name-based tagging is partial)
  2. rain trigger recall          share of claims whose county saw >= T mm (1-day or 3-day accumulation)
                                  within the 2 days up to the claim date, per cause code
  3. lag                          days between the county's 3-day rain peak and the claim date
  4. candidate size               how many county-days a trigger threshold produces (training-set size)

    python scripts/23_flood_definition_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402


def main() -> None:
    days, lat, lon, P = fp.load_precip()
    ct = fp.county_table()
    masks = fp.county_masks(lat, lon)
    ct = ct[ct.cty.isin(masks)].reset_index(drop=True)
    cid = {c: i for i, c in enumerate(ct.cty)}
    print(f"precipitation {P.shape} {days[0]:%Y-%m-%d}..{days[-1]:%Y-%m-%d}; {len(ct)} counties")
    D1 = fp.county_daily(P, {c: masks[c] for c in ct.cty}, 1).to_numpy()
    D3 = fp.county_daily(P, {c: masks[c] for c in ct.cty}, 3).to_numpy()
    np.savez_compressed(wb2.cache_path("labels_extra") / "flood" / "pilot" / "county_daily.npz", D1=D1, D3=D3,
                        cty=ct.cty.values, days=days.values)

    c = fp.claims_tx()
    c = c[c.cty.isin(cid) & c.date.between(days[0], days[-1])].copy()
    c["di"] = (c.date - days[0]).dt.days.to_numpy()
    c["ci"] = c.cty.map(cid).to_numpy()
    print(f"\nTexas non-tropical-named inland claims on the 2010 geography: {len(c):,} "
          f"(${c.paid.sum() / 1e9:.2f}B), {c.bg10.nunique():,} block groups, {c.cty.nunique()} counties")
    print("by year:", c.groupby(c.date.dt.year).size().to_dict())

    print("\n== 1. tropical-cyclone windows applied to the claims that are NOT name-tagged tropical")
    rows = []
    for r in (400, 800, 1200):
        for w in (1, 2, 4):
            f = fp.tc_flags(days, ct, r, w)
            hit = f[c.di.to_numpy(), c.ci.to_numpy()]
            rows.append({"radius_km": r, "window_days": w, "claims_in_window": hit.mean(),
                         "paid_in_window": c.paid[hit].sum() / c.paid.sum(), "county_days_flagged": f.mean()})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    flag = fp.tc_flags(days, ct, 800, 2)
    c["tc"] = flag[c.di.to_numpy(), c.ci.to_numpy()]
    np.save(wb2.cache_path("labels_extra") / "flood" / "pilot" / "tc_flag_800_2.npy", flag)

    print("\n== 2. rain trigger recall (claims outside the 800 km / +-2 day tropical window)")
    k = c[~c.tc]
    rows = []
    for name, D in (("1-day max", D1), ("3-day max", D3)):
        for T in (10, 20, 30, 40, 50, 75, 100):
            ok = np.zeros(len(k), bool)
            for lag in (0, 1, 2):
                ok |= D[np.clip(k.di.to_numpy() - lag, 0, None), k.ci.to_numpy()] >= T
            rows.append({"accumulation": name, "T_mm": T, "recall_all": ok.mean(),
                         **{f"cause{x}": ok[(k.cause == x).to_numpy()].mean() for x in ("2", "3", "4")},
                         "county_days": int((D >= T).sum() - (flag & (D >= T)).sum())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\n== 3. lag between the county's 3-day rain peak (previous 7 days) and the claim date")
    best = np.zeros(len(k), int)
    cm = k.ci.to_numpy()
    di = k.di.to_numpy()
    for i in range(len(k)):
        lo = max(di[i] - 7, 0)
        seg = D3[lo:di[i] + 1, cm[i]]
        best[i] = di[i] - (lo + int(seg.argmax()))
    s = pd.Series(best)
    print("lag quantiles (days) 10/25/50/75/90/95%:", s.quantile([.1, .25, .5, .75, .9, .95]).astype(int).tolist())
    print("share by lag:", s.value_counts(normalize=True).sort_index().head(8).round(3).to_dict())
    for x in ("2", "4"):
        print(f"  cause {x}: median lag {int(s[(k.cause == x).to_numpy()].median())} d, 90th {int(s[(k.cause == x).to_numpy()].quantile(.9))} d")


if __name__ == "__main__":
    main()
