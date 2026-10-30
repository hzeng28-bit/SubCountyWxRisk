"""Share of NFIP claims outside the mapped floodplain, by state and weather regime (stage 29).

Compares our own claims with the literature that says most flood damage lies outside the 100-year floodplain (Texas: Harris County studies).
Zone from the claim's ratedFloodZone: SFHA = A*, V*, AE, AH, AO, AR, A99; outside = X (any), B, C, D (and blank is counted apart).
Regime: named tropical event (stage 26 pattern) vs not; inland causes 2/3/4, 2010-2023.
Inputs : nfip_national/nfip_20*.parquet
Run    : python scripts/67_claims_outside_sfha.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

FL = wb2.cache_path("labels_extra") / "flood"
x = pd.concat([pd.read_parquet(FL / "nfip_national" / f"nfip_{y}.parquet", columns=["state", "ratedFloodZone", "causeOfDamage", "floodEvent", "yearOfLoss"])
               for y in range(2010, 2024)])
x = x[x.causeOfDamage.astype(str).isin(["2", "3", "4"])].copy()
z = x.ratedFloodZone.fillna("").astype(str).str.upper().str.strip()
x["zone"] = "blank"
x.loc[z.str.match(r"^(A|V)"), "zone"] = "SFHA"
x.loc[z.str.match(r"^(X|B|C|D)") & ~z.str.match(r"^(A|V)"), "zone"] = "outside"
x["trop"] = x.floodEvent.fillna("").str.contains(fp.TC_PATTERN)
x["scope"] = x.trop.map({True: "named tropical", False: "not tropical"})
t = x[x.state.isin(["TX", "LA", "FL", "MS", "NJ", "NY", "NC", "SC"])]
r = t.groupby(["state", "scope"]).zone.value_counts(normalize=True).unstack().fillna(0).round(3)
r["claims"] = t.groupby(["state", "scope"]).size()
print(r.to_string())
a = x[x.zone != "blank"]
print("\nall states: share outside SFHA among claims with a zone: %.3f (named tropical %.3f, not tropical %.3f)" % (
    (a.zone == "outside").mean(), (a[a.trop].zone == "outside").mean(), (a[~a.trop].zone == "outside").mean()))
big = t[t.zone != "blank"].assign(yr=lambda d: d.yearOfLoss)
print("\nTexas outside-SFHA share by loss year:")
print(big[big.state == "TX"].groupby("yr").zone.apply(lambda s: (s == "outside").mean()).round(3).to_string())
