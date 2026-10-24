"""Does being flooded make people buy NFIP? Event study around Hurricane Harvey (Texas, Aug 25 - Sep 5 2017).

CANDIDATE-pool analysis for the "insure after an incident" prior (see docs/FLOOD_PRIORS.md). Block groups are
binned by Harvey claim intensity (claims per in-force policy on 2017-07-31, all zones). For each bin we follow
the number of in-force single-family policies in non-SFHA zones (X/B/C/D) month by month and divide by the
2017-07-31 value. The no-claim bin is the control; its drift is the background trend (and spillover from news).

  excess(t) = index_bin(t) / index_nolossbin(t) - 1     take-up added by direct experience, relative to the control
  share(t)  = 1 - index_nolossbin(t) / index_bin(t)     fraction of the bin's stock that is "excess"

Policies of 2017 carry 2010 block groups, and so do the claims of that year, so they match each other. Voluntary
purchase is approximated by mandatoryPurchaseFlag == False; rows are policy terms, so renewals keep the stock constant.

    python scripts/35_takeup_after_flood.py | tee $WXRISK_DATA/audit/takeup_after_flood.txt
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

FLOOD = wb2.cache_path("labels_extra") / "flood"
BASE = pd.Timestamp("2017-07-31")
MONTHS = pd.date_range("2016-07-31", "2020-12-31", freq="ME")
RATE_BINS = [-1, 0, 0.02, 0.10, 10]
RATE_LAB = ["no claims", "0-2%", "2-10%", ">10%"]


def main() -> None:
    cols = ["censusGeoid", "ratedFloodZone", "occupancyType", "policyEffectiveDate", "policyTerminationDate",
            "policyCount", "mandatoryPurchaseFlag"]
    p = ds.dataset(FLOOD / "pilot" / "policies_tx.parquet").to_table(columns=cols).to_pandas()
    p["occ"] = pd.to_numeric(p.occupancyType, errors="coerce")
    p = p[p.occ.isin([1, 11])].copy()
    p["geoid"] = p.censusGeoid.astype(str).str.zfill(12)
    eff = pd.to_datetime(p.policyEffectiveDate).values.astype("datetime64[D]")
    term = pd.to_datetime(p.policyTerminationDate).fillna(pd.Timestamp("2100-01-01")).values.astype("datetime64[D]")
    w = p.policyCount.astype(float).values
    sfha = p.ratedFloodZone.astype(str).str.upper().str[:1].isin(["A", "V"]).values
    vol = ~p.mandatoryPurchaseFlag.astype(bool).values

    def in_force(day: pd.Timestamp, mask: np.ndarray) -> pd.Series:
        d = np.datetime64(day.date())
        m = mask & (eff <= d) & (term > d)
        return pd.Series(w[m]).groupby(p.geoid.values[m]).sum()

    all_base = in_force(BASE, np.ones(len(p), bool))
    c = pd.concat([pd.read_parquet(f, columns=["state", "dateOfLoss", "censusGeoid"])
                   for f in sorted(glob.glob(str(FLOOD / "nfip_national" / "*2017.parquet")))])
    c = c[(c.state == "TX") & pd.to_datetime(c.dateOfLoss).between("2017-08-25", "2017-09-05")]
    n_claims = c.censusGeoid.astype(str).str.zfill(12).value_counts()
    g = pd.DataFrame({"base": all_base})
    g = g[g.base >= 20]
    g["rate"] = (n_claims.reindex(g.index).fillna(0) / g.base)
    g["bin"] = pd.cut(g.rate, RATE_BINS, labels=RATE_LAB)
    print(f"Harvey claims in file: {len(c):,}; block groups with >= 20 policies on {BASE.date()}: {len(g):,}")
    print(g.bin.value_counts().reindex(RATE_LAB).to_dict())

    for name, mask in [("non-SFHA voluntary", ~sfha & vol), ("non-SFHA all", ~sfha), ("SFHA all", sfha)]:
        rows = {}
        base_stock = in_force(BASE, mask)
        for t in MONTHS:
            s = in_force(t, mask).reindex(g.index).fillna(0)
            rows[t] = s.groupby(g.bin, observed=True).sum()
        tab = pd.DataFrame(rows).T
        b0 = base_stock.reindex(g.index).fillna(0).groupby(g.bin, observed=True).sum()
        idx = tab / b0
        print(f"\n## {name}: in-force stock / stock on {BASE.date()}, by Harvey claim rate")
        keep = [m for m in idx.index if m.month in (1, 7) or m.strftime("%Y-%m") in ("2017-08", "2017-09", "2017-10", "2017-12")]
        out = idx.loc[keep].round(3)
        out.index = [m.strftime("%Y-%m") for m in out.index]
        print(out.to_string())
        ctrl = idx["no claims"]
        for b in RATE_LAB[1:]:
            exc = idx[b] / ctrl - 1
            peak = exc.idxmax()
            print(f"  {b}: excess vs control peaks at {exc.max():+.1%} ({peak:%Y-%m}); 2020-12: {exc.iloc[-1]:+.1%}")


if __name__ == "__main__":
    main()
