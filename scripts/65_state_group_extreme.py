"""State screening when tropical / extreme storm floods are kept in the label (stage 27).

Scope: all inland-cause (2/3/4) NFIP claims 2010-2023, named tropical events included. An event is a claim cluster (same county, gaps <= 3
days) with >= 10 claims in >= 5 block groups. Its regime is `extreme` when at least half of its claims are named tropical or fall in a
tropical-cyclone best-track window (800 km, +-2 days, the stage 8 definition, all counties), else `ordinary`. A state is usable for learning
the contrast between ordinary and extreme weather only if it has events of both regimes in both the training (<= 2016) and test (>= 2017) period.
Also compares candidate groups of states (pooled events by regime and period, claims-weighted penetration).

Inputs : nfip_national/nfip_20*.parquet, nfip_penetration.parquet, hurdat2*.txt, census county polygons
Output : $WXRISK_DATA/audit/state_group_extreme.txt, state_screen_extreme.csv
Run    : python scripts/65_state_group_extreme.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import labels_flood, wb2  # noqa: E402

FL = wb2.cache_path("labels_extra") / "flood"
OUT = []
ST_FIPS = {"01": "AL", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "16": "ID",
           "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
           "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
           "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT",
           "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY"}


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def main() -> None:
    polys = labels_flood.county_polys(tuple(ST_FIPS))
    ct = pd.DataFrame([(k, p.centroid.x, p.centroid.y) for k, p in polys.items()], columns=["cty", "lon", "lat"])
    days = pd.date_range("2009-12-01", "2023-12-31")
    tc = fp.tc_flags(days, ct, 800, 2)
    ci = {c: i for i, c in enumerate(ct.cty)}
    say(f"counties with centroid: {len(ct)}; share of county-days inside a TC window: {tc.mean():.3f}")

    x = pd.concat([pd.read_parquet(FL / "nfip_national" / f"nfip_{y}.parquet",
                                   columns=["state", "dateOfLoss", "censusGeoid", "countyCode", "causeOfDamage", "floodEvent"])
                   for y in range(2010, 2024)])
    x["date"] = pd.to_datetime(x.dateOfLoss, errors="coerce").dt.tz_localize(None).dt.normalize()
    x = x[x.date.notna() & x.causeOfDamage.astype(str).isin(["2", "3", "4"])].copy()
    bg = x.censusGeoid.astype("string").str.split(".").str[0].str.zfill(12)
    cty = bg.str[:5].fillna(x.countyCode.astype("string").str.split(".").str[0].str.zfill(5))
    x["bg"], x["cty"] = bg.fillna("none"), cty
    x = x[x.cty.notna() & x.cty.isin(ci)].copy()
    x["named"] = x.floodEvent.fillna("").str.contains(fp.TC_PATTERN)
    i = (x.date - days[0]).dt.days.to_numpy()
    x["track"] = tc[i, x.cty.map(ci).to_numpy().astype(int)]
    x["ext"] = x.named | x.track
    say(f"inland claims 2010-2023 with a county and date: {len(x):,}; named tropical {x.named.mean():.3f}; in TC window or named {x.ext.mean():.3f}")

    x = x.sort_values(["cty", "date"])
    gap = x.groupby("cty").date.diff().dt.days
    x["cl"] = ((gap > 3) | gap.isna()).cumsum()
    ev = x.groupby("cl").agg(st=("state", "first"), n=("date", "size"), nbg=("bg", "nunique"), ext=("ext", "mean"), yr=("date", lambda s: s.min().year))
    ev["regime"] = np.where(ev.ext >= 0.5, "extreme", "ordinary")
    good = ev[(ev.n >= 10) & (ev.nbg >= 5)].copy()
    good["period"] = np.where(good.yr <= 2016, "train", "test")

    pen = pd.read_parquet(FL / "nfip_penetration.parquet")
    for c in ["resContractsInForce", "totalResStructures"]:
        pen[c] = pd.to_numeric(pen[c], errors="coerce")
    pen["st"] = pen.fipsCode.astype(str).str.zfill(5).str[:2].map(ST_FIPS)
    pg = pen.groupby("st").agg(pol=("resContractsInForce", "sum"), struct=("totalResStructures", "sum"))
    pg["pen"] = pg.pol / pg.struct

    s = pd.DataFrame({"claims": x.groupby("state").size(), "ext_claim_share": x.groupby("state").ext.mean()})
    for r in ("ordinary", "extreme"):
        for p in ("train", "test"):
            s[f"{r[:3]}_{p}"] = good[(good.regime == r) & (good.period == p)].groupby("st").size()
    s = s.fillna(0).join(pg[["pen"]])
    s["balance"] = s[["ord_train", "ord_test", "ext_train", "ext_test"]].min(axis=1)
    s["events"] = s[["ord_train", "ord_test", "ext_train", "ext_test"]].sum(axis=1)
    s = s[s.index.isin(ST_FIPS.values())].sort_values("events", ascending=False)
    say("\n== state screen (events = clusters >= 10 claims in >= 5 block groups; ord/ext = regime, train <= 2016 / test >= 2017)")
    say(s.head(26).round(3).to_string())
    s.to_csv(wb2.cache_path("audit") / "state_screen_extreme.csv")

    say("\n== candidate groups (pooled)")
    groups = {"TX+LA": ["TX", "LA"], "TX+LA+MS": ["TX", "LA", "MS"], "TX+LA+FL": ["TX", "LA", "FL"], "TX+LA+MS+FL": ["TX", "LA", "MS", "FL"],
              "Gulf+South Atlantic (TX LA MS AL FL GA SC NC)": ["TX", "LA", "MS", "AL", "FL", "GA", "SC", "NC"],
              "above + VA NJ NY": ["TX", "LA", "MS", "AL", "FL", "GA", "SC", "NC", "VA", "NJ", "NY"],
              "TX alone": ["TX"]}
    rows = []
    for k, v in groups.items():
        z = s.loc[v]
        pp = pg.loc[v]
        rows.append({"group": k, "states": len(v), "claims_k": z.claims.sum() / 1000, "events": z.events.sum(),
                     "ord_tr": z.ord_train.sum(), "ord_te": z.ord_test.sum(), "ext_tr": z.ext_train.sum(), "ext_te": z.ext_test.sum(),
                     "ext_claim_share": float((z.ext_claim_share * z.claims).sum() / z.claims.sum()),
                     "pen_weighted": pp.pol.sum() / pp.struct.sum(), "pen_min": pp.pen.min(), "pen_max": pp.pen.max()})
    say(pd.DataFrame(rows).round(3).to_string(index=False))
    p = wb2.cache_path("audit") / "state_group_extreme.txt"
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
