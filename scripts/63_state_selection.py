"""Which states have high NFIP coverage and why, and which could be studied as a group? (stage 25)

County table (3,159 counties) from FEMA's penetration table, the policy file (policies effective 2022-2023) and the claim files:
  sfha_share     residential structures in the special flood hazard area / all (exposure that is mapped)
  pen_sfha, pen_non   take-up inside the SFHA and outside it
  mand, crs, prim, post   shares of policies with mandatory purchase, in a CRS community, primary residence, post-FIRM
  exp0009        claims 2000-2009 per 1,000 residential structures (earlier flood experience; partly a take-up proxy)
Parts: 1 decomposition of the between-state penetration gap into exposure share, SFHA take-up and non-SFHA take-up (counterfactuals);
2 county regressions on state effects and covariates; 3 state screening table (coverage, claim volume, inland non-tropical share,
claim clusters usable as events, stability of the policy base) for choosing a group.
Inputs : nfip_penetration.parquet, nfip_policies_v3.parquet, nfip_national/nfip_20*.parquet
Output : $WXRISK_DATA/audit/state_selection.txt, $WXRISK_DATA/audit/state_screen.csv
Run    : python scripts/63_state_selection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

FL = wb2.cache_path("labels_extra") / "flood"
OUT = []
ABBR = {"Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT",
        "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
        "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
        "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def county_table() -> pd.DataFrame:
    pen = pd.read_parquet(FL / "nfip_penetration.parquet")
    for c in ["resContractsInForce", "totalResStructures", "resContractsInForceSfha", "totalResStructuresSfha"]:
        pen[c] = pd.to_numeric(pen[c], errors="coerce")
    pen = pen[pen.state.isin(ABBR)].copy()
    pen["st"] = pen.state.map(ABBR)
    pen["fips"] = pen.fipsCode.astype(str).str.zfill(5)
    pen["sfha_share"] = pen.totalResStructuresSfha / pen.totalResStructures
    pen["pen"] = pen.resContractsInForce / pen.totalResStructures
    pen["pen_sfha"] = pen.resContractsInForceSfha / pen.totalResStructuresSfha
    pen["pen_non"] = (pen.resContractsInForce - pen.resContractsInForceSfha) / (pen.totalResStructures - pen.totalResStructuresSfha)
    q = f"""select substr(lpad(cast(censusGeoid as varchar), 12, '0'), 1, 5) fips,
                   sum(coalesce(policyCount, 1)) n,
                   avg(case when mandatoryPurchaseFlag then 1.0 else 0.0 end) mand,
                   avg(case when crsClassCode is not null and crsClassCode < 10 then 1.0 else 0.0 end) crs,
                   avg(case when primaryResidenceIndicator then 1.0 else 0.0 end) prim,
                   avg(case when postFIRMConstructionIndicator then 1.0 else 0.0 end) post
            from read_parquet('{FL / 'nfip_policies_v3.parquet'}')
            where year(cast(policyEffectiveDate as date)) in (2022, 2023) and censusGeoid is not null group by 1"""
    pol = duckdb.connect().execute(q).df()
    cl = []
    for y in range(2000, 2010):
        d = pd.read_parquet(FL / "nfip_national" / f"nfip_{y}.parquet", columns=["censusGeoid"])
        cl.append(d.censusGeoid.dropna().astype(str).str.split(".").str[0].str.zfill(12).str[:5])
    e = pd.concat(cl).value_counts().rename("claims0009")
    t = pen.merge(pol, on="fips", how="left").merge(e, left_on="fips", right_index=True, how="left")
    t["claims0009"] = t.claims0009.fillna(0)
    t["exp0009"] = 1000 * t.claims0009 / t.totalResStructures
    return t


def state_table(t: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("st").agg(pol=("resContractsInForce", "sum"), st_all=("totalResStructures", "sum"), pol_s=("resContractsInForceSfha", "sum"),
                            st_s=("totalResStructuresSfha", "sum"))
    g["pen"] = g.pol / g.st_all
    g["share"] = g.st_s / g.st_all
    g["pen_s"] = g.pol_s / g.st_s
    g["pen_n"] = (g.pol - g.pol_s) / (g.st_all - g.st_s)
    return g


def decompose(g: pd.DataFrame) -> pd.DataFrame:
    nat = {"share": g.st_s.sum() / g.st_all.sum(), "pen_s": g.pol_s.sum() / g.st_s.sum(),
           "pen_n": (g.pol.sum() - g.pol_s.sum()) / (g.st_all.sum() - g.st_s.sum())}

    def pen(sh, ps, pn):
        return sh * ps + (1 - sh) * pn
    out = pd.DataFrame(index=g.index)
    base = pen(nat["share"], nat["pen_s"], nat["pen_n"])
    out["pen"] = g.pen
    out["vs_nat"] = g.pen - base
    out["from_exposure_share"] = pen(g.share, nat["pen_s"], nat["pen_n"]) - base
    out["from_sfha_takeup"] = pen(nat["share"], g.pen_s, nat["pen_n"]) - base
    out["from_non_sfha_takeup"] = pen(nat["share"], nat["pen_s"], g.pen_n) - base
    return out


def main() -> None:
    t = county_table()
    g = state_table(t)
    say(f"== 0. {len(t)} counties, {t.n.notna().sum()} with policies in the 2022-23 policy file")

    say("\n== 1. what makes a state's penetration differ from the national one (counterfactual: replace one factor by the national value)")
    d = decompose(g[g.st_all >= 3e5]).sort_values("pen", ascending=False)
    say("   pen = share of structures in SFHA x SFHA take-up + the rest x non-SFHA take-up; columns are percentage points of penetration")
    say(pd.concat([g.loc[d.index, ["share", "pen_s", "pen_n"]], d], axis=1).assign(
        **{c: lambda x, c=c: x[c] * 100 for c in ["vs_nat", "from_exposure_share", "from_sfha_takeup", "from_non_sfha_takeup", "pen"]}).round(3).head(25).to_string())
    top = d[d.pen > 0.04].index.tolist()
    say(f"  states above 4%: {top}; their gap to the national value is made of: exposure share {d.loc[top].from_exposure_share.mean() * 100:+.2f}, "
        f"SFHA take-up {d.loc[top].from_sfha_takeup.mean() * 100:+.2f}, non-SFHA take-up {d.loc[top].from_non_sfha_takeup.mean() * 100:+.2f} points")

    say("\n== 2. county log-penetration: state effects and covariates (counties >= 1000 structures, penetration > 0)")
    c = t[(t.totalResStructures >= 1000) & (t.pen > 0) & t.mand.notna()].copy()
    c["lp"] = np.log(c.pen)
    c["l_sfha"] = np.log(c.sfha_share.clip(lower=1e-3))
    c["l_exp"] = np.log1p(c.exp0009)
    y = c.lp.to_numpy()

    def r2(X):
        X = np.column_stack([np.ones(len(X)), X])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    D = pd.get_dummies(c.st, drop_first=True).to_numpy(float)
    rows = {"state effects only": r2(D), "SFHA structure share": r2(c[["l_sfha"]].to_numpy()),
            "+ earlier claims per structure (2000-09)": r2(c[["l_sfha", "l_exp"]].to_numpy()),
            "+ CRS share": r2(c[["l_sfha", "l_exp", "crs"]].to_numpy()),
            "state effects + SFHA share + earlier claims": r2(np.column_stack([D, c[["l_sfha", "l_exp"]].to_numpy()])),
            "state effects + SFHA share + earlier claims + CRS": r2(np.column_stack([D, c[["l_sfha", "l_exp", "crs"]].to_numpy()]))}
    for k, v in rows.items():
        say(f"  R2 {v:.3f}  {k}")
    X = np.column_stack([np.ones(len(c)), (c[["l_sfha", "l_exp", "crs"]] - c[["l_sfha", "l_exp", "crs"]].mean()) / c[["l_sfha", "l_exp", "crs"]].std()])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    say(f"  standardised coefficients on log penetration (no state effects): SFHA share {b[1]:+.2f}, earlier claims {b[2]:+.2f}, CRS share {b[3]:+.2f}")

    say("\n== 3. state screen")
    cl = []
    for yv in range(2010, 2024):
        x = pd.read_parquet(FL / "nfip_national" / f"nfip_{yv}.parquet",
                            columns=["state", "dateOfLoss", "censusGeoid", "causeOfDamage", "floodEvent"])
        cl.append(x)
    x = pd.concat(cl)
    x["date"] = pd.to_datetime(x.dateOfLoss, errors="coerce")
    x["inl"] = x.causeOfDamage.astype(str).isin(["2", "3", "4"])
    x["trop"] = x.floodEvent.fillna("").str.contains(fp.TC_PATTERN)
    x["bg"] = x.censusGeoid.astype(str).str.split(".").str[0].str.zfill(12)
    x["cty"] = x.bg.str[:5]
    keep = x[x.inl & ~x.trop & x.date.notna()].sort_values(["cty", "date"])
    gap = keep.groupby("cty").date.diff().dt.days
    keep["cl"] = ((gap > 3) | gap.isna()).cumsum()
    ev = keep.groupby("cl").agg(st=("state", "first"), n=("date", "size"), nbg=("bg", "nunique"), yr=("date", lambda s: s.min().year))
    good = ev[(ev.n >= 10) & (ev.nbg >= 5)]
    s = pd.DataFrame({"claims_all": x.groupby("state").size(), "inland_nontropical": keep.groupby("state").size()})
    s["inland_share"] = s.inland_nontropical / s.claims_all
    s["events10_5"] = good.groupby("st").size()
    s["ev_train_le2016"] = good[good.yr <= 2016].groupby("st").size()
    s["ev_test_ge2017"] = good[good.yr >= 2017].groupby("st").size()
    s["share_claims_in_events"] = good.groupby("st").n.sum() / s.inland_nontropical
    s = s.join(g[["pen", "pen_s", "share"]]).join(
        t.groupby("st").apply(lambda z: (z.pen >= 0.01).mean(), include_groups=False).rename("cty_ge1pct"))
    s = s[s.index.isin(ABBR.values())].fillna(0).sort_values("inland_nontropical", ascending=False)
    say(s.head(22).round(3).to_string())
    s.to_csv(wb2.cache_path("audit") / "state_screen.csv")
    p = wb2.cache_path("audit") / "state_selection.txt"
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
