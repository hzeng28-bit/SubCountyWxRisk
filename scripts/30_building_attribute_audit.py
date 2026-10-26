"""Which building attributes does NFIP give us, how complete are they, and do they carry a loss signal?

Candidate PRIORS for the flood work (foundation, elevation, basement), not forecast inputs. Sections:

  fill      policy-side fill rates and the 'unknown' foundation share by effective year (Texas)
  claims    Texas claims 2010+: damage/coverage by foundation x exterior depth, year fixed effects
  known     known vs unknown foundation on the same years: is 'unknown' missing at random?
  fe        damage/coverage vs slab with depth-bin and state-month fixed effects (national + per state)
  plateau   median building damage ($) by INTERIOR depth, slab vs basement vs elevated
  glm       the same foundation ratios with a log-link GLM (ratio of MEANS, dollars) next to the log-OLS (ratio of geometric means)
  occ       in the claims file 'unknown' foundation IS the legacy occupancy code 1: loss ratio of occupancy 1 vs 11
            under state-month fixed effects, and the elevated-indicator effect inside each family

Foundation codes (OpenFEMA v3 dictionary): 1 slab, 2 basement, 3 crawlspace, 4-6 elevated (4 without enclosure, 5-6 with
enclosure); 0 is NOT in the dictionary (an unknown/default code, see docs/CLEANING_REGISTER.md). 65% of crawlspace claims
carry elevatedBuildingIndicator, against 93-95% for 4-6 and 1% for slab and basement, so 3 is its own group.

Reads the files written by scripts/20_fetch_nfip_claims.py and the OpenFEMA policy bulk file.

    python scripts/30_building_attribute_audit.py | tee $WXRISK_DATA/audit/building_attrs.txt
    python scripts/30_building_attribute_audit.py --sections claims fe
"""

from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

warnings.filterwarnings("ignore")
FLOOD = wb2.cache_path("labels_extra") / "flood"
POLICIES = FLOOD / "nfip_policies_v3.parquet"
CLAIMS = sorted(glob.glob(str(FLOOD / "nfip_national" / "*.parquet")))
GROUP = {0: "unk", 1: "slab", 2: "bsmt", 3: "crawl", 4: "elev", 5: "elev", 6: "elev"}
DEPTH_BINS = [0, 6, 12, 24, 48, 1e4]
DEPTH_LAB = ["1-6", "7-12", "13-24", "25-48", ">48"]
NUM = ["interiorWaterDepth", "exteriorWaterDepth", "buildingDamageAmount", "totalBuildingInsuranceCoverage"]
CLAIM_COLS = ["state", "yearOfLoss", "dateOfLoss", "foundationType", "occupancyType", "elevatedBuildingIndicator", *NUM]


def load_claims(since: int = 2010) -> pd.DataFrame:
    c = pd.concat([pd.read_parquet(f, columns=CLAIM_COLS) for f in CLAIMS])
    c = c[c.yearOfLoss >= since].copy()
    for k in NUM:
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c["g"] = pd.to_numeric(c.foundationType, errors="coerce").map(GROUP)
    c["occ"] = pd.to_numeric(c.occupancyType, errors="coerce")
    c["dr"] = (c.buildingDamageAmount / c.totalBuildingInsuranceCoverage).clip(0, 1.5)
    c["ev"] = c.state.astype(str) + "_" + pd.to_datetime(c.dateOfLoss).dt.strftime("%Y-%m")
    return c


def single_family(c: pd.DataFrame) -> pd.DataFrame:
    """Occupancy 1 (and 11 under Risk Rating 2.0) with a usable damage ratio."""
    return c[c.occ.isin([1, 11]) & c.dr.notna() & c.totalBuildingInsuranceCoverage.gt(0)].copy()


def sec_fill() -> None:
    print("\n## fill: Texas policies, effective 2009+")
    cols = ["foundationType", "elevatedBuildingIndicator", "basementEnclosureCrawlspaceType", "enclosureTypeCode",
            "lowestFloorElevation", "elevationDifference", "numberOfFloorsInInsuredBuilding", "construction",
            "buildingDescriptionCode", "policyEffectiveDate"]
    t = ds.dataset(POLICIES).to_table(columns=cols, filter=pc.field("propertyState") == "TX").to_pandas()
    print(f"rows {len(t):,}")
    print("non-null share:\n", t.drop(columns="policyEffectiveDate").notna().mean().round(3).to_string())
    yr = pd.to_datetime(t.policyEffectiveDate).dt.year
    unk = (t.foundationType.astype(str) == "0").groupby(yr).mean().round(2)
    print("foundationType == 0 (unknown) share by effective year:\n", unk.to_dict())
    print("`construction` is a boolean 'under construction' flag, share true:",
          round(float(t.construction.astype(bool).mean()), 4), "-> NOT wood/brick; no such field exists")
    ct = pd.crosstab(t.foundationType.astype(str), t.elevatedBuildingIndicator, normalize="index").round(2)
    print("foundationType x elevatedBuildingIndicator (row share):\n", ct.to_string())


def sec_claims(c: pd.DataFrame) -> None:
    print("\n## claims: Texas, single-family, 2010+, foundation known")
    s = single_family(c[c.state == "TX"])
    k = s[s.g.isin(["slab", "bsmt", "elev"]) & s.exteriorWaterDepth.gt(0)].copy()
    k["db"] = pd.cut(k.exteriorWaterDepth, DEPTH_BINS, labels=DEPTH_LAB)
    print(f"n {len(k):,}; basement n {int((k.g == 'bsmt').sum())} (too few to trust)")
    print("median damage/coverage by exterior depth bin:\n",
          k.pivot_table(index="db", columns="g", values="dr", aggfunc="median").round(2).to_string())
    print("slab-only median damage/coverage by year (event/year drift dwarfs most effects):\n",
          s[s.g == "slab"].groupby("yearOfLoss").dr.median().round(2).to_dict())
    d = k[k.dr > 0].copy()
    d["y"] = np.log(d.dr)
    x = pd.get_dummies(d[["db", "g"]].astype(str)).drop(columns=["g_slab", "db_1-6"])
    x = pd.concat([x, pd.get_dummies(d.yearOfLoss.astype(str), prefix="y", drop_first=True)], axis=1).astype(float)
    x.insert(0, "c", 1.0)
    b = pd.Series(np.linalg.lstsq(x.values, d.y.values, rcond=None)[0], x.columns)
    g = [i for i in b.index if i.startswith("g_")]
    print("ratio vs slab (depth bin + year FE):", np.exp(b[g]).round(2).to_dict())


def sec_known(c: pd.DataFrame) -> None:
    print("\n## known: is 'unknown' foundation missing at random? (Texas, single-family, 2017-2021)")
    s = single_family(c[(c.state == "TX") & c.yearOfLoss.between(2017, 2021)])
    print(s.groupby("g", dropna=False).agg(n=("dr", "size"), dr_median=("dr", "median"),
                                           ext_depth_median=("exteriorWaterDepth", "median")).round(3).to_string())
    print("-> unknown has higher damage ratio, but in claims 'unknown' == occupancy code 1 (section occ): this is a contrast "
          "between occupancy-code families, not between unknown and known buildings")


def fe_ratio(d: pd.DataFrame, min_events: int = 30) -> tuple[int, object]:
    d = d[d.groupby("ev").ev.transform("size") >= min_events].copy()
    d["db"] = pd.cut(d.exteriorWaterDepth, DEPTH_BINS, labels=False).astype(float)
    x = pd.get_dummies(d.g).astype(float)
    if "slab" not in x:
        return len(d), "no slab"
    x = x.drop(columns="slab")
    if x.shape[1] == 0 or (x.sum() < 20).any():
        return len(d), "too few"
    for b in [1, 2, 3, 4]:
        x[f"d{b}"] = (d.db == b).astype(float)
    xd = x - x.groupby(d.ev.values).transform("mean")
    yd = d.y - d.groupby("ev").y.transform("mean")
    b = pd.Series(np.linalg.lstsq(xd.values, yd.values, rcond=None)[0], x.columns)
    return len(d), np.exp(b[[i for i in ["bsmt", "elev"] if i in b.index]]).round(2).to_dict()


def sec_fe(c: pd.DataFrame) -> None:
    print("\n## fe: damage/coverage ratio vs slab, state-month fixed effects (floodEvent covers only ~81% / 72 events)")
    s = single_family(c)
    s = s[s.g.isin(["slab", "bsmt", "elev"]) & s.buildingDamageAmount.gt(0) & s.exteriorWaterDepth.gt(0)].copy()
    s["y"] = np.log(s.dr.clip(1e-4, 1.5))
    print("national:", fe_ratio(s))
    cnt = s.groupby("state").g.value_counts().unstack(fill_value=0)
    cnt["n"] = cnt.sum(axis=1)
    print("basement-heavy states (basement n >= 200):")
    for st in cnt[cnt.get("bsmt", 0) >= 200].sort_values("bsmt", ascending=False).index:
        print(f"  {st} basement_share={cnt.loc[st, 'bsmt'] / cnt.loc[st, 'n']:.2f}", fe_ratio(s[s.state == st]))


def sec_plateau(c: pd.DataFrame) -> None:
    print("\n## plateau: median building damage ($) by INTERIOR depth, single-family, 2010+, national")
    s = single_family(c)
    s = s[s.g.isin(["slab", "bsmt", "elev"]) & s.interiorWaterDepth.notna()].copy()
    s["ib"] = pd.cut(s.interiorWaterDepth, [-1, 3, 12, 24, 48, 1e4], labels=["0-3", "4-12", "13-24", "25-48", ">48"])
    print(s.pivot_table(index="ib", columns="g", values="buildingDamageAmount", aggfunc="median").round(0).to_string())
    print("Basement damage is flat while slab keeps rising. NFIP covers only limited basement items, so this cannot be"
          " separated from a coverage cap or from 'equipment sits at floor level in the basement'.")
    print("Depth 0 / missing is unreliable (damage is large at 0): treat it as missing, not as 'dry'.")


def _fe_diff(d: pd.DataFrame, flag: pd.Series) -> tuple[float, int]:
    """exp(coef) of log(dr) on a 0/1 flag with state-month and depth-bin fixed effects (ratio of geometric means)."""
    d = d.assign(f=np.asarray(flag, dtype=float)).reset_index(drop=True)
    d = d[d.groupby("ev").ev.transform("size") >= 30].reset_index(drop=True)
    f = d.f
    db = pd.get_dummies(pd.cut(d.exteriorWaterDepth, DEPTH_BINS, labels=False)).astype(float).iloc[:, 1:]
    x = pd.concat([f.rename("f"), db], axis=1)
    xd = x - x.groupby(d.ev.values).transform("mean")
    yd = d.y - d.groupby("ev").y.transform("mean")
    b = np.linalg.lstsq(xd.values, yd.values, rcond=None)[0]
    return round(float(np.exp(b[0])), 2), len(d)


def sec_occ(c: pd.DataFrame) -> None:
    print("\n## occ: foundation 'unknown' vs occupancy code in the claims file (single-family, 2010+, national)")
    s = single_family(c)
    print("share foundation == 0 by occupancy code:", s.groupby("occ").g.apply(lambda g: round((g == "unk").mean(), 3)).to_dict())
    print("rows by occupancy code and year (code 1 = legacy, 11 = Risk Rating 2.0 coding):")
    print(s.groupby(["yearOfLoss", "occ"]).size().unstack(fill_value=0).to_string())
    d = s[s.buildingDamageAmount.gt(0) & s.exteriorWaterDepth.gt(0)].copy()
    d["y"] = np.log(d.dr.clip(1e-4, 1.5))
    r, n = _fe_diff(d, d.occ == 11)
    print(f"occupancy 11 vs 1, ratio of geometric-mean loss ratios, state-month + depth-bin FE: {r} (n {n:,})"
          "  <- the size of the 'unknown vs known' contrast that needs no foundation effect at all")
    for occ in (1, 11):
        e = d[d.occ == occ]
        flag = e.elevatedBuildingIndicator.astype(str).isin(["1", "True", "true", "Y"])
        if flag.sum() >= 50 and (~flag).sum() >= 50:
            r, n = _fe_diff(e, flag)
            print(f"elevated indicator vs not, occupancy {occ}: ratio {r} (n {n:,}, elevated {int(flag.sum()):,})")
        else:
            print(f"elevated indicator, occupancy {occ}: too few elevated claims ({int(flag.sum())})")
    print("foundation groups (slab/bsmt/crawl/elev) exist only inside occupancy 11, so every foundation ratio of"
          " sections claims/fe is an occupancy-11 estimate.")


def sec_glm(c: pd.DataFrame) -> None:
    from scipy import sparse
    from sklearn.linear_model import LinearRegression, PoissonRegressor
    from sklearn.preprocessing import OneHotEncoder

    print("\n## glm: foundation vs slab, depth-bin + fixed effects; exp(coef) of log-OLS (ratio of geometric means) vs log-link GLM (ratio of means)")
    s = single_family(c)
    s = s[s.g.isin(["slab", "bsmt", "crawl", "elev"]) & s.buildingDamageAmount.gt(0) & s.exteriorWaterDepth.gt(0)].copy()
    s["dr"] = s.dr.clip(1e-4, 1.5)
    for label, d, fe in (("Texas, year FE", s[s.state == "TX"], "yearOfLoss"), ("national, state-month FE", s, "ev")):
        d = d[d.groupby(fe)[fe].transform("size") >= 30]
        g = pd.get_dummies(d.g).astype(float).drop(columns="slab")
        db = pd.get_dummies(pd.cut(d.exteriorWaterDepth, DEPTH_BINS, labels=False)).astype(float).iloc[:, 1:]
        f = OneHotEncoder(drop="first", sparse_output=True).fit_transform(d[[fe]].astype(str))
        X = sparse.hstack([sparse.csr_matrix(pd.concat([g, db], axis=1).to_numpy()), f]).tocsr()
        ols = LinearRegression().fit(X, np.log(d.dr))
        glm = PoissonRegressor(alpha=0, max_iter=500).fit(X, d.dr)
        print(f"{label} (n {len(d):,}):", {k: {"geo_mean_ratio": round(float(np.exp(ols.coef_[i])), 2),
                                              "mean_ratio": round(float(np.exp(glm.coef_[i])), 2)} for i, k in enumerate(g.columns)})
    print("Dollar losses follow the ratio of means; the geometric-mean ratio overstates a reduction when the spread differs between groups.")


SECTIONS = ["fill", "claims", "known", "fe", "plateau", "occ", "glm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    a = ap.parse_args()
    if "fill" in a.sections:
        sec_fill()
    if set(a.sections) - {"fill"}:
        c = load_claims()
        for name, fn in [("claims", sec_claims), ("known", sec_known), ("fe", sec_fe), ("plateau", sec_plateau), ("occ", sec_occ), ("glm", sec_glm)]:
            if name in a.sections:
                fn(c)


if __name__ == "__main__":
    main()
