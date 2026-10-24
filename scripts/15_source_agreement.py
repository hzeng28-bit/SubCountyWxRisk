"""Do independent-ish sources agree with the Storm Events labels?

  SPC   storm reports (county FIPS in f1..f4) vs Storm Events SCS3 county-days, with +-1 day slack
        because SPC dates follow its own convective-day convention
  RMA   crop-insurance indemnity for Hail / Wind causes (county x month of loss) vs Storm Events
        hail / wind reports in the same county-month

Scope: CO/NM/AZ counties, May-Sep, 2000-2022. Files come from $WXRISK_DATA/labels_extra.

    python scripts/15_source_agreement.py
"""

from __future__ import annotations

import glob
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import labels_alt, wb2  # noqa: E402

EXTRA = wb2.cache_path("labels_extra")
STATES = {"CO": 8, "NM": 35, "AZ": 4}


def spc_county_days() -> pd.DataFrame:
    fr = []
    for nm, y0 in (("hail", 1955), ("wind", 1955), ("torn", 1950)):
        z = zipfile.ZipFile(EXTRA / "spc" / f"{y0}-2025_{nm}.csv.zip")
        d = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        d = d[d.st.isin(STATES) & d.yr.between(2000, 2022) & d.mo.between(5, 9)].copy()
        d["kind"] = nm
        fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["date"] = pd.to_datetime(d.date)
    out = []
    for col in ("f1", "f2", "f3", "f4"):
        s = d[d[col] > 0].copy()
        s["cty"] = s.stf.astype(int).astype(str).str.zfill(2) + s[col].astype(int).astype(str).str.zfill(3)
        out.append(s[["cty", "date", "kind", "mag"]])
    return pd.concat(out, ignore_index=True)


def section_spc(se: pd.DataFrame) -> None:
    print("\n== SPC vs Storm Events, county-days (May-Sep 2000-2022, CO/NM/AZ)")
    spc = spc_county_days()
    n_rec = {k: int((spc.kind == k).sum()) for k in ("hail", "wind", "torn")}
    print("SPC county-assigned records:", n_rec)
    a = set(map(tuple, se[["cty", "date"]].drop_duplicates().to_numpy()))
    b = set(map(tuple, spc[["cty", "date"]].drop_duplicates().to_numpy()))
    both = a & b

    def shifted(s, k):
        return {(c, d + pd.Timedelta(days=k)) for c, d in s}

    near = a & (b | shifted(b, 1) | shifted(b, -1))
    near_b = b & (a | shifted(a, 1) | shifted(a, -1))
    print(f"Storm Events county-days {len(a)}; SPC county-days {len(b)}")
    print(f"exact-date overlap {len(both)}  ({len(both) / len(a):.3f} of SE, {len(both) / len(b):.3f} of SPC)")
    print(f"within +-1 day: {len(near) / len(a):.3f} of SE found in SPC, "
          f"{len(near_b) / len(b):.3f} of SPC found in SE")


def rma_month_losses() -> pd.DataFrame:
    fr = []
    for f in sorted(glob.glob(str(EXTRA / "rma" / "colsom_*.zip"))):
        z = zipfile.ZipFile(f)
        d = pd.read_csv(z.open(z.namelist()[0]), sep="|", header=None, dtype=str, low_memory=False,
                        encoding="latin1")
        d = d[d[2].isin(STATES)]
        if len(d):
            fr.append(d[[0, 1, 3, 12, 13, 15, 16, 17, d.shape[1] - 2]])
    d = pd.concat(fr, ignore_index=True)
    d.columns = ["cyear", "state", "county", "cause", "month", "lyear", "pol_earn", "pol_ind", "indem"]
    d["indem"] = pd.to_numeric(d.indem, errors="coerce").fillna(0)
    d["pol_ind"] = pd.to_numeric(d.pol_ind, errors="coerce").fillna(0)
    d["cty"] = d.state.str.zfill(2) + d.county.str.zfill(3)
    d["lyear"] = pd.to_numeric(d.lyear, errors="coerce").fillna(pd.to_numeric(d.cyear, errors="coerce"))
    d["month"] = pd.to_numeric(d.month, errors="coerce")
    d["cause"] = d.cause.fillna("").str.strip()
    return d


def section_rma(se: pd.DataFrame, years=range(2000, 2023)) -> None:
    print("\n== RMA crop-insurance indemnity vs Storm Events, county-months (May-Sep)")
    r = rma_month_losses()
    print("distinct causes in CO/NM/AZ containing hail/wind/tornado:",
          sorted(c for c in r.cause.unique() if any(k in c.lower() for k in ("hail", "wind", "tornado"))))
    r = r[r.lyear.between(years[0], years[-1]) & r.month.between(5, 9)]
    se = se.copy()
    se["lyear"], se["month"] = se.date.dt.year, se.date.dt.month
    full = pd.MultiIndex.from_product([sorted(se.cty.unique()), list(years), range(5, 10)],
                                      names=["cty", "lyear", "month"]).to_frame(index=False)
    # exact cause names: "Hot Wind" (dry heat) is not convective wind
    for kind, key, se_types in (("hail", "Hail", ["Hail"]), ("wind", "Wind/Excess Wind", ["Thunderstorm Wind"])):
        m = r[r.cause == key].groupby(["cty", "lyear", "month"], as_index=False).agg(
            indem=("indem", "sum"), pol=("pol_ind", "sum"))
        s = se[se.EVENT_TYPE.isin(se_types)].groupby(["cty", "lyear", "month"], as_index=False).agg(
            n_rep=("prop", "size"), dmg=("prop", "sum"))
        t = full.merge(m, how="left", on=["cty", "lyear", "month"]).merge(s, how="left", on=["cty", "lyear", "month"])
        t = t.fillna({"indem": 0, "pol": 0, "n_rep": 0, "dmg": 0})
        pos = t.indem > 0
        print(f"\n{kind}: county-months {len(t)}, with RMA {kind} indemnity {pos.sum()}, "
              f"with SE {kind} report {int((t.n_rep > 0).sum())}")
        print(f"  P(SE report | RMA indemnity>0) = {(t.n_rep[pos] > 0).mean():.3f}   "
              f"P(RMA indemnity>0 | SE report) = {pos[t.n_rep > 0].mean():.3f}")
        both = t[(t.indem > 0) & (t.n_rep > 0)]
        print(f"  Spearman(RMA indemnity, SE report count): all county-months "
              f"{t[['indem', 'n_rep']].corr('spearman').iloc[0, 1]:.3f}; "
              f"months with both >0 (n={len(both)}) {both[['indem', 'n_rep']].corr('spearman').iloc[0, 1]:.3f}; "
              f"vs SE damage $ (both >0): {both[['indem', 'dmg']].corr('spearman').iloc[0, 1]:.3f}")
        by = t.assign(st=t.cty.str[:2].map({"08": "CO", "35": "NM", "04": "AZ"})).groupby("st").apply(
            lambda g: pd.Series({"rma_pos_share": (g.indem > 0).mean(), "se_rep_share": (g.n_rep > 0).mean()}),
            include_groups=False)
        print(by.round(3).to_string())


def main() -> None:
    se = labels_alt.storm_events(range(2000, 2023))
    print(f"Storm Events SCS3 records {len(se)}")
    section_spc(se)
    section_rma(se)


if __name__ == "__main__":
    main()
