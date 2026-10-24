"""County-day flood labels from Storm Events, NFIP claims, USGS gauges and FEMA declarations.

Universe and date convention match labels_alt: 112 CO/NM/AZ counties, May-Sep, 2000-2022, local
standard-time date. Every function returns a sparse table keyed (cty, date); callers left-join it
onto the full county x day grid.

  se_flood_days      Storm Events Flash Flood / Flood (county records); multi-day floods expand over
                     their begin..end days, capped at 30
  nfip_days          NFIP claims by date of loss (count, dollars paid)
  gauge_rise_days    USGS daily-mean discharge that jumps >= RISE_FACTOR x the previous day and
                     reaches the site's own 95th percentile: a flashy, storm-driven rise (snowmelt
                     rises slowly, so it does not trigger). Daily means miss short flash floods.
  fema_flood_days    county-level FEMA flood declarations, expanded over the incident window
"""

from __future__ import annotations

import glob
import io
import zipfile

import numpy as np
import pandas as pd

from . import labels_alt, wb2

EXTRA = wb2.cache_path("labels_extra")
FLOOD = EXTRA / "flood"
RISE_FACTOR, MIN_Q, PCTL = 3.0, 5.0, 0.95
SE_COLS = ["BEGIN_YEARMONTH", "BEGIN_DAY", "END_YEARMONTH", "END_DAY", "STATE_FIPS", "CZ_FIPS",
           "CZ_TYPE", "EVENT_TYPE", "DAMAGE_PROPERTY"]


def season(d: pd.Series) -> pd.Series:
    return d.dt.month.isin([5, 6, 7, 8, 9]) & d.dt.year.between(2000, 2022)


def se_flood_days() -> pd.DataFrame:
    fr = []
    for y in range(2000, 2023):
        f = glob.glob(str(wb2.cache_path("storm_events")) + f"/StormEvents_details*_d{y}_*.csv.gz")[0]
        fr.append(pd.read_csv(f, usecols=SE_COLS, dtype=str, low_memory=False))
    ev = pd.concat(fr, ignore_index=True)
    ev = ev[ev.EVENT_TYPE.isin(["Flash Flood", "Flood"]) & (ev.CZ_TYPE == "C")
            & ev.STATE_FIPS.str.zfill(2).isin(labels_alt.STATES)].reset_index(drop=True)
    ev["cty"] = ev.STATE_FIPS.str.zfill(2) + ev.CZ_FIPS.str.zfill(3)
    b = pd.to_datetime(ev.BEGIN_YEARMONTH + ev.BEGIN_DAY.str.zfill(2), format="%Y%m%d", errors="coerce")
    e = pd.to_datetime(ev.END_YEARMONTH + ev.END_DAY.str.zfill(2), format="%Y%m%d", errors="coerce")
    ev["prop"] = ev.DAMAGE_PROPERTY.map(labels_alt.parse_damage)
    span = ((e - b).dt.days.clip(0, 30)).fillna(0).astype(int)
    rows = []
    for k in range(31):
        s = ev[span >= k]
        rows.append(pd.DataFrame({"cty": s.cty, "date": b[s.index] + pd.Timedelta(days=k),
                                  "ff": (s.EVENT_TYPE == "Flash Flood") & (k == 0),
                                  "fl": s.EVENT_TYPE == "Flood", "prop": s.prop * (1.0 if k == 0 else 0.0)}))
    d = pd.concat(rows, ignore_index=True)
    d = d[season(d.date)]
    return d.groupby(["cty", "date"], as_index=False).agg(
        f_ff=("ff", "max"), f_fl=("fl", "max"), f_dmg=("prop", "sum")).assign(
        f_any=lambda x: (x.f_ff | x.f_fl).astype(np.int8),
        f_ff=lambda x: x.f_ff.astype(np.int8), f_fl=lambda x: x.f_fl.astype(np.int8),
        f_10k=lambda x: (x.f_dmg >= 1e4).astype(np.int8),
        f_100k=lambda x: (x.f_dmg >= 1e5).astype(np.int8))


def nfip_days() -> pd.DataFrame:
    d = pd.read_parquet(FLOOD / "nfip_claims_conmaz.parquet")
    d["date"] = pd.to_datetime(d.dateOfLoss).dt.tz_localize(None).dt.normalize()
    d["cty"] = d.countyCode.astype(str).str.zfill(5)
    d["paid"] = pd.to_numeric(d.amountPaidOnBuildingClaim, errors="coerce").fillna(0) + pd.to_numeric(
        d.amountPaidOnContentsClaim, errors="coerce").fillna(0)
    d = d[season(d.date)]
    return d.groupby(["cty", "date"], as_index=False).agg(n_claims=("paid", "size"), paid=("paid", "sum"))


def county_polys(states: tuple[str, ...] = ("04", "08", "35")):
    import shapefile
    from shapely.geometry import shape

    z = zipfile.ZipFile(EXTRA / "census" / "cb_2020_us_county_20m.zip")
    rd = shapefile.Reader(shp=io.BytesIO(z.read("cb_2020_us_county_20m.shp")),
                          shx=io.BytesIO(z.read("cb_2020_us_county_20m.shx")),
                          dbf=io.BytesIO(z.read("cb_2020_us_county_20m.dbf")))
    names = [f[0] for f in rd.fields[1:]]
    out = {}
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        if rec["STATEFP"] in states:
            out[rec["GEOID"]] = shape(sr.shape.__geo_interface__)
    return out


def _read_rdb(path) -> pd.DataFrame:
    """USGS RDB: '#' comments, a header, then a width row such as '5s 15s 20d' that is not data."""
    d = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    return d[~d.iloc[:, 0].fillna("").str.match(r"^\d+[sdn]$")].reset_index(drop=True)


def read_dv(path) -> pd.DataFrame:
    """Daily-mean discharge rows (site_no, datetime, q). A file can hold several header blocks."""
    rows, qi = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if f[0] == "agency_cd":
                qi = [i for i, c in enumerate(f) if c.endswith("_00060_00003")]
                continue
            if f[0][:1].isdigit() or not qi or len(f) <= qi[0]:  # width row such as '5s 15s 20d'
                continue
            rows.append((f[1], f[2], f[qi[0]]))
    d = pd.DataFrame(rows, columns=["site_no", "datetime", "q"])
    d["q"] = pd.to_numeric(d.q, errors="coerce")
    return d


def gauge_rise_days() -> pd.DataFrame:
    from shapely.geometry import Point

    polys = county_polys()
    sites = pd.concat([_read_rdb(p) for p in sorted(glob.glob(str(FLOOD / "nwis" / "sites_*.rdb")))])
    sites = sites.dropna(subset=["dec_lat_va", "dec_long_va"]).drop_duplicates("site_no")
    cmap = {}
    for r in sites.itertuples():
        pt = Point(float(r.dec_long_va), float(r.dec_lat_va))
        for k, p in polys.items():
            if p.contains(pt):
                cmap[r.site_no] = k
                break
    q = pd.concat([read_dv(p) for p in sorted(glob.glob(str(FLOOD / "nwis" / "dv_*.rdb")))],
                  ignore_index=True).dropna()
    q["date"] = pd.to_datetime(q.datetime)
    q = q.drop_duplicates(["site_no", "date"]).sort_values(["site_no", "date"])
    prev = q.groupby("site_no").q.shift(1)
    gap = q.groupby("site_no").date.diff().dt.days
    p95 = q.groupby("site_no").q.transform(lambda s: s.quantile(PCTL))
    q["rise"] = ((q.q >= RISE_FACTOR * prev) & (q.q >= MIN_Q) & (q.q >= p95) & (gap == 1))
    q["cty"] = q.site_no.map(cmap)
    q = q.dropna(subset=["cty"])
    q = q[season(q.date)]
    return q.groupby(["cty", "date"], as_index=False).agg(
        g_rise=("rise", "max"), n_gauges=("rise", "size")).assign(g_rise=lambda x: x.g_rise.astype(np.int8))


def fema_flood_days() -> pd.DataFrame:
    d = pd.read_parquet(FLOOD / "fema_flood_declarations.parquet")
    d = d[d.fipsCountyCode.astype(str) != "000"].copy()
    d["cty"] = d.fipsStateCode.astype(str).str.zfill(2) + d.fipsCountyCode.astype(str).str.zfill(3)
    b = pd.to_datetime(d.incidentBeginDate).dt.tz_localize(None).dt.normalize()
    e = pd.to_datetime(d.incidentEndDate).dt.tz_localize(None).dt.normalize().fillna(b)
    rows = []
    for c, s, t in zip(d.cty, b, e, strict=True):
        if pd.isna(s) or s.year > 2022 or t.year < 2000:
            continue
        days = pd.date_range(max(s, pd.Timestamp("2000-01-01")), min(t, pd.Timestamp("2022-12-31")))
        rows += [(c, x) for x in days if x.month in (5, 6, 7, 8, 9)]
    return pd.DataFrame(rows, columns=["cty", "date"]).drop_duplicates().assign(h_decl=1)
