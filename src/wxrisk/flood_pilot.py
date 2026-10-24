"""Inputs of the Texas sub-county flood pilot: block groups, gridMET rain, tropical-cyclone days, claims.

Geography is the 2010 block group (see bg_crosswalk). Rain is gridMET daily precipitation (4 km).
Everything is keyed by calendar day (UTC date of the gridMET day, 2000-2023).
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd

from . import bg_crosswalk, wb2

PILOT = wb2.cache_path("labels_extra") / "flood" / "pilot"
TX = "48"
YEARS = range(2000, 2024)


def bg_geometry() -> pd.DataFrame:
    """2010 Texas block groups: code, county, centroid, land area (km2)."""
    import shapefile
    from shapely.geometry import shape

    z = zipfile.ZipFile(PILOT / "bg2010_tx.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                          dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    rows = []
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        c = shape(sr.shape.__geo_interface__).centroid
        rows.append((rec["STATE"] + rec["COUNTY"] + rec["TRACT"] + rec["BLKGRP"], rec["STATE"] + rec["COUNTY"],
                     c.x, c.y, float(rec["CENSUSAREA"]) * 2.58999))
    return pd.DataFrame(rows, columns=["bg10", "cty", "lon", "lat", "area_km2"])


def load_precip() -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """days, lat (descending), lon, P[day, lat, lon] in mm; cached as .npy after the first read."""
    import xarray as xr

    cache = PILOT / "pr_tx_2000_2023.npy"
    meta = PILOT / "pr_tx_meta.npz"
    if cache.exists() and meta.exists():
        m = np.load(meta, allow_pickle=True)
        return pd.DatetimeIndex(m["days"]), m["lat"], m["lon"], np.load(cache, mmap_mode="r")
    parts, days = [], []
    for y in YEARS:
        ds = xr.open_dataset(PILOT / "gridmet" / f"pr_tx_{y}.nc", engine="netcdf4")
        parts.append(ds["precipitation_amount"].values.astype(np.float32))
        days.append(pd.DatetimeIndex(ds.day.values))
        lat, lon = ds.lat.values, ds.lon.values
    P = np.concatenate(parts)
    days = days[0].append(days[1:])
    np.save(cache, P)
    np.savez(meta, days=days.values, lat=lat, lon=lon)
    return days, lat, lon, P


def grid_index(lat: np.ndarray, lon: np.ndarray, ylat: np.ndarray, xlon: np.ndarray):
    iy = np.abs(lat[None, :] - ylat[:, None]).argmin(1)
    ix = np.abs(lon[None, :] - xlon[:, None]).argmin(1)
    return iy, ix


def county_table() -> pd.DataFrame:
    from .labels_flood import county_polys

    polys = county_polys(("48",))
    return pd.DataFrame([(k, p.centroid.x, p.centroid.y) for k, p in polys.items()], columns=["cty", "lon", "lat"])


def county_masks(lat: np.ndarray, lon: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean grid mask per Texas county."""
    from shapely import contains_xy

    from .labels_flood import county_polys

    gx, gy = np.meshgrid(lon, lat)
    return {k: contains_xy(p, gx, gy) for k, p in county_polys(("48",)).items()}


def county_daily(P: np.ndarray, masks: dict[str, np.ndarray], k: int) -> pd.DataFrame:
    """Per county-day max of the k-day trailing accumulation (k=1: the day itself)."""
    out = {}
    for c, m in masks.items():
        x = np.nan_to_num(np.asarray(P[:, m]), nan=0.0)
        if k > 1:
            cs = np.cumsum(x, axis=0)
            x = cs.copy()
            x[k:] = cs[k:] - cs[:-k]
        out[c] = x.max(axis=1) if x.shape[1] else np.zeros(len(x))
    return pd.DataFrame(out)


def read_hurdat(path) -> pd.DataFrame:
    rows = []
    for ln in open(path):
        f = [x.strip() for x in ln.split(",")]
        if len(f) > 6 and f[0][:1].isdigit():
            lat = float(f[4][:-1]) * (1 if f[4].endswith("N") else -1)
            lon = float(f[5][:-1]) * (1 if f[5].endswith("E") else -1)
            rows.append((pd.Timestamp(f[0]), lat, lon, f[3]))
    return pd.DataFrame(rows, columns=["date", "lat", "lon", "status"])


def tc_flags(days: pd.DatetimeIndex, ctab: pd.DataFrame, radius_km: float, window: int) -> np.ndarray:
    """bool[day, county]: a tropical-cyclone track point lay within radius_km of the county centroid
    on a day within `window` days of this one (Atlantic and eastern Pacific best tracks)."""
    tr = pd.concat([read_hurdat(PILOT / "hurdat2.txt"), read_hurdat(PILOT / "hurdat2_nepac.txt")], ignore_index=True)
    tr = tr[(tr.date >= days[0] - pd.Timedelta(days=window)) & (tr.date <= days[-1] + pd.Timedelta(days=window))]
    la, lo = np.radians(ctab.lat.values), np.radians(ctab.lon.values)
    flags = np.zeros((len(days), len(ctab)), bool)
    d0 = days[0]
    for t in tr.itertuples():
        a, b = np.radians(t.lat), np.radians(t.lon)
        d = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la - a) / 2) ** 2 + np.cos(a) * np.cos(la) * np.sin((lo - b) / 2) ** 2))
        near = np.flatnonzero(d <= radius_km)
        if near.size:
            i = (t.date.normalize() - d0).days
            flags[max(i - window, 0):min(i + window + 1, len(days))][:, near] = True
    return flags


def claims_tx() -> pd.DataFrame:
    """Texas non-tropical-named inland claims (cause 2/3/4) on the 2010 block-group geography."""
    s = pd.read_parquet(wb2.cache_path("nfip_inland_nontropical.parquet"))
    s = s[s.state == "TX"].copy()
    cw = bg_crosswalk.build().set_index("code")
    s["bg10"] = s.bg.map(cw.bg10)
    s["share"] = s.bg.map(cw.share)
    s["cty"] = s.bg10.str[:5]
    return s[s.bg10.notna()]


EXPO_START = pd.Timestamp("2010-01-01")  # the policy bulk file begins with 2009 effective dates
EXPO_END = pd.Timestamp("2023-12-31")


def expo_days() -> pd.DatetimeIndex:
    return pd.date_range(EXPO_START, EXPO_END)


def exposure_matrices(bg_ids: list[str]) -> dict[str, np.ndarray]:
    """In-force policy statistics per block group and day, float32[bg, day] for 2010-2023.

    n_pol policies in force, n_sfha in a special flood hazard area (zone A*/V*), n_single single-family
    occupancy, cov_sum building coverage ($ millions). Policy terms are 365-366 days, so in force on
    day d means effective in (d-365, d].
    """
    pol = pd.read_parquet(PILOT / "policies_tx.parquet")
    cw = bg_crosswalk.build().set_index("code")
    code = pol.censusGeoid.astype("string").str.zfill(12)
    bg = code.map(cw.bg10)
    index = {b: i for i, b in enumerate(bg_ids)}
    bi = bg.map(index)
    keep = bi.notna().to_numpy()
    pol, bi = pol[keep], bi[keep].astype(np.int64).to_numpy()
    d0 = EXPO_START
    n = len(expo_days())
    e = (pd.to_datetime(pol.policyEffectiveDate) - d0).dt.days.to_numpy()
    end = e + 366
    cancel = pd.to_datetime(pol.cancellationDateOfFloodPolicy)
    cend = (cancel - d0).dt.days.to_numpy(dtype="float64")
    end = np.where(np.isnan(cend), end, np.minimum(end, cend)).astype(np.int64)
    s0, s1 = np.clip(e, 0, n), np.clip(end, 0, n)
    ok = s1 > s0
    cnt = pol.policyCount.to_numpy().astype(np.float64)
    zone = pol.ratedFloodZone.astype(str).str[:1].isin(["A", "V"]).to_numpy()
    single = pol.occupancyType.isin([1, 11]).to_numpy()  # 11 = single-family under Risk Rating 2.0 (2021+)
    cov = pol.totalBuildingInsuranceCoverage.to_numpy().astype(np.float64) / 1e6 * cnt
    out = {}
    nb = len(bg_ids)
    for name, w in (("n_pol", cnt), ("n_sfha", cnt * zone), ("n_single", cnt * single), ("cov_sum", cov)):
        up = np.bincount(bi[ok] * (n + 1) + s0[ok], weights=w[ok], minlength=nb * (n + 1))
        dn = np.bincount(bi[ok] * (n + 1) + s1[ok], weights=w[ok], minlength=nb * (n + 1))
        m = np.cumsum((up - dn).reshape(nb, n + 1), axis=1, dtype=np.float64)[:, :n]
        out[name] = m.astype(np.float32)
    return out


def claim_cumulative(claims: pd.DataFrame, bg_ids: list[str]) -> np.ndarray:
    """int32[bg, day+1] cumulative claim counts, so claims in [a, b] = C[:, b+1] - C[:, a]."""
    index = {b: i for i, b in enumerate(bg_ids)}
    n = len(expo_days())
    c = claims[(claims.date >= EXPO_START) & (claims.date <= EXPO_END)]
    bi = c.bg10.map(index).to_numpy()
    di = (c.date - EXPO_START).dt.days.to_numpy()
    m = np.zeros((len(bg_ids), n + 1), np.int32)
    np.add.at(m, (bi, di + 1), 1)
    return np.cumsum(m, axis=1, dtype=np.int32)
