"""E0 of docs/CNN_HYBRID_DESIGN.md: day tensors, block-group indices and the rows of the all-days unit, with registration checks.

For every forecast day S (00Z run of S-1, leads 36-84 h; the 3,639 days 2010-01-01 .. 2019-12-27 that script 77 uses) on the GEFS grid (53 x 61 cells, 0.25 deg):
  channels (float16, `gefs_day_tensors.npy`, [day, 18, 53, 61]):
    0 fc_mean_peak     ensemble mean of the per-member peak over S and S+1          (forecast)
    1 fc_mean_sum      ensemble mean of the 2-day sum                               (forecast)
    2-6 member peaks   per-member peaks sorted from high to low                     (forecast)
    7 log1p policies per cell, 8 SFHA share per cell, 9 log1p claims 2000-2009 (hist00) per cell        (exposure, issue time)
    10 obs_peak, 11 obs_sum   gridMET 4 km averaged to the GEFS cells, peak / sum over S and S+1        (OBSERVED: E1 and reference only)
    12-17 ERA5 tcwv, q850, u850, v850, u500, v500, daily means averaged over S and S+1                   (valid time, not a forecast: E3 ceiling only)
  `obs4k_day_maps.npy` float16 [day, 2, 260, 320]: gridMET 4 km peak and sum over S and S+1 (E1 at full resolution)
  rows (CSR by day, `rows_*.npy`): the block groups with >= 1 policy on S outside tropical-cyclone county-days (as script 77), tabular features
    [log policies, SFHA share, single-family share, mean coverage, block groups in the county, hist00, prior3, month, policies], claim count and paid amount in [S, S+3]
  `static.npz`: grid, masks (Texas cells, ERA5 coverage, gridMET coverage), block-group cell indices (0.25 deg and 4 km), county index, channel names
Checks (printed, written to the audit file): coordinate registration of ERA5 on the GEFS grid, direct-sample comparison of the ERA5 and rain channels, rows and
labels against the script 77 test rows, channel statistics, and a map figure (`e0_maps.png`).
Inputs : gefs/apcp_*.npy, gridMET, cds_gulf/*.nc, clean exposure and claims, ranking_test_rows.parquet (script 77)
Output : $WXRISK_DATA/cnn/*, $WXRISK_DATA/audit/e0_day_tensors.txt
Run    : python scripts/80_build_day_tensors.py [--check-only]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import zstio  # noqa: E402
from wxrisk import wb2  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e75, e48 = _load("e75", "75_task_metrics.py"), _load("e48", "48_flood_forecast_pilot.py")
OUT_DIR = wb2.cache_path("cnn")
CDS = wb2.cache_path("cds_gulf")
CHANNELS = ["fc_mean_peak", "fc_mean_sum", "fc_member1", "fc_member2", "fc_member3", "fc_member4", "fc_member5", "log_policies", "sfha_share", "log_hist00",
            "obs_peak", "obs_sum", "era5_tcwv", "era5_q850", "era5_u850", "era5_v850", "era5_u500", "era5_v500"]
TAB = ["log_pol", "sfha_share", "single_share", "mean_cov", "cty_nbg", "hist00", "prior3", "month", "n_pol"]
WINDOW = 4
LOG: list[str] = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def era5_daily() -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Daily means of tcwv, q850, u850, v850, u500, v500 on the ERA5 grid: [day, 6, 53, 113], days from 2009-01-01."""
    days = pd.date_range("2009-01-01", "2023-12-31")
    E = np.full((len(days), 6, 53, 113), np.nan, np.float32)
    lat = lon = None
    for y in range(2009, 2024):
        with xr.open_dataset(CDS / f"single_{y}.nc") as ds:
            d = ds.tcwv.resample(valid_time="1D").mean().load()
            lat, lon = ds.latitude.to_numpy(), ds.longitude.to_numpy()
            idx = (pd.DatetimeIndex(d.valid_time.to_numpy()) - days[0]).days
            E[idx, 0] = d.to_numpy()
        for q in range(1, 5):
            with xr.open_dataset(CDS / f"plev_{y}_q{q}.nc") as ds:
                d = ds.resample(valid_time="1D").mean().load()
                idx = (pd.DatetimeIndex(d.valid_time.to_numpy()) - days[0]).days
                E[idx, 1] = d.q.sel(pressure_level=850).to_numpy()
                E[idx, 2] = d.u.sel(pressure_level=850).to_numpy()
                E[idx, 3] = d.v.sel(pressure_level=850).to_numpy()
                E[idx, 4] = d.u.sel(pressure_level=500).to_numpy()
                E[idx, 5] = d.v.sel(pressure_level=500).to_numpy()
    return E, days, lat, lon


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = fp.bg_geometry()
    ids = g.bg10.tolist()
    nbg = len(ids)
    cty_of = g.cty.to_numpy()
    ctys = sorted(g.cty.unique())
    cty_idx = np.array([ctys.index(c) for c in cty_of], np.int16)
    say("loading exposure and claims ...")
    ex = fp.exposure_matrices(ids)
    cl = e75.claims("S0")
    C = fp.claim_cumulative(cl, ids)
    Cp = np.zeros((nbg, C.shape[1]), np.float64)
    index = {b: i for i, b in enumerate(ids)}
    c = cl[(cl.date >= fp.EXPO_START) & (cl.date <= fp.EXPO_END)]
    np.add.at(Cp, (c.bg10.map(index).to_numpy(), (c.date - fp.EXPO_START).dt.days.to_numpy() + 1), c.paid.to_numpy())
    Cp = np.cumsum(Cp, axis=1)
    # claim history features as in script 49 (claims from 2000, clean table)
    ch = fp.claims_tx()
    ch = ch[ch.date >= "2000-01-01"]
    bi = ch.bg10.map(index).to_numpy()
    dd = (ch.date - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    keys = np.sort(bi.astype(np.int64) * 20000 + dd)
    hist00 = np.log1p(np.bincount(bi[dd < 3653], minlength=nbg)).astype(np.float32)

    meta = zstio.load(e48.GEFS / "gefs_meta.npz", allow_pickle=True)
    glat, glon = meta["lat"], meta["lon"]
    ny, nx = len(glat), len(glon)
    days, lat, lon, P = fp.load_precip()
    py = np.abs(glat[None, :] - lat[:, None]).argmin(1)
    px = np.abs(glon[None, :] - lon[:, None]).argmin(1)
    cell_of_pixel = (py[:, None] * nx + px[None, :]).ravel()
    ncell = ny * nx
    cnt = np.bincount(cell_of_pixel, minlength=ncell).astype(np.float32)
    cy = np.abs(glat[None, :] - g.lat.to_numpy()[:, None]).argmin(1)
    cx = np.abs(glon[None, :] - g.lon.to_numpy()[:, None]).argmin(1)
    bcell = cy * nx + cx
    by, bx = fp.grid_index(lat, lon, g.lat.to_numpy(), g.lon.to_numpy())
    cty_cells = {cc: np.unique(bcell[i]) for cc, i in g.groupby("cty").indices.items()}
    valid = np.zeros(ncell, bool)
    for v in cty_cells.values():
        valid[v] = True
    d_ = zstio.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    tc = zstio.load(fp.PILOT / "tc_flag_800_2.npy")
    tc_ci = {cc: i for i, cc in enumerate(d_["cty"])}
    csum = np.concatenate([np.zeros((1, tc.shape[1]), np.int32), np.cumsum(tc, axis=0, dtype=np.int32)])
    tc_cty = np.array([tc_ci.get(cc, -1) for cc in cty_of])
    say("reading ERA5 daily means ...")
    E, edays, elat, elon = era5_daily()
    # registration: GEFS cell (i, k) <-> ERA5 index (i - 2, k - 2); the two outer rows (north) and columns (west) of the GEFS grid are outside the ERA5 box
    assert np.allclose(elat[: ny - 2], glat[2:]) and np.allclose(elon[: nx - 2], glon[2:]), "ERA5 / GEFS coordinates do not line up"
    era_valid = np.zeros((ny, nx), bool)
    era_valid[2:, 2:] = True

    files = sorted(e48.GEFS.glob("apcp_*.npy"))
    sdays = [pd.Timestamp(f.stem.split("_")[1]) + pd.Timedelta(1, "D") for f in files]
    keep = [(f, s) for f, s in zip(files, sdays, strict=True) if pd.Timestamp("2010-01-01") <= s <= pd.Timestamp("2019-12-27") and s in days]
    D = len(keep)
    say(f"{D} forecast days")
    G = np.lib.format.open_memmap(OUT_DIR / "gefs_day_tensors.npy", mode="w+", dtype=np.float16, shape=(D, len(CHANNELS), ny, nx))
    O = np.lib.format.open_memmap(OUT_DIR / "obs4k_day_maps.npy", mode="w+", dtype=np.float16, shape=(D, 2, *P.shape[1:]))
    ptr, rbg, rtab, rn, rpaid = [0], [], [], [], []
    cache: dict = {}

    def obs_cell(k):
        if k not in cache:
            f = np.nan_to_num(np.asarray(P[k]), nan=0.0).ravel()
            cache[k] = np.bincount(cell_of_pixel, weights=f, minlength=ncell) / np.maximum(cnt, 1)
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[k]

    obs4_valid = np.isfinite(np.asarray(P[0]))  # gridMET covers only part of the GEFS box: zero-filled elsewhere, so the mask is saved
    assert (np.isfinite(np.asarray(P[4000])) == obs4_valid).all(), "gridMET coverage changes between days"
    obs_cover = (np.bincount(cell_of_pixel, weights=obs4_valid.ravel().astype(np.float32), minlength=ncell) / np.maximum(cnt, 1)).reshape(ny, nx)
    hist_cell = np.log1p(np.bincount(bcell, weights=np.expm1(hist00), minlength=ncell)).reshape(ny, nx)
    base = pd.Timestamp("2000-01-01")
    for n, (f, s0) in enumerate(keep):
        es = (s0 - fp.EXPO_START).days
        di = days.get_loc(s0)
        a = zstio.load(f).astype(np.float32)
        dayf = np.stack([a[:, :4].sum(1), a[:, 4:].sum(1)], 1)
        F1m = dayf.max(1)
        G[n, 0] = F1m.mean(0)
        G[n, 1] = dayf.sum(1).mean(0)
        G[n, 2:7] = np.sort(F1m, axis=0)[::-1]
        npol_all = ex["n_pol"][:, es]
        pol_cell = np.bincount(bcell, weights=npol_all, minlength=ncell)
        sfha_cell = np.bincount(bcell, weights=ex["n_sfha"][:, es], minlength=ncell)
        G[n, 7] = np.log1p(pol_cell).reshape(ny, nx)
        G[n, 8] = np.where(pol_cell > 0, sfha_cell / np.maximum(pol_cell, 1), 0).reshape(ny, nx)
        G[n, 9] = hist_cell
        oc = np.stack([obs_cell(di), obs_cell(di + 1)])
        G[n, 10] = oc.max(0).reshape(ny, nx)
        G[n, 11] = oc.sum(0).reshape(ny, nx)
        ei = (s0 - edays[0]).days
        em = np.nanmean(E[ei:ei + 2], axis=0)  # [6, 53, 113]
        era = np.full((6, ny, nx), np.nan, np.float32)
        era[:, 2:, 2:] = em[:, : ny - 2, : nx - 2]
        G[n, 12:18] = era
        pix = np.nan_to_num(np.asarray(P[di:di + 2]), nan=0.0)
        O[n, 0], O[n, 1] = pix.max(0), pix.sum(0)
        # rows of the day (script 77 rules)
        bgs = np.flatnonzero(npol_all >= 1)
        ti = (s0 - base).days
        flag = np.zeros(len(bgs), bool)
        okc = tc_cty[bgs] >= 0
        flag[okc] = (csum[ti + WINDOW, tc_cty[bgs][okc]] - csum[ti, tc_cty[bgs][okc]]) > 0
        bgs = bgs[~flag]
        npol = npol_all[bgs]
        cs = pd.Series(cty_of[bgs]).map(pd.Series(cty_of[bgs]).value_counts()).to_numpy().astype(np.float32)
        b = bgs.astype(np.int64) * 20000
        prior3 = np.log1p((np.searchsorted(keys, b + ti) - np.searchsorted(keys, b + ti - 1095)).astype(np.float32))
        rtab.append(np.column_stack([np.log1p(npol), ex["n_sfha"][bgs, es] / npol, ex["n_single"][bgs, es] / npol, ex["cov_sum"][bgs, es] / npol, cs,
                                     hist00[bgs], prior3, np.full(len(bgs), s0.month), npol]).astype(np.float32))
        rbg.append(bgs.astype(np.int32))
        rn.append((C[bgs, es + WINDOW] - C[bgs, es]).astype(np.float32))
        rpaid.append((Cp[bgs, es + WINDOW] - Cp[bgs, es]).astype(np.float32))
        ptr.append(ptr[-1] + len(bgs))
        if n % 500 == 0:
            say(f"  {n} / {D}")
    G.flush()
    O.flush()
    np.save(OUT_DIR / "rows_ptr.npy", np.array(ptr, np.int64))
    np.save(OUT_DIR / "rows_bg.npy", np.concatenate(rbg))
    np.save(OUT_DIR / "rows_tab.npy", np.concatenate(rtab))
    np.save(OUT_DIR / "rows_n.npy", np.concatenate(rn))
    np.save(OUT_DIR / "rows_paid.npy", np.concatenate(rpaid))
    np.savez(OUT_DIR / "static.npz", days=np.array([s for _, s in keep], dtype="datetime64[D]"), glat=glat, glon=glon, valid_tx=valid.reshape(ny, nx),
             era_valid=era_valid, obs4_valid=obs4_valid, obs_cover=obs_cover.astype(np.float32), bcell_y=cy.astype(np.int16), bcell_x=cx.astype(np.int16), b4_y=by.astype(np.int16), b4_x=bx.astype(np.int16),
             cty_idx=cty_idx, ctys=np.array(ctys), bg10=np.array(ids), hist00=hist00, channels=np.array(CHANNELS), tab=np.array(TAB),
             lat4=lat, lon4=lon)
    say(f"saved {D} days, {ptr[-1]:,} rows to {OUT_DIR}")


def check() -> None:
    say("\n== E0 checks")
    G = zstio.load(OUT_DIR / "gefs_day_tensors.npy", mmap_mode="r")
    O = zstio.load(OUT_DIR / "obs4k_day_maps.npy", mmap_mode="r")
    st = zstio.load(OUT_DIR / "static.npz", allow_pickle=True)
    ptr, rbg, rtab = zstio.load(OUT_DIR / "rows_ptr.npy"), zstio.load(OUT_DIR / "rows_bg.npy"), zstio.load(OUT_DIR / "rows_tab.npy", mmap_mode="r")
    rn = zstio.load(OUT_DIR / "rows_n.npy")
    days = pd.DatetimeIndex(st["days"])
    D = len(days)
    say(f"  tensors {G.shape} float16 ({G.nbytes / 1e6:.0f} MB), 4 km maps {O.shape} ({O.nbytes / 1e6:.0f} MB), rows {len(rbg):,} ({ptr[-1]:,}), days {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}")
    sample = np.sort(np.random.default_rng(0).choice(D, 200, replace=False))
    stats = []
    for c, name in enumerate(st["channels"]):
        x = np.asarray(G[sample, c], np.float32)
        valid = st["valid_tx"][None] if c < 12 else (st["valid_tx"] & st["era_valid"])[None]
        v = x[np.broadcast_to(valid, x.shape)]
        stats.append((name, np.isnan(v).mean(), np.nanmean(v), np.nanstd(v), np.nanmin(v), np.nanmax(v)))
    say("  channel statistics over 200 random days, valid Texas cells (share NaN, mean, sd, min, max):")
    for s in stats:
        say(f"    {s[0]:14s} {s[1]:.4f} {s[2]:9.3f} {s[3]:9.3f} {s[4]:9.3f} {s[5]:9.3f}")
    cov = st["obs_cover"]
    say(f"  observed-rain coverage (gridMET): cells with full coverage {int((cov >= 0.99).sum())} of {cov.size}; Texas cells with coverage < 0.99: {int(((cov < 0.99) & st['valid_tx']).sum())}; "
        f"4 km valid pixels {st['obs4_valid'].mean():.1%} (the observed channels are zero outside this mask)")
    say(f"  GEFS grid cells in Texas {int(st['valid_tx'].sum())} of {st['valid_tx'].size}; ERA5-covered cells {int(st['era_valid'].sum())}; block groups with a cell index inside the grid: "
        f"{np.mean((st['bcell_y'] >= 0) & (st['bcell_y'] < 53) & (st['bcell_x'] >= 0) & (st['bcell_x'] < 61)):.0%}")
    # ERA5 registration: tensor value at a block group's cell against the daily mean sampled directly from the ERA5 file at the nearest ERA5 point
    rng = np.random.default_rng(1)
    g = fp.bg_geometry()
    diffs, n_ok = [], 0
    for n in rng.choice(D, 40, replace=False):
        s0 = days[n]
        with xr.open_dataset(CDS / f"single_{s0.year}.nc") as ds:
            day = ds.tcwv.sel(valid_time=slice(s0, s0 + pd.Timedelta(hours=23))).mean("valid_time")
            day2 = ds.tcwv.sel(valid_time=slice(s0 + pd.Timedelta(days=1), s0 + pd.Timedelta(days=1, hours=23))).mean("valid_time") if s0.month < 12 or s0.day < 31 else day
            for b in rng.choice(len(g), 8, replace=False):
                lat_b, lon_b = g.lat.iloc[b], g.lon.iloc[b]
                direct = float(((day + day2) / 2).sel(latitude=lat_b, longitude=lon_b, method="nearest"))
                cell = float(G[n, 12, st["bcell_y"][b], st["bcell_x"][b]])
                if np.isfinite(cell) and np.isfinite(direct):
                    diffs.append(abs(cell - direct))
                    n_ok += 1
    say(f"  ERA5 tcwv, tensor cell value against a direct nearest-point read of the ERA5 file at the block group (320 comparisons, {n_ok} finite): "
        f"max |difference| {np.max(diffs):.3f} kg/m2, median {np.median(diffs):.4f}, share of exact matches within 0.1: {np.mean(np.array(diffs) < 0.1):.1%}")
    # consistency with script 77: rows, labels and the forecast peak of the test days
    t = pd.read_parquet(fp.PILOT / "ranking_test_rows.parquet", columns=["sysid", "bg", "n", "n_pol", "fc_f1"])
    tdays = [d for d in days if d.year >= 2017]
    ok_rows = ok_lab = ok_fc = 0
    maxd = 0.0
    pick = rng.choice(len(tdays), 12, replace=False)
    for i in pick:
        d = tdays[i]
        n = days.get_loc(d)
        tt = t[t.sysid == d.strftime("%Y%m%d")].sort_values("bg")
        rows = slice(ptr[n], ptr[n + 1])
        bg = rbg[rows]
        o = np.argsort(bg)
        same = len(tt) == len(bg) and (tt.bg.to_numpy() == bg[o]).all()
        ok_rows += int(same)
        if same:
            ok_lab += int(np.allclose(tt.n.to_numpy(), rn[rows][o]))
            fc = np.asarray(G[n, 0], np.float32)[st["bcell_y"][bg[o]], st["bcell_x"][bg[o]]]
            dfc = np.abs(fc - tt.fc_f1.to_numpy())
            maxd = max(maxd, float(dfc.max()))
            ok_fc += int(dfc.max() < 0.5)
    say(f"  against the script 77 test rows on 12 random test days: same block groups {ok_rows} / 12, same claim counts {ok_lab} / 12, forecast peak at the cell equal within 0.5 mm "
        f"{ok_fc} / 12 (largest difference {maxd:.3f} mm; float16 storage)")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        picks = {"Memorial Day 2015-05-24": "2015-05-24", "Tax Day 2016-04-18": "2016-04-18", "Harvey 2017-08-26": "2017-08-26", "Imelda 2019-09-19": "2019-09-19"}
        panels = [(0, "forecast mean peak (mm)", "Blues"), (10, "observed peak, 0.25 deg (mm)", "Blues"), (12, "ERA5 tcwv (kg/m2)", "viridis"), (7, "log1p policies per cell", "magma")]
        fig, axs = plt.subplots(len(picks), len(panels), figsize=(14, 3.2 * len(picks)))
        for r, (lab, dstr) in enumerate(picks.items()):
            n = days.get_loc(pd.Timestamp(dstr))
            for c, (ch, title, cmap) in enumerate(panels):
                a = np.asarray(G[n, ch], np.float32)
                im = axs[r, c].imshow(a, cmap=cmap, extent=[st["glon"][0], st["glon"][-1], st["glat"][-1], st["glat"][0]], aspect="auto")
                axs[r, c].set_title(f"{lab}: {title}" if c == 0 else title, fontsize=8)
                fig.colorbar(im, ax=axs[r, c], fraction=0.046)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "e0_maps.png", dpi=90)
        say(f"  figure: {OUT_DIR / 'e0_maps.png'}")
    except Exception as e:  # noqa: BLE001
        say(f"  figure skipped: {e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if not a.check_only:
        build()
    check()
    (wb2.cache_path("audit") / "e0_day_tensors.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
