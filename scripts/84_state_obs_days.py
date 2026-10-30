"""Observed-rain day products of one state for the scale-up (RESULTS stage 45): the E0 rows of script 80 without the forecast side.

For every day S (2010-01-01 .. 2023-12-27, so the gridMET years after the GEFS reforecast count) of the state:
  `obs4k_day_maps.npy`  float16 [day, 2, ny, nx]  gridMET 4 km peak and sum over S and S+1 (zero outside the land mask; mask in static.npz)
  rows (CSR by day)     the block groups with >= 1 policy on S; unlike script 80 the tropical-cyclone days are KEPT and flagged, so both
                        label scopes come from one table:
      `rows_bg.npy` block-group index, `rows_tab.npy` [log policies, SFHA share, single-family share, mean coverage, block groups in the county
      that day, hist00, prior3, month, policies] (script 80 definitions), `rows_tc.npy` bool (county within 800 km of a best-track point,
      +-2 days around [S, S+3]), `rows_n0.npy` / `rows_paid0.npy` claim count / paid building amount in [S, S+3] WITHOUT tropical-named
      claims (scope S0), `rows_nall.npy` claim count with them (scope S2)
  `static.npz`          grid, block-group pixel indices, county index, block-group ids, day list
Texas (`TX`) goes through the same code, so the pooled training data has one format.
Inputs : gridMET pr_<tag>_<year>.nc (scripts/22), clean policies and claims (scripts/40), TIGER block groups and crosswalk (wxrisk.states)
Output : $WXRISK_DATA/cnn_<st>/*
Run    : python scripts/84_state_obs_days.py LA
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import states, wb2  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402

WINDOW = 4  # claims in [S, S+3] (script 80)
TAB = ["log_pol", "sfha_share", "single_share", "mean_cov", "cty_nbg", "hist00", "prior3", "month", "n_pol"]
TAG = {"TX": "tx", "LA": "lafl", "FL": "lafl"}  # LA and FL share one gridMET box


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    a = ap.parse_args()
    st = states.get(a.state)
    tx = st.abbr == "TX"
    out_dir = wb2.cache_path(f"cnn_{st.low}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cwp = None if tx else st.relationship
    g = fp.bg_geometry(None if tx else st.bg2010)
    ids = g.bg10.tolist()
    nbg = len(ids)
    index = {b: i for i, b in enumerate(ids)}
    cty_of = g.cty.to_numpy()
    ctys = sorted(g.cty.unique())
    cty_idx = np.array([ctys.index(c) for c in cty_of], np.int16)
    print(f"{st.name}: {nbg} block groups, {len(ctys)} counties", flush=True)

    ex = fp.exposure_matrices(ids, table=f"policies_{st.low}", crosswalk=cwp)
    cl = fp.claims_state(st.abbr, cwp)
    cl = cl[(cl.date >= "2000-01-01") & (cl.date <= fp.EXPO_END)]
    n_cl = len(cl)
    cl = cl[cl.bg10.isin(index)]  # the cartographic file lacks a few block groups that the Census relationship file has
    print(f"claims {n_cl:,}, on a block group of the geometry {len(cl):,}", flush=True)
    s0 = cl[~cl.tc_named]
    C0 = fp.claim_cumulative(s0, ids)
    Call = fp.claim_cumulative(cl, ids)
    Cp = np.zeros((nbg, C0.shape[1]), np.float64)
    c = s0[s0.date >= fp.EXPO_START]
    np.add.at(Cp, (c.bg10.map(index).to_numpy(), (c.date - fp.EXPO_START).dt.days.to_numpy() + 1), c.paid.to_numpy())
    Cp = np.cumsum(Cp, axis=1)
    base = pd.Timestamp("2000-01-01")
    bi = s0.bg10.map(index).to_numpy()
    dd = (s0.date - base).dt.days.to_numpy()
    keys = np.sort(bi.astype(np.int64) * 20000 + dd)
    hist00 = np.log1p(np.bincount(bi[dd < 3653], minlength=nbg)).astype(np.float32)

    days, lat, lon, P = fp.load_precip(tag=TAG[st.abbr])
    by, bx = fp.grid_index(lat, lon, g.lat.to_numpy(), g.lon.to_numpy())
    # tropical-cyclone flag per county and day, as for the Texas pilot (800 km, +-2 days)
    ctab = g.groupby("cty")[["lon", "lat"]].mean().reindex(ctys).reset_index()
    tc = fp.tc_flags(days, ctab, 800, 2)
    csum = np.concatenate([np.zeros((1, tc.shape[1]), np.int32), np.cumsum(tc, axis=0, dtype=np.int32)])

    keep = [s for s in pd.date_range(fp.EXPO_START, "2023-12-27") if s in days]
    D = len(keep)
    O = np.lib.format.open_memmap(out_dir / "obs4k_day_maps.npy", mode="w+", dtype=np.float16, shape=(D, 2, *P.shape[1:]))
    ptr, rbg, rtab, rtc, rn0, rpaid, rnall = [0], [], [], [], [], [], []
    for n, s in enumerate(keep):
        es = (s - fp.EXPO_START).days
        di = days.get_loc(s)
        pix = np.nan_to_num(np.asarray(P[di:di + 2]), nan=0.0)
        O[n, 0], O[n, 1] = pix.max(0), pix.sum(0)
        npol_all = ex["n_pol"][:, es]
        bgs = np.flatnonzero(npol_all >= 1)
        ti = (s - base).days
        npol = npol_all[bgs]
        ci = cty_idx[bgs].astype(np.int64)
        flag = (csum[ti + WINDOW, ci] - csum[ti, ci]) > 0
        cs = np.bincount(ci, minlength=len(ctys))[ci].astype(np.float32)
        b = bgs.astype(np.int64) * 20000
        prior3 = np.log1p((np.searchsorted(keys, b + ti) - np.searchsorted(keys, b + ti - 1095)).astype(np.float32))
        rtab.append(np.column_stack([np.log1p(npol), ex["n_sfha"][bgs, es] / npol, ex["n_single"][bgs, es] / npol, ex["cov_sum"][bgs, es] / npol, cs,
                                     hist00[bgs], prior3, np.full(len(bgs), s.month), npol]).astype(np.float32))
        rbg.append(bgs.astype(np.int32))
        rtc.append(flag)
        rn0.append((C0[bgs, es + WINDOW] - C0[bgs, es]).astype(np.float32))
        rnall.append((Call[bgs, es + WINDOW] - Call[bgs, es]).astype(np.float32))
        rpaid.append((Cp[bgs, es + WINDOW] - Cp[bgs, es]).astype(np.float32))
        ptr.append(ptr[-1] + len(bgs))
        if n % 1000 == 0:
            print(f"  {n} / {D}", flush=True)
    O.flush()
    for name, arr in (("ptr", np.array(ptr, np.int64)), ("bg", np.concatenate(rbg)), ("tab", np.concatenate(rtab)), ("tc", np.concatenate(rtc)),
                      ("n0", np.concatenate(rn0)), ("nall", np.concatenate(rnall)), ("paid0", np.concatenate(rpaid))):
        np.save(out_dir / f"rows_{name}.npy", arr)
    np.savez(out_dir / "static.npz", days=np.array(keep, dtype="datetime64[D]"), lat4=lat, lon4=lon, b4_y=by.astype(np.int16), b4_x=bx.astype(np.int16),
             cty_idx=cty_idx, ctys=np.array(ctys), bg10=np.array(ids), hist00=hist00, tab=np.array(TAB), obs4_valid=np.isfinite(np.asarray(P[0])))
    n0, na = np.concatenate(rn0), np.concatenate(rnall)
    print(f"saved {D} days, {ptr[-1]:,} rows to {out_dir}; claims in rows: S0 {n0.sum():,.0f}, with tropical-named {na.sum():,.0f}; "
          f"rows with TC flag {np.concatenate(rtc).mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
