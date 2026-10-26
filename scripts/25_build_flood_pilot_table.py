"""Build the block-group x rain-episode table for the Texas sub-county flood pilot (2010-2023).

Episode   a run of days (gaps <= 1) on which a county's 1-day maximum gridMET rainfall is >= T mm, outside
          the tropical-cyclone window (800 km, +-2 days). Window for claims: [start, end + 2] days.
Row       every block group of the county with >= 1 NFIP policy in force at the episode start.
Outcome   y = at least one non-tropical inland NFIP claim in that block group in the window.
Features  rain at the block-group cell, neighbourhood mean / max / exceedance fraction (the observed
          analogue of neighbourhood ensemble probability), antecedent rain, county rain, exposure
          (policies, flood-zone share, coverage), location.

    python scripts/25_build_flood_pilot_table.py --threshold 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402


def episodes(D1: np.ndarray, tc: np.ndarray, days: pd.DatetimeIndex, T: float):
    """(county index, start index, end index) runs of trigger days, in precipitation-day indices."""
    out = []
    trig = (D1 >= T) & ~tc
    lo = (fp.EXPO_START - days[0]).days
    for c in range(D1.shape[1]):
        idx = np.flatnonzero(trig[:, c])
        idx = idx[(idx >= lo + 15) & (idx <= len(days) - 4)]
        if not idx.size:
            continue
        cut = np.flatnonzero(np.diff(idx) > 2)
        starts = np.r_[idx[0], idx[cut + 1]]
        ends = np.r_[idx[cut], idx[-1]]
        out += [(c, s, e) for s, e in zip(starts, ends, strict=True)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=30.0)
    ap.add_argument("--out", default="flood_pilot_table.parquet")
    ap.add_argument("--tc-file", default="tc_flag_800_2.npy")
    ap.add_argument("--inputs", choices=["clean", "stored"], default="clean",
                    help="exposure and claim arrays: rebuilt from the clean tables (default; scripts/45 compares them with the stored "
                         "ones) or read from the stored expo_2010_2023.npz / claims_cum.npy the first pilot table was built from")
    ap.add_argument("--min-share", type=float, default=0.0,
                    help="keep only claims whose block-group crosswalk share is at least this")
    a = ap.parse_args()

    days, lat, lon, P = fp.load_precip()
    d = np.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    D1, cty = d["D1"], list(d["cty"])
    tc = np.load(fp.PILOT / a.tc_file)
    g = fp.bg_geometry()
    ids = g.bg10.tolist()
    iy, ix = fp.grid_index(lat, lon, g.lat.to_numpy(), g.lon.to_numpy())
    if a.inputs == "stored":
        e = np.load(fp.PILOT / "expo_2010_2023.npz")
        assert list(e["bg"]) == ids
    else:
        e = fp.exposure_matrices(ids)
    n_pol, n_sfha, n_single, cov = e["n_pol"], e["n_sfha"], e["n_single"], e["cov_sum"]
    if a.min_share > 0 or a.inputs == "clean":
        cl = fp.claims_tx()
        C = fp.claim_cumulative(cl[cl.share >= a.min_share], ids)
        print(f"claims kept with crosswalk share >= {a.min_share}: {int(C[:, -1].sum()):,}", flush=True)
    else:
        C = np.load(fp.PILOT / "claims_cum.npy", mmap_mode="r")
    masks = fp.county_masks(lat, lon)
    bg_by_cty = {c: np.flatnonzero((g.cty == c).to_numpy()) for c in cty}
    pd0 = (fp.EXPO_START - days[0]).days  # precipitation index of the first exposure day

    eps = episodes(D1, tc, days, a.threshold)
    print(f"{len(eps):,} episodes at T={a.threshold} mm", flush=True)
    rows = []
    for k, (c, s, en) in enumerate(eps):
        bgs = bg_by_cty[cty[c]]
        if not bgs.size:
            continue
        es = s - pd0  # exposure-day index of the start
        bgs = bgs[n_pol[bgs, es] >= 1]
        if not bgs.size:
            continue
        sl = np.nan_to_num(np.asarray(P[s - 2:en + 1]), nan=0.0)  # includes two days before for 3-day sums
        cs = np.cumsum(sl, axis=0)
        acc3 = cs.copy()
        acc3[3:] = cs[3:] - cs[:-3]
        F1 = sl[2:].max(axis=0)
        F3 = acc3[2:].max(axis=0)
        S = np.nan_to_num(np.asarray(P[s - 1:en + 1]), nan=0.0).sum(axis=0)
        A14 = np.nan_to_num(np.asarray(P[s - 14:s]), nan=0.0).sum(axis=0)
        nb = {"nb3_mean": uniform_filter(F1, 3), "nb7_mean": uniform_filter(F1, 7), "nb15_mean": uniform_filter(F1, 15),
              "nb7_max": maximum_filter(F1, 7), "nb15_max": maximum_filter(F1, 15)}
        for sz in (7, 15):
            for thr in (25, 50):
                nb[f"nep{sz}_{thr}"] = uniform_filter((F1 >= thr).astype(np.float32), sz)
        yy, xx = iy[bgs], ix[bgs]
        m = masks[cty[c]]
        cf1 = F1[m]
        w0, w1 = max(es - 0, 0), min(es + (en - s) + 2, C.shape[1] - 2)
        n_claims = C[bgs, w1 + 1] - C[bgs, w0]
        r = {"bg": bgs, "ep": k, "cty": cty[c], "start": days[s], "end": days[en], "dur": en - s + 1,
             "y": (n_claims > 0).astype(np.int8), "n_claims": n_claims,
             "bg_f1": F1[yy, xx], "bg_f3": F3[yy, xx], "bg_s": S[yy, xx], "bg_a14": A14[yy, xx],
             **{kk: v[yy, xx] for kk, v in nb.items()},
             "cty_f1_max": float(cf1.max()), "cty_f1_mean": float(cf1.mean()), "cty_s_mean": float(S[m].mean()),
             "cty_a14_mean": float(A14[m].mean()), "cty_nbg": int(bgs.size),
             "n_pol": n_pol[bgs, es], "sfha_share": n_sfha[bgs, es] / n_pol[bgs, es],
             "single_share": n_single[bgs, es] / n_pol[bgs, es], "mean_cov": cov[bgs, es] / n_pol[bgs, es]}
        rows.append(pd.DataFrame(r))
        if k % 5000 == 0:
            print(f"  {k:,} episodes", flush=True)
    t = pd.concat(rows, ignore_index=True)
    t = t.join(g[["lat", "lon", "area_km2"]], on="bg").assign(bg10=lambda x: g.bg10.to_numpy()[x.bg])
    t["year"], t["month"] = t.start.dt.year, t.start.dt.month
    t["f1_vs_cty"] = t.bg_f1 - t.cty_f1_mean
    t.to_parquet(fp.PILOT / a.out)
    print(f"{len(t):,} rows, {t.ep.nunique():,} episodes, positives {int(t.y.sum()):,} ({t.y.mean():.4f}); "
          f"claims inside windows {int(t.n_claims.sum()):,}")
    print("rows/positives by year:\n", t.groupby("year").agg(rows=("y", "size"), pos=("y", "sum"), eps=("ep", "nunique")).T.to_string())
    print(f"saved {fp.PILOT / a.out}")


if __name__ == "__main__":
    main()
