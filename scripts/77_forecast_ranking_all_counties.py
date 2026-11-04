"""Forecast-mode ranking on an ex-ante set of counties (stage 38).

Stage 37: the pilot's county set (counties with an observed episode) removes the forecast's between-county skill. Here every unit of a forecast day is
a row, whatever the rain turned out to be:
  rows   every block group with >= 1 policy on day S, for each stored GEFS initialisation day (00Z of S-1; leads 36-84 h), 2010-2019, except county-days
         in a tropical-cyclone window (800 km, +-2 d of S..S+3: scope S0 as before); features known at issue time only (no `dur`, no observed rain):
         forecast rain at the cell (peak and sum over S, S+1, ensemble mean), forecast county peak and mean, exposure, county size, month
  label  claim count and paid building amount in [S, S+3] (the window of a two-day episode), clean claims, scope S0
  event  one forecast day: all counties of Texas ranked against each other; a second view restricts to the counties with a forecast county peak >= 10 mm
         (an ex-ante footprint)
  models forecast model; observed-rain model on the same rows (reference; observed rain on the GEFS cells of S and S+1); trained 2010-2016 (negatives
         subsampled 1 in 10 with weight 10), tested 2017-2019 on every row
Metrics: those of scripts/75 (cross-county AUC, top-k capture of claim count and paid amount, NDCG, county AUC, top-k counties, probability companion).
Inputs : gefs/apcp_*.npy, gridMET, exposure arrays and claims from the clean tables, tc_flag_800_2.npy
Output : $WXRISK_DATA/audit/forecast_ranking_all_counties.txt, pilot/ranking_test_rows.parquet (test rows with predictions)
Run    : python scripts/77_forecast_ranking_all_counties.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier

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
OUT: list[str] = []
e75.OUT = OUT
WINDOW = 4  # days S .. S+3
CACHE = fp.PILOT / "ranking_test_rows.parquet"


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def paid_cumulative(cl: pd.DataFrame, ids: list[str]) -> np.ndarray:
    index = {b: i for i, b in enumerate(ids)}
    n = len(fp.expo_days())
    c = cl[(cl.date >= fp.EXPO_START) & (cl.date <= fp.EXPO_END)]
    m = np.zeros((len(ids), n + 1), np.float64)
    np.add.at(m, (c.bg10.map(index).to_numpy(), (c.date - fp.EXPO_START).dt.days.to_numpy() + 1), c.paid.to_numpy())
    return np.cumsum(m, axis=1)


def build() -> pd.DataFrame:
    g = fp.bg_geometry()
    ids = g.bg10.tolist()
    cty_of = g.cty.to_numpy()
    say("loading exposure and claims ...")
    ex = fp.exposure_matrices(ids)
    cl = e75.claims("S0")
    C = fp.claim_cumulative(cl, ids)
    Cp = paid_cumulative(cl, ids)
    meta = zstio.load(e48.GEFS / "gefs_meta.npz", allow_pickle=True)
    glat, glon = meta["lat"], meta["lon"]
    days, lat, lon, P = fp.load_precip()
    py = np.abs(glat[None, :] - lat[:, None]).argmin(1)
    px = np.abs(glon[None, :] - lon[:, None]).argmin(1)
    cell_of_pixel = (py[:, None] * len(glon) + px[None, :]).ravel()
    ncell = len(glat) * len(glon)
    cnt = np.bincount(cell_of_pixel, minlength=ncell).astype(np.float32)
    cy = np.abs(glat[None, :] - g.lat.to_numpy()[:, None]).argmin(1)
    cx = np.abs(glon[None, :] - g.lon.to_numpy()[:, None]).argmin(1)
    bcell = cy * len(glon) + cx
    cty_cells = {c: np.unique(bcell[i]) for c, i in g.groupby("cty").indices.items()}
    d = zstio.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    tc = zstio.load(fp.PILOT / "tc_flag_800_2.npy")
    tc_ci = {c: i for i, c in enumerate(d["cty"])}
    csum = np.concatenate([np.zeros((1, tc.shape[1]), np.int32), np.cumsum(tc, axis=0, dtype=np.int32)])
    tc_cty = np.array([tc_ci.get(c, -1) for c in cty_of])
    cache: dict = {}

    def obs_cell(k):
        if k not in cache:
            f = np.nan_to_num(np.asarray(P[k]), nan=0.0).ravel()
            cache[k] = (np.bincount(cell_of_pixel, weights=f, minlength=ncell) / np.maximum(cnt, 1))
            if len(cache) > 8:
                cache.pop(next(iter(cache)))
        return cache[k]

    def county_stats(cellvals):  # {county: (max, mean)} over the county's GEFS cells
        return {c: (float(np.nanmax(cellvals[i])), float(np.nanmean(cellvals[i]))) for c, i in cty_cells.items()}

    parts = []
    files = sorted(e48.GEFS.glob("apcp_*.npy"))
    base = pd.Timestamp("2000-01-01")
    for f in files:
        s0 = pd.Timestamp(f.stem.split("_")[1]) + pd.Timedelta(1, "D")
        if not (pd.Timestamp("2010-01-01") <= s0 <= pd.Timestamp("2019-12-27")) or s0 not in days:
            continue
        es = (s0 - fp.EXPO_START).days
        di = days.get_loc(s0)
        npol = ex["n_pol"][:, es]
        bgs = np.flatnonzero(npol >= 1)
        ti = (s0 - base).days
        flag = np.zeros(len(bgs), bool)
        okc = tc_cty[bgs] >= 0
        flag[okc] = (csum[ti + WINDOW, tc_cty[bgs][okc]] - csum[ti, tc_cty[bgs][okc]]) > 0
        bgs = bgs[~flag]
        if not len(bgs):
            continue
        a = zstio.load(f).astype(np.float32)
        dayf = np.stack([a[:, :4].sum(1), a[:, 4:].sum(1)], 1)  # [member, day, y, x]
        F1m = dayf.max(1)  # per member peak over S, S+1: [member, y, x]
        fc_f1 = F1m.mean(0).ravel()
        fc_s = dayf.sum(1).mean(0).ravel()
        # ensemble extremes (script 79): member maximum, share of members >= 25 / 50 mm, spread, 3x3 mean of the 25 mm share
        v25, v50 = (F1m >= 25).mean(0), (F1m >= 50).mean(0)
        fc_mx, fc_v25, fc_v50, fc_sd = F1m.max(0).ravel(), v25.ravel(), v50.ravel(), F1m.std(0).ravel()
        fc_nb3 = uniform_filter(v25, 3).ravel()
        c_mx, c_25, c_50 = county_stats(fc_mx), county_stats(fc_v25), county_stats(fc_v50)
        cx_mx = np.array([c_mx[c] for c in cty_of[bgs]])
        cx_25 = np.array([c_25[c] for c in cty_of[bgs]])
        cx_50 = np.array([c_50[c] for c in cty_of[bgs]])
        oc = np.stack([obs_cell(di), obs_cell(di + 1)])
        oc_f1, oc_s = oc.max(0), oc.sum(0)
        fcc, occ = county_stats(fc_f1), county_stats(oc_f1)
        fcm = np.array([fcc[c] for c in cty_of[bgs]])
        ocm = np.array([occ[c] for c in cty_of[bgs]])
        cs = pd.Series(cty_of[bgs]).map(pd.Series(cty_of[bgs]).value_counts()).to_numpy()
        n = npol[bgs]
        parts.append(pd.DataFrame({
            "day": s0, "bg": bgs.astype(np.int32), "cty": cty_of[bgs], "month": s0.month, "cty_nbg": cs.astype(np.float32),
            "n_pol": n, "log_pol": np.log1p(n), "sfha_share": ex["n_sfha"][bgs, es] / n, "single_share": ex["n_single"][bgs, es] / n,
            "mean_cov": ex["cov_sum"][bgs, es] / n,
            "fc_f1": fc_f1[bcell[bgs]], "fc_s": fc_s[bcell[bgs]], "fc_cty_max": fcm[:, 0], "fc_cty_mean": fcm[:, 1],
            "oc_f1": oc_f1[bcell[bgs]], "oc_s": oc_s[bcell[bgs]], "oc_cty_max": ocm[:, 0], "oc_cty_mean": ocm[:, 1],
            "fc_mx": fc_mx[bcell[bgs]], "fc_v25": fc_v25[bcell[bgs]], "fc_v50": fc_v50[bcell[bgs]], "fc_sd": fc_sd[bcell[bgs]], "fc_nb3": fc_nb3[bcell[bgs]],
            "fc_cty_mx": cx_mx[:, 0], "fc_cty_v25": cx_25[:, 0], "fc_cty_v25mean": cx_25[:, 1], "fc_cty_v50": cx_50[:, 0],
            "n": (C[bgs, es + WINDOW] - C[bgs, es]).astype(np.float32), "paid": (Cp[bgs, es + WINDOW] - Cp[bgs, es]).astype(np.float32)}))
    df = pd.concat(parts, ignore_index=True)
    for c in df.columns[df.dtypes == np.float64]:  # halve the memory of the 40 M-row frame
        df[c] = df[c].astype(np.float32)
    df["year"] = df.day.dt.year
    df["sysid"] = df.day.dt.strftime("%Y%m%d")
    return df


def main() -> None:
    df = build()
    say(f"rows {len(df):,} on {df.day.nunique():,} forecast days (mean {len(df) / df.day.nunique():,.0f} block groups, {df.groupby('day').cty.nunique().mean():.0f} counties per day); "
        f"claimed block groups {int((df.n > 0).sum()):,} ({(df.n > 0).mean() * 100:.3f}%), claims {int(df.n.sum()):,}, paid ${df.paid.sum() / 1e6:.1f} M")
    prev = df.groupby("year").apply(lambda z: (z.n > 0).mean(), include_groups=False)
    say("yearly claimed-block-group rate (%): " + ", ".join(f"{a} {v * 100:.3f}" for a, v in prev.items()))
    y = (df.n.to_numpy() > 0).astype(float)
    tr, va, te = (df.year <= 2016).to_numpy(), ((df.year >= 2014) & (df.year <= 2016)).to_numpy(), (df.year >= 2017).to_numpy()
    rng = np.random.default_rng(0)
    keep = tr & ((y == 1) | (rng.random(len(df)) < 0.1))
    w = np.where(y == 1, 1.0, 10.0)
    base = ["month", "cty_nbg", "log_pol", "sfha_share", "single_share", "mean_cov"]
    fsets = {"p_fc": base + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"], "p_oc": base + ["oc_f1", "oc_s", "oc_cty_max", "oc_cty_mean"]}
    for col, cols in fsets.items():
        X = df[cols].to_numpy(np.float32)
        mk = lambda n: HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300, l2_regularization=1.0,  # noqa: E731
                                                      early_stopping=False, random_state=0)
        fit_rows = keep & ~va
        m = mk(300).fit(X[fit_rows], y[fit_rows], sample_weight=w[fit_rows])
        vm = va
        sc = [np.average((p[:, 1] - y[vm]) ** 2) for p in m.staged_predict_proba(X[vm])]
        n_it = int(np.argmin(sc)) + 1
        mdl = mk(n_it).fit(X[keep], y[keep], sample_weight=w[keep])
        df.loc[te, col] = mdl.predict_proba(X[te])[:, 1]
        say(f"fitted {col} ({n_it} iterations; training rows {int(keep.sum()):,})")
    T = df[te].reset_index(drop=True)
    keep_cols = ["sysid", "day", "year", "bg", "cty", "n", "paid", "n_pol", "fc_f1", "fc_cty_max", "p_fc", "p_oc"]
    T[keep_cols].to_parquet(CACHE)  # test rows with predictions, for the stratified analysis of script 78
    scores = {"forecast model": ("p_fc", "sum"), "observed-rain model (reference)": ("p_oc", "sum"), "forecast cell rain only": ("fc_f1", "max"),
              "forecast county peak only": ("fc_cty_max", "max"), "policies only": ("n_pol", "sum")}
    say(f"\ntest 2017-2019: {len(T):,} rows, {T.day.nunique()} forecast days, claimed block groups {int((T.n > 0).sum()):,}")
    e75.report("ALL COUNTIES of each forecast day (event = the day, ranked across every county of Texas)", T, scores, "forecast model", "policies only")
    e75.auxiliary("forecast model, all counties", T, T.p_fc.to_numpy(), prev)
    fp10 = T[T.fc_cty_max >= 10].reset_index(drop=True)
    say(f"\nex-ante footprint: counties with a forecast county peak >= 10 mm: {len(fp10):,} rows ({len(fp10) / len(T):.1%}), "
        f"{fp10.groupby('day').cty.nunique().mean():.0f} counties per day, holding {fp10.n.sum() / T.n.sum():.1%} of the claim count of the test rows")
    e75.report("FORECAST FOOTPRINT (forecast county peak >= 10 mm)", fp10, scores, "forecast model", "policies only")
    e75.auxiliary("forecast model, footprint", fp10, fp10.p_fc.to_numpy(), prev)
    pth = wb2.cache_path("audit") / "forecast_ranking_all_counties.txt"
    pth.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
