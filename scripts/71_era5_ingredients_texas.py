"""ERA5 moisture and wind ingredients against the Texas pilot episodes (stage 32).

From cds_gulf (script 70): at each Texas county centroid, per day, the maximum total column water vapour (tcwv), the mean 850 hPa wind and humidity,
and the mean 500 hPa wind. Per county episode (table_notc.parquet, window [start - 1 d, end]):
  tcwv, tcwv_z   episode maximum and its z-score against the cell's own climatology of the same calendar month (2009-2016 only)
  q850, flux850  mean 850 hPa specific humidity and moisture flux q x |V|  (low-level transport)
  w850, w500     mean wind speed at 850 and 500 hPa (500 hPa speed as the steering-flow / stall proxy)
Outcome: any claim in the county episode and the number of claimed block groups (all inland claims, tropical included). Leave-one-year-out
boosting; base = county rain, duration, month and block groups. Also within the tropical-window episodes and the 100 episodes with the largest rain.
Inputs : cds_gulf/*.nc, table_notc.parquet, clean NFIP claims
Run    : python scripts/71_era5_ingredients_texas.py
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

spec = importlib.util.spec_from_file_location("e64", Path(__file__).with_name("64_label_scope_relaxation.py"))
e64 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e64)
D = wb2.cache_path("cds_gulf")
DAYS = pd.date_range("2009-01-01", "2023-12-31")


def county_series(ct: pd.DataFrame) -> dict[str, np.ndarray]:
    la = xr.DataArray(ct.lat.to_numpy(), dims="c")
    lo = xr.DataArray(ct.lon.to_numpy(), dims="c")
    out = {k: np.full((len(DAYS), len(ct)), np.nan, np.float32) for k in ("tcwv", "q850", "w850", "w500")}
    d0 = DAYS[0]
    for f in sorted(glob.glob(str(D / "single_*.nc"))):
        with xr.open_dataset(f) as a:
            x = a.tcwv.sel(latitude=la, longitude=lo, method="nearest").load()
            day = ((x.valid_time.dt.floor("D").to_numpy() - np.datetime64(d0)) / np.timedelta64(1, "D")).astype(int)
            for i in np.unique(day):
                out["tcwv"][i] = x.values[day == i].max(axis=0)
    for f in sorted(glob.glob(str(D / "plev_*.nc"))):
        with xr.open_dataset(f) as a:
            x = a.sel(latitude=la, longitude=lo, method="nearest").load()
            day = ((x.valid_time.dt.floor("D").to_numpy() - np.datetime64(d0)) / np.timedelta64(1, "D")).astype(int)
            w = np.hypot(x.u, x.v)
            for i in np.unique(day):
                m = day == i
                out["q850"][i] = x.q.sel(pressure_level=850).values[m].mean(axis=0)
                out["w850"][i] = w.sel(pressure_level=850).values[m].mean(axis=0)
                out["w500"][i] = w.sel(pressure_level=500).values[m].mean(axis=0)
    return out


def main() -> None:
    ct = fp.county_table().sort_values("cty").reset_index(drop=True)
    ci = {c: i for i, c in enumerate(ct.cty)}
    S = county_series(ct)
    print({k: float(np.isnan(v).mean()) for k, v in S.items()}, "share of missing county-days (should be ~0)")
    # climatology of tcwv by calendar month and county, 2009-2016
    mon = DAYS.month.to_numpy()
    tr = DAYS.year.to_numpy() <= 2016
    mu = np.stack([S["tcwv"][tr & (mon == m)].mean(axis=0) for m in range(1, 13)])
    sd = np.stack([S["tcwv"][tr & (mon == m)].std(axis=0) for m in range(1, 13)])

    t = pd.read_parquet(fp.PILOT / "table_notc.parquet")
    cl = e64.all_claims()
    t["pos"] = e64.counts(t, cl[cl.cause.isin(["2", "3", "4"])]) > 0
    e = t.groupby("ep").agg(cty=("cty", "first"), start=("start", "first"), end=("end", "first"), year=("year", "first"), month=("month", "first"),
                            peak=("cty_f1_max", "first"), msum=("cty_s_mean", "first"), dur=("dur", "first"), nbg=("cty_nbg", "first"),
                            npos=("pos", "sum")).reset_index()
    e = e[e.cty.isin(ci)].copy()
    j = e.cty.map(ci).to_numpy()
    s0 = ((e.start - DAYS[0]).dt.days - 1).to_numpy()
    s1 = (e.end - DAYS[0]).dt.days.to_numpy()
    f = {k: np.array([np.nanmax(S[k][a:b + 1, c]) if k == "tcwv" else np.nanmean(S[k][a:b + 1, c]) for a, b, c in zip(s0, s1, j, strict=True)])
         for k in S}
    e["tcwv"], e["q850"], e["w850"], e["w500"] = f["tcwv"], f["q850"], f["w850"], f["w500"]
    m_ = e.month.to_numpy() - 1
    e["tcwv_z"] = (e.tcwv - mu[m_, j]) / sd[m_, j]
    e["flux850"] = e.q850 * e.w850
    tc = np.load(fp.PILOT / "tc_flag_800_2.npy")
    cs = np.concatenate([np.zeros((1, tc.shape[1]), np.int32), np.cumsum(tc, axis=0, dtype=np.int32)])
    d = np.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    cj = e.cty.map({c: i for i, c in enumerate(d["cty"])}).to_numpy().astype(int)
    si, ei = (e.start - pd.Timestamp("2000-01-01")).dt.days.to_numpy(), (e.end - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    e["tc"] = (cs[ei + 3, cj] - cs[si, cj]) > 0
    e["any"] = (e.npos > 0).astype(int)
    e["lognpos"] = np.log1p(e.npos)
    print(f"{len(e):,} episodes; with a claim {e['any'].mean():.3f}")
    print(e[["tcwv", "tcwv_z", "q850", "w850", "w500", "flux850"]].describe().round(2).T.to_string())

    print("\nSpearman with the number of claimed block groups (all episodes | tropical-window episodes):")
    for c in ["peak", "msum", "tcwv", "tcwv_z", "q850", "w850", "w500", "flux850"]:
        print(f"  {c:8s} {spearmanr(e[c], e.lognpos, nan_policy='omit')[0]:+.3f} | {spearmanr(e.loc[e.tc, c], e.loc[e.tc, 'lognpos'], nan_policy='omit')[0]:+.3f}")
    print("\ntcwv z-score of named events:")
    for name, (a, b) in {"Memorial Day 2015": ("2015-05-23", "2015-05-27"), "Tax Day 2016": ("2016-04-17", "2016-04-19"),
                         "Harvey": ("2017-08-25", "2017-09-02"), "Imelda": ("2019-09-17", "2019-09-21")}.items():
        z = e[(e.start >= a) & (e.start <= b)]
        print(f"  {name:18s} tcwv {z.tcwv.median():5.1f} kg/m2 (z median {z.tcwv_z.median():+.2f}, p90 {z.tcwv_z.quantile(.9):+.2f}); 500 hPa wind {z.w500.median():4.1f} m/s")

    base = ["peak", "msum", "dur", "month", "nbg"]
    sets = {"county rain": base, "+ tcwv, tcwv_z": base + ["tcwv", "tcwv_z"], "+ 850 hPa humidity, wind, flux": base + ["q850", "w850", "flux850"],
            "+ 500 hPa wind": base + ["w500"], "+ all ERA5 ingredients": base + ["tcwv", "tcwv_z", "q850", "w850", "flux850", "w500"]}
    print("\nleave-one-year-out boosting, AUC of any claim | tropical-window episodes | the 500 episodes with the largest county rain")
    big = e.peak.rank(ascending=False) <= 500
    for name, cols in sets.items():
        p = np.zeros(len(e))
        for y in sorted(e.year.unique()):
            te = (e.year == y).to_numpy()
            m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=40, random_state=0)
            p[te] = m.fit(e.loc[~te, cols], e["any"][~te]).predict_proba(e.loc[te, cols])[:, 1]
        a = roc_auc_score(e["any"], p)
        b = roc_auc_score(e["any"][e.tc], p[e.tc.to_numpy()])
        c = roc_auc_score(e["any"][big], p[big.to_numpy()])
        print(f"  {name:34s} {a:.4f} | {b:.4f} | {c:.4f}   Spearman with log claimed block groups {spearmanr(p, e.lognpos)[0]:.3f}")


if __name__ == "__main__":
    main()
