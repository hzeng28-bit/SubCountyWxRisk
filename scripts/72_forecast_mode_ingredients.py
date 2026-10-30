"""Forecast mode with moisture and wind ingredients (stage 33).

Stage 16 setting (block group x rain episode, 2010-2019, train 2010-2016, test 2017-2019, boosting depth 5, rain = GEFSv12 reforecast ensemble
mean, leads 36-84 h from the 00Z run of the day before the episode) plus episode-level ERA5 ingredients from cds_gulf, in two timings:
  perfect-prog  ingredients over the episode window [start - 1, end] (what a perfect moisture / wind forecast would give; optimistic)
  init          the 00Z analysis of the day before the episode start, i.e. the state the reforecast run starts from (no forecast error,
                nothing from after the run time)
Ingredients: tcwv, tcwv z-score (same-month climatology 2009-2016 of the cell), 850 hPa moisture flux q x |V|, 500 hPa wind speed.
Rows: fcmean (BASE + fc_f1, fc_s, fc_cty_max, fc_cty_mean) with and without the ingredients; obsC (observed rain at the GEFS cell) for reference.
Within-event AUC, event-clustered paired bootstrap (stage 16 helpers).
Inputs : flood_pilot_table.parquet, flood_pilot_fc_features.parquet, cds_gulf/*.nc
Output : $WXRISK_DATA/audit/forecast_mode_ingredients.txt
Run    : python scripts/72_forecast_mode_ingredients.py
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e48, e71 = _load("e48", "48_flood_forecast_pilot.py"), _load("e71", "71_era5_ingredients_texas.py")
D = wb2.cache_path("cds_gulf")
DAYS = e71.DAYS
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def init_series(ct: pd.DataFrame) -> dict[str, np.ndarray]:
    """Same four fields as e71.county_series but from the 00Z step only."""
    la, lo = xr.DataArray(ct.lat.to_numpy(), dims="c"), xr.DataArray(ct.lon.to_numpy(), dims="c")
    out = {k: np.full((len(DAYS), len(ct)), np.nan, np.float32) for k in ("tcwv", "q850", "w850", "w500")}
    for f in sorted(glob.glob(str(D / "single_*.nc"))):
        with xr.open_dataset(f) as a:
            x = a.tcwv.sel(valid_time=a.valid_time.dt.hour == 0).sel(latitude=la, longitude=lo, method="nearest").load()
            i = ((x.valid_time.dt.floor("D").to_numpy() - np.datetime64(DAYS[0])) / np.timedelta64(1, "D")).astype(int)
            out["tcwv"][i] = x.values
    for f in sorted(glob.glob(str(D / "plev_*.nc"))):
        with xr.open_dataset(f) as a:
            x = a.sel(valid_time=a.valid_time.dt.hour == 0).sel(latitude=la, longitude=lo, method="nearest").load()
            i = ((x.valid_time.dt.floor("D").to_numpy() - np.datetime64(DAYS[0])) / np.timedelta64(1, "D")).astype(int)
            w = np.hypot(x.u, x.v)
            out["q850"][i] = x.q.sel(pressure_level=850).values
            out["w850"][i] = w.sel(pressure_level=850).values
            out["w500"][i] = w.sel(pressure_level=500).values
    return out


def episode_features(e: pd.DataFrame, S: dict, mu, sd, j, lag: int, window: bool) -> pd.DataFrame:
    s0 = ((e.start - DAYS[0]).dt.days).to_numpy() + lag
    s1 = ((e.end - DAYS[0]).dt.days).to_numpy() if window else s0
    f = {k: np.array([np.nanmax(S[k][a:b + 1, c]) if k == "tcwv" else np.nanmean(S[k][a:b + 1, c]) for a, b, c in zip(s0, s1, j, strict=True)]) for k in S}
    m_ = e.month.to_numpy() - 1
    return pd.DataFrame({"tcwv": f["tcwv"], "tcwv_z": (f["tcwv"] - mu[m_, j]) / sd[m_, j], "flux850": f["q850"] * f["w850"], "w500": f["w500"]}, index=e.index)


def main() -> None:
    ct = fp.county_table().sort_values("cty").reset_index(drop=True)
    ci = {c: i for i, c in enumerate(ct.cty)}
    Sp = e71.county_series(ct)
    Si = init_series(ct)
    mon, trn = DAYS.month.to_numpy(), DAYS.year.to_numpy() <= 2016
    mu = np.stack([Sp["tcwv"][trn & (mon == m)].mean(axis=0) for m in range(1, 13)])
    sd = np.stack([Sp["tcwv"][trn & (mon == m)].std(axis=0) for m in range(1, 13)])

    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t = t[t.year <= e48.LAST_YEAR].merge(pd.read_parquet(e48.FEATS), on=["ep", "bg"], how="inner")
    t["log_pol"] = np.log1p(t.n_pol)
    e = t.drop_duplicates("ep")[["ep", "cty", "start", "end", "month"]].set_index("ep", drop=False)
    j = e.cty.map(ci).to_numpy()
    pp = episode_features(e, Sp, mu, sd, j, lag=-1, window=True).add_suffix("_pp")
    ini = episode_features(e, Si, mu, sd, j, lag=-1, window=False).add_suffix("_i")
    ing_pp, ing_i = list(pp.columns), list(ini.columns)
    t = t.join(pp, on="ep").join(ini, on="ep")
    y = t.y.to_numpy().astype(np.float64)
    yr = t.year.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    ep_te = t.ep.to_numpy()[te]
    say(f"{len(t):,} rows, {t.ep.nunique():,} episodes; test positives {int(y[te].sum())}; missing ingredient values {t[ing_pp + ing_i].isna().mean().mean():.4f}")
    fc = e48.BASE + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"]
    oc = e48.BASE + ["oc_f1", "oc_s", "oc_cty_max", "oc_cty_mean"]
    sets = {"fcmean": fc, "fcmean + ERA5 perfect-prog": fc + ing_pp, "fcmean + ERA5 at run init (00Z day before)": fc + ing_i,
            "obsC": oc, "obsC + ERA5 perfect-prog": oc + ing_pp}
    pe = {}
    for name, cols in sets.items():
        p, n = e48.exp26.fit_predict(t[cols].to_numpy(np.float32), y, tr, te, va)
        pe[name] = e48.per_episode(ep_te, y[te], p)
        say(f"  {name:44s} within-event AUC {pe[name][1].sum() / pe[name][2].sum():.4f}  ({n} iterations)")
    say("\npaired event-clustered bootstrap (difference, 95% interval)")
    for a, b in (("fcmean", "fcmean + ERA5 perfect-prog"), ("fcmean", "fcmean + ERA5 at run init (00Z day before)"), ("obsC", "obsC + ERA5 perfect-prog")):
        say("  {} -> {}: {:+.4f} [{:+.4f}, {:+.4f}]".format(a, b, *e48.boot_diff(pe[a], pe[b])))

    # between events: does a county episode have a claim? (the ingredients are constant inside an episode, so this is where they can act)
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    g = t.groupby("ep")
    ee = g.agg(year=("year", "first"), month=("month", "first"), dur=("dur", "first"), nbg=("cty_nbg", "first"), anyc=("y", "max"), nclaimed=("y", "sum"),
               fcmax=("fc_cty_max", "first"), fcmean=("fc_cty_mean", "first"), ocmax=("oc_cty_max", "first"), ocmean=("oc_cty_mean", "first"),
               fcsd=("fc_sd", "mean"), fcv25=("fc_v25", "max"), **{c: (c, "first") for c in ing_pp + ing_i})
    ee["lognclaimed"] = np.log1p(ee.nclaimed)
    base_f = ["month", "dur", "nbg", "fcmax", "fcmean", "fcsd", "fcv25"]
    base_o = ["month", "dur", "nbg", "ocmax", "ocmean"]
    say("\nepisode level, leave-one-year-out (2010-2019), AUC of any claim | Spearman of the prediction with log claimed block groups")
    for name, cols in {"forecast rain": base_f, "forecast rain + perfect-prog ingredients": base_f + ing_pp,
                       "forecast rain + ingredients at run init": base_f + ing_i, "observed rain (GEFS cell)": base_o,
                       "observed rain + perfect-prog ingredients": base_o + ing_pp}.items():
        pr = np.zeros(len(ee))
        for yv in sorted(ee.year.unique()):
            m_te = (ee.year == yv).to_numpy()
            clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=4, min_samples_leaf=40, random_state=0)
            pr[m_te] = clf.fit(ee.loc[~m_te, cols], ee.anyc[~m_te]).predict_proba(ee.loc[m_te, cols])[:, 1]
        say(f"  {name:44s} {roc_auc_score(ee.anyc, pr):.4f} | {spearmanr(pr, ee.lognclaimed)[0]:.3f}")
    p = wb2.cache_path("audit") / "forecast_mode_ingredients.txt"
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
