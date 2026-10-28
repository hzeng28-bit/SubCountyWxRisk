"""Spatial features for the flood pilot: neighbour-smoothed claim history, tested under a year split and a year + region split (stage 21).

Stage 17 found that the same-episode labels of the 5 nearest block groups (an oracle, not a feature) add +0.022 under observed rain and
+0.085 under forecast rain. This script tests what a usable, ex ante version of that neighbourhood adds on top of the stage 19 model
(base + hist00 + prior3):
  nbhist00   log1p of the mean claim count 2000-2009 over the 8 nearest block groups
  nbprior3   same for the 3 years before the episode
  nb_dist    mean distance (km) to those 8 neighbours, a density proxy
  lat, lon   location (arm 3)
Two evaluations per rain arm (observed 2017-2023, forecast fcmean 2017-2019):
  time split   train <= 2016, test later (the repo's standard; test block groups have neighbours in the training set)
  region split train <= 2016 and outside the test region, test later and inside it (5 k-means regions of county centroids,
               each region held out in turn, predictions pooled); the neighbour features are still computed from all block groups,
               as they would be for a new region with a claim record.

Inputs : flood_pilot_table.parquet, flood_pilot_fc_features.parquet, clean NFIP claims
Output : $WXRISK_DATA/audit/spatial_features.txt
Run    : python scripts/56_spatial_features.py [--obs-only] [--no-region]      # Mac CPU: tens of minutes
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

K = 8
REGIONS = 5
HIST = ["hist00", "prior3"]
NB = ["nbhist00", "nbprior3", "nb_dist"]
OUT = []


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def neighbours(geo: pd.DataFrame):
    """K nearest block groups (self excluded) and the distance to them in km."""
    xy = np.column_stack([geo.lat.to_numpy() * 111.0, geo.lon.to_numpy() * 111.0 * 0.87])  # ~cos(31 deg)
    d, nb = cKDTree(xy).query(xy, k=K + 1)
    return nb[:, 1:], d[:, 1:]


def spatial_features(t: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    nb, dist = neighbours(geo)
    cl = fp.claims_tx()
    cl = cl[cl.date >= "2000-01-01"]
    idx = {b: i for i, b in enumerate(geo.bg10)}
    bi = cl.bg10.map(idx).to_numpy()
    day = (cl.date - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    keys = np.sort(bi.astype(np.int64) * 20000 + day)
    c00 = np.bincount(bi[day < 3653], minlength=len(geo)).astype(np.float64)
    bg = t.bg.to_numpy().astype(np.int64)
    start = (pd.to_datetime(t.start) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    p3 = np.zeros(len(t))
    for j in range(K):
        b = nb[bg, j].astype(np.int64) * 20000
        p3 += np.searchsorted(keys, b + start) - np.searchsorted(keys, b + start - 1095)
    return pd.DataFrame({"nbhist00": np.log1p(c00[nb][bg].mean(1)), "nbprior3": np.log1p(p3 / K),
                         "nb_dist": dist.mean(1)[bg]}, index=t.index)


def regions(t: pd.DataFrame) -> np.ndarray:
    """Contiguous regions: k-means on county centroids, so a held-out region has no training neighbours at its core."""
    c = t.groupby("cty")[["lat", "lon"]].mean()
    lab = KMeans(REGIONS, n_init=10, random_state=0).fit_predict(np.column_stack([c.lat, c.lon * 0.87]))
    return pd.Series(lab, index=c.index).reindex(t.cty).to_numpy()


def run(t, base, arms, label, e26, e48, region_split):
    y = t.y.to_numpy().astype(np.float64)
    yr = t.year.to_numpy()
    ep = t.ep.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    reg = regions(t)
    splits = {"time split": None}
    if region_split:
        splits["year + region split"] = reg
    for sname, rg in splits.items():
        res = {}
        for name, cols in arms.items():
            X = t[base + cols].to_numpy(np.float32)
            if rg is None:
                p, n = e26.fit_predict(X, y, tr, te, (yr >= 2014) & tr)
                s = p
                say(f"    {label} / {sname} / {name}: {n} iterations")
            else:
                s = np.full(len(t), np.nan)
                for r in range(REGIONS):
                    trn, tst = tr & (rg != r), te & (rg == r)
                    s[tst], _ = e26.fit_predict(X, y, trn, tst, (yr >= 2014) & trn)
                    say(f"    {label} / {sname} / {name}: region {r} done")
                s = s[te]
            res[name] = e48.per_episode(ep[te], y[te], s)
        ref = res["base (stage 19)"]
        say(f"\n  {label}, {sname}: test positives {int(y[te].sum()):,}")
        say(f"  {'arm':34s} within-event AUC   difference to the stage 19 base [95% interval]")
        for name, v in res.items():
            a = v[1].sum() / v[2].sum()
            if name == "base (stage 19)":
                say(f"  {name:34s} {a:.4f}")
            else:
                d, lo, hi = e48.boot_diff(ref, v)
                say(f"  {name:34s} {a:.4f}            {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-only", action="store_true", help="skip the forecast-rain arm")
    ap.add_argument("--no-region", action="store_true", help="skip the region split")
    a = ap.parse_args()
    e26, e48, e49 = _load("e26", "26_flood_pilot_experiment.py"), _load("e48", "48_flood_forecast_pilot.py"), _load("e49", "49_information_audit.py")
    geo = fp.bg_geometry()
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t = t.join(e49.history(t)).join(spatial_features(t, geo))
    arms = {"base (stage 19)": [], "+ neighbour history": NB, "+ neighbour history + lat/lon": [*NB, "lat", "lon"]}
    obs = e26.COUNTY + e26.EXPO + e26.CELL + HIST

    say("== observed rain, test 2017-2023")
    run(t, obs, arms, "observed rain", e26, e48, not a.no_region)
    if not a.obs_only:
        say("\n== forecast rain (fcmean), test 2017-2019")
        u = t[t.year <= 2019].merge(pd.read_parquet(e48.FEATS), on=["ep", "bg"], how="inner")
        run(u, e48.BASE + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"] + HIST, arms, "forecast rain", e26, e48, not a.no_region)
    p = wb2.cache_path("audit") / "spatial_features.txt"
    p.parent.mkdir(exist_ok=True)
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
