"""Does the 'slow / stalled motion' ingredient, taken from the best track, explain how much a tropical episode floods? (stage 31)

For every Texas county episode inside a tropical-cyclone window (table_notc.parquet, tc flag 800 km +-2 d), from HURDAT2 (Atlantic + eastern Pacific):
  hours_near   6 h x best-track points within 500 km of the county centroid during [start - 3 d, end + 3 d]   (dwell / stall)
  min_dist     closest approach (km)
  speed        mean translation speed over those points (km/h)
  vmax         maximum sustained wind among them (kt)
Outcome: any claim in the county episode and the number of claimed block groups (all inland claims). Base = county rain features; the test is whether
the motion features add to them, by leave-one-year-out within the tropical episodes (small sample: read with that in mind).
Inputs : table_notc.parquet, hurdat2.txt, hurdat2_nepac.txt, clean NFIP claims, tc_flag_800_2.npy
Run    : python scripts/69_tc_motion_ingredient.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402

spec = importlib.util.spec_from_file_location("e64", Path(__file__).with_name("64_label_scope_relaxation.py"))
e64 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e64)
D0 = pd.Timestamp("2000-01-01")


def track() -> pd.DataFrame:
    rows = []
    for f in ("hurdat2.txt", "hurdat2_nepac.txt"):
        sid = None
        for ln in open(fp.PILOT / f):
            p = [x.strip() for x in ln.split(",")]
            if not p[0][:1].isdigit():
                sid = p[0]
                continue
            if len(p) < 7:
                continue
            hh = p[1].zfill(4)
            t = pd.Timestamp(p[0]) + pd.Timedelta(hours=int(hh[:2]), minutes=int(hh[2:]))
            if int(hh) % 600:  # keep the synoptic 00/06/12/18 UTC points
                continue
            lat = float(p[4][:-1]) * (1 if p[4].endswith("N") else -1)
            lon = float(p[5][:-1]) * (1 if p[5].endswith("E") else -1)
            rows.append((sid, t, lat, lon, float(p[6])))
    d = pd.DataFrame(rows, columns=["sid", "t", "lat", "lon", "vmax"])
    d = d.sort_values(["sid", "t"])
    la, lo = np.radians(d.lat.to_numpy()), np.radians(d.lon.to_numpy())
    dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin(np.diff(la, prepend=la[0]) / 2) ** 2 + np.cos(la) * np.cos(np.roll(la, 1)) * np.sin(np.diff(lo, prepend=lo[0]) / 2) ** 2))
    first = d.sid.ne(d.sid.shift()).to_numpy()
    d["speed"] = np.where(first, np.nan, dist / 6.0)
    return d.reset_index(drop=True)


def main() -> None:
    t = pd.read_parquet(fp.PILOT / "table_notc.parquet")
    cl = e64.all_claims()
    cl = cl[cl.cause.isin(["2", "3", "4"])]
    t["pos"] = e64.counts(t, cl) > 0
    d = np.load(fp.PILOT / "county_daily.npz", allow_pickle=True)
    ci = {c: i for i, c in enumerate(d["cty"])}
    tc = np.load(fp.PILOT / "tc_flag_800_2.npy")
    csum = np.concatenate([np.zeros((1, tc.shape[1]), np.int32), np.cumsum(tc, axis=0, dtype=np.int32)])
    e = t.groupby("ep").agg(cty=("cty", "first"), start=("start", "first"), end=("end", "first"), year=("year", "first"), peak=("cty_f1_max", "first"),
                            msum=("cty_s_mean", "first"), dur=("dur", "first"), nbg=("cty_nbg", "first"), month=("month", "first"),
                            npos=("pos", "sum")).reset_index()
    si, ei = (e.start - D0).dt.days.to_numpy(), (e.end - D0).dt.days.to_numpy()
    cj = e.cty.map(ci).to_numpy().astype(int)
    e = e[(csum[ei + 3, cj] - csum[si, cj]) > 0].copy()
    ct = fp.county_table().set_index("cty")
    tr = track()
    tl, tn = np.radians(tr.lat.to_numpy()), np.radians(tr.lon.to_numpy())
    tt = tr.t.to_numpy()
    feats = []
    for r in e.itertuples():
        la, lo = np.radians(ct.loc[r.cty, "lat"]), np.radians(ct.loc[r.cty, "lon"])
        dist = 6371 * 2 * np.arcsin(np.sqrt(np.sin((tl - la) / 2) ** 2 + np.cos(la) * np.cos(tl) * np.sin((tn - lo) / 2) ** 2))
        m = (tt >= (r.start - pd.Timedelta(days=3)).to_datetime64()) & (tt <= (r.end + pd.Timedelta(days=3)).to_datetime64()) & (dist <= 500)
        feats.append((6.0 * m.sum(), dist[m].min() if m.any() else np.nan, np.nanmean(tr.speed.to_numpy()[m]) if m.any() else np.nan,
                      tr.vmax.to_numpy()[m].max() if m.any() else np.nan))
    e[["hours_near", "min_dist", "speed", "vmax"]] = feats
    e["min_dist"] = e.min_dist.fillna(500.0)  # no track point within 500 km: capped distance, no dwell, no speed or wind
    e[["speed", "vmax"]] = e[["speed", "vmax"]].fillna(0.0)
    e["any"] = (e.npos > 0).astype(int)
    e["lognpos"] = np.log1p(e.npos)
    print(f"tropical-window county episodes: {len(e):,}; with a track point within 500 km: {(e.hours_near > 0).sum():,}; with a claim {e['any'].mean():.3f}")
    print(e[["hours_near", "min_dist", "speed", "vmax"]].describe().round(1).T.to_string())
    print("\nSpearman with the number of claimed block groups:")
    for c in ["peak", "msum", "hours_near", "min_dist", "speed", "vmax"]:
        print(f"  {c:11s} {spearmanr(e[c], e.lognpos)[0]:+.3f}")
    base = ["peak", "msum", "dur", "nbg"]
    mot = base + ["hours_near", "min_dist", "speed", "vmax"]
    for name, cols in {"county rain": base, "+ motion / track (hours near, closest approach, speed, wind)": mot, "motion alone": ["hours_near", "min_dist", "speed", "vmax"]}.items():
        X = e[cols].copy()
        X = X.assign(**{c: np.log1p(X[c].clip(lower=0)) for c in X.columns if c in ("peak", "msum", "hours_near", "min_dist", "nbg")})
        p = np.zeros(len(e))
        for y in sorted(e.year.unique()):
            te = (e.year == y).to_numpy()
            sc = StandardScaler().fit(X[~te])
            p[te] = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[~te]), e["any"][~te]).predict_proba(sc.transform(X[te]))[:, 1]
        print(f"  leave-one-year-out AUC of any claim, {name}: {roc_auc_score(e['any'], p):.4f}; Spearman with log claimed block groups {spearmanr(p, e.lognpos)[0]:.3f}")


if __name__ == "__main__":
    main()
