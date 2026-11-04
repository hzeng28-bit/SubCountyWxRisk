"""Scenario analysis for the Texas city forecaster (scenario report, RESULTS stage 56): timing against NWS flash flood warnings, ranking inside warned counties,
and two ways of showing the sub-county result (discrete points vs geographic interpolation).

A. NWS storm-based Flash Flood Warnings (IEM archive, labels_extra/iem, PHENOM FF, SIG W) 2017-2023, polygons intersected with the Texas counties.
B. Timing, per test event day S (>= 10 claimed block groups): T0 = 12Z of S (start of the rain window the model sees: GEFS 00Z run of S-1, leads 36-84 h).
   Hours from T0 to the first warning that covers a county with claims, and to the first warning anywhere in Texas, in the window T0-24 h .. T0+48 h.
C. Accuracy inside the warned counties: the same ranking metrics as script 96 restricted to the counties covered by a warning of that event day (what a forecaster
   holding the warning would use the ranking for), forecast mode 2017-2019 and observed rain 2017-2023, against policies only.
D. Display: (1) block-group points as they are, (2) the points smoothed with a distance kernel (5 / 10 / 20 / 40 km), scored at the block groups (ranking skill);
   (3) interpolation as a filler: 30% of the block groups hidden, their probability rebuilt from the neighbours, against policies only.
Inputs : labels_extra/iem/*_tsmf_sbw.zip, census county file, script 96 builders
Output : audit/scenario_analysis.txt, audit/ffw_tx.parquet, docs/figures/scenario_*.png
Run    : python scripts/98_scenario_analysis.py
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402
from wxrisk import zstio  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT: list[str] = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tx_counties():
    import shapefile
    from shapely.geometry import shape

    z = zipfile.ZipFile(wb2.cache_path("labels_extra") / "census" / "cb_2020_us_county_20m.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")), dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    geo, code = [], []
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        if rec["STATEFP"] == "48":
            geo.append(shape(sr.shape.__geo_interface__))
            code.append("48" + rec["COUNTYFP"])
    return geo, code


def load_ffw(years=range(2017, 2024)) -> pd.DataFrame:
    import shapefile
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    cache = wb2.cache_path("audit") / "ffw_tx.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    geo, code = tx_counties()
    tree = STRtree(geo)
    rows = []
    for y in years:
        z = zipfile.ZipFile(wb2.cache_path("labels_extra") / "iem" / f"{y}_tsmf_sbw.zip")
        stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
        rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")), dbf=io.BytesIO(z.read(stem + ".dbf")))
        names = [f[0] for f in rd.fields[1:]]
        csv = pd.read_csv(io.BytesIO(z.read([n for n in z.namelist() if n.endswith(".csv")][0])), low_memory=False, usecols=["WFO", "ETN", "PHENOM", "SIG", "DAMAGETAG", "IS_EMERGENCY"])
        csv = csv[(csv.PHENOM == "FF") & (csv.SIG == "W")]
        tags = {(w, int(e)): (str(d) if isinstance(d, str) else "", bool(em)) for w, e, d, em in zip(csv.WFO, csv.ETN, csv.DAMAGETAG, csv.IS_EMERGENCY, strict=True)}  # the shapefile drops these fields
        for i, rec in enumerate(rd.iterRecords()):
            r = dict(zip(names, rec, strict=True))
            if r["PHENOM"] != "FF" or r["SIG"] != "W":
                continue
            poly = shape(rd.shape(i).__geo_interface__)
            hit = [code[j] for j in tree.query(poly, predicate="intersects")]
            if hit:
                rows.append({"issued": pd.Timestamp(r["ISSUED"]), "expired": pd.Timestamp(r["EXPIRED"]), "wfo": r["WFO"], "emergency": tags.get((r["WFO"], int(r["ETN"])), ("", False))[1], "damage": tags.get((r["WFO"], int(r["ETN"])), ("", False))[0], "counties": ",".join(sorted(hit))})
    df = pd.DataFrame(rows).sort_values("issued").reset_index(drop=True)
    df.to_parquet(cache)
    return df


def warned_by_day(ffw: pd.DataFrame, days, h0=-24, h1=48):
    """Per day S: set of counties covered by a warning issued in [T0+h0, T0+h1] (T0 = 12Z of S), and the first issue time per county."""
    iss = ffw.issued.to_numpy()
    out = {}
    for sid in days:
        t0 = pd.Timestamp(sid) + pd.Timedelta(12, "h")
        m = (iss >= np.datetime64(t0 + pd.Timedelta(h0, "h"))) & (iss <= np.datetime64(t0 + pd.Timedelta(h1, "h")))
        first = {}
        for t, cs in zip(ffw.issued[m], ffw.counties[m], strict=True):
            for c in cs.split(","):
                if c not in first or t < first[c]:
                    first[c] = t
        out[sid] = first
    return out


def timing(T, ffw, label):
    ev = T.assign(y=T.n > 0).groupby("sysid").agg(ncl=("y", "sum"))
    big = ev[ev.ncl >= 10].index
    W = warned_by_day(ffw, big, h0=-48)  # timing looks further back than the warned-county set (T0-24 h) so warnings before the model run are counted
    claimed = T[T.n > 0].groupby("sysid").cty.unique()
    h_aff, h_any, nw = [], [], 0
    for sid in big:
        t0 = pd.Timestamp(sid) + pd.Timedelta(12, "h")
        f = W[sid]
        if not f:
            continue
        nw += 1
        h_any.append((min(f.values()) - t0).total_seconds() / 3600)
        aff = [f[c] for c in claimed.get(sid, []) if c in f]
        if aff:
            h_aff.append((min(aff) - t0).total_seconds() / 3600)
    say(f"\n== {label}: {len(big)} event days with >= 10 claimed block groups")
    say(f"   with >= 1 Texas FF Warning in [T0-48h, T0+48h]: {nw} ({nw / len(big):.0%}); with a warning covering a county that had claims: {len(h_aff)} ({len(h_aff) / len(big):.0%})")
    for nm, h in (("first warning anywhere in Texas", h_any), ("first warning for an affected county", h_aff)):
        h = np.array(h)
        say(f"   hours from T0 to the {nm}: p10 {np.percentile(h, 10):+.1f}, median {np.median(h):+.1f}, p90 {np.percentile(h, 90):+.1f}; issued before T0 in {np.mean(h < 0):.0%}, "
            f"more than 6 h after T0 in {np.mean(h > 6):.0%}")
    return np.array(h_aff), np.array(h_any), W


def restrict(T, W):
    keep = np.zeros(len(T), bool)
    sid, cty = T.sysid.to_numpy(), T.cty.to_numpy()
    idx = pd.Series(np.arange(len(T))).groupby(sid).apply(lambda s: s.to_numpy())
    for s, ii in idx.items():
        w = W.get(s)
        if w:
            keep[ii] = np.isin(cty[ii], list(w))
    return T[keep]


def warned_metrics(T, scores, W, e75, label, model):
    R = restrict(T, W)
    ev_all = T.assign(y=T.n > 0).groupby("sysid").y.sum()
    R = R[R.sysid.map(ev_all) >= 1]
    say(f"\n== {label}: rows in warned counties {len(R):,}, event days {R.sysid.nunique()}")
    tabs = e75.event_table(R, scores)
    base = "policies only"
    rows = []
    for lab, col, w in (("BG AUC", "bg_auc", "bg_w"), ("BG top-5% capture of claims", "bg_cap_n_0.05", None), ("BG top-10% capture", "bg_cap_n_0.1", None), ("BG NDCG@5% (paid)", "bg_ndcg_p", None),
                        ("county AUC", "c_auc", "c_w"), ("county top-20% capture", "c_cap_n_0.2", None)):
        res, dd = e75.boot(tabs, col, w, ref=base)
        say(f"   {lab}: " + " | ".join(f"{n} {res[n][0]:.3f} [{res[n][1]:.3f}, {res[n][2]:.3f}]" for n in scores))
        say(f"      {model} - policies only: {dd[model][0]:+.3f} [{dd[model][1]:+.3f}, {dd[model][2]:+.3f}]")
        rows.append((lab, {n: res[n] for n in scores}, dd[model]))
    return rows, R


def bg_xy():
    S = zstio.load(wb2.cache_path("cnn") / "static.npz", allow_pickle=True)
    lat, lon = S["lat4"][S["b4_y"]], S["lon4"][S["b4_x"]]
    return lon, lat


def display_tests(T, col, label, e75, n_ev=150):
    from scipy.spatial import cKDTree

    lon, lat = bg_xy()
    xy = np.column_stack([(lon + 100) * 111.32 * np.cos(np.deg2rad(31)), (lat - 31) * 110.57])
    ev = T.assign(y=T.n > 0).groupby("sysid").y.sum()
    ids = ev[ev >= 10].index.to_numpy()
    rng = np.random.default_rng(0)
    ids = rng.choice(ids, min(n_ev, len(ids)), replace=False)
    res = {k: [] for k in ("points", "k5", "k10", "k20", "k40", "pol")}
    miss = {k: [] for k in ("hidden_interp", "hidden_nearest", "hidden_pol")}
    for sid in ids:
        g = T[T.sysid == sid]
        p = g[col].to_numpy().astype(float)
        P = xy[g.bg.to_numpy()]
        tree = cKDTree(P)
        n = g.n.to_numpy()
        yb = (n > 0).astype(int)
        vals = {"points": p, "pol": g.n_pol.to_numpy().astype(float)}
        d, j = tree.query(P, k=min(30, len(P)))
        for r in (5, 10, 20, 40):
            w = np.exp(-0.5 * (d / (r / 2)) ** 2)
            vals[f"k{r}"] = (w * p[j]).sum(1) / w.sum(1)
        for k, s in vals.items():
            a, _ = e75.pair_auc(s, yb)
            m = max(int(np.ceil(0.05 * len(s))), 1)
            top = e75.top_idx(s, m)
            res[k].append((a, n[top].sum() / n.sum()))
        # hide 30%, rebuild their score from the visible block groups (kernel 10 km; nearest as the reference)
        hide = rng.random(len(p)) < 0.3
        if hide.sum() > 5 and yb[hide].sum() > 0 and (~hide).sum() > 30:
            t2 = cKDTree(P[~hide])
            d2, j2 = t2.query(P[hide], k=min(15, int((~hide).sum())))
            w2 = np.exp(-0.5 * (d2 / 5.0) ** 2) + 1e-12
            interp = (w2 * p[~hide][j2]).sum(1) / w2.sum(1)
            near = p[~hide][j2[:, 0]]
            for k, s in (("hidden_interp", interp), ("hidden_nearest", near), ("hidden_pol", g.n_pol.to_numpy()[hide].astype(float))):
                a, _ = e75.pair_auc(s, yb[hide])
                miss[k].append(a)
    say(f"\n== display test, {label}: {len(ids)} sampled event days with >= 10 claimed block groups")
    for k, nm in (("points", "block-group points as they are"), ("k5", "smoothed 5 km"), ("k10", "smoothed 10 km"), ("k20", "smoothed 20 km"), ("k40", "smoothed 40 km"), ("pol", "policies only")):
        a = np.array(res[k])
        say(f"   {nm:34s} BG AUC {np.nanmean(a[:, 0]):.3f}   top-5% capture {np.nanmean(a[:, 1]):.3f}")
    for k, nm in (("hidden_interp", "30% hidden: kernel interpolation"), ("hidden_nearest", "30% hidden: nearest visible point"), ("hidden_pol", "30% hidden: policies only")):
        say(f"   {nm:34s} AUC on the hidden block groups {np.nanmean(miss[k]):.3f} ({len(miss[k])} events)")
    return res, miss


def main() -> None:
    e75, s96 = _load("e75", "75_task_metrics.py"), _load("s96", "96_task_scorecard.py")
    ffw = load_ffw()
    say(f"Texas FF Warnings 2017-2023: {len(ffw)} (warnings per year: {ffw.issued.dt.year.value_counts().sort_index().to_dict()}); emergencies {int(ffw.emergency.sum())}")
    dur = (ffw.expired - ffw.issued).dt.total_seconds() / 3600
    say(f"warning duration (h): median {dur.median():.1f}, p90 {dur.quantile(.9):.1f}")
    out = {}
    for label, builder, years in (("forecast mode 2017-2019", s96.build_forecast, (2017, 2019)), ("observed rain 2017-2023", lambda: s96.build_observed(_load("e85", "85_state_group_baseline.py")), (2017, 2023))):
        T, scores, model = builder()
        T["year"] = T.sysid.str[:4].astype(int)
        h_aff, h_any, W = timing(T, ffw, label)
        full = {k: v for k, v in scores.items() if k != "blind guess"}
        rows, R = warned_metrics(T, full, W, e75, f"inside the warned counties, {label}", model)
        col = scores[model][0]
        disp = display_tests(T, col, label, e75)
        out[label] = (h_aff, h_any, rows, disp)
        del T, R
    (wb2.cache_path("audit") / "scenario_analysis.txt").write_text("\n".join(OUT) + "\n")
    import pickle

    pickle.dump(out, open(wb2.cache_path("audit") / "scenario_analysis.pkl", "wb"))


if __name__ == "__main__":
    main()
