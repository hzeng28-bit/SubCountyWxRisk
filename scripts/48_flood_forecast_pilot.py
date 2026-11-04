"""Forecast-mode flood pilot (stage 16): replace observed rain by GEFSv12 reforecast rain and measure what is lost.

Same unit, label, split logic and learner as scripts 25/26 (block group x rain episode; any inland non-tropical NFIP claim;
gradient boosting depth 5), restricted to episodes 2010-2019 (the reforecast ends in 2019): train 2010-2016, test 2017-2019.
Rain enters through the two days a 00Z run of the day before the episode can see (days S and S+1, gridMET day = 12Z..12Z):
  F1 = largest daily rain over those days, S = their sum (truncated at the episode end, so a 4-day episode is cut to 2 days).

Rain sources, from least to most forecast-like:
  obs4k   gridMET at the block group's 4 km cell            (the stage 8 input, truncated to the same days)
  obsC    gridMET averaged over the 0.25 deg GEFS cell      (what resolution alone costs)
  fcmean  ensemble-mean forecast at the cell                (5 members, lead 36-84 h)
  fcens   fcmean + member vote (>= 25 / 50 mm), spread, 3x3 neighbourhood mean
County-level rain features (max / mean over the county's cells) come from the same source as the cell value.

Inputs : flood_pilot_table.parquet, gridMET (pr_tx_2000_2023.npy), GEFS crops from scripts/47
Outputs: flood_pilot_fc_features.parquet (cache), $WXRISK_DATA/audit/flood_forecast_pilot.{txt,json}
Run    : python scripts/48_flood_forecast_pilot.py --build      # features (needs scripts/47 output)
         python scripts/48_flood_forecast_pilot.py              # models and tables
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import zstio  # noqa: E402
from wxrisk import wb2  # noqa: E402

GEFS = wb2.cache_path("labels_extra") / "flood" / "gefs"
FEATS = fp.PILOT / "flood_pilot_fc_features.parquet"
AUDIT = wb2.cache_path("audit")
LAST_YEAR = 2019
BASE = ["dur", "month", "cty_nbg", "log_pol", "sfha_share", "single_share", "mean_cov"]
REPS = 1000

_spec = importlib.util.spec_from_file_location("exp26", Path(__file__).with_name("26_flood_pilot_experiment.py"))
exp26 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp26)


def build() -> None:
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet",
                        columns=["bg", "ep", "cty", "start", "end", "dur", "year"])
    t = t[t.year <= LAST_YEAR].copy()
    meta = zstio.load(GEFS / "gefs_meta.npz", allow_pickle=True)
    glat, glon = meta["lat"], meta["lon"]
    days, lat, lon, P = fp.load_precip()
    g = fp.bg_geometry()
    # nearest GEFS cell of every gridMET pixel and of every block group centroid
    py = np.abs(glat[None, :] - lat[:, None]).argmin(1)
    px = np.abs(glon[None, :] - lon[:, None]).argmin(1)
    cell_of_pixel = (py[:, None] * len(glon) + px[None, :]).ravel()
    ncell = len(glat) * len(glon)
    cnt = np.bincount(cell_of_pixel, minlength=ncell).astype(np.float32)
    by, bx = fp.grid_index(lat, lon, g.lat.to_numpy(), g.lon.to_numpy())
    cy = np.abs(glat[None, :] - g.lat.to_numpy()[:, None]).argmin(1)
    cx = np.abs(glon[None, :] - g.lon.to_numpy()[:, None]).argmin(1)
    cty_idx = g.groupby("cty").indices  # county -> block-group positions in `g` (the `bg` column of the table)
    cty_cells = {c: np.unique(cy[i] * len(glon) + cx[i]) for c, i in cty_idx.items()}
    ep_rows = {e: v.to_numpy() for e, v in t.groupby("ep", sort=False)["bg"]}

    cache: dict = {}

    def obs_cell(k: int) -> np.ndarray:
        if k not in cache:
            f = np.nan_to_num(np.asarray(P[k]), nan=0.0).ravel()
            cache[k] = (np.bincount(cell_of_pixel, weights=f, minlength=ncell) / np.maximum(cnt, 1)).reshape(len(glat), len(glon))
        return cache[k]

    ep = t.drop_duplicates("ep")[["ep", "cty", "start", "end", "dur"]]
    out, dropped = [], 0
    for r in ep.itertuples():
        s0 = pd.Timestamp(r.start).normalize()
        f = GEFS / f"apcp_{s0 - pd.Timedelta(1, 'D'):%Y%m%d}.npy"
        if not f.exists():
            dropped += 1
            continue
        a = zstio.load(f).astype(np.float32)  # [5, 8, ny, nx]
        nd = 1 if r.dur == 1 else 2
        dayf = np.stack([a[:, :4].sum(1), a[:, 4:].sum(1)], 1)[:, :nd]  # [member, day, y, x]
        di = days.get_loc(s0)
        used = list(range(di, di + nd))
        F1m, Sm = dayf.max(1), dayf.sum(1)
        fc_f1, fc_s = F1m.mean(0), Sm.mean(0)
        oc = np.stack([obs_cell(k) for k in used])
        oc_f1, oc_s = oc.max(0), oc.sum(0)
        pix = np.nan_to_num(np.asarray(P[used[0]:used[-1] + 1]), nan=0.0)
        o4_f1, o4_s = pix.max(0), pix.sum(0)
        rows = ep_rows[r.ep]
        yy, xx = by[rows], bx[rows]
        ly, lx = cy[rows], cx[rows]
        cc = cty_cells[r.cty]
        cyy, cxx = np.divmod(cc, len(glon))
        d = {"ep": np.full(len(rows), r.ep), "bg": rows,
             "o4_f1": o4_f1[yy, xx], "o4_s": o4_s[yy, xx], "oc_f1": oc_f1[ly, lx], "oc_s": oc_s[ly, lx],
             "fc_f1": fc_f1[ly, lx], "fc_s": fc_s[ly, lx],
             "fc_v25": (F1m >= 25).mean(0)[ly, lx], "fc_v50": (F1m >= 50).mean(0)[ly, lx], "fc_sd": F1m.std(0)[ly, lx],
             "fc_nb3": uniform_filter(fc_f1, 3)[ly, lx], "fc_v25_nb3": uniform_filter((F1m >= 25).mean(0), 3)[ly, lx],
             "fc_cty_max": float(fc_f1[cyy, cxx].max()), "fc_cty_mean": float(fc_f1[cyy, cxx].mean()),
             "oc_cty_max": float(oc_f1[cyy, cxx].max()), "oc_cty_mean": float(oc_f1[cyy, cxx].mean()),
             "o4_cty_max": float(o4_f1[by[cty_idx[r.cty]], bx[cty_idx[r.cty]]].max()),
             "o4_cty_mean": float(o4_f1[by[cty_idx[r.cty]], bx[cty_idx[r.cty]]].mean()),
             "nd": nd}
        out.append(pd.DataFrame(d))
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(FEATS)
    print(f"{res.ep.nunique():,} episodes with a reforecast, {dropped:,} without; {len(res):,} rows -> {FEATS}")


def per_episode(ep, y, s, min_bg=10):
    df = pd.DataFrame({"ep": ep, "y": y, "s": s})
    ids, num, den = [], [], []
    for e, grp in df.groupby("ep", sort=False):
        n1 = int(grp.y.sum())
        n0 = len(grp) - n1
        if len(grp) < min_bg or n1 == 0 or n0 == 0:
            continue
        r = rankdata(grp.s.to_numpy())
        auc = (r[grp.y.to_numpy() == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        ids.append(e)
        num.append(auc * n1 * n0)
        den.append(n1 * n0)
    return np.array(ids), np.array(num), np.array(den)


def boot_diff(a, b, reps=REPS, seed=0):
    """Event-clustered paired bootstrap of within-event AUC (pair weighted) for two score sets over the same episodes."""
    ia, na, da = a
    ib, nb, db = b
    common = np.intersect1d(ia, ib)
    ka, kb = np.searchsorted(ia, common), np.searchsorted(ib, common)
    na, da, nb, db = na[ka], da[ka], nb[kb], db[kb]
    rng = np.random.default_rng(seed)
    n = len(common)
    d = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        d.append(nb[i].sum() / db[i].sum() - na[i].sum() / da[i].sum())
    d = np.array(d)
    return float(nb.sum() / db.sum() - na.sum() / da.sum()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
        return
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t = t[t.year <= LAST_YEAR].merge(pd.read_parquet(FEATS), on=["ep", "bg"], how="inner")
    t["log_pol"] = np.log1p(t.n_pol)
    y = t.y.to_numpy().astype(np.float64)
    yr = t.year.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    print(f"{len(t):,} rows, {t.ep.nunique():,} episodes; train {tr.sum():,} ({int(y[tr].sum()):,} pos); "
          f"test {te.sum():,} ({int(y[te].sum()):,} pos)")

    roles = ["f1", "s", "cty_max", "cty_mean"]
    src = {"obs4k": "o4", "obsC": "oc", "fcmean": "fc"}
    cols = {k: [f"{p}_{r}" for r in roles] for k, p in src.items()}
    cols["fcens"] = cols["fcmean"] + ["fc_v25", "fc_v50", "fc_sd", "fc_nb3", "fc_v25_nb3"]
    sets = {"exposure only": BASE, **{k: BASE + v for k, v in cols.items()}}
    ep_te = t.ep.to_numpy()[te]
    pe, scores = {}, {}
    for name, c in sets.items():
        X = t[c].to_numpy(np.float32)
        p, n = exp26.fit_predict(X, y, tr, te, va)
        scores[name] = p
        pe[name] = per_episode(ep_te, y[te], p)
    # train on observed rain, score with the forecast substituted for the same role (no retraining)
    for tag, trn in (("obsC", "obsC"), ("obs4k", "obs4k")):
        cT, cF = BASE + cols[trn], BASE + cols["fcmean"]
        X = t[cT].to_numpy(np.float32)
        Xf = t[cF].to_numpy(np.float32)
        _, n = exp26.fit_predict(X, y, tr, te, va)
        m = exp26.HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300,
                                                 l2_regularization=1.0, early_stopping=False, random_state=0).fit(X[tr], y[tr])
        p = m.predict_proba(Xf[te])[:, 1]
        nm = f"train {tag} -> score fcmean"
        scores[nm] = p
        pe[nm] = per_episode(ep_te, y[te], p)
    # no-training scores
    for nm, col in {"point value obs4k": "o4_f1", "point value obsC": "oc_f1", "point value fcmean": "fc_f1",
                    "vote >=25 mm": "fc_v25"}.items():
        s = t[col].to_numpy()[te]
        scores[nm] = s
        pe[nm] = per_episode(ep_te, y[te], s)

    rows = []
    for nm, v in pe.items():
        s = scores[nm]
        rows.append({"model": nm, "within_AUC": v[1].sum() / v[2].sum(), "episodes": len(v[0]),
                     "AUC": roc_auc_score(y[te], s)})
    tab = pd.DataFrame(rows)
    pairs = [("obs4k", "obsC"), ("obsC", "fcmean"), ("obs4k", "fcmean"), ("fcmean", "fcens"),
             ("obsC", "train obsC -> score fcmean"), ("train obsC -> score fcmean", "fcmean"),
             ("exposure only", "fcmean"), ("exposure only", "obs4k")]
    diffs = {f"{b} - {a_}": boot_diff(pe[a_], pe[b]) for a_, b in pairs}
    pd.set_option("display.width", 200)
    txt = [tab.round(4).to_string(index=False), "", "paired event-clustered bootstrap, within-event AUC (difference, 95% interval)"]
    txt += [f"  {k}: {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in diffs.items()]
    out = "\n".join(txt)
    print(out)
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "flood_forecast_pilot.txt").write_text(out + "\n")
    (AUDIT / "flood_forecast_pilot.json").write_text(json.dumps(
        {"table": tab.to_dict("records"), "diffs": diffs, "n_test_pos": int(y[te].sum())}, indent=1))


if __name__ == "__main__":
    main()
