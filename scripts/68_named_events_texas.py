"""Named Texas flood events from the meteorological literature, seen through the pilot table (stage 30).

Each event window is taken from NWS / NHC reports; episodes of table_notc.parquet whose start lies in the window are pooled. Per event:
county rain-peak distribution, positives, claimed block groups, and the within-event AUC (pair weighted over the county episodes) of two
single scores: cell rain (what the rain says about WHERE in a county) and policies in the block group (what exposure says).
Then, over all 2,570 episodes with an AUC, the within-event AUC of the two scores by the spread of rain inside the county (dispersion_bins), and a test of
the rain-spread features in the stage 8 L2 model (spread_feature_test).
Inputs : table_notc.parquet, flood_pilot_table.parquet, clean NFIP claims, tc_flag_800_2.npy
Run    : python scripts/68_named_events_texas.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e64, e48 = _load("e64", "64_label_scope_relaxation.py"), _load("e48", "48_flood_forecast_pilot.py")
EVENTS = {
    "Memorial Day 2015 (MCS, Hill Country / Houston)": ("2015-05-23", "2015-05-27", "convective"),
    "October 2015 (Patricia remnants + front)": ("2015-10-22", "2015-10-26", "tropical moisture"),
    "Tax Day 2016 (MCS, Houston)": ("2016-04-17", "2016-04-19", "convective"),
    "Harvey 2017 (stalled hurricane)": ("2017-08-25", "2017-09-02", "tropical, stalled"),
    "Imelda 2019 (slow tropical storm, training bands)": ("2019-09-17", "2019-09-21", "tropical, training"),
}


def main() -> None:
    t = pd.read_parquet(fp.PILOT / "table_notc.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    cl = e64.all_claims()
    cl = cl[cl.cause.isin(["2", "3", "4"])]
    t["pos"] = e64.counts(t, cl) > 0
    rows = []
    for name, (a, b, kind) in EVENTS.items():
        m = (t.start >= a) & (t.start <= b)
        d = t[m]
        ep = d.drop_duplicates("ep")
        out = {"event": name, "type": kind, "county episodes": ep.ep.nunique(), "counties with a positive": int(d.groupby("ep").pos.any().sum()),
               "block groups": len(d), "claimed block groups": int(d.pos.sum()),
               "county peak mm p50/p90/max": "{:.0f}/{:.0f}/{:.0f}".format(*ep.cty_f1_max.quantile([.5, .9, 1]).to_numpy()),
               "share of block groups claimed": d.pos.mean()}
        for lab, col in (("AUC cell rain", "bg_f1"), ("AUC policies", "log_pol")):
            r = e48.per_episode(d.ep.to_numpy(), d.pos.to_numpy().astype(float), d[col].to_numpy())
            out[lab] = r[1].sum() / r[2].sum() if len(r[0]) else np.nan
            out["episodes with an AUC"] = len(r[0])
        rows.append(out)
    pd.set_option("display.width", 250)
    df = pd.DataFrame(rows).round(3); df["event"] = df.event.str.split(" \\(").str[0]; print(df.drop(columns=["type"]).to_string(index=False))



def dispersion_bins() -> None:
    """All episodes: does the spatial spread of rain inside the county decide which score ranks block groups better?"""
    from scipy.stats import rankdata

    t = pd.read_parquet(fp.PILOT / "table_notc.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    cl = e64.all_claims()
    t["pos"] = e64.counts(t, cl[cl.cause.isin(["2", "3", "4"])]) > 0
    g = t.groupby("ep")
    disp = (g.bg_f1.std() / g.bg_f1.mean().clip(lower=1)).rename("cv")
    res = []
    for ep, d in g:
        n1 = int(d.pos.sum())
        n0 = len(d) - n1
        if len(d) < 10 or n1 == 0 or n0 == 0:
            continue
        row = {"ep": ep, "pairs": n1 * n0, "cv": disp[ep], "peak": d.cty_f1_max.iloc[0]}
        for lab, col in (("rain", "bg_f1"), ("exposure", "log_pol")):
            r = rankdata(d[col].to_numpy())
            row[lab] = (r[d.pos.to_numpy()].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        res.append(row)
    r = pd.DataFrame(res)
    r["bin"] = pd.qcut(r.cv, 4, labels=["lowest spread", "low-mid", "high-mid", "highest spread"])
    out = r.groupby("bin", observed=True).apply(lambda z: pd.Series({
        "episodes": len(z), "cv median": z.cv.median(), "AUC cell rain": np.average(z.rain, weights=z.pairs),
        "AUC policies": np.average(z.exposure, weights=z.pairs)}), include_groups=False)
    print("\nepisodes with an AUC:", len(r), "\n", out.round(3).to_string())



def spread_feature_test() -> None:
    """Add the within-episode spread of rain (std and CV of block-group cell rain, known when the rain is) to the stage 8 L2 model."""
    e26 = _load("e26", "26_flood_pilot_experiment.py")
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    g = t.groupby("ep").bg_f1
    t["f1_std"] = g.transform("std").fillna(0)
    t["f1_cv"] = t.f1_std / g.transform("mean").clip(lower=1)
    t["f1_range"] = g.transform("max") - g.transform("min")
    base = e26.COUNTY + e26.EXPO + e26.CELL
    y, yr, ep = t.y.to_numpy().astype(float), t.year.to_numpy(), t.ep.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    va = (yr >= 2014) & tr
    per = {}
    for name, cols in {"L2 base": base, "+ rain spread (std, cv, range)": base + ["f1_std", "f1_cv", "f1_range"]}.items():
        p, n = e26.fit_predict(t[cols].to_numpy(np.float32), y, tr, te, va)
        per[name] = e48.per_episode(ep[te], y[te], p)
        print(f"{name}: within-event AUC {per[name][1].sum() / per[name][2].sum():.4f} ({n} iterations)")
    d = e48.boot_diff(per["L2 base"], per["+ rain spread (std, cv, range)"])
    print("difference {:+.4f} [{:+.4f}, {:+.4f}]".format(*d))


if __name__ == "__main__":
    main()
    dispersion_bins()
    spread_feature_test()
