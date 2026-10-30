"""Do ensemble extremes lift the forecast ranking in big events? (stage 40)

Stage 39: the forecast model recovers about a fifth of the observed-rain gain over exposure in the largest events, where the five-member MEAN underestimates the rain
(about 0.37 of the observed county peak). Here the all-days model of script 77 gets ensemble-extreme features:
  mean model       BASE + ensemble-mean peak and sum, county peak and mean of the mean            (script 77)
  + extremes       + member maximum, share of members >= 25 mm and >= 50 mm, member spread, 3x3 mean of the 25 mm share at the cell;
                   county maximum of the member maximum, county maximum and mean of the 25 mm share, county maximum of the 50 mm share
  observed-rain    reference, as in script 77
Same rows, split, subsampling and metrics (scripts 75 / 77 / 78); compared overall and by observed event size and by the day's largest forecast county peak,
extremes minus mean model and extremes minus policies only, 500 bootstrap draws over days.
Inputs : as script 77 (GEFS archive, gridMET, exposure and claims)
Output : $WXRISK_DATA/audit/ensemble_extremes.txt, pilot/ranking_test_rows_ext.parquet
Run    : python scripts/79_ensemble_extremes.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e77 = _load("e77", "77_forecast_ranking_all_counties.py")
e75 = e77.e75
OUT: list[str] = []
e75.OUT = OUT
SCORES = {"mean model": ("p_mean", "sum"), "mean + extremes": ("p_ext", "sum"), "observed-rain model": ("p_obs", "sum"), "policies only": ("n_pol", "sum")}
METRICS = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"),
           ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def stratum(name: str, T: pd.DataFrame, days) -> None:
    sub = T if days is None else T[T.sysid.isin(days)]
    tabs = e75.event_table(sub, SCORES)
    n_ev = int(tabs["mean model"].bg_auc.notna().sum())
    say(f"\n  {name}: {sub.sysid.nunique()} days, {n_ev} scored, {sub.n.sum() / T.n.sum():.1%} of the test claim count; point values mean / +extremes / observed / policies")
    for col, w, lab in METRICS:
        pt = {k: e75.pooled(t, col, w) for k, t in tabs.items()}
        _, d1 = e75.boot({"mean + extremes": tabs["mean + extremes"], "mean model": tabs["mean model"]}, col, w, ref="mean model")
        _, d2 = e75.boot({"mean + extremes": tabs["mean + extremes"], "policies only": tabs["policies only"]}, col, w, ref="policies only")
        say(f"    {lab:24s} {pt['mean model']:.3f} / {pt['mean + extremes']:.3f} / {pt['observed-rain model']:.3f} / {pt['policies only']:.3f}   "
            f"extremes - mean {d1['mean + extremes'][0]:+.3f} [{d1['mean + extremes'][1]:+.3f}, {d1['mean + extremes'][2]:+.3f}]   "
            f"extremes - policies {d2['mean + extremes'][0]:+.3f} [{d2['mean + extremes'][1]:+.3f}, {d2['mean + extremes'][2]:+.3f}]")


def main() -> None:
    df = e77.build()
    y = (df.n.to_numpy() > 0).astype(np.float32)
    yr = df.year.to_numpy()
    tr, va, te = yr <= 2016, (yr >= 2014) & (yr <= 2016), yr >= 2017
    rng = np.random.default_rng(0)
    keep = tr & ((y == 1) | (rng.random(len(df)) < 0.1))
    w = np.where(y == 1, 1.0, 10.0).astype(np.float32)
    base = ["month", "cty_nbg", "log_pol", "sfha_share", "single_share", "mean_cov"]
    mean = base + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"]
    ext = mean + ["fc_mx", "fc_v25", "fc_v50", "fc_sd", "fc_nb3", "fc_cty_mx", "fc_cty_v25", "fc_cty_v25mean", "fc_cty_v50"]
    obs = base + ["oc_f1", "oc_s", "oc_cty_max", "oc_cty_mean"]
    say(f"rows {len(df):,}, forecast days {df.day.nunique():,}; training rows {int(keep.sum()):,}; test rows {int(te.sum()):,}")
    T = df.loc[te, ["sysid", "day", "year", "bg", "cty", "n", "paid", "n_pol", "fc_f1", "fc_cty_max"]].reset_index(drop=True)
    fit_rows, val_rows = keep & ~va, keep & va
    for col, cols in (("p_mean", mean), ("p_ext", ext), ("p_obs", obs)):
        mk = lambda n: HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300, l2_regularization=1.0,  # noqa: E731
                                                      early_stopping=False, random_state=0)
        Xf, Xv = df.loc[fit_rows, cols].to_numpy(np.float32), df.loc[val_rows, cols].to_numpy(np.float32)
        m = mk(300).fit(Xf, y[fit_rows], sample_weight=w[fit_rows])
        sc = [np.average((p[:, 1] - y[val_rows]) ** 2, weights=w[val_rows]) for p in m.staged_predict_proba(Xv)]
        n_it = int(np.argmin(sc)) + 1
        del Xf, Xv
        mdl = mk(n_it).fit(df.loc[keep, cols].to_numpy(np.float32), y[keep], sample_weight=w[keep])
        T[col] = mdl.predict_proba(df.loc[te, cols].to_numpy(np.float32))[:, 1].astype(np.float32)
        say(f"fitted {col} ({len(cols)} features, {n_it} iterations)")
    T.to_parquet(fp.PILOT / "ranking_test_rows_ext.parquet")
    g = T.groupby("sysid")
    day = pd.DataFrame({"claimed": g.n.apply(lambda s: int((s > 0).sum())), "fcmax": g.fc_cty_max.max()})
    say("\n== overall, all counties of a day (point values: mean model / + extremes / observed-rain / policies only)")
    stratum("all test days", T, None)
    say("\n== by observed size of the day (claimed block groups)")
    for lab, lo, hi in (("1-2", 1, 2), ("3-9", 3, 9), ("10-29", 10, 29), (">= 30", 30, 10 ** 9)):
        stratum(lab, T, day.index[(day.claimed >= lo) & (day.claimed <= hi)])
    say("\n== by the day's largest forecast county peak of the ensemble mean (known at issue time)")
    for lab, lo, hi in (("< 10 mm", 0, 10), ("10-25 mm", 10, 25), ("25-50 mm", 25, 50), (">= 50 mm", 50, 1e9)):
        stratum(lab, T, day.index[(day.fcmax >= lo) & (day.fcmax < hi)])
    pth = wb2.cache_path("audit") / "ensemble_extremes.txt"
    pth.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
