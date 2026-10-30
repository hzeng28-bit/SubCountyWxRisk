"""Does the forecast model's gain over the exposure ranking grow with the size of the event? (stage 39)

Test rows and predictions of script 77 (every block group of every forecast day 2017-2019, ex-ante features). Days are stratified two ways:
  observed size   number of claimed block groups of the day (1-2, 3-9, 10-29, >= 30): descriptive, uses the outcome
  forecast size   the day's largest forecast county peak (ensemble mean of the 2-day peak): known at issue time (< 10, 10-25, 25-50, >= 50 mm)
Per stratum, per day: cross-county AUC of block groups, top-5% capture of the claim count, NDCG at 5% with paid amount, county AUC, top-10% county capture;
forecast model, observed-rain model (reference) and policies only; differences to policies only with 500 bootstrap draws over days (scripts/75 helpers).
Inputs : pilot/ranking_test_rows.parquet (script 77)
Output : $WXRISK_DATA/audit/ranking_by_event_size.txt
Run    : python scripts/78_ranking_by_event_size.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

spec = importlib.util.spec_from_file_location("e75", Path(__file__).with_name("75_task_metrics.py"))
e75 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e75)
OUT: list[str] = []
e75.OUT = OUT
SCORES = {"forecast model": ("p_fc", "sum"), "observed-rain model": ("p_oc", "sum"), "policies only": ("n_pol", "sum")}
METRICS = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"),
           ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def stratum(name: str, T: pd.DataFrame, days: pd.Index) -> None:
    sub = T[T.sysid.isin(days)]
    tabs = e75.event_table(sub, SCORES)
    n_ev = int(tabs["forecast model"].bg_auc.notna().sum())
    share = sub.n.sum() / T.n.sum()
    say(f"\n  {name}: {len(days)} days, {n_ev} scored, {share:.1%} of the test claim count; forecast model / observed-rain model / policies only (point values), "
        f"then forecast model minus policies only [95%]")
    cells = []
    for col, w, lab in METRICS:
        pt = {k: e75.pooled(t, col, w) for k, t in tabs.items()}
        _, d = e75.boot({"forecast model": tabs["forecast model"], "policies only": tabs["policies only"]}, col, w, ref="policies only")
        _, d2 = e75.boot({"observed-rain model": tabs["observed-rain model"], "policies only": tabs["policies only"]}, col, w, ref="policies only")
        cells.append(f"    {lab:24s} {pt['forecast model']:.3f} / {pt['observed-rain model']:.3f} / {pt['policies only']:.3f}   "
                     f"forecast {d['forecast model'][0]:+.3f} [{d['forecast model'][1]:+.3f}, {d['forecast model'][2]:+.3f}]   "
                     f"observed {d2['observed-rain model'][0]:+.3f} [{d2['observed-rain model'][1]:+.3f}, {d2['observed-rain model'][2]:+.3f}]")
    say("\n".join(cells))


def main() -> None:
    T = pd.read_parquet(fp.PILOT / "ranking_test_rows.parquet")
    say(f"test rows {len(T):,}, days {T.sysid.nunique()}, claimed block groups {int((T.n > 0).sum()):,}")
    g = T.groupby("sysid")
    day = pd.DataFrame({"claimed": g.n.apply(lambda s: int((s > 0).sum())), "fcmax": g.fc_cty_max.max()})
    say("days by observed size (claimed block groups): " + ", ".join(f"{k}: {v}" for k, v in pd.cut(day.claimed, [-1, 0, 2, 9, 29, 1e9], labels=["0", "1-2", "3-9", "10-29", ">= 30"]).value_counts().sort_index().items()))
    say("== A. strata by observed event size")
    for lab, lo, hi in (("1-2 claimed block groups", 1, 2), ("3-9", 3, 9), ("10-29", 10, 29), (">= 30", 30, 10 ** 9)):
        stratum(lab, T, day.index[(day.claimed >= lo) & (day.claimed <= hi)])
    say("\n== B. strata by the day's largest forecast county peak (known at issue time)")
    for lab, lo, hi in (("< 10 mm", 0, 10), ("10-25 mm", 10, 25), ("25-50 mm", 25, 50), (">= 50 mm", 50, 1e9)):
        stratum(lab, T, day.index[(day.fcmax >= lo) & (day.fcmax < hi)])
    p = wb2.cache_path("audit") / "ranking_by_event_size.txt"
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
