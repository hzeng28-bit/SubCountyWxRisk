"""Figure of the cadence simulation (script 101): interfaces, product age, skill against lead, skill on the timeline before the hazard T-0, claim opening, the two T-0s.
Input  : audit/cadence_simulation.pkl     Output : docs/figures/cadence_summary.png
Run    : python scripts/102_cadence_figures.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"


def lead_for(delta, ht0_h, cadence, latency):
    t = ht0_h - delta
    k = np.floor((t - latency + 12.0) / cadence)
    return max(24.0 - (-12.0 + k * cadence), 1.0)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = pickle.load(open(wb2.cache_path("audit") / "cadence_simulation.pkl", "rb"))
    L, cur, D = R["LEADS"], R["curves"], R["DELTAS"]
    fig, axs = plt.subplots(2, 3, figsize=(21, 11.5))
    ax = axs[0, 0]
    names = list(R["res_if"])
    med = [np.median(R["res_if"][k]) for k in names]
    p95 = [np.percentile(R["res_if"][k], 95) for k in names]
    y = np.arange(len(names))[::-1]
    ax.barh(y + 0.2, med, 0.38, color="#4c78a8", label="median")
    ax.barh(y - 0.2, p95, 0.38, color="#e8a33d", label="95th percentile")
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace(" (+5-60 s propagation, assumed)", "").replace(" (NWS: within 10 s)", "") for n in names], fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("seconds from warning issuance to the organisation seeing it")
    ax.set_title("Detection delay of a new warning by interface", fontsize=11)
    ax.legend(fontsize=8)
    ax = axs[0, 1]
    an = list(R["ages"])
    ax.barh(np.arange(len(an))[::-1] + 0.2, [np.median(R["ages"][k]) for k in an], 0.38, color="#4c78a8", label="median")
    ax.barh(np.arange(len(an))[::-1] - 0.2, [np.percentile(R["ages"][k], 95) for k in an], 0.38, color="#e8a33d", label="95th percentile")
    ax.set_yticks(np.arange(len(an))[::-1])
    ax.set_yticklabels([n.split(" (")[0] + "\n(" + n.split(" (")[1] for n in an], fontsize=8)
    ax.set_xlabel("age of the freshest product at a random moment (hours)")
    ax.set_title("How old is the newest information", fontsize=11)
    ax.legend(fontsize=8)
    ax = axs[0, 2]
    a2 = ax.twinx()
    for al, st in ((0.5, ":"), (0.75, "-"), (1.0, "--")):
        ax.plot(L, cur[al][:, 0], color="#d1495b", ls=st, marker="o", ms=3, label=f"BG AUC, alpha {al}")
        a2.plot(L, cur[al][:, 1], color="#2a9d8f", ls=st, marker="s", ms=3, label=f"top-5% capture, alpha {al}")
    ax.axhline(0.826, color="#d1495b", lw=0.8, alpha=0.5)
    ax.axhline(R["pol"][0], color="#e8a33d", lw=1)
    ax.text(84, R["pol"][0] + 0.003, "policies only", ha="right", fontsize=8, color="#e8a33d")
    ax.text(84, 0.829, "observed rain (eps 0)", ha="right", fontsize=8, color="#d1495b")
    ax.scatter([60], [0.778], s=90, facecolors="none", edgecolors="k", zorder=5)
    ax.annotate("real GEFS-mode run\n(calibration point)", (60, 0.778), (63, 0.745), fontsize=8, arrowprops={"arrowstyle": "-"})
    ax.set_xlabel("lead of the information, hours before the middle of the 2-day rain window")
    ax.set_ylabel("block-group AUC", color="#d1495b")
    a2.set_ylabel("share of claims in the top-5% block groups", color="#2a9d8f")
    ax.set_title("Synthetic fields: ranking skill against the lead (eps grows as lead^alpha)", fontsize=11)
    ax.legend(fontsize=7, loc="lower left")
    a2.legend(fontsize=7, loc="lower right")
    # timeline before the hazard T-0, both metrics
    ax = axs[1, 0]
    ev = R["events"]
    hh = np.array([(pd.Timestamp(v["ht0"]) - pd.Timestamp(v["t0"])).total_seconds() / 3600 for v in ev.values()])
    a2 = ax.twinx()
    for name, (cad, lat), col in (("6-hourly stream (+5 h)", (6.0, 5.0), "#4c78a8"), ("hourly stream (+2 h)", (1.0, 2.0), "#d1495b")):
        auc = np.array([np.mean([np.interp(lead_for(dl, h_, cad, lat), L, cur[0.75][:, 0]) for h_ in hh]) for dl in D])
        cap = np.array([np.mean([np.interp(lead_for(dl, h_, cad, lat), L, cur[0.75][:, 1]) for h_ in hh]) for dl in D])
        lo = np.array([np.mean([np.interp(lead_for(dl, h_, cad, lat), L, cur[1.0][:, 1]) for h_ in hh]) for dl in D])
        hi = np.array([np.mean([np.interp(lead_for(dl, h_, cad, lat), L, cur[0.5][:, 1]) for h_ in hh]) for dl in D])
        ax.plot(-D, auc, color=col, marker="o", label=f"{name}: BG AUC")
        a2.plot(-D, cap, color=col, ls="--", marker="s", label=f"{name}: top-5% capture")
        a2.fill_between(-D, np.minimum(lo, hi), np.maximum(lo, hi), color=col, alpha=0.12)
    ax.set_xlabel("decision time, hours relative to the hazard T-0 (negative = before)")
    ax.set_ylabel("expected block-group AUC")
    a2.set_ylabel("expected top-5% capture (band: alpha 0.5 - 1)")
    ax.set_title("What the freshest update is worth on the way to the hazard T-0", fontsize=11)
    ax.legend(fontsize=7, loc="upper left")
    a2.legend(fontsize=7, loc="lower right")
    ax = axs[1, 1]
    cc, ct10, ct50, lag = R["claims"]
    days = np.arange(cc.shape[1])
    ax.fill_between(days, np.percentile(cc, 25, 0) * 100, np.percentile(cc, 75, 0) * 100, color="#8d6a9f", alpha=0.3, label="25-75% of events")
    ax.plot(days, np.median(cc, 0) * 100, color="#8d6a9f", lw=2.5, label="median event")
    for x, lab, col in ((-1.3 + 0.5, "forecast ranking\n(T-31 h)", "#2a9d8f"), (0.5 + 54 / 24, "observed-rain ranking\n(T+54 h)", "#d1495b")):
        ax.axvline(x, color=col, lw=1.5, ls="--")
        ax.text(x + 0.1, 8, lab, fontsize=8, color=col)
    ax.set_xlim(-2, 14)
    ax.set_xlabel("days after the first loss day S (00Z of S = 0)")
    ax.set_ylabel("share of the event's NFIP claims already opened (%)")
    ax.set_title("Claims T-0: the insurer's own claim stream starts late (394 events)", fontsize=11)
    ax.legend(fontsize=8, loc="center right")
    ax = axs[1, 2]
    ax.axis("off")
    ax.set_xlim(-40, 130)
    ax.set_ylim(0, 10)
    ax.axis("on")
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.axhline(5, color="#46586a", lw=2)
    marks = [(-31, "forecast ranking\non screen", "#2a9d8f", 6.2), (0, "T0 (12Z of S)\nhazard T-0 median: T0 %+.0f h" % np.median(hh), "#222", 7.6), (54, "observed-rain\nranking (assumed QPE delay)", "#d1495b", 6.2),
             (24 * np.median(ct10) + 18, "claims T-0 (10%% opened)\nmedian day S+%d" % np.median(ct10), "#8d6a9f", 7.6), (24 * np.median(ct50) + 18, "50%% opened\nS+%d" % np.median(ct50), "#8d6a9f", 6.2)]
    for x, lab, col, yy in marks:
        ax.plot([x, x], [4.5, 5.5], color=col, lw=3)
        ax.text(x, yy, lab, ha="center", fontsize=8.5, color=col)
    ax.fill_betweenx([4.2, 4.6], np.percentile(hh, 10), np.percentile(hh, 90), color="#222", alpha=0.25)
    ax.text(np.percentile(hh, 10), 3.6, "hazard T-0 p10-p90 (hours from T0)", fontsize=8)
    ax.fill_betweenx([3.2, 3.6], 24 * np.percentile(ct10, 10) + 18, 24 * np.percentile(ct10, 90) + 18, color="#8d6a9f", alpha=0.3)
    ax.text(24 * np.percentile(ct10, 10) + 18, 2.6, "claims T-0 p10-p90", fontsize=8)
    ax.set_xlabel("hours from T0 (day-resolution claim dates are placed at the end of the day, 06Z)")
    ax.set_title("The two T-0s on one axis (medians)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "cadence_summary.png", dpi=105)


if __name__ == "__main__":
    main()
