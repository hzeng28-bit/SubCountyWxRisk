"""Figures of the scenario report from the results of script 98 (audit/scenario_analysis.pkl): lead time of the model ranking over the NWS warnings, accuracy inside the
warned counties, and the display test (points vs interpolation).
Output : docs/figures/scenario_summary.png
Run    : python scripts/100_scenario_figures.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
READY = -31.0  # hours from T0 (GEFS 00Z of S-1 + 5 h)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pickle.load(open(wb2.cache_path("audit") / "scenario_analysis.pkl", "rb"))
    fk, ok = "forecast mode 2017-2019", "observed rain 2017-2023"
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.2))
    ax = axs[0]
    for key, col in ((fk, "#2a9d8f"), (ok, "#d1495b")):
        h = out[key][0] - READY
        ax.hist(h, bins=np.arange(-48, 90, 6), alpha=0.55, color=col, label=f"event days {key[-9:]}: median {np.median(h):.0f} h, p10 {np.percentile(h, 10):.0f} h, n={len(h)}")
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("hours from 'ranking ready' (T0-31 h) to the first warning for an affected county")
    ax.set_ylabel("event days with >= 10 claimed block groups")
    ax.set_title("How early the ranking is on screen (positive = before the warning)", fontsize=11)
    ax.legend(fontsize=8)
    ax = axs[1]
    labs = ["BG AUC", "BG top-5%\ncapture", "BG top-10%\ncapture", "county AUC"]
    idx = [0, 1, 2, 4]
    w = 0.2
    series = [(fk, "policies only", "#e8a33d"), (fk, "CNN (forecast)", "#2a9d8f"), (ok, "boosting", "#4c78a8"), (ok, "CNN (observed)", "#d1495b")]
    for i, (key, name, col) in enumerate(series):
        rows = out[key][2]
        pts = [rows[j][1][name][0] for j in idx]
        lo = [rows[j][1][name][0] - rows[j][1][name][1] for j in idx]
        hi = [rows[j][1][name][2] - rows[j][1][name][0] for j in idx]
        ax.bar(np.arange(4) + i * w, pts, w, yerr=[lo, hi], color=col, capsize=2, label=f"{name} ({'forecast 2017-19' if key == fk else 'observed 2017-23'})")
    ax.set_xticks(np.arange(4) + 1.5 * w)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy inside the counties that hold a warning (95% over event days)", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axs[2]
    names = ["points", "k5", "k10", "k20", "k40", "pol"]
    lab = ["points\nas they are", "smoothed\n5 km", "10 km", "20 km", "40 km", "policies\nonly"]
    for i, (key, col) in enumerate(((fk, "#2a9d8f"), (ok, "#d1495b"))):
        res = out[key][3][0]
        auc = [np.nanmean(np.array(res[n])[:, 0]) for n in names]
        ax.bar(np.arange(len(names)) + i * 0.4, auc, 0.4, color=col, label=key)
        for x, v in zip(np.arange(len(names)) + i * 0.4, auc, strict=True):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(names)) + 0.2)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel("block-group AUC (150 sampled event days)")
    ax.set_title("Showing the result: points vs interpolated surface", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "scenario_summary.png", dpi=110)


if __name__ == "__main__":
    main()
