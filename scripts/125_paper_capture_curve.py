"""Capture curve of the task scorecard redrawn for the paper (RESULTS stage 55), x axis in plain percent instead of 10^k.

Reuses the observed-rain rows and scores of script 96 (A) and its `gain_curves`; only the figure differs.
Output : Overleaf/figs/fig_scorecard_paper.png (left: capture curve, right: bars from audit/task_scorecard.csv)
Run    : python scripts/125_paper_capture_curve.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    s96, e85 = _load("s96", "96_task_scorecard.py"), _load("e85", "85_state_group_baseline.py")
    T, scores, model = s96.build_observed(e85)
    gain = s96.gain_curves(T, scores)
    colr = {"blind guess": "#9aa0a6", "policies only": "#e8a33d", "rain only": "#8d6a9f", "boosting": "#4c78a8", "CNN (observed)": "#d1495b"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.05]})
    for n, v in gain.items():
        ax.plot(s96.KGRID * 100, v * 100, marker="o", ms=3, color=colr[n], label="4 km CNN" if n == model else n,
                lw=2.2 if n == model else 1.3, ls="--" if n == "blind guess" else "-")
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20, 40, 100]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}%"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda x, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}%"))
    ax.set_xlabel("share of block groups reviewed (highest score first)")
    ax.set_ylabel("share of the day's claims found")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    import pandas as pd
    df = pd.read_csv(s96.wb2.cache_path("audit") / "task_scorecard.csv")
    df = df[df.run.str.startswith("A ")]
    mets = [("BG AUC (pair-weighted)", "BG AUC"), ("BG capture of claim count, top 5%", "BG top-5%\ncapture"),
            ("county AUC (pair-weighted)", "County AUC"), ("county capture of claim count, top 10%", "County top-10%\ncapture")]
    print(sorted(df.metric.unique()))
    names = ["blind guess", "policies only", "rain only", "boosting", "CNN (observed)"]
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        pts, lo, hi = [], [], []
        for m, _ in mets:
            r = df[(df.metric == m) & (df.score == n)].iloc[0]
            pts.append(r.point); lo.append(r.point - r.lo); hi.append(r.hi - r.point)
        bx.bar(np.arange(len(mets)) + i * w, pts, w, yerr=[lo, hi], color=colr[n], capsize=2, label="4 km CNN" if n == model else n)
    bx.set_xticks(np.arange(len(mets)) + 0.4 - w / 2)
    bx.set_xticklabels([l for _, l in mets], fontsize=9)
    bx.axhline(0.5, color="#9aa0a6", lw=0.8)
    bx.set_ylim(0, 1)
    bx.set_ylabel("score (95% interval over days)")
    bx.grid(axis="y", alpha=0.3)
    ax.set_title("(a) Claims found for a fixed review budget", fontsize=10)
    bx.set_title("(b) Ranking metrics", fontsize=10)
    fig.tight_layout()
    out = ROOT / "Overleaf" / "figs" / "fig_scorecard_paper.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
