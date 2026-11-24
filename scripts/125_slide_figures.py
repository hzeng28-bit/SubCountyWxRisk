"""Three presentation figures sized for the CSE 472 Project 2 slides (4.0 in wide, large type), redrawn from stored results.

s2_auc.png   block-group AUC by model, observed rain 2017-2023 (audit/task_scorecard.csv, script 96)
s3_map.png   Texas county claim density with the small screened USGS gauges (gis/*.geojson, script 122)
s4_verify.png claim rate by observed gauge flow, forecast alert x flow confirmation (RESULTS stage 64, script 119 log)
Output : docs/figures/slides/*.png  (300 dpi)
Run    : uv run python scripts/125_slide_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = wb2.DATA if hasattr(wb2, "DATA") else Path.home() / "wxrisk_data"
FIG = ROOT / "docs" / "figures" / "slides"
FIG.mkdir(parents=True, exist_ok=True)
INK, MUTED = "#1f2937", "#6b7280"
COL = {"blind guess": "#9aa0a6", "rain only": "#8d6a9f", "policies only": "#e8a33d", "boosting": "#4c78a8", "CNN (observed)": "#d1495b"}


def base():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_auc():
    plt = base()
    t = pd.read_csv(DATA / "audit" / "task_scorecard.csv")
    t = t[t.run.str.startswith("A") & t.metric.str.startswith("BG AUC")].set_index("score")
    order = ["blind guess", "rain only", "policies only", "boosting", "CNN (observed)"]
    labels = ["Blind guess", "Rain only", "Policies only", "Boosting\n(rain + policies)", "CNN"]
    fig, ax = plt.subplots(figsize=(4.0, 3.3))
    y = np.arange(len(order))[::-1]
    for yi, k in zip(y, order):
        p, lo, hi = t.loc[k, ["point", "lo", "hi"]]
        ax.barh(yi, p, color=COL[k], height=0.62, xerr=[[p - lo], [hi - p]], error_kw={"ecolor": INK, "lw": 1, "capsize": 2})
        ax.text(0.02, yi, f"{p:.2f}", va="center", ha="left", color="white", fontweight="bold", fontsize=11)
    ax.set_yticks(y, labels, fontsize=10)
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("AUC (0.5 = blind guess)")
    ax.set_title("Texas test events 2017-2023", fontsize=11, loc="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "s2_auc.png", dpi=300)


def fig_map():
    plt = base()
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon as MPoly

    C = json.load(open(DATA / "gis" / "tx_counties_claims_gauges.geojson"))["features"]
    G = json.load(open(DATA / "gis" / "tx_gauges.geojson"))["features"]
    patches, vals = [], []
    for f in C:
        g, v = f["geometry"], max(f["properties"]["claims_per_1000km2"], 0.5)
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for p in polys:
            patches.append(MPoly(np.asarray(p[0]), closed=True))
            vals.append(v)
    fig, ax = plt.subplots(figsize=(4.0, 2.95))
    coll = PatchCollection(patches, array=np.array(vals), cmap="YlOrRd", norm=LogNorm(0.5, max(vals)), edgecolor="#b0b0b0", linewidth=0.2)
    ax.add_collection(coll)
    pts = np.array([f["geometry"]["coordinates"][:2] for f in G if "small screened" in str(f["properties"]["class"])])
    ax.scatter(pts[:, 0], pts[:, 1], s=7, c="#0b3d91", edgecolor="white", linewidth=0.25, zorder=4, label=f"small gauge (n={len(pts)})")
    ax.set_xlim(-106.8, -93.3)
    ax.set_ylim(25.6, 36.7)
    ax.set_aspect(1 / np.cos(np.radians(31)))
    ax.axis("off")
    cb = fig.colorbar(coll, ax=ax, fraction=0.04, pad=0.0, shrink=0.8)
    cb.set_label("claims per 1000 km$^2$", fontsize=9)
    cb.ax.tick_params(labelsize=9)
    ax.legend(loc="lower left", fontsize=9, frameon=False, handletextpad=0.2, borderaxespad=0.2)
    fig.subplots_adjust(left=0.0, right=0.86, top=1.0, bottom=0.0)
    fig.savefig(FIG / "s3_map.png", dpi=300)


def fig_verify():
    plt = base()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    # RESULTS stage 64, V4 table (test years 2017-2019): claim rate of gauge-day units by flow confirmation
    groups = [("Flow below\n99th percentile", [(0.8, "no alert", "n=27,865"), (1.3, "alert", "n=2,284")], "#9aa0a6"),
              ("Flow at or above\n99th percentile", [(12.0, "no alert", "n=1,622"), (10.7, "alert", "n=1,003")], "#1f5aa6")]
    x0 = 0
    ticks, tl = [], []
    for name, bars, c in groups:
        for i, (v, lab, n) in enumerate(bars):
            ax.bar(x0 + i * 0.9, v, 0.8, color=c)
            ax.text(x0 + i * 0.9, v + 0.3, f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
            ax.text(x0 + i * 0.9, -0.4, f"{lab}", ha="center", va="top", fontsize=9, color=MUTED)
        ticks.append(x0 + 0.45)
        tl.append(name)
        x0 += 2.4
    ax.set_xticks(ticks, tl, fontsize=10)
    ax.tick_params(axis="x", pad=16, length=0)
    ax.set_ylim(0, 15)
    ax.set_ylabel("gauge-days with a claim (%)", fontsize=10)
    ax.set_title("Gauge flow vs claims", fontsize=11, loc="left", color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG / "s4_verify.png", dpi=300)


if __name__ == "__main__":
    fig_auc()
    fig_map()
    fig_verify()
    for f in sorted(FIG.glob("s*.png")):
        print(f)
