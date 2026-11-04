"""T-0 countdown replay of one real Texas event for the scenario report (RESULTS stage 56): what a forecaster / county command post would have seen, clock by clock.

Retrospective replay, not a live product: the model outputs are the stored predictions of the evaluation (forecast mode `fc + expo` from the GEFS 00Z run of S-1, and the
observed-rain model `rain + expo` for the later update), the NWS Flash Flood Warnings are the real ones of that period, the claims are the real NFIP claims of [S, S+3].
Clock: T0 = 12Z of day S (the start of the 2-day rain window the model reads). Timeline assumptions, all labelled on screen:
  - GEFS output of the 00Z cycle of S-1 is available 5 h after initialisation (T0-31 h); the exact publication delay was not verified (sensitivity 4-8 h).
  - one pipeline run costs what script 97 measured (download of 5 members 9 s, feature preparation 3 ms, network 30 ms on CPU); the map render time is measured here.
  - the observed-rain update needs the 2-day rain window plus a daily QPE product: T0+48 h + 6 h (assumed; validated only with daily gridMET, not with MRMS / Stage IV).
Panels: county ranking (sum of block-group probabilities), discrete block-group points, 10 km kernel-interpolated surface (context only), clock, timeline, dispatch list, cost box.
Inputs : script 96 builders (predictions), audit/ffw_tx.parquet (script 98), census county file
Output : docs/media/countdown_<event>.mp4, docs/media/countdown_<event>_final.png
Run    : python scripts/99_countdown_video.py --event 20190604 ; python scripts/99_countdown_video.py --event 20190609
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

MEDIA = Path(__file__).resolve().parents[1] / "docs" / "media"
GEFS_DELAY_H, QPE_DELAY_H = 5.0, 6.0
T_READY = -36.0 + GEFS_DELAY_H  # hours from T0
T_OBS = 48.0 + QPE_DELAY_H
T_END = 120.0
FPS = 24
# (video second, clock hours from T0)
KEYS = [(0, -36), (3, -36), (6.5, T_READY), (12, T_READY), (30, 0), (33, 0), (47, T_OBS), (51, T_OBS), (58, T_END), (66, T_END)]
READY_SEC, END_SEC = 12.0, KEYS[-1][0]
BG, FG, MUTED, ACC = "#0f1720", "#e8edf2", "#8b98a5", "#f2c14e"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def clock_at(sec: float) -> float:
    return float(np.interp(sec, [k[0] for k in KEYS], [k[1] for k in KEYS]))


def fmt_clock(h: float) -> str:
    s = "-" if h < 0 else "+"
    m = int(round(abs(h) * 60))
    return f"T{s}{m // 60:02d}:{m % 60:02d}"


def counties():
    import shapefile
    from shapely.geometry import shape
    from shapely.ops import unary_union

    z = zipfile.ZipFile(wb2.cache_path("labels_extra") / "census" / "cb_2020_us_county_20m.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")), dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    parts, nm, geo = {}, {}, []
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        if rec["STATEFP"] != "48":
            continue
        code = "48" + rec["COUNTYFP"]
        pts = np.array(sr.shape.points)
        idx = list(sr.shape.parts) + [len(pts)]
        parts[code] = [pts[a:b] for a, b in zip(idx[:-1], idx[1:], strict=True)]
        nm[code] = rec["NAME"]
        geo.append(shape(sr.shape.__geo_interface__))
    return parts, nm, unary_union(geo)


def surface(lon, lat, p, tx, grid):
    """10 km Gaussian-kernel surface of the block-group probabilities on a lon / lat grid, masked to Texas and to 25 km around a block group."""
    import shapely
    from scipy.spatial import cKDTree

    glon, glat = grid
    X, Y = np.meshgrid(glon, glat)
    k = lambda lo, la: np.column_stack([(lo + 100) * 111.32 * np.cos(np.deg2rad(31)), (la - 31) * 110.57])  # noqa: E731
    tree = cKDTree(k(lon, lat))
    d, j = tree.query(k(X.ravel(), Y.ravel()), k=20)
    w = np.exp(-0.5 * (d / 5.0) ** 2) + 1e-12
    z = (w * p[j]).sum(1) / w.sum(1)
    ok = (d[:, 0] < 25) & shapely.contains_xy(tx, X.ravel(), Y.ravel())
    z[~ok] = np.nan
    return z.reshape(X.shape)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, help="event day S as YYYYMMDD (forecast-mode test days 2017-2019)")
    ap.add_argument("--still", action="store_true", help="only write the final frame")
    a = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    s96, s98 = _load("s96", "96_task_scorecard.py"), _load("s98", "98_scenario_analysis.py")
    T, _, _ = s96.build_forecast()
    g = T[T.sysid == a.event].reset_index(drop=True)
    del T
    assert len(g), "event day not in the forecast-mode rows"
    e75 = _load("e75", "75_task_metrics.py")
    yb = (g.n.to_numpy() > 0).astype(int)
    auc_f = e75.pair_auc(g.fc.to_numpy(), yb)[0]
    auc_o = e75.pair_auc(g.obs.to_numpy(), yb)[0]
    top = e75.top_idx(g.fc.to_numpy(), max(int(np.ceil(0.05 * len(g))), 1))
    cap_f = g.n.to_numpy()[top].sum() / g.n.sum()
    top = e75.top_idx(g.obs.to_numpy(), max(int(np.ceil(0.05 * len(g))), 1))
    cap_o = g.n.to_numpy()[top].sum() / g.n.sum()
    top = e75.top_idx(g.n_pol.to_numpy().astype(float), max(int(np.ceil(0.05 * len(g))), 1))
    cap_p = g.n.to_numpy()[top].sum() / g.n.sum()
    print(f"event {a.event}: rows {len(g)}, claimed BGs {int(yb.sum())}; AUC forecast {auc_f:.3f}, observed {auc_o:.3f}; top-5% capture forecast {cap_f:.3f}, observed {cap_o:.3f}, policies {cap_p:.3f}")

    S = np.load(wb2.cache_path("cnn") / "static.npz", allow_pickle=True)
    lon_all, lat_all = S["lon4"][S["b4_x"]], S["lat4"][S["b4_y"]]
    lon, lat = lon_all[g.bg.to_numpy()], lat_all[g.bg.to_numpy()]
    parts, names, tx = counties()
    grid = (np.arange(-106.8, -93.4, 0.08), np.arange(25.8, 36.7, 0.08))
    t_ = time.perf_counter()
    Zf = surface(lon, lat, g.fc.to_numpy().astype(float), tx, grid)
    Zo = surface(lon, lat, g.obs.to_numpy().astype(float), tx, grid)
    print(f"surfaces {time.perf_counter() - t_:.1f} s")
    cs_f = g.groupby("cty").fc.sum()
    cs_o = g.groupby("cty").obs.sum()
    claimed_c = set(g[g.n > 0].cty)
    ncl_c = g[g.n > 0].groupby("cty").n.sum()

    ffw = pd.read_parquet(wb2.cache_path("audit") / "ffw_tx.parquet")
    t0 = pd.Timestamp(a.event) + pd.Timedelta(12, "h")
    ffw["h"] = (ffw.issued - t0).dt.total_seconds() / 3600
    ev = ffw[(ffw.h >= -36) & (ffw.h <= T_END)].sort_values("h")
    first_aff = min((r.h for r in ev.itertuples() if claimed_c & set(r.counties.split(","))), default=None)
    print(f"warnings in the window: {len(ev)}; first covering an affected county at T{first_aff:+.1f} h" if first_aff is not None else f"warnings in the window: {len(ev)}; none covers an affected county")

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    ax = [fig.add_axes([0.01 + 0.325 * i, 0.38, 0.32, 0.56], facecolor=BG) for i in range(3)]
    ext = [-106.8, -93.4, 25.8, 36.7]
    allp = [p for c in parts.values() for p in c]
    codes = [c for c, ps in parts.items() for _ in ps]
    rank_cmap = plt.get_cmap("YlOrRd")
    pcs = []
    for i, x in enumerate(ax):
        x.set_xlim(ext[0], ext[1]); x.set_ylim(ext[2], ext[3]); x.set_aspect("equal"); x.axis("off")
        pc = PolyCollection(allp, facecolors="none" if i else "#1b2733", edgecolors="#46586a", linewidths=0.35, zorder=1)
        x.add_collection(pc)
        pcs.append(pc)
    norm = LogNorm(1e-4, 0.5)
    order = np.argsort(g.fc.to_numpy())
    sc = ax[1].scatter(lon[order], lat[order], c=g.fc.to_numpy()[order], s=3.5, cmap="viridis", norm=norm, zorder=3, visible=False)
    im = ax[2].imshow(np.where(np.isnan(Zf), np.nan, Zf), extent=[grid[0][0] - 0.04, grid[0][-1] + 0.04, grid[1][0] - 0.04, grid[1][-1] + 0.04], origin="lower", cmap="viridis", norm=norm,
                      zorder=2, visible=False, interpolation="bilinear")
    claimed = g.n.to_numpy() > 0
    rings = ax[1].scatter(lon[claimed], lat[claimed], s=60, facecolors="none", edgecolors="#ff4d4d", linewidths=1.6, zorder=5, visible=False)
    rings2 = ax[2].scatter(lon[claimed], lat[claimed], s=60, facecolors="none", edgecolors="#ff4d4d", linewidths=1.6, zorder=5, visible=False)
    # warned counties outline layer on the county panel
    warn_pc = PolyCollection(allp, facecolors="none", edgecolors="#ffffff", linewidths=0.0, zorder=4)
    ax[0].add_collection(warn_pc)
    titles = ["County ranking (sum of block-group probabilities)", "Sub-county: discrete block-group points", "Sub-county: 10 km interpolated surface (context only)"]
    for x, t in zip(ax, titles, strict=True):
        x.text(0.5, 1.0, t, transform=x.transAxes, ha="center", va="bottom", fontsize=13, color=FG)
    cmt = fig.text(0.5, 0.965, "", ha="center", fontsize=15, color=MUTED)
    clock = fig.text(0.02, 0.33, "", fontsize=56, color=ACC, family="DejaVu Sans Mono", va="top")
    sub = fig.text(0.02, 0.22, "", fontsize=11.5, color=FG, va="top", linespacing=1.5)
    tl = fig.add_axes([0.33, 0.14, 0.36, 0.17], facecolor=BG)
    tl.set_xlim(-36, T_END); tl.set_ylim(0, 1); tl.axis("off")
    tl.plot([-36, T_END], [0.5, 0.5], color="#46586a", lw=3, zorder=1)
    for h, lab, y, col, ha in ((-36, "GEFS 00Z init", 0.98, MUTED, "left"), (T_READY, "ranking ready", 0.76, ACC, "center"), (0, "T0: rain window opens", 0.98, FG, "left"),
                               (T_OBS, "observed-rain update", 0.98, "#4fd1c5", "center"), (T_END, "claims", 0.98, "#ff4d4d", "right")):
        tl.plot([h, h], [0.4, 0.6], color=col, lw=3, zorder=2)
        tl.text(h, y, lab, color=col, fontsize=10, ha=ha, va="top")
    wx = ev.h.to_numpy()
    tl.scatter(wx, np.full(len(wx), 0.28), s=22, color="#f28e2b", zorder=3, marker="v")
    tl.text(-36, 0.12, f"NWS Flash Flood Warnings issued (real, {len(ev)} in this window)", color="#f28e2b", fontsize=10, ha="left", va="center")
    cursor = tl.axvline(-36, color="#ffffff", lw=2, zorder=5)
    lst = fig.text(0.72, 0.30, "", fontsize=13, color=FG, va="top", family="DejaVu Sans Mono", linespacing=1.55)
    cost = fig.text(0.33, 0.1, "", fontsize=11, color=MUTED, va="top", linespacing=1.5)
    foot = fig.text(0.01, 0.012, "Retrospective replay of stored model outputs, real NWS warnings and real NFIP claims. Not a live product. Texas, scope S0, GEFSv12 reforecast (5 members, 0.25 deg).", fontsize=10, color=MUTED)

    def county_colors(cs):
        r = cs.rank(pct=True)
        return [rank_cmap(r[c]) if c in r.index else (0.11, 0.15, 0.2, 1.0) for c in codes]

    col_f, col_o = county_colors(cs_f), county_colors(cs_o)
    pcs[0].set_facecolor(col_f)
    lat_ms = {"download": 9.0, "prep": 0.003, "net": 0.030}

    def render(sec: float) -> None:
        h = clock_at(sec)
        clock.set_text(fmt_clock(h))
        cursor.set_xdata([h, h])
        ready = sec >= READY_SEC
        use_obs = h >= T_OBS and sec >= KEYS[6][0]
        revealed = h >= T_END - 1e-6 and sec >= 58
        # step-by-step processing display
        if sec < 3:
            msg = "00Z GEFS ensemble initialised"
            sub.set_text("Laptop idle.\nWaiting for the model run to be published.")
        elif sec < READY_SEC:
            steps = [("GEFS 00Z output out (assumed +5 h)", 6.5), ("download 5 members, Texas: 9 s", 7.5), ("features + exposure: 3 ms", 9.0), ("network forward on CPU: 30 ms", 10.0), ("maps rendered", 11.2)]
            sub.set_text("Pipeline running on the laptop:\n" + "\n".join(("[x] " if sec >= s else "[ ] ") + t for t, s in steps))
            msg = "Model run: GEFS 00Z of day S-1 (leads 36-84 h)"
        else:
            sub.set_text("Pipeline idle. The next GEFS cycle (06Z)\nwould refresh the ranking (not simulated:\nthe reforecast has only the 00Z run)." if not use_obs else "Observed-rain update running\n(daily QPE of the rain window).")
            msg = ("Forecast mode: ranking issued " + f"{abs(T_READY):.0f} h before T0" if not use_obs else "Observed-rain mode: ranking after the rain (damage-assessment use)")
        cmt.set_text(msg)
        # maps
        sc.set_visible(ready); im.set_visible(ready)
        p = g.obs.to_numpy() if use_obs else g.fc.to_numpy()
        if ready:
            sc.set_array(p[order] if not use_obs else p[np.argsort(p)])
            sc.set_offsets(np.column_stack([lon[order], lat[order]]) if not use_obs else np.column_stack([lon[np.argsort(p)], lat[np.argsort(p)]]))
            im.set_data(Zo if use_obs else Zf)
            pcs[0].set_facecolor(col_o if use_obs else col_f)
        else:
            pcs[0].set_facecolor("#1b2733")
        rings.set_visible(revealed); rings2.set_visible(revealed)
        # warned counties so far
        warned = set()
        for r in ev.itertuples():
            if r.h <= h:
                warned |= set(r.counties.split(","))
        warn_pc.set_edgecolor([("#ffffff" if c in warned else (0, 0, 0, 0)) for c in codes])
        warn_pc.set_linewidth([1.6 if c in warned else 0 for c in codes])
        nw = int((ev.h <= h).sum())
        # dispatch list
        if ready:
            cs = cs_o if use_obs else cs_f
            tp = cs.sort_values(ascending=False).head(5)
            lines = ["Dispatch suggestion: top-5 counties", "(expected claimed block groups)"]
            for c, v in tp.items():
                mark = ("  claims: %d" % ncl_c.get(c, 0)) if revealed else ""
                lines.append(f"{names[c][:14]:14s} {v:6.2f}{mark}")
            lst.set_text("\n".join(lines))
        else:
            lst.set_text("Dispatch suggestion\n\n(waiting for the model run)")
        note = f"Active NWS warnings so far in the window: {nw}"
        if first_aff is not None and h >= first_aff:
            note += f"   |   first warning for an affected county: T{first_aff:+.1f} h"
        if first_aff is not None and h >= first_aff:
            lead = first_aff - T_READY
            note += f"\nModel ranking was available {lead:.0f} h before the first warning for an affected county" if lead > 0 else f"\nFirst affected-county warning came {abs(lead):.0f} h before the model run was ready"
        cost.set_text(note + (f"\nThis event: block-group AUC {auc_o if use_obs else auc_f:.2f}; top-5% of block groups hold {100 * (cap_o if use_obs else cap_f):.0f}% of the claims "
                              f"(policies only: {100 * cap_p:.0f}%)" if ready else ""))

    if a.still:
        render(END_SEC)
        fig.savefig(MEDIA / f"countdown_{a.event}_final.png", dpi=100, facecolor=BG)
        return
    out = MEDIA / f"countdown_{a.event}.mp4"
    w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"])
    n = int(END_SEC * FPS)
    t_ = time.perf_counter()
    with w.saving(fig, str(out), 100):
        for i in range(n):
            render(i / FPS)
            w.grab_frame(facecolor=BG)
            if i == 24:
                print(f"first second rendered in {time.perf_counter() - t_:.1f} s", flush=True)
    render(END_SEC)
    fig.savefig(MEDIA / f"countdown_{a.event}_final.png", dpi=100, facecolor=BG)
    print(f"wrote {out} ({n} frames, {n / FPS:.0f} s) in {time.perf_counter() - t_:.0f} s")


if __name__ == "__main__":
    main()
