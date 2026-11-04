"""Countdown replay v2: two T-0s and the update cadence (scenario report part 2, RESULTS stage 57).

One real Texas event (test years 2017-2019), run on the synthetic-field simulation of script 101: the observed 4 km rain field is the known truth, and every update of a stream is that field with the
error of its lead (eps = eps60 * (lead / 60 h)^0.75, eps60 calibrated to the real GEFS-mode skill). Two streams are replayed side by side: a 6-hourly one (available 5 h after its cycle) and an hourly one
(2 h latency); the maps follow the hourly stream, the quality chart shows both. Real data in the clip: NWS Flash Flood Warnings (arrival lane), the claimed block groups, and the NFIP claim open dates (a red ring
appears when the first claim of a block group was opened, day resolution placed at midday). Two clocks: hazard T-0 (first warning for an affected county, else T0 = 12Z of the first loss day S) and claims T-0
(day by which 10% of the event's claims are opened). Assumptions are on screen: 5 h / 2 h stream latencies, 6 h QPE delay for the observed-rain ranking at T0 + 54 h.
Inputs : script 101 (pipeline, perturbation), audit/cadence_simulation.pkl, audit/ffw_tx.parquet, labels_extra/flood/nfip_tx_claim_dates.parquet
Output : docs/media/countdown2_<event>.mp4, docs/media/countdown2_<event>_final.png
Run    : python scripts/103_countdown_video_v2.py            (picks a typical event)   |   --event 20190604
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

MEDIA = Path(__file__).resolve().parents[1] / "docs" / "media"
FPS = 24
BG, FG, MUTED, ACC = "#0f1720", "#e8edf2", "#8b98a5", "#f2c14e"
ALPHA = 0.75
KEYS = [(0, -48), (3, -48), (33, 0), (37, 0), (63, 168), (70, 168)]  # (video second, hours from the hazard T-0)
T_OBS_REL_T0 = 54.0


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def clock_at(sec: float) -> float:
    return float(np.interp(sec, [k[0] for k in KEYS], [k[1] for k in KEYS]))


def hm(h: float) -> str:
    s = "-" if h < 0 else "+"
    m = int(round(abs(h) * 60))
    return f"{s}{m // 60:02d}:{m % 60:02d}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import LogNorm

    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default=None)
    ap.add_argument("--still", action="store_true")
    a = ap.parse_args()
    MEDIA.mkdir(parents=True, exist_ok=True)
    s101, s98, s99, e75 = _load("s101", "101_cadence_simulation.py"), _load("s98", "98_scenario_analysis.py"), _load("s99", "99_countdown_video.py"), _load("e75", "75_task_metrics.py")
    R = pickle.load(open(wb2.cache_path("audit") / "cadence_simulation.pkl", "rb"))
    eps60 = R["eps60"]
    pipe = s101.RainPipeline()
    d = pipe.d
    ev = R["events"]
    ffw = pd.read_parquet(wb2.cache_path("audit") / "ffw_tx.parquet")

    def eps_of(lead):
        return eps60 * (max(lead, 1.0) / 60.0) ** ALPHA

    # pick the event
    cand = []
    for n, v in ev.items():
        n = int(n)
        t0, ht0 = pd.Timestamp(v["t0"]), pd.Timestamp(v["ht0"])
        h_ = (ht0 - t0).total_seconds() / 3600
        cnt = d.rn[d.ptr[n]:d.ptr[n + 1]]
        if v["has_warning"] == "True" and abs(h_) <= 6 and (cnt > 0).sum() >= 25:
            cand.append((n, h_))
    if a.event:
        n = int(np.flatnonzero(d.days == pd.Timestamp(a.event))[0])
    else:
        best = None
        for n_, h_ in cand:  # typical = mean synthetic AUC at 60 h near the expectation (0.778) and exposure-only AUC near its mean (0.731)
            au = np.mean([s101.event_metrics(e75, pipe.day(n_, s101.perturb(pipe.rain_maps(n_), eps_of(60.0), s101.noise(d.valid4.shape, np.random.default_rng(7000 + j)), pipe.valid))[1], d.rn[d.ptr[n_]:d.ptr[n_ + 1]])[0] for j in range(3)])
            po = s101.event_metrics(e75, d.rtab[d.ptr[n_]:d.ptr[n_ + 1], 8].astype(float), d.rn[d.ptr[n_]:d.ptr[n_ + 1]])[0]
            sc_ = abs(au - 0.778) / 0.02 + abs(po - 0.731) / 0.03
            if best is None or sc_ < best[0]:
                best = (sc_, n_, au, po)
        n = best[1]
        print(f"picked day index {n} ({d.days[n]}), mean synthetic AUC at 60 h {best[2]:.3f}, exposure-only AUC {best[3]:.3f}")
    S = pd.Timestamp(d.days[n])
    T0 = S + pd.Timedelta(12, "h")
    v = ev[n] if n in ev else ev[str(n)]
    ht0 = pd.Timestamp(v["ht0"])
    H0 = (ht0 - T0).total_seconds() / 3600  # hazard T-0 in hours from T0
    bgs, tab, cnt, _ = d.rows(n)
    y = cnt > 0
    ctys = np.array(d.ctys)[d.cty_idx[bgs]]
    S_ = np.load(wb2.cache_path("cnn") / "static.npz", allow_pickle=True)
    lon_all, lat_all = S_["lon4"][S_["b4_x"]], S_["lat4"][S_["b4_y"]]
    lon, lat = lon_all[bgs], lat_all[bgs]
    parts, names, tx = s99.counties()
    grid = (np.arange(-106.8, -93.4, 0.08), np.arange(25.8, 36.7, 0.08))
    z = s101.noise(d.valid4.shape, np.random.default_rng(1000 + n))
    raw = pipe.rain_maps(n)
    bgcode = S_["bg10"][bgs]

    # real claims: first open time per claimed block group (hours from T0)
    from wxrisk import bg_crosswalk

    cw = bg_crosswalk.build().set_index("code")
    cl = pd.read_parquet(wb2.cache_path("labels_extra") / "flood" / "nfip_tx_claim_dates.parquet")
    cl = cl[cl.causeOfDamage.astype(str).isin(["2", "3", "4"])].copy()
    cl["dl"] = pd.to_datetime(cl.dateOfLoss, utc=True).dt.tz_localize(None).dt.normalize()
    cl["od"] = pd.to_datetime(cl.openDate, utc=True).dt.tz_localize(None).dt.normalize()
    cl = cl[(cl.dl >= S) & (cl.dl <= S + pd.Timedelta(3, "D"))].copy()
    cl["bg10"] = cl.censusGeoid.map(cw.bg10)
    cl = cl[cl.bg10.isin(set(bgcode[y]))]
    first_open = cl.groupby("bg10").od.min()
    open_h = np.full(len(bgs), np.inf)
    for i in np.flatnonzero(y):
        o_ = first_open.get(bgcode[i])
        if o_ is not None and not pd.isna(o_):
            open_h[i] = (o_ + pd.Timedelta(12, "h") - T0).total_seconds() / 3600
    cl_cty = cl.groupby("countyCode").size()
    d_open = (cl.od - S).dt.days.to_numpy()
    cum = np.array([np.mean(d_open <= k) for k in range(31)])
    ct10 = int(np.argmax(cum >= 0.10))
    CT0 = 24 * ct10 + 18  # end of day S+ct10 (06Z next day) in hours from T0
    print(f"event {S.date()}: rows {len(bgs)}, claimed BGs {int(y.sum())}, hazard T-0 = T0{H0:+.1f} h, claims T-0 (10% opened) = day S+{ct10} -> T0{CT0:+.0f} h; claims with open dates {len(cl)}")

    # stream updates and predictions
    def updates(cad, lat_):
        ks = np.arange(-80, 260)
        avail = (-12.0 + ks * cad) + lat_
        return avail[(avail >= -48 + H0) & (avail <= 168 + H0)]

    UP = {"6-hourly (+5 h)": (6.0, 5.0), "hourly (+2 h)": (1.0, 2.0)}
    zs_extra = [s101.noise(d.valid4.shape, np.random.default_rng(5000 + j)) for j in range(7)]
    pre = {}
    surf_cache = {}
    t_ = time.perf_counter()
    for name, (cad, lat_) in UP.items():
        rows = []
        for av in updates(cad, lat_):
            init = av - lat_
            lead = max(24.0 - init, 1.0)
            o = s101.perturb(raw, eps_of(lead), z, pipe.valid)
            _, p, _ = pipe.day(n, o)
            ms = [s101.event_metrics(e75, p, cnt)]
            for zj in zs_extra:  # the chart shows the mean of 8 synthetic draws; the maps use draw 0
                _, pj, _ = pipe.day(n, s101.perturb(raw, eps_of(lead), zj, pipe.valid))
                ms.append(s101.event_metrics(e75, pj, cnt))
            rows.append((float(av), float(lead), p, float(np.mean([m[0] for m in ms])), float(np.mean([m[1] for m in ms]))))
        pre[name] = rows
    _, p_obs, _ = pipe.day(n)
    obs_m = s101.event_metrics(e75, p_obs, cnt)
    print(f"updates precomputed in {time.perf_counter() - t_:.0f} s; observed-rain ranking AUC {obs_m[0]:.3f}, top-5% {obs_m[1]:.3f}")
    pol = s101.event_metrics(e75, tab[:, 8].astype(float), cnt)

    def surface_for(key, p):
        if key not in surf_cache:
            surf_cache[key] = s99.surface(lon, lat, p.astype(float), tx, grid)
        return surf_cache[key]

    def latest(name, t_abs):
        r = [x for x in pre[name] if x[0] <= t_abs + 1e-9]
        return r[-1] if r else None

    # figure
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    ax = [fig.add_axes([0.01 + 0.325 * i, 0.43, 0.32, 0.52], facecolor=BG) for i in range(3)]
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
    order0 = np.argsort(p_obs)
    sc = ax[1].scatter(lon[order0], lat[order0], c=p_obs[order0], s=3.5, cmap="viridis", norm=norm, zorder=3, visible=False)
    zz = surface_for("obs", p_obs)
    ext_im = [grid[0][0] - 0.04, grid[0][-1] + 0.04, grid[1][0] - 0.04, grid[1][-1] + 0.04]
    im = ax[2].imshow(zz, extent=ext_im, origin="lower", cmap="viridis", norm=norm, zorder=2, visible=False, interpolation="bilinear")
    rings = ax[1].scatter(lon[y], lat[y], s=70, facecolors="none", edgecolors="#ff4d4d", linewidths=1.8, zorder=5, visible=True)
    rings.set_offsets(np.empty((0, 2)))
    rings2 = ax[2].scatter([], [], s=70, facecolors="none", edgecolors="#ff4d4d", linewidths=1.8, zorder=5)
    warn_pc = PolyCollection(allp, facecolors="none", edgecolors="#ffffff", linewidths=0.0, zorder=4)
    ax[0].add_collection(warn_pc)
    for x, t in zip(ax, ["County ranking (sum of block-group probabilities)", "Sub-county: discrete block-group points", "Sub-county: 10 km interpolated surface (context)"], strict=True):
        x.text(0.5, 1.0, t, transform=x.transAxes, ha="center", va="bottom", fontsize=13, color=FG)
    head = fig.text(0.5, 0.968, "", ha="center", fontsize=15, color=MUTED)
    topc = ax[0].text(0.99, 0.02, "", transform=ax[0].transAxes, fontsize=11, color=FG, ha="right", va="bottom", family="DejaVu Sans Mono")
    # clocks
    fig.text(0.015, 0.425, "HAZARD T-0", fontsize=14, color=ACC, va="top", weight="bold")
    fig.text(0.115, 0.423, "decisions + wireless alerts done before", fontsize=9, color=ACC, va="top")
    clk1 = fig.text(0.015, 0.405, "", fontsize=38, color=ACC, family="DejaVu Sans Mono", va="top")
    fig.text(0.015, 0.31, "CLAIMS T-0", fontsize=14, color="#c792ea", va="top", weight="bold")
    fig.text(0.115, 0.308, "10% of the event's claims opened", fontsize=9, color="#c792ea", va="top")
    clk2 = fig.text(0.015, 0.29, "", fontsize=38, color="#c792ea", family="DejaVu Sans Mono", va="top")
    stat = fig.text(0.015, 0.19, "", fontsize=9.5, color=FG, va="top", linespacing=1.45)
    # lanes
    tl = fig.add_axes([0.43, 0.075, 0.30, 0.27], facecolor=BG)
    tmin, tmax = -48 + H0, 168 + H0  # hours from T0
    tl.set_xlim(tmin, tmax); tl.set_ylim(0, 6); tl.axis("off")
    lanes = ["NWWS-OI push (<10 s)", "api.weather.gov poll 5 min", "6-hourly stream (+5 h)", "hourly stream (+2 h)", "claims opened (day res.)"]
    ys = [5.2, 4.2, 3.2, 2.2, 1.2]
    for yy, lb in zip(ys, lanes, strict=True):
        tl.plot([tmin, tmax], [yy, yy], color="#2a3947", lw=1)
        tl.text(tmin - 1, yy, lb, ha="right", va="center", fontsize=9, color=MUTED)
    for hh_, col, lb in ((H0, ACC, "hazard T-0"), (CT0, "#c792ea", "claims T-0")):
        tl.axvline(hh_, color=col, lw=2)
        tl.text(hh_, 5.95, lb, color=col, fontsize=9, ha="center", va="top")
    tl.axvline(T_OBS_REL_T0, color="#4fd1c5", lw=1.2, ls="--")
    tl.text(T_OBS_REL_T0, 0.35, "observed-rain ranking\n(QPE +6 h, assumed)", color="#4fd1c5", fontsize=8, ha="center")
    ffw["h"] = (ffw.issued - T0).dt.total_seconds() / 3600
    wsel = ffw[(ffw.h >= tmin) & (ffw.h <= tmax)]
    aff = set(ctys[y])
    cursor = tl.axvline(tmin, color="#ffffff", lw=2, zorder=9)
    lane_art = {}
    for yy, key in ((ys[0], "nwws"), (ys[1], "api")):
        lane_art[key] = tl.scatter([], [], s=26, color="#f28e2b", marker="v", zorder=5)
    lane_art["g"] = tl.scatter([], [], s=34, color="#4c78a8", zorder=5)
    lane_art["h"] = tl.scatter([], [], s=14, color="#d1495b", zorder=5)
    lane_art["c"] = tl.scatter([], [], s=14, color="#c792ea", zorder=5)
    ok_open = np.sort(open_h[np.isfinite(open_h)])
    # quality chart
    qa = fig.add_axes([0.765, 0.095, 0.22, 0.235], facecolor="#131d27")
    qa.set_xlim(-48, 0); qa.set_ylim(0.70, 0.90)
    for s_ in qa.spines.values():
        s_.set_color("#46586a")
    qa.tick_params(colors=MUTED, labelsize=8)
    qa.set_title("this event: BG AUC of each update (mean of 8 synthetic draws);\nfaint = expectation over 210 events", fontsize=9, color=FG)
    for nm_, col_ in (("6-hourly stream (+5 h)", "#4c78a8"), ("hourly stream (+2 h)", "#d1495b")):
        qa.plot(-R["DELTAS"], R["timeline"][nm_][0], color=col_, lw=2.5, alpha=0.25)
    qa.axhline(pol[0], color="#e8a33d", lw=1); qa.text(-47, pol[0] + 0.004, "policies only", color="#e8a33d", fontsize=8)
    qa.axhline(obs_m[0], color="#d1495b", lw=1, alpha=0.6); qa.text(-47, obs_m[0] + 0.004, "observed-rain model", color="#d1495b", fontsize=8)
    ln6, = qa.plot([], [], color="#4c78a8", marker="o", ms=4, lw=1.6, label="6-hourly")
    lnh, = qa.plot([], [], color="#d1495b", marker="o", ms=2.5, lw=1.2, label="hourly")
    qa.legend(fontsize=8, loc="lower right", facecolor="#131d27", edgecolor="#46586a")
    qa.set_xlabel("hours relative to the hazard T-0", fontsize=8)
    foot = fig.text(0.01, 0.012, "Replay of a simulated rain-forecast stream (known truth = observed gridMET, error grows with lead, calibrated to the real GEFS-mode skill); real NWS warnings, real claims. "
                    "Latencies are assumptions. Texas, S0.", fontsize=9.5, color=MUTED)

    def county_colors(cs):
        r = cs.rank(pct=True)
        return [rank_cmap(r[c]) if c in r.index else (0.11, 0.15, 0.2, 1.0) for c in codes]

    state = {"key": None}

    def render(sec: float) -> None:
        h = clock_at(sec)  # hours from the hazard T-0
        t_abs = h + H0  # hours from T0
        clk1.set_text("T" + hm(h))
        clk2.set_text("T" + hm(t_abs - CT0))
        cursor.set_xdata([t_abs, t_abs])
        r_h = latest("hourly (+2 h)", t_abs)
        r_6 = latest("6-hourly (+5 h)", t_abs)
        use_obs = t_abs >= T_OBS_REL_T0
        p = p_obs if use_obs else (r_h[2] if r_h else None)
        key = ("obs",) if use_obs else (("h", r_h[0]) if r_h else None)
        if key != state["key"]:
            state["key"] = key
            if p is None:
                sc.set_visible(False); im.set_visible(False)
                pcs[0].set_facecolor("#1b2733")
            else:
                o_ = np.argsort(p)
                sc.set_visible(True); im.set_visible(True)
                sc.set_offsets(np.column_stack([lon[o_], lat[o_]])); sc.set_array(p[o_])
                im.set_data(surface_for(key, p))
                cs = pd.Series(p).groupby(ctys).sum()
                pcs[0].set_facecolor(county_colors(cs))
                top = cs.sort_values(ascending=False).head(3)
                topc.set_text("top counties now:\n" + "\n".join(f"{names[c][:12]:12s} {v:5.2f}" for c, v in top.items()))
        if p is None:
            topc.set_text("top counties now:\n(waiting for the first run)")
        head.set_text(("Observed-rain ranking (after the rain)" if use_obs else (f"Freshest hourly update: issued {t_abs - (r_h[0] - 2.0):.1f} h ago, lead {r_h[1]:.0f} h" if r_h else "Waiting for the first model run")))
        # warnings
        seen = wsel[wsel.h <= t_abs]
        lane_art["nwws"].set_offsets(np.column_stack([seen.h.to_numpy() + 10 / 3600, np.full(len(seen), ys[0])]) if len(seen) else np.empty((0, 2)))
        lane_art["api"].set_offsets(np.column_stack([np.ceil(seen.h.to_numpy() * 12) / 12, np.full(len(seen), ys[1])]) if len(seen) else np.empty((0, 2)))
        warned = set()
        for r in seen.itertuples():
            warned |= set(r.counties.split(","))
        warn_pc.set_edgecolor([("#ffffff" if c_ in warned else (0, 0, 0, 0)) for c_ in codes])
        warn_pc.set_linewidth([1.5 if c_ in warned else 0 for c_ in codes])
        g_ = [x[0] for x in pre["6-hourly (+5 h)"] if x[0] <= t_abs]
        h_ = [x[0] for x in pre["hourly (+2 h)"] if x[0] <= t_abs]
        lane_art["g"].set_offsets(np.column_stack([g_, np.full(len(g_), ys[2])]) if g_ else np.empty((0, 2)))
        lane_art["h"].set_offsets(np.column_stack([h_, np.full(len(h_), ys[3])]) if h_ else np.empty((0, 2)))
        op = ok_open[ok_open <= t_abs]
        lane_art["c"].set_offsets(np.column_stack([op, np.full(len(op), ys[4])]) if len(op) else np.empty((0, 2)))
        # claims rings
        shown = open_h <= t_abs
        rings.set_offsets(np.column_stack([lon[shown], lat[shown]]) if shown.any() else np.empty((0, 2)))
        rings2.set_offsets(np.column_stack([lon[shown], lat[shown]]) if shown.any() else np.empty((0, 2)))
        # quality chart (relative to the hazard T-0)
        for ln, name in ((ln6, "6-hourly (+5 h)"), (lnh, "hourly (+2 h)")):
            xs = [x[0] - H0 for x in pre[name] if x[0] <= t_abs and x[0] - H0 >= -48]
            ys_ = [x[3] for x in pre[name] if x[0] <= t_abs and x[0] - H0 >= -48]
            ln.set_data(xs, ys_)
        nopen = int((open_h <= t_abs).sum())
        extra = "" if not r_h else f"\nupdates received so far: 6-hourly {len(g_)}, hourly {len(h_)}"
        stat.set_text("Before hazard T-0: staging, shifts, alert text.\nBefore claims T-0: preliminary estimate,\nimagery / aerial reservation (lead assumed)." + extra.replace("updates received so far: ", "updates so far: ")
                      + f"\nopened claims: {nopen} of {int(y.sum())} block groups\nwarnings so far: {len(seen)}")

    if a.still:
        render(KEYS[-1][0] - 0.5)
        fig.savefig(MEDIA / f"countdown2_{S:%Y%m%d}_final.png", dpi=100, facecolor=BG)
        return
    out = MEDIA / f"countdown2_{S:%Y%m%d}.mp4"
    w = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"])
    nfr = int(KEYS[-1][0] * FPS)
    t_ = time.perf_counter()
    with w.saving(fig, str(out), 100):
        for i in range(nfr):
            render(i / FPS)
            w.grab_frame(facecolor=BG)
    render(KEYS[-1][0] - 0.5)
    fig.savefig(MEDIA / f"countdown2_{S:%Y%m%d}_final.png", dpi=100, facecolor=BG)
    print(f"wrote {out} ({nfr} frames, {nfr / FPS:.0f} s) in {time.perf_counter() - t_:.0f} s")


if __name__ == "__main__":
    main()
