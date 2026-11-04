"""Countdown replay v4: two groups (model output | real-world consequence), no scan sweep, a featured-county act (scenario report part 2, RESULTS stage 57).

v4 against v3: the scan sweep is gone; the screen is split into MODEL OUTPUT (county ranking, sub-county probability, run counter and log, quality chart) and REAL-WORLD CONSEQUENCE (dispatch set with
the block groups that entered / left it, movement since the previous run, the two T-0 clocks, warning / run / claim lanes); a last act of 14 s zooms into one county (chosen by the rule: most claimed
block groups in the event, not by skill) at the hazard T-0 minus 24 h and shows what a county-level dispatch cannot say and the sub-county ranking can. Encoding: Apple VideoToolbox HEVC (hvc1) by default.

v3 description:

What is new against `103_countdown_video_v2.py`: every run of the hourly stream is made visible: a run counter and a log line per run (with the measured cost), a scan sweep across the maps,
the dispatch set (top 5% block groups) drawn bright on a dimmed map with green rings for block groups that entered it and red marks for those that left, a diverging map of how the
ranking moved since the previous run (percentile points, interpolated), a heartbeat lane whose bar height is the churn of the dispatch set per run, and county list arrows.

Original description of v2:

One real Texas event (test years 2017-2019), run on the synthetic-field simulation of script 101 (v3: successive runs of a stream share only part of their error, AR(1) with tau = 12 h, so every run brings new information): the observed 4 km rain field is the known truth, and every update of a stream is that field with the
error of its lead (eps = eps60 * (lead / 60 h)^0.75, eps60 calibrated to the real GEFS-mode skill). Two streams are replayed side by side: a 6-hourly one (available 5 h after its cycle) and an hourly one
(2 h latency); the maps follow the hourly stream, the quality chart shows both. Real data in the clip: NWS Flash Flood Warnings (arrival lane), the claimed block groups, and the NFIP claim open dates (a red ring
appears when the first claim of a block group was opened, day resolution placed at midday). Two clocks: hazard T-0 (first warning for an affected county, else T0 = 12Z of the first loss day S) and claims T-0
(day by which 10% of the event's claims are opened). Assumptions are on screen: 5 h / 2 h stream latencies, 6 h QPE delay for the observed-rain ranking at T0 + 54 h.
Inputs : script 101 (pipeline, perturbation), audit/cadence_simulation.pkl, audit/ffw_tx.parquet, labels_extra/flood/nfip_tx_claim_dates.parquet
Output : docs/media/countdown4_<event>.mp4 (HEVC), docs/media/countdown4_<event>_h264.mp4 (fallback, transcoded), docs/media/countdown4_<event>_final.png
Run    : python scripts/105_countdown_video_v4.py --event 20190511 [--codec hevc|h264] [--still SECOND]
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
KEYS = [(0, -48), (3, -48), (33, 0), (37, 0), (57, 54), (60, 54), (68, 168), (72, 168)]
ACT3, END = 72.0, 86.0  # featured county act  # (video second, hours from the hazard T-0)
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
    ap.add_argument("--event", default="20190511")
    ap.add_argument("--still", type=float, default=None, help="render only the frame at this video second")
    ap.add_argument("--codec", default="hevc", choices=["hevc", "h264"])
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
    TAU = 12.0  # hours: how long an error persists from one run to the next (ASSUMPTION); each run has fresh innovations, the marginal error of a run is unchanged

    def chains(seed, n_up, dt):
        rho = float(np.exp(-dt / TAU))
        rng = np.random.default_rng(seed)
        cur = s101.noise(d.valid4.shape, rng)
        out = [cur]
        for _ in range(n_up - 1):
            nw = s101.noise(d.valid4.shape, rng)
            cur = {"shift": np.sqrt(rho) * cur["shift"] + np.sqrt(1 - rho) * nw["shift"], "amp": (np.sqrt(rho) * cur["amp"] + np.sqrt(1 - rho) * nw["amp"]).astype(np.float32)}
            out.append(cur)
        return out

    for name, (cad, lat_) in UP.items():
        rows = []
        avs = updates(cad, lat_)
        ch0 = chains(1000 + n, len(avs), cad)
        chx = [chains(5000 + j, len(avs), cad) for j in range(7)]
        for i_, av in enumerate(avs):
            init = av - lat_
            lead = max(24.0 - init, 1.0)
            o = s101.perturb(raw, eps_of(lead), ch0[i_], pipe.valid)
            _, p, _ = pipe.day(n, o)
            ms = [s101.event_metrics(e75, p, cnt)]
            for j_ in range(7):  # the chart shows the mean of 8 synthetic draws; the maps use draw 0
                _, pj, _ = pipe.day(n, s101.perturb(raw, eps_of(lead), chx[j_][i_], pipe.valid))
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

    # ------------------------------------------------------------------ per-run quantities of the hourly stream
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyBboxPatch
    from scipy.stats import rankdata

    CY, GR, RD, YL, MG = "#00e5ff", "#39ff88", "#ff5c5c", "#ffd166", "#e879f9"
    PANEL = "#070d14"
    t_ = time.perf_counter()
    for _ in range(20):
        pipe.day(n, s101.perturb(raw, eps_of(24.0), z, pipe.valid))
    run_s = (time.perf_counter() - t_) / 20  # measured: perturbation + features + network of one run on this machine (CPU)
    HR = pre["hourly (+2 h)"]
    N = len(bgs)
    K = int(np.ceil(0.05 * N))
    pct = [rankdata(r_[2]) / N for r_ in HR]
    top = [pc_ >= 1 - K / N + 1e-12 for pc_ in pct]
    ent = [np.zeros(N, bool)] + [top[i] & ~top[i - 1] for i in range(1, len(HR))]
    lef = [np.zeros(N, bool)] + [~top[i] & top[i - 1] for i in range(1, len(HR))]
    churn = np.array([0.0] + [(ent[i].sum() + lef[i].sum()) / (2 * K) for i in range(1, len(HR))])
    dpct = [np.zeros(N)] + [pct[i] - pct[i - 1] for i in range(1, len(HR))]
    cs_all = [pd.Series(r_[2]).groupby(ctys).sum() for r_ in HR]
    av_h = np.array([r_[0] for r_ in HR])
    print(f"hourly runs {len(HR)}, dispatch set K = {K} block groups, churn per run: median {np.median(churn[1:]):.3f}, p90 {np.percentile(churn[1:], 90):.3f}; measured run {run_s * 1000:.0f} ms")
    DSURF = {}

    def dsurface(k):
        if k not in DSURF:
            DSURF[k] = s99.surface(lon, lat, dpct[k], tx, grid)
        return DSURF[k]

    # ---- featured county act: the (event day, county) pair with the most claimed block groups among the forecast-mode events (a rule that does not look at skill),
    # shown with the REAL forecast-mode predictions (GEFS run, `fc + expo`, ready T-31 h), not with the simulated stream
    s96 = _load("s96", "96_task_scorecard.py")
    pairs_df = pd.read_parquet(wb2.cache_path("audit") / "within_county_pairs_fc.parquet")
    pr = pairs_df.sort_values("ncl", ascending=False).iloc[0]
    FE_DAY, fcode = str(pr.event), str(pr.cty)
    T_all, _, _ = s96.build_forecast()
    gE = T_all[T_all.sysid == FE_DAY].reset_index(drop=True)
    del T_all
    lonE, latE = lon_all[gE.bg.to_numpy()], lat_all[gE.bg.to_numpy()]
    fi = np.flatnonzero(gE.cty.to_numpy() == fcode)
    yE = gE.n.to_numpy() > 0
    yf = yE[fi]
    mm = max(int(np.ceil(0.1 * len(fi))), 1)
    sel = {"forecast": fi[e75.top_idx(gE.fc.to_numpy()[fi].astype(float), mm)], "policies": fi[e75.top_idx(gE.n_pol.to_numpy()[fi].astype(float), mm)],
           "observed": fi[e75.top_idx(gE.obs.to_numpy()[fi].astype(float), mm)]}
    claimed_f = fi[yf]
    order_rv = np.random.default_rng(3).permutation(len(claimed_f))
    claimed_f = claimed_f[order_rv]
    in_sel = {k_: np.isin(claimed_f, v_) for k_, v_ in sel.items()}
    f_auc = e75.pair_auc(gE.fc.to_numpy()[fi], yf.astype(int))[0]
    o_auc = e75.pair_auc(gE.obs.to_numpy()[fi], yf.astype(int))[0]
    pol_auc = e75.pair_auc(gE.n_pol.to_numpy()[fi].astype(float), yf.astype(int))[0]
    fname = names[fcode]
    bx = np.vstack(parts[fcode])
    fb = (bx[:, 0].min(), bx[:, 0].max(), bx[:, 1].min(), bx[:, 1].max())
    padx, pady = 0.12 * (fb[1] - fb[0]), 0.12 * (fb[3] - fb[2])
    fb = (fb[0] - padx, fb[1] + padx, fb[2] - pady, fb[3] + pady)
    exp_rand = mm / len(fi) * int(yf.sum())
    cs_E = gE.groupby("cty").fc.sum()
    pct_auc = float(np.mean(pairs_df["auc_forecast (fc+expo)"] <= f_auc))
    print(f"featured: {FE_DAY} {fname} County (rule: most claimed block groups): {len(fi)} block groups, {int(yf.sum())} claimed; top 10% = {mm}: forecast {in_sel['forecast'].sum()}, policies {in_sel['policies'].sum()}, observed {in_sel['observed'].sum()} hits "
          f"(random {exp_rand:.1f}); within-county AUC forecast {f_auc:.2f} (percentile {pct_auc:.0%} among all {len(pairs_df)} county-days), observed {o_auc:.2f}, policies {pol_auc:.2f}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": FG, "axes.labelcolor": FG})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="#04080d")
    for x0, ec in ((0.008, CY), (0.503, ACC)):
        fig.add_artist(FancyBboxPatch((x0 + 0.004, 0.045), 0.481, 0.935, boxstyle="round,pad=0.002,rounding_size=0.01", transform=fig.transFigure, facecolor="#09121b", edgecolor=ec, linewidth=1.6, alpha=0.95, zorder=0))
    fig.text(0.02, 0.972, "MODEL OUTPUT", fontsize=17, color=CY, weight="bold", va="top")
    fig.text(0.515, 0.972, "REAL-WORLD CONSEQUENCE", fontsize=17, color=ACC, weight="bold", va="top")
    sub_l = fig.text(0.02, 0.942, "", fontsize=11, color=MUTED, va="top")
    fig.text(0.515, 0.942, "dispatch changes and trends", fontsize=11, color=MUTED, va="top")
    dot = fig.text(0.185, 0.974, "●", fontsize=17, color=GR, va="top")
    axpos = [[0.012, 0.49, 0.236, 0.42], [0.252, 0.49, 0.236, 0.42], [0.507, 0.49, 0.236, 0.42], [0.747, 0.49, 0.236, 0.42]]
    ax = [fig.add_axes(p_, facecolor=PANEL) for p_ in axpos]
    ext = [-106.8, -93.4, 25.8, 36.7]
    allp = [p for c in parts.values() for p in c]
    codes = [c for c, ps in parts.items() for _ in ps]
    rank_cmap = plt.get_cmap("magma")
    pcs = []
    for i, x in enumerate(ax):
        x.set_xlim(ext[0], ext[1]); x.set_ylim(ext[2], ext[3]); x.set_aspect("equal"); x.axis("off")
        pc = PolyCollection(allp, facecolors="#101a25" if i == 0 else "none", edgecolors="#2c3e50", linewidths=0.35, zorder=1)
        x.add_collection(pc)
        pcs.append(pc)
    ttl = ["County ranking (colour = rank)", "Sub-county probability (log colour)", "Dispatch set: top 5% (ring entered, x left)", "Movement since the previous run"]
    ttl_t = [x.text(0.5, 1.0, t_, transform=x.transAxes, ha="center", va="bottom", fontsize=10.5, color=FG) for x, t_ in zip(ax, ttl, strict=True)]
    norm = LogNorm(1e-4, 0.5)
    probsc = ax[1].scatter(lon, lat, s=3.5, c=np.full(N, 1e-4), cmap="viridis", norm=norm, zorder=3, visible=False)
    cax = fig.add_axes([0.275, 0.478, 0.19, 0.007])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=7, colors=MUTED, length=2)
    cb.set_label("probability of a claim at the block group", fontsize=7, color=MUTED, labelpad=1)
    base = ax[2].scatter(lon, lat, s=3.0, c=np.zeros(N), cmap="Blues", vmin=-0.2, vmax=1.0, alpha=0.5, zorder=3, visible=False)
    topsc = ax[2].scatter([], [], s=15, color=CY, alpha=0.95, zorder=5, edgecolors="none")
    entsc = ax[2].scatter([], [], s=60, facecolors="none", edgecolors=GR, linewidths=2.0, zorder=7)
    lefsc = ax[2].scatter([], [], s=40, marker="x", color=RD, linewidths=1.6, zorder=7)
    claim_rings = [ax[2].scatter([], [], s=60, facecolors="none", edgecolors=YL, linewidths=1.8, zorder=8), ax[1].scatter([], [], s=60, facecolors="none", edgecolors=YL, linewidths=1.8, zorder=8)]
    ext_im = [grid[0][0] - 0.04, grid[0][-1] + 0.04, grid[1][0] - 0.04, grid[1][-1] + 0.04]
    im = ax[3].imshow(np.full((len(grid[1]), len(grid[0])), np.nan), extent=ext_im, origin="lower", cmap="RdBu_r", vmin=-0.04, vmax=0.04, zorder=2, visible=False, interpolation="bilinear")
    warn_pc = PolyCollection(allp, facecolors="none", edgecolors="#ffffff", linewidths=0.0, zorder=4)
    ax[2].add_collection(warn_pc)
    base_d = ax[3].scatter([], [], s=3.0, color="#3b5b78", alpha=0.5, zorder=3)
    pol_top = ax[3].scatter([], [], s=15, color=ACC, alpha=0.95, zorder=5, edgecolors="none")
    ring_d = ax[3].scatter([], [], s=60, facecolors="none", edgecolors=YL, linewidths=1.8, zorder=8)
    feat_pc = PolyCollection(parts[fcode], facecolors="none", edgecolors=CY, linewidths=0.0, zorder=6)
    ax[0].add_collection(feat_pc)
    topc = ax[0].text(0.99, 0.02, "", transform=ax[0].transAxes, fontsize=10, color=FG, ha="right", va="bottom", family="DejaVu Sans Mono",
                      bbox={"facecolor": PANEL, "alpha": 0.95, "edgecolor": "#2c3e50", "pad": 4})
    # left bottom: run counter + log (act 3: featured county text)
    runs_t = fig.text(0.02, 0.435, "", fontsize=24, color=GR, family="DejaVu Sans Mono", va="top", weight="bold")
    log_t = fig.text(0.02, 0.375, "", fontsize=8.6, color="#9fb3c8", family="DejaVu Sans Mono", va="top", linespacing=1.5)
    feat_t = fig.text(0.02, 0.435, "", fontsize=11.5, color=FG, va="top", linespacing=1.55)
    ba = fig.add_axes([0.62, 0.10, 0.35, 0.30], facecolor="#0c141d")
    ba.set_visible(False)
    qa = fig.add_axes([0.305, 0.095, 0.18, 0.31], facecolor="#0c141d")
    qa.set_xlim(-48, 0); qa.set_ylim(0.70, 0.90)
    for s_ in qa.spines.values():
        s_.set_color("#46586a")
    qa.tick_params(colors=MUTED, labelsize=8)
    qa.set_title("BG AUC of each run (this event, mean of 8\nsynthetic draws); faint = expectation, 210 events", fontsize=8.5, color=FG)
    for nm_, col_ in (("6-hourly stream (+5 h)", "#4c78a8"), ("hourly stream (+2 h)", GR)):
        qa.plot(-R["DELTAS"], R["timeline"][nm_][0], color=col_, lw=2.5, alpha=0.25)
    qa.axhline(pol[0], color="#e8a33d", lw=1); qa.text(-47, pol[0] + 0.004, "policies only", color="#e8a33d", fontsize=8)
    qa.axhline(obs_m[0], color="#d1495b", lw=1, alpha=0.6); qa.text(-47, obs_m[0] + 0.004, "observed-rain model", color="#d1495b", fontsize=8)
    ln6, = qa.plot([], [], color="#4c78a8", marker="o", ms=4, lw=1.6, label="6-hourly")
    lnh, = qa.plot([], [], color=GR, marker="o", ms=2.5, lw=1.2, label="hourly")
    qa.legend(fontsize=8, loc="lower right", facecolor="#0c141d", edgecolor="#46586a")
    qa.set_xlabel("hours relative to the hazard T-0", fontsize=8)
    # right bottom: clocks, text, lanes
    lab_h = fig.text(0.515, 0.445, "HAZARD T-0", fontsize=13, color=ACC, va="top", weight="bold")
    clk1 = fig.text(0.515, 0.42, "", fontsize=34, color=ACC, family="DejaVu Sans Mono", va="top")
    lab_c = fig.text(0.515, 0.325, "CLAIMS T-0", fontsize=13, color=MG, va="top", weight="bold")
    clk2 = fig.text(0.515, 0.30, "", fontsize=34, color=MG, family="DejaVu Sans Mono", va="top")
    note_t = fig.text(0.515, 0.205, "Before hazard T-0:\nstaging, shifts, alert text\nBefore claims T-0:\nestimate, imagery, flights", fontsize=9, color=FG, va="top", linespacing=1.5)
    tl = fig.add_axes([0.785, 0.085, 0.195, 0.345], facecolor=PANEL)
    tmin, tmax = -48 + H0, 168 + H0
    tl.set_xlim(tmin, tmax); tl.set_ylim(0, 6); tl.axis("off")
    lanes = ["NWS warnings (push)", "NWS warnings (poll 5 min)", "6-hourly stream", "hourly runs (bar = churn)", "claims opened (day res.)"]
    ys = [5.4, 4.5, 3.7, 1.9, 0.8]
    for yy, lb in zip(ys, lanes, strict=True):
        tl.plot([tmin, tmax], [yy, yy], color="#22313f", lw=1)
        tl.text(tmin - 1, yy, lb, ha="right", va="center", fontsize=8, color=MUTED)
    for hh_, col, lb in ((H0, ACC, "hazard"), (CT0, MG, "claims")):
        tl.axvline(hh_, color=col, lw=2)
        tl.text(hh_, 5.95, lb, color=col, fontsize=8, ha="center", va="top")
    tl.axvline(T_OBS_REL_T0, color="#4fd1c5", lw=1.2, ls="--")
    tl.text(T_OBS_REL_T0, -0.05, "observed-rain\nranking (QPE +6 h)", color="#4fd1c5", fontsize=7.5, ha="center", va="top")
    ffw["h"] = (ffw.issued - T0).dt.total_seconds() / 3600
    wsel = ffw[(ffw.h >= tmin) & (ffw.h <= tmax)]
    cursor = tl.axvline(tmin, color="#ffffff", lw=2, zorder=9)
    lane_art = {"nwws": tl.scatter([], [], s=24, color="#f28e2b", marker="v", zorder=5), "api": tl.scatter([], [], s=24, color="#f28e2b", marker="v", zorder=5),
                "g": tl.scatter([], [], s=30, color="#4c78a8", zorder=5), "c": tl.scatter([], [], s=16, color=YL, zorder=5)}
    bars = LineCollection([], colors=GR, linewidths=1.2, zorder=5)
    tl.add_collection(bars)
    ok_open = np.sort(open_h[np.isfinite(open_h)])
    fig.text(0.01, 0.012, "Replay of a simulated rain-forecast stream (known truth = observed gridMET, error grows with lead, calibrated to the real GEFS-mode skill); real NWS warnings, real claims. "
             f"Latencies are assumptions; run time {run_s * 1000:.0f} ms and fetch 9 s are measured. Block-group positions are 4 km pixel centres. Texas, S0.", fontsize=8.8, color=MUTED)

    def county_colors(cs):
        r = cs.rank(pct=True)
        return [rank_cmap(0.15 + 0.8 * r[c]) if c in r.index else (0.06, 0.1, 0.14, 1.0) for c in codes]

    state = {"k": -2, "obs": False, "act3": False}
    log_lines: list[str] = []

    def log_line(k):
        z_ = HR[k]
        c_ = cs_all[k].sort_values(ascending=False).index[0]
        v_ = cs_all[k][c_]
        ts = T0 + pd.Timedelta(z_[0], "h")
        return f"{ts:%b%d %H:%MZ} run {k + 1:3d}  lead {z_[1]:3.0f} h  {run_s * 1000:.0f} ms  set +{int(ent[k].sum())}/-{int(lef[k].sum())}  {names[c_][:9]} {v_:4.1f}"

    def show_run(k):
        """Fill every panel with run k of the hourly stream."""
        p = HR[k][2]
        o_ = np.argsort(p)
        probsc.set_visible(True)
        probsc.set_offsets(np.column_stack([lon[o_], lat[o_]])); probsc.set_array(p[o_])
        base.set_visible(True)
        base.set_array(np.clip(pct[k] ** 2, 0, 1))
        topsc.set_color(CY)
        topsc.set_offsets(np.column_stack([lon[top[k]], lat[top[k]]]))
        entsc.set_offsets(np.column_stack([lon[ent[k]], lat[ent[k]]]) if ent[k].any() else np.empty((0, 2)))
        lefsc.set_offsets(np.column_stack([lon[lef[k]], lat[lef[k]]]) if lef[k].any() else np.empty((0, 2)))
        pcs[0].set_facecolor(county_colors(cs_all[k]))
        if k > 0:
            im.set_data(dsurface(k)); im.set_visible(True)
        tc = cs_all[k].sort_values(ascending=False).head(4)
        lines = []
        for c_, v_ in tc.items():
            dv = v_ - cs_all[k - 1].get(c_, 0.0) if k > 0 else 0.0
            lines.append(f"{names[c_][:11]:11s} {v_:5.2f} {'▲' if dv > 0.005 else ('▼' if dv < -0.005 else '─')}{abs(dv):4.2f}")
        topc.set_text("expected claimed BGs | change vs previous run\n" + "\n".join(lines))
        for kk in range(len(log_lines), k + 1):
            log_lines.append(log_line(kk))

    def set_zoom(e):
        x0 = ext[0] + e * (fb[0] - ext[0]); x1 = ext[1] + e * (fb[1] - ext[1])
        y0 = ext[2] + e * (fb[2] - ext[2]); y1 = ext[3] + e * (fb[3] - ext[3])
        for i in (1, 2, 3):
            ax[i].set_xlim(x0, x1); ax[i].set_ylim(y0, y1)
        sz = 1 + 9 * e
        probsc.set_sizes(np.full(N, 3.5 * sz)); base.set_sizes(np.full(N, 3.0 * sz)); topsc.set_sizes(np.full(max(len(topsc.get_offsets()), 1), 15 * sz))
        for cr in claim_rings + [ring_d]:
            cr.set_sizes(np.full(max(len(cr.get_offsets()), 1), 60 * (1 + 1.2 * e)))
        base_d.set_sizes(np.full(max(len(base_d.get_offsets()), 1), 3.0 * sz)); pol_top.set_sizes(np.full(max(len(pol_top.get_offsets()), 1), 15 * sz))

    def render(sec: float) -> None:
        act3 = sec >= ACT3
        if act3:
            if not state["act3"]:
                state["act3"] = True
                for sc_ in (entsc, lefsc):
                    sc_.set_offsets(np.empty((0, 2)))
                for a_ in (im, ):
                    a_.set_visible(False)
                # swap every panel to the featured event day (real forecast-mode predictions)
                o_ = np.argsort(gE.fc.to_numpy())
                probsc.set_visible(True)
                probsc.set_offsets(np.column_stack([lonE[o_], latE[o_]])); probsc.set_array(gE.fc.to_numpy()[o_])
                base.set_visible(True)
                base.set_offsets(np.column_stack([lonE, latE])); base.set_array(np.clip(rankdata(gE.fc.to_numpy()) / len(gE), 0, 1) ** 2)
                topsc.set_color(CY)
                topsc.set_offsets(np.column_stack([lonE[sel["forecast"]], latE[sel["forecast"]]]))
                base_d.set_offsets(np.column_stack([lonE, latE]))
                pol_top.set_offsets(np.column_stack([lonE[sel["policies"]], latE[sel["policies"]]]))
                pcs[0].set_facecolor(county_colors(cs_E))
                feat_pc.set_linewidth(3.0)
                warn_pc.set_linewidth(0.0)
                for cr in claim_rings + [ring_d]:
                    cr.set_offsets(np.empty((0, 2)))
                ttl_t[0].set_text("County ranking (real forecast run)")
                ttl_t[1].set_text(f"Sub-county probability, {fname} County")
                ttl_t[2].set_text("Sub-county dispatch: top 10% by the forecast")
                ttl_t[3].set_text("Same budget, ranked by exposure only")
                topc.set_text("")
                sub_l.set_text(f"FEATURED COUNTY: {fname}, {FE_DAY[:4]}-{FE_DAY[4:6]}-{FE_DAY[6:]}, real forecast run ready at T-31 h")
                runs_t.set_text(""); log_t.set_text("")
                tl.set_visible(False); qa.set_visible(False); ba.set_visible(True)
                for o_t in (clk1, clk2, lab_h, lab_c, note_t):
                    o_t.set_visible(False)
                ba.set_title("claimed block groups inside the top-10% budget", fontsize=9, color=FG)
                ba.set_ylim(0, max(len(claimed_f) * 0.45, 5)); ba.set_xlim(-0.6, 3.6)
                ba.set_xticks(range(4)); ba.set_xticklabels(["random", "exposure only", "forecast\n(T-31 h)", "observed rain\n(after the rain)"], fontsize=8)
                ba.tick_params(colors=MUTED, labelsize=8)
                for s_ in ba.spines.values():
                    s_.set_color("#46586a")
                state["bars"] = ba.bar(range(4), [0, 0, 0, 0], color=["#9aa0a6", ACC, CY, "#ffffff"], width=0.6)
                state["btxt"] = [ba.text(i_, 0, "", ha="center", va="bottom", fontsize=10, color=FG) for i_ in range(4)]
            e = float(np.clip((sec - ACT3) / 2.0, 0, 1))
            e = e * e * (3 - 2 * e)
            set_zoom(e)
            rv = float(np.clip((sec - 79.0) / 4.0, 0, 1))
            nshow = int(round(rv * len(claimed_f)))
            off = np.column_stack([lonE[claimed_f[:nshow]], latE[claimed_f[:nshow]]]) if nshow else np.empty((0, 2))
            for cr in claim_rings + [ring_d]:
                cr.set_offsets(off)
            vals = [exp_rand * rv, int(in_sel["policies"][:nshow].sum()), int(in_sel["forecast"][:nshow].sum()), int(in_sel["observed"][:nshow].sum())]
            for b_, v_, tx_ in zip(state["bars"], vals, state["btxt"], strict=True):
                b_.set_height(v_)
                tx_.set_y(v_); tx_.set_text(f"{v_:.0f}" if sec >= 79.0 else "")
            txt = (f"{fname} County: {len(fi):,} block groups, {int(yf.sum())} with claims\n\n"
                   f"County-level dispatch: the whole county, all {len(fi):,} block groups,\nno order inside it\n"
                   f"Sub-county dispatch: the top 10% = {mm} block groups")
            if sec >= 79.0:
                txt += (f"\n  forecast {int(in_sel['forecast'].sum())} | exposure only {int(in_sel['policies'].sum())} | random {exp_rand:.0f} | observed rain {int(in_sel['observed'].sum())}"
                        f"\nWithin-county AUC: forecast {f_auc:.2f}, exposure {pol_auc:.2f}, observed {o_auc:.2f}\n(this county-day: percentile {pct_auc:.0%} of {len(pairs_df)})")
            feat_t.set_text(txt)
            dot.set_color(GR)
            return
        h = clock_at(sec)
        t_abs = h + H0
        clk1.set_text("T" + hm(h))
        clk2.set_text("T" + hm(t_abs - CT0))
        cursor.set_xdata([t_abs, t_abs])
        k = int(np.searchsorted(av_h, t_abs + 1e-9, side="right") - 1)
        use_obs = t_abs >= T_OBS_REL_T0
        if k != state["k"] and use_obs:
            state["k"] = k
            for kk in range(len(log_lines), k + 1):
                log_lines.append(log_line(kk))
        elif k != state["k"]:
            state["k"] = k
            if k < 0:
                for a_ in (probsc, base, im):
                    a_.set_visible(False)
                topsc.set_offsets(np.empty((0, 2)))
                pcs[0].set_facecolor("#101a25")
            else:
                show_run(k)
        if use_obs and not state["obs"]:
            state["obs"] = True
            pc_o = rankdata(p_obs) / N
            o_ = np.argsort(p_obs)
            probsc.set_offsets(np.column_stack([lon[o_], lat[o_]])); probsc.set_array(p_obs[o_])
            base.set_array(np.clip(pc_o ** 2, 0, 1))
            tm = pc_o >= 1 - K / N + 1e-12
            topsc.set_offsets(np.column_stack([lon[tm], lat[tm]])); topsc.set_color("#ffffff")
            entsc.set_offsets(np.empty((0, 2))); lefsc.set_offsets(np.empty((0, 2)))
            im.set_visible(False)
            ttl_t[2].set_text("Dispatch set (white) from observed rain")
            ttl_t[3].set_text("Forecast runs no longer drive the ranking")
            pcs[0].set_facecolor(county_colors(pd.Series(p_obs).groupby(ctys).sum()))
            topc.set_text("observed-rain ranking")
        rate = max((clock_at(sec + 0.1) - clock_at(sec - 0.1)) / 0.2, 0.0)
        age = (t_abs - av_h[k]) / max(rate, 0.5) if k >= 0 else 99.0
        pulse = float(np.exp(-age / 0.45)) if not use_obs else 0.0
        entsc.set_alpha(min(1.0, pulse * 1.2)); entsc.set_sizes(np.full(max(len(entsc.get_offsets()), 1), 60 + 380 * (1 - pulse)))
        lefsc.set_alpha(min(1.0, pulse * 1.2))
        if k >= 1 and not use_obs:
            im.set_alpha(0.30 + 0.70 * pulse)
        live = (k >= 0) and (not use_obs)
        dot.set_color(GR if (live and pulse > 0.15) or (live and (sec * 2) % 1 < 0.5) else "#1d5a3a")
        if k < 0:
            sub_l.set_text("waiting for the first model run"); runs_t.set_text("RUNS 0")
        elif use_obs:
            sub_l.set_text("observed-rain ranking (after the rain)"); runs_t.set_text(f"RUNS {k + 1}")
        else:
            sub_l.set_text(f"MODEL RUN #{k + 1}   input issued {t_abs - (av_h[k] - 2.0):.1f} h ago, lead {HR[k][1]:.0f} h"); runs_t.set_text(f"RUNS {k + 1}")
        log_t.set_text("\n".join(log_lines[max(0, min(k + 1, len(log_lines)) - 7):min(k + 1, len(log_lines))]) if k >= 0 else "(log empty)")
        seen = wsel[wsel.h <= t_abs]
        lane_art["nwws"].set_offsets(np.column_stack([seen.h.to_numpy() + 10 / 3600, np.full(len(seen), ys[0])]) if len(seen) else np.empty((0, 2)))
        lane_art["api"].set_offsets(np.column_stack([np.ceil(seen.h.to_numpy() * 12) / 12, np.full(len(seen), ys[1])]) if len(seen) else np.empty((0, 2)))
        warned = set()
        for r in seen.itertuples():
            warned |= set(r.counties.split(","))
        warn_pc.set_edgecolor([("#ffffff" if c_ in warned else (0, 0, 0, 0)) for c_ in codes])
        warn_pc.set_linewidth([1.5 if c_ in warned else 0 for c_ in codes])
        g_ = [x[0] for x in pre["6-hourly (+5 h)"] if x[0] <= t_abs]
        lane_art["g"].set_offsets(np.column_stack([g_, np.full(len(g_), ys[2])]) if g_ else np.empty((0, 2)))
        segs = [[(av_h[i], ys[3]), (av_h[i], ys[3] + 0.15 + min(1.0, 6.0 * churn[i]))] for i in range(0, k + 1)] if k >= 0 else []
        bars.set_segments(segs)
        op = ok_open[ok_open <= t_abs]
        lane_art["c"].set_offsets(np.column_stack([op, np.full(len(op), ys[4])]) if len(op) else np.empty((0, 2)))
        shown = open_h <= t_abs
        off = np.column_stack([lon[shown], lat[shown]]) if shown.any() else np.empty((0, 2))
        for cr in claim_rings:
            cr.set_offsets(off)
        for ln, name in ((ln6, "6-hourly (+5 h)"), (lnh, "hourly (+2 h)")):
            xs = [x[0] - H0 for x in pre[name] if x[0] <= t_abs and x[0] - H0 >= -48]
            ys_ = [x[3] for x in pre[name] if x[0] <= t_abs and x[0] - H0 >= -48]
            ln.set_data(xs, ys_)

    if a.still is not None:
        render(a.still)
        fig.savefig(MEDIA / f"countdown4_{S:%Y%m%d}_still_{a.still:.0f}.png", dpi=100, facecolor=fig.get_facecolor())
        return
    nfr = int(END * FPS)
    if a.codec == "hevc":
        codec, extra, suffix = "hevc_videotoolbox", ["-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-b:v", "6M", "-allow_sw", "0"], ""
    else:
        codec, extra, suffix = "libx264", ["-pix_fmt", "yuv420p", "-crf", "20", "-preset", "medium"], "_h264"
    out = MEDIA / f"countdown4_{S:%Y%m%d}{suffix}.mp4"
    w = FFMpegWriter(fps=FPS, codec=codec, extra_args=extra)
    t_ = time.perf_counter()
    with w.saving(fig, str(out), 100):
        for i in range(nfr):
            render(i / FPS)
            w.grab_frame(facecolor=fig.get_facecolor())
    render(END - 0.3)
    fig.savefig(MEDIA / f"countdown4_{S:%Y%m%d}_final.png", dpi=100, facecolor=fig.get_facecolor())
    print(f"wrote {out} ({nfr} frames, {nfr / FPS:.0f} s, encoder {codec}) in {time.perf_counter() - t_:.0f} s")


if __name__ == "__main__":
    main()
