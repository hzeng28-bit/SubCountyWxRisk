"""Update cadence and the two T-0s (scenario report part 2, RESULTS stage 57).

Task framing (Impact Forecasting.pdf): the loss / impact model is checked on simulated fields with a known answer. Here the rain field is the simulated input: the observed 4 km gridMET field
of a test day is the known truth, and a synthetic "forecast issued l hours before the rain" is that truth with a displacement, a correlated amplitude error and a representativeness blur whose
size grows with the lead (one scalar eps; eps at the GEFS-mode lead (60 h) is calibrated so that the observed-rain network `rain + expo` on the synthetic field reproduces the real GEFS-mode block-group
AUC of the stored `fc + expo` run; the growth law eps ~ lead^alpha is an ASSUMPTION, shown for alpha = 0.5 / 0.75 / 1). The claims are the real NFIP claims, so every number has a known answer.
A. Interfaces: detection delay of the real Texas Flash Flood Warnings (2020-2023) under NWWS-OI push (NWS figure: within 10 s) and under polling of api.weather.gov every 1 / 5 / 10 / 15 / 30 min
   (+ an assumed 5-60 s propagation); mean age of the freshest information of each product from its cadence and latency.
B. Skill of the ranking against the lead of the information (synthetic fields, 210 event days 2017-2019 with >= 10 claimed block groups, Texas, S0).
C. Skill on the timeline before the hazard T-0 for a 6-hourly stream (GEFS-like, 5 h latency) and an hourly stream (rapid-refresh-like, 2 h latency).
D. Two T-0s. Hazard T-0 = first Flash Flood Warning for an affected county in [T0-24 h, T0+48 h] (T0 = 12Z of S) when one exists, else T0. Claims T-0 = when the insurer's own claim stream becomes
   usable: the day by which 10% / 50% of the event's claims (loss dates S..S+3, affected counties) have been opened (OpenFEMA `openDate`, day resolution).
Inputs : cnn/ rows and weights (script 91), audit/ffw_tx.parquet (script 98), labels_extra/flood/nfip_tx_claim_dates.parquet (scripts/fetch/fetch_tx_claim_dates.py)
Output : audit/cadence_simulation.txt / .pkl, docs/figures/cadence_summary.png
Run    : python scripts/101_cadence_simulation.py
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT: list[str] = []
LEADS = np.array([1, 3, 6, 12, 24, 36, 48, 60, 72, 84], float)
ALPHAS = (0.5, 0.75, 1.0)
GEFS_LEAD = 60.0  # mean lead of the 2-day window of the stored GEFS-mode run: init T0-36 h, window T0..T0+48 h
TARGET_AUC = 0.778  # mean block-group AUC of fc + expo over the 210 event days (stage 54)


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class RainPipeline:
    """`rain + expo` network (script 91), one day, with an optional replacement of the 4 km rain field."""

    def __init__(self, seed: int = 0):
        import torch

        from wxrisk.encoders import GridSpec, MultiSourceNet

        self.torch = torch
        self.e81 = _load("e81", "81_cnn_e1.py")
        self.d = d = self.e81.Data()
        rng = np.random.default_rng(seed)
        train_idx = np.flatnonzero(d.is_train)
        sample_days = rng.choice(train_idx, 300, replace=False)
        sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in sample_days])
        tf = self.e81.tab_features(sample)
        self.mu, self.sd = tf.mean(0), tf.std(0) + 1e-6
        self.model = MultiSourceNet([GridSpec("rain", 3), GridSpec("expo", 3)], [], 9, proj=8, ch=32, dilations=(1, 2, 4, 8, 16), modality_dropout=0.0)
        self.model.load_state_dict(torch.load(self.e81.D / f"ms_rain+expo_s{seed}.pt", map_location="cpu"))
        self.model.eval()
        self.up = torch.from_numpy(d.cell4)
        self.valid = d.valid4.astype(np.float32)

    def rain_maps(self, n: int) -> np.ndarray:
        return np.asarray(self.d.O[n], np.float32)

    def day(self, n: int, o: np.ndarray | None = None):
        torch, d = self.torch, self.d
        bgs, tab, cnt, _ = d.rows(n)
        num = (self.e81.tab_features(tab) - self.mu) / self.sd
        img = d.image("4km", n)
        if o is not None:
            img = img.copy()
            img[0], img[1] = np.log1p(np.clip(o[0], 0, None)) / 3, np.log1p(np.clip(o[1], 0, None)) / 3
        f = {"rain": torch.from_numpy(img[0:3][None]), "expo": torch.from_numpy(img[3:6][None])}
        with torch.no_grad():
            out = self.model(f, {}, self.up, torch.from_numpy(d.bpix4[bgs]), torch.from_numpy(num.astype(np.float32)), torch.zeros((len(bgs), 0), dtype=torch.long))
            p = torch.sigmoid(out).numpy()
        return bgs, p, cnt


def perturb(o: np.ndarray, eps: float, z: dict, valid: np.ndarray) -> np.ndarray:
    """Synthetic forecast of the observed 2-channel field: rigid displacement (sd 25 km x eps), correlated amplitude error (sd 0.5 x eps, 50 km), blur (4 px x eps)."""
    if eps <= 0:
        return o
    px = 4.0  # km per pixel
    dy, dx = z["shift"] * eps * 25.0 / px
    amp = np.exp(0.5 * eps * z["amp"])
    out = []
    for c in range(2):
        f = shift(o[c], (dy, dx), order=1, mode="constant", cval=0.0)
        f = f * amp
        if eps * 4.0 > 0.3:
            f = gaussian_filter(f, eps * 4.0)
        out.append(f * valid)
    return np.stack(out).astype(np.float32)


def noise(shape, rng) -> dict:
    g = gaussian_filter(rng.standard_normal(shape), 12.0)
    return {"shift": rng.standard_normal(2), "amp": (g / g.std()).astype(np.float32)}


def event_metrics(e75, p, cnt):
    y = (cnt > 0).astype(int)
    a = e75.pair_auc(p, y)[0]
    m = max(int(np.ceil(0.05 * len(p))), 1)
    top = e75.top_idx(p, m)
    return a, cnt[top].sum() / cnt.sum()


def interface_part(ffw: pd.DataFrame):
    rng = np.random.default_rng(0)
    w = ffw[ffw.issued >= "2020-01-01"]
    wea = w[w.damage.isin(["CONSIDERABLE", "CATASTROPHIC"])]
    say(f"Texas FF Warnings 2020-2023: {len(w)}; with a considerable / catastrophic damage tag (the only ones that trigger a Wireless Emergency Alert): {len(wea)} ({len(wea) / len(w):.0%}); emergencies {int(w.emergency.sum())}")
    res = {}
    n = 20000
    res["NWWS-OI push (NWS: within 10 s)"] = rng.uniform(0, 10, n)
    for P in (1, 5, 10, 15, 30):
        res[f"api.weather.gov poll every {P} min (+5-60 s propagation, assumed)"] = rng.uniform(0, P * 60, n) + rng.uniform(5, 60, n)
    say("detection delay of a new warning after issuance (seconds; median / 95th percentile):")
    for k, v in res.items():
        say(f"   {k:62s} {np.median(v):7.0f} / {np.percentile(v, 95):7.0f}")
    dur = (w.expired - w.issued).dt.total_seconds()
    say(f"warning duration (s): median {dur.median():.0f}; delay of the 30-min poll at its 95th percentile uses {np.percentile(res['api.weather.gov poll every 30 min (+5-60 s propagation, assumed)'], 95) / dur.median():.1%} of it")
    # age of the freshest information from cadence and latency (hours)
    prod = {"GEFS-like cycle (6 h cycle, +5 h, assumed)": (6.0, 5.0), "rapid-refresh-like hourly run (+1.9 h, third-party figure)": (1.0, 1.9),
            "MRMS multi-sensor QPE Pass 1 (hourly, +20 min, NSSL)": (1.0, 1 / 3), "MRMS multi-sensor QPE Pass 2 (hourly, +60 min, NSSL)": (1.0, 1.0)}
    ages = {}
    for k, (cad, lat) in prod.items():
        t = rng.uniform(0, 24 * 7, 50000)
        ages[k] = (t - np.floor((t - lat) / cad) * cad - lat) + lat
    say("age of the freshest product at a random moment (hours; median / 95th percentile):")
    for k, a in ages.items():
        say(f"   {k:62s} {np.median(a):5.2f} / {np.percentile(a, 95):5.2f}")
    return res, ages, w, wea


def claims_part(T_events: pd.DataFrame, ffw: pd.DataFrame):
    c = pd.read_parquet(wb2.cache_path("labels_extra") / "flood" / "nfip_tx_claim_dates.parquet")
    c = c[c.causeOfDamage.astype(str).isin(["2", "3", "4"])].copy()
    c["dl"] = pd.to_datetime(c.dateOfLoss, utc=True).dt.tz_localize(None).dt.normalize()
    c["od"] = pd.to_datetime(c.openDate, utc=True).dt.tz_localize(None).dt.normalize()
    c = c[c.od.notna() & c.dl.notna()]
    lag = (c.od - c.dl).dt.days
    say(f"\nTexas NFIP flood claims, loss years 2017-2023: {len(c):,} with an open date; claim opened on the loss day or before: {np.mean(lag <= 0):.0%}; "
        f"lag (days) p10 {lag.quantile(.1):.0f}, median {lag.median():.0f}, p90 {lag.quantile(.9):.0f}")
    curves, ct10, ct50 = [], [], []
    ks = np.arange(0, 31)
    for sid, g in T_events.groupby("sysid"):
        S = pd.Timestamp(sid)
        cc = c[(c.dl >= S) & (c.dl <= S + pd.Timedelta(3, "D")) & c.countyCode.isin(set(g.cty))]
        if len(cc) < 5:
            continue
        d_open = (cc.od - S).dt.days.to_numpy()
        cum = np.array([np.mean(d_open <= k) for k in ks])
        curves.append(cum)
        ct10.append(ks[np.argmax(cum >= 0.10)])
        ct50.append(ks[np.argmax(cum >= 0.50)])
    curves = np.array(curves)
    say(f"events with >= 5 claims in the affected counties: {len(curves)}")
    for k in (0, 1, 2, 3, 7, 14):
        say(f"   share of the event's claims opened by day S+{k}: median {np.median(curves[:, k]):.0%} (p25 {np.percentile(curves[:, k], 25):.0%}, p75 {np.percentile(curves[:, k], 75):.0%})")
    say(f"   day by which 10% of the claims are opened: median S+{np.median(ct10):.0f} (p10 S+{np.percentile(ct10, 10):.0f}, p90 S+{np.percentile(ct10, 90):.0f}); 50%: median S+{np.median(ct50):.0f} (p10 S+{np.percentile(ct50, 10):.0f}, p90 S+{np.percentile(ct50, 90):.0f})")
    return curves, np.array(ct10), np.array(ct50), lag


def main() -> None:
    e75, s96, s98 = _load("e75", "75_task_metrics.py"), _load("s96", "96_task_scorecard.py"), _load("s98", "98_scenario_analysis.py")
    ffw = s98.load_ffw()
    res_if, ages, w, wea = interface_part(ffw)

    # ---- B. synthetic fields
    pipe = RainPipeline()
    d = pipe.d
    test = np.flatnonzero(d.is_test)
    big = [int(n) for n in test if (d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).sum() >= 10]
    say(f"\nsynthetic-field test: {len(big)} event days with >= 10 claimed block groups (2017-2019)")
    # check the pipeline against the stored predictions (observed rain, eps = 0)
    stored = np.load(pipe.e81.D / "ms_rain+expo_s0.npy")
    lo = d.ptr[test[0]]
    n0 = big[0]
    _, p0, _ = pipe.day(n0)
    say(f"pipeline vs stored predictions on day {n0}: max abs diff {np.abs(p0 - stored[d.ptr[n0] - lo:d.ptr[n0 + 1] - lo]).max():.1e}")
    rngs = {n: np.random.default_rng(1000 + n) for n in big}
    Z = {n: noise(d.valid4.shape, rngs[n]) for n in big}
    raw = {n: pipe.rain_maps(n) for n in big}

    def run(eps: float):
        t_ = time.perf_counter()
        a, c = [], []
        for n in big:
            o = perturb(raw[n], eps, Z[n], pipe.valid)
            _, p, cnt = pipe.day(n, o)
            r = event_metrics(e75, p, cnt)
            a.append(r[0]); c.append(r[1])
        return float(np.nanmean(a)), float(np.nanmean(c)), time.perf_counter() - t_

    # anchors
    pol = [event_metrics(e75, d.rtab[d.ptr[n]:d.ptr[n + 1], 8].astype(float), d.rn[d.ptr[n]:d.ptr[n + 1]]) for n in big]
    say(f"policies only: BG AUC {np.nanmean([x[0] for x in pol]):.3f}, top-5% capture {np.nanmean([x[1] for x in pol]):.3f}; real GEFS-mode (fc + expo): AUC {TARGET_AUC:.3f}, top-5% 0.274 (stage 54)")
    grid_eps = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    cal = {}
    for e in grid_eps:
        a, c, t = run(e)
        cal[e] = (a, c)
        say(f"  eps {e:3.1f}: BG AUC {a:.3f}, top-5% capture {c:.3f}  ({t:.0f} s)")
    es, au = np.array(grid_eps), np.array([cal[e][0] for e in grid_eps])
    eps60 = float(np.interp(TARGET_AUC, au[::-1], es[::-1]))
    say(f"calibration: eps at the GEFS-mode lead (60 h) = {eps60:.2f} reproduces the real BG AUC {TARGET_AUC}")
    curves = {}
    for al in ALPHAS:
        pts = []
        for l in LEADS:
            e = eps60 * (l / GEFS_LEAD) ** al
            pts.append(run(e)[:2])
        curves[al] = np.array(pts)
        say(f"  alpha {al}: leads {LEADS.astype(int).tolist()} -> BG AUC " + " ".join(f"{x[0]:.3f}" for x in curves[al]) + " | top-5% " + " ".join(f"{x[1]:.3f}" for x in curves[al]))

    # ---- C. timeline before the hazard T-0
    # per event: hazard T-0 and the stream states
    s_events = {}
    sids = [pd.Timestamp(d.days[n]) for n in big]
    for n, S in zip(big, sids, strict=True):
        t0 = S + pd.Timedelta(12, "h")
        bgs, _, cnt, _ = d.rows(n)
        aff = set(np.array(d.ctys)[d.cty_idx[bgs[cnt > 0]]])
        m = (ffw.issued >= t0 - pd.Timedelta(24, "h")) & (ffw.issued <= t0 + pd.Timedelta(48, "h"))
        cand = [(r.issued, r) for r in ffw[m].itertuples() if aff & set(r.counties.split(","))]
        ht0 = min(cand, key=lambda x: x[0])[0] if cand else t0
        wea_ = [r.issued for _, r in cand if r.damage in ("CONSIDERABLE", "CATASTROPHIC")]
        s_events[n] = {"t0": t0, "ht0": ht0, "has_warning": bool(cand), "wea": min(wea_) if wea_ else None}
    hh = np.array([(v["ht0"] - v["t0"]).total_seconds() / 3600 for v in s_events.values()])
    say(f"\nhazard T-0 (first affected-county warning, else T0): {np.mean([v['has_warning'] for v in s_events.values()]):.0%} of the events have a warning; hours from T0: p10 {np.percentile(hh, 10):+.1f}, median {np.median(hh):+.1f}, p90 {np.percentile(hh, 90):+.1f}")
    DELTAS = np.array([48, 36, 24, 12, 6, 3, 1, 0.5])

    def stream_lead(delta, ht0_h, cadence, latency):
        """lead (h) of the freshest run available delta hours before the hazard T-0, relative to the middle of the 2-day rain window (T0 + 24 h)."""
        t = ht0_h - delta  # clock hours from T0
        s = np.floor((t - latency) / cadence) * cadence  # init time of the freshest run (cycles on multiples of the cadence from 00Z; T0 = 12Z)
        s = s + 0.0
        return max((24.0 - s), 1.0)

    # cycles are anchored at 00Z: T0 is 12Z, so a cycle time is -12 + k * cadence relative to T0
    def lead_for(delta, ht0_h, cadence, latency):
        t = ht0_h - delta
        k = np.floor((t - latency + 12.0) / cadence)
        s = -12.0 + k * cadence
        return max(24.0 - s, 1.0)

    mid = curves[0.75]
    timeline = {}
    for name, (cad, lat) in {"6-hourly stream (+5 h)": (6.0, 5.0), "hourly stream (+2 h)": (1.0, 2.0)}.items():
        arr = np.zeros((len(DELTAS), len(big)))
        arr_lo = np.zeros_like(arr)
        arr_hi = np.zeros_like(arr)
        for j, (n, v) in enumerate(s_events.items()):
            h_ = (v["ht0"] - v["t0"]).total_seconds() / 3600
            for i, dl in enumerate(DELTAS):
                l = lead_for(dl, h_, cad, lat)
                arr[i, j] = np.interp(l, LEADS, mid[:, 0])
                arr_lo[i, j] = np.interp(l, LEADS, curves[1.0][:, 0])
                arr_hi[i, j] = np.interp(l, LEADS, curves[0.5][:, 0])
        timeline[name] = (arr.mean(1), arr_lo.mean(1), arr_hi.mean(1))
        say(f"{name}: expected block-group AUC at decision time (hours before the hazard T-0 {DELTAS.tolist()}): " + " ".join(f"{x:.3f}" for x in arr.mean(1)) + f"  (alpha 1.0: {arr_lo.mean(1)[0]:.3f}..{arr_lo.mean(1)[-1]:.3f}; alpha 0.5: {arr_hi.mean(1)[0]:.3f}..{arr_hi.mean(1)[-1]:.3f})")
    # number of updates received between T0-36 h and the hazard T-0
    for name, (cad, lat) in {"6-hourly stream (+5 h)": (6.0, 5.0), "hourly stream (+2 h)": (1.0, 2.0)}.items():
        cnts = []
        for n, v in s_events.items():
            h_ = (v["ht0"] - v["t0"]).total_seconds() / 3600
            ks = np.arange(-60, 60)
            avail = (-12.0 + ks * cad) + lat
            cnts.append(int(((avail >= -36) & (avail <= h_)).sum()))
        say(f"{name}: updates available between T0-36 h and the hazard T-0: median {np.median(cnts):.0f} (p10 {np.percentile(cnts, 10):.0f}, p90 {np.percentile(cnts, 90):.0f})")

    # ---- D. claims
    S_ev = pd.DataFrame({"sysid": [S.strftime("%Y%m%d") for S in sids]})
    t_obs, _, _ = s96.build_observed(_load("e85", "85_state_group_baseline.py"))
    t_obs = t_obs.assign(y=t_obs.n > 0)
    evs = t_obs.groupby("sysid").y.sum()
    evs = evs[evs >= 10].index
    Tclaims = t_obs[t_obs.sysid.isin(evs) & (t_obs.n > 0)][["sysid", "cty"]].drop_duplicates()
    curves_c, ct10, ct50, lag = claims_part(Tclaims, ffw)
    pickle.dump({"res_if": res_if, "ages": ages, "cal": cal, "eps60": eps60, "curves": curves, "LEADS": LEADS, "timeline": timeline, "DELTAS": DELTAS, "hh": hh, "claims": (curves_c, ct10, ct50, lag.to_numpy()),
                 "events": {int(n): {k: (str(x) if x is not None else None) for k, x in v.items()} for n, v in s_events.items()}, "wea_share": len(wea) / len(w), "pol": (np.nanmean([x[0] for x in pol]), np.nanmean([x[1] for x in pol]))},
                open(wb2.cache_path("audit") / "cadence_simulation.pkl", "wb"))
    (wb2.cache_path("audit") / "cadence_simulation.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
