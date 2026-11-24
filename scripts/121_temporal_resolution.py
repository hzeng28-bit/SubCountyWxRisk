"""How fine in time does the hydrologic information have to be? 15-minute discharge in the 14 largest claim events (stage 66).

Data: 15-minute discharge (script 111, 2012-2019 events, day -2..+5 around each event day) at the small unregulated gauges of
script 114, converted to local time (America/Chicago); claims by loss date (local, one day resolution) in the gauge's basin.
  A  flood-wave time scales by drainage area: rise time (10% -> 100% of the peak above base), width at half height, and the
     ratio peak / local-day mean flow (what a daily mean hides)
  B  day-boundary sensitivity: the daily max recomputed with the day starting at 00, 06, 12, 18 local; Spearman with claims per
     day and share of gauge-events where the day of peak flow equals the day with most claims; by hour of the peak
  C  does sub-daily shape (1-hour rise rate, share of the day above half the peak) add to policies + rain + daily-max percentile
     for 'any claim in the basin that day'? leave-one-event-out AUC, 14 events
Run : python scripts/121_temporal_resolution.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


g116 = _load("g116", "116_gauge_outcomes_vs_claims.py")
D, DAYS, auc = g116.D, g116.DAYS, g116.auc
rng = np.random.default_rng(0)


def main():
    g, S, R1, R3, mem, ring, bg, pos, ex, ncl, paid = g116.build()
    fr = [f for s in g.index if (f := g116.gauge_frame(s, S, R1, R3, mem, bg, pos, ex, ncl, paid)) is not None]
    F = pd.concat(fr, ignore_index=True)
    F = F[F.pol.groupby(F.site).transform("mean") >= 20]
    F["rain_pct"] = F.groupby("site").p3.rank(pct=True)
    Fi = F.set_index(["site", "day"])
    area = g.drainage_area
    iv = pd.read_parquet(D / "tx_iv_events.parquet")
    iv = iv[iv.monitoring_location_id.isin(F.site.unique())].copy()
    iv["t"] = iv.time.dt.tz_convert("America/Chicago")
    print(f"15-min rows at the selected gauges: {len(iv):,}; gauges {iv.monitoring_location_id.nunique()}; events {iv.event.nunique()}")

    waves, days_rows, align = [], [], []
    for (s, ev), d in iv.groupby(["monitoring_location_id", "event"]):
        d = d.dropna(subset=["value"]).sort_values("t")
        if len(d) < 300:
            continue
        t = d.t.dt.tz_localize(None).to_numpy()  # naive local wall time
        q = d.value.to_numpy()
        base, pk = np.percentile(q, 5), q.max()
        ip = int(q.argmax())
        dlt = pk - base
        ok = pk >= 5 and dlt >= 2 * max(base, 0.5)
        # local days with a full record
        ser = pd.Series(q, index=pd.DatetimeIndex(t))
        full = ser.groupby(ser.index.normalize()).size()
        full = full[full >= 90].index
        dm = ser.groupby(ser.index.normalize()).mean()
        if ok:
            th = base + 0.1 * dlt
            j = ip
            while j > 0 and q[j] > th:
                j -= 1
            rise_h = (t[ip] - t[j]) / np.timedelta64(1, "h")
            hh = base + 0.5 * dlt
            a = ip
            while a > 0 and q[a] >= hh:
                a -= 1
            b = ip
            while b < len(q) - 1 and q[b] >= hh:
                b += 1
            width_h = (t[b] - t[a]) / np.timedelta64(1, "h")
            pday = pd.Timestamp(t[ip]).normalize()
            waves.append((s, ev, area.get(s, np.nan), rise_h, width_h, pk / max(dm.get(pday, np.nan), 1e-6), pk, pd.Timestamp(t[ip]).hour))
        # per-day features and offset daily maxima
        for off in (0, 6, 12, 18):
            sh = ser.copy()
            sh.index = sh.index - pd.Timedelta(hours=off)
            cnt = sh.groupby(sh.index.normalize()).size()
            mx = sh.groupby(sh.index.normalize()).max()
            mx = mx[cnt >= 90]
            for dday, v in mx.items():
                days_rows.append((s, ev, off, dday, v))
        # 1-h rise and duration above half the day's max, local days 00-24
        for dday in full:
            x = ser[(ser.index >= dday) & (ser.index < dday + pd.Timedelta(days=1))]
            if len(x) < 90 or x.max() <= 0:
                continue
            xv = x.to_numpy()
            rise = np.max(xv[4:] - xv[:-4]) / x.max() if len(xv) > 4 else 0
            dur = (xv >= 0.5 * x.max()).mean()
            align.append((s, ev, dday, x.max(), x.mean(), rise, dur, x.idxmax().hour))
    W = pd.DataFrame(waves, columns=["site", "event", "area", "rise_h", "width_h", "peak_over_dmean", "peak", "peak_hour"])
    W["abin"] = pd.cut(W.area, [0, 25, 100, 250], labels=["<=25 mi2", "25-100", "100-250"])
    print(f"\n=== A. Flood-wave time scales ({len(W)} gauge-events with a clear wave, peak >= 5 cfs and >= 2x base) ===")
    t = W.groupby("abin", observed=True).agg(n=("site", "size"), rise_h_median=("rise_h", "median"), rise_h_q25=("rise_h", lambda x: x.quantile(.25)),
                                             width_h_median=("width_h", "median"), width_h_q25=("width_h", lambda x: x.quantile(.25)),
                                             peak_over_daily_mean_median=("peak_over_dmean", "median"),
                                             share_width_lt_6h=("width_h", lambda x: (x < 6).mean()),
                                             share_width_lt_24h=("width_h", lambda x: (x < 24).mean()))
    print(t.round(2).to_string())
    print(f"  all: rise {W.rise_h.median():.1f} h, half-width {W.width_h.median():.1f} h, peak / daily mean {W.peak_over_dmean.median():.2f} "
          f"(90th pct {W.peak_over_dmean.quantile(.9):.2f})")

    # ---- B ----
    O = pd.DataFrame(days_rows, columns=["site", "event", "off", "day", "qmax"])
    A = pd.DataFrame(align, columns=["site", "event", "day", "qmax", "qmean", "rise", "dur", "peak_hour"])
    cl = F.set_index(["site", "day"]).claims
    print("\n=== B. Day-boundary sensitivity (gauge-events with >= 10 claims in the window) ===")
    rows = []
    for off in (0, 6, 12, 18):
        o = O[O.off == off]
        rs, hit, n = [], 0, 0
        for (s, ev), d in o.groupby(["site", "event"]):
            c = np.array([cl.get((s, pd.Timestamp(x) + pd.Timedelta(hours=off).floor("D")), np.nan) if False else cl.get((s, pd.Timestamp(x).normalize()), np.nan) for x in d.day])
            if np.isnan(c).any() or c.sum() < 10 or len(d) < 5:
                continue
            rs.append(pd.Series(d.qmax.to_numpy()).corr(pd.Series(c), method="spearman"))
            hit += int(d.day.to_numpy()[d.qmax.to_numpy().argmax()] == d.day.to_numpy()[c.argmax()])
            n += 1
        rows.append((f"day starts {off:02d}:00 local", n, np.nanmean(rs), hit / max(n, 1)))
    print(pd.DataFrame(rows, columns=["day definition", "gauge-events", "mean Spearman daily max vs claims/day", "peak-day = claim-peak-day"]).round(3).to_string(index=False))
    # hour of peak vs claim-peak day
    rows = []
    for (s, ev), d in A.groupby(["site", "event"]):
        c = np.array([cl.get((s, pd.Timestamp(x)), np.nan) for x in d.day])
        if np.isnan(c).any() or c.sum() < 10 or len(d) < 5:
            continue
        ip = int(d.qmax.to_numpy().argmax())
        ic = int(c.argmax())
        rows.append((d.peak_hour.iloc[ip], ic - ip))
    H = pd.DataFrame(rows, columns=["peak_hour", "claim_day_minus_peak_day"])
    H["hour"] = pd.cut(H.peak_hour, [-1, 5, 11, 17, 23], labels=["00-06", "06-12", "12-18", "18-24"])
    print(f"  claim-peak day minus flow-peak day by local hour of the flow peak (n = {len(H)}):")
    print(H.groupby("hour", observed=True).claim_day_minus_peak_day.agg(n="size", same_day=lambda x: (x == 0).mean(),
                                                                       next_day=lambda x: (x == 1).mean(), prev_day=lambda x: (x == -1).mean()).round(2).to_string())

    # ---- C ----
    print("\n=== C. Does sub-daily shape add to policies + rain + daily-max percentile? (event gauge-days, leave-one-event-out AUC) ===")
    A["day"] = pd.to_datetime(A.day)
    M = A.merge(F[["site", "day", "claims", "pol", "rain_pct", "qmax_pct"]], on=["site", "day"], how="inner").dropna(subset=["qmax_pct", "rain_pct"])
    M = M[M.pol > 0]
    M["y"] = (M.claims > 0).astype(int)
    M["lp"] = np.log1p(M.pol)
    M["peak_ratio"] = M.qmax / M.qmean.clip(lower=1e-6)
    print(f"  gauge-days {len(M):,}; with a claim {int(M.y.sum())}; events {M.event.nunique()}")
    sets = {"pol + rain": ["lp", "rain_pct"], "+ daily-max percentile": ["lp", "rain_pct", "qmax_pct"],
            "+ 1-h rise rate": ["lp", "rain_pct", "qmax_pct", "rise"], "+ share of day above half peak": ["lp", "rain_pct", "qmax_pct", "dur"],
            "+ peak / daily mean": ["lp", "rain_pct", "qmax_pct", "peak_ratio"], "+ all three": ["lp", "rain_pct", "qmax_pct", "rise", "dur", "peak_ratio"]}
    evs = M.event.unique()
    sc = {k: np.zeros(len(M)) for k in sets}
    for e in evs:
        te = (M.event == e).to_numpy()
        for k, c in sets.items():
            s_ = StandardScaler().fit(M.loc[~te, c])
            sc[k][te] = LogisticRegression(max_iter=500).fit(s_.transform(M.loc[~te, c]), M.y[~te]).decision_function(s_.transform(M.loc[te, c]))
    y, sit = M.y.to_numpy(), M.site.to_numpy()
    us = np.unique(sit)
    gi = {u: np.flatnonzero(sit == u) for u in us}
    draws = [np.concatenate([gi[u] for u in rng.choice(us, len(us))]) for _ in range(200)]
    b = sc["+ daily-max percentile"]
    for k, s_ in sc.items():
        d = [auc(y[i], s_[i]) - auc(y[i], b[i]) for i in draws] if k not in ("pol + rain", "+ daily-max percentile") else [np.nan, np.nan]
        print(f"  {k:<34} AUC {auc(y, s_):.3f}   vs daily-max model {auc(y, s_) - auc(y, b):+.4f} [{np.nanpercentile(d, 2.5) if not np.isnan(d[0]) else np.nan:+.4f}, {np.nanpercentile(d, 97.5) if not np.isnan(d[0]) else np.nan:+.4f}]")


if __name__ == "__main__":
    main()
