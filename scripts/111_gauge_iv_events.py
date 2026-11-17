"""15-minute discharge at Texas gauges in the largest claim events, and lag-aligned conservation (stage 59, part 2).

Daily means mix travel time and loss. Here the event windows (largest claim days of the pilot, 2012-2023) are fetched at
15 minutes (`waterdata.get_continuous`), the travel lag of each gauge / immediate-upstream group is estimated per event by
cross-correlation, and the volume ratio  V_down / V_up(lag)  and peak ratio are compared with the daily-mean flag.

Inputs : usgs_tx/tx_gauges_q, tx_basins.geojson, tx_mass_balance (scripts 108, 109), flood_pilot_table
Outputs: usgs_tx/tx_iv_events.parquet, usgs_tx/tx_iv_balance.parquet
Run    : python scripts/111_gauge_iv_events.py fetch | balance
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

D = wb2.cache_path("labels_extra") / "flood" / "usgs_tx"
spec = importlib.util.spec_from_file_location("g109", Path(__file__).with_name("109_gauge_network_test.py"))
g109 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g109)

N_EVENTS = 14


def events() -> pd.DataFrame:
    """Largest claim days by total claims (one per 5-day cluster), padded 2 days before and 5 after."""
    t = pd.read_parquet(wb2.cache_path("labels_extra") / "flood" / "pilot" / "flood_pilot_table.parquet",
                        columns=["start", "n_claims"])
    t["start"] = pd.to_datetime(t.start)
    s = t[t.start >= "2012-01-01"].groupby("start").n_claims.sum().sort_values(ascending=False)
    keep = []
    for d, v in s.items():
        if all(abs((d - k).days) > 5 for k, _ in keep):
            keep.append((d, v))
        if len(keep) == N_EVENTS:
            break
    return pd.DataFrame(keep, columns=["day", "claims"]).sort_values("day").reset_index(drop=True)


def fetch() -> None:
    from dataretrieval import waterdata

    g, _ = g109.load()
    ids = list(g.index)
    ev = events()
    print(ev.to_string())
    out = D / "tx_iv_events.parquet"
    rows = pd.read_parquet(out) if out.exists() else None
    have = set() if rows is None else set(zip(rows.event.astype(str), rows.monitoring_location_id))
    parts = [] if rows is None else [rows]
    for k, e in ev.iterrows():
        w = f"{(e.day - pd.Timedelta(days=2)).strftime('%Y-%m-%dT00:00:00Z')}/{(e.day + pd.Timedelta(days=5)).strftime('%Y-%m-%dT00:00:00Z')}"
        todo = [i for i in ids if (str(e.day.date()), i) not in have]
        for j in range(0, len(todo), 25):
            for attempt in range(4):
                try:
                    c, _ = waterdata.get_continuous(monitoring_location_id=todo[j:j + 25], parameter_code="00060",
                                                    time=w)
                    break
                except Exception as ex:
                    print(f"  retry {type(ex).__name__}", flush=True)
                    time.sleep(30 * (attempt + 1))
            else:
                raise RuntimeError("continuous fetch failed")
            if len(c):
                c = c[["monitoring_location_id", "time", "value"]].copy()
                c["event"] = str(e.day.date())
                c["value"] = pd.to_numeric(c["value"], errors="coerce")
                parts.append(c)
        pd.concat(parts, ignore_index=True).to_parquet(out)
        print(f"event {k + 1}/{len(ev)} {e.day.date()} done, {sum(len(p) for p in parts):,} rows", flush=True)


def balance() -> None:
    g, _ = g109.load()
    iv = pd.read_parquet(D / "tx_iv_events.parquet")
    iv["time"] = pd.to_datetime(iv["time"], utc=True)
    pairs = g109.immediate_pairs(g)
    mb = pd.read_parquet(D / "tx_mass_balance.parquet").set_index("site")
    rows = []
    for ev, x in iv.groupby("event"):
        q = x.pivot_table(index="time", columns="monitoring_location_id", values="value").resample("15min").mean()
        for s, imm in pairs:
            if s not in q.columns or not imm or s not in mb.index:
                continue
            imm = [u for u in imm if u in q.columns]
            if not imm:
                continue
            up = q[imm].sum(axis=1, min_count=len(imm))
            d = pd.concat([q[s].rename("dn"), up.rename("up")], axis=1).dropna()
            if len(d) < 200 or d.up.max() < 50:
                continue
            # lag in 15-min steps, 0-4 days, maximising correlation of the series
            best = None
            for lag in range(0, 4 * 96, 2):
                a = d.dn.iloc[lag:].to_numpy()
                b = d.up.iloc[: len(d) - lag].to_numpy()
                if len(a) < 100:
                    break
                c = np.corrcoef(a, b)[0, 1]
                if best is None or c > best[1]:
                    best = (lag, c)
            lag, c = best
            a, b = d.dn.iloc[lag:], d.up.iloc[: len(d) - lag]
            rows.append(dict(event=ev, site=s, n_up=len(imm), lag_h=lag / 4, corr=c, vol_ratio=a.sum() / b.sum(),
                             peak_ratio=a.max() / b.max(), up_peak=b.max(), cov=mb.loc[s, "cov"],
                             daily_loss_share=mb.loc[s, "loss_share"], daily_ratio_all=mb.loc[s, "ratio_med_all"]))
    t = pd.DataFrame(rows)
    t.to_parquet(D / "tx_iv_balance.parquet")
    print(f"{len(t)} group-events, {t.site.nunique()} groups")
    print(t[["lag_h", "corr", "vol_ratio", "peak_ratio"]].describe().round(3))
    nat = t[(t["corr"] >= 0.85) & t.daily_ratio_all.between(0.8, 1.25)]
    print(f"\nnatural-looking groups (daily screen) with IV: {nat.site.nunique()}, group-events {len(nat)}")
    print(nat[["lag_h", "vol_ratio", "peak_ratio"]].describe().round(3))
    per = nat.groupby("site").agg(n=("event", "size"), lag_h=("lag_h", "median"), vol=("vol_ratio", "median"),
                                  peak=("peak_ratio", "median"), vol_min=("vol_ratio", "min"),
                                  daily_loss=("daily_loss_share", "first"))
    print(per.sort_values("vol").head(12).round(2).to_string())
    print("\nSpearman between daily loss share and IV median volume ratio:",
          round(per[["daily_loss", "vol"]].corr("spearman").iloc[0, 1], 3))


if __name__ == "__main__":
    {"fetch": fetch, "balance": balance}[sys.argv[1]]()
