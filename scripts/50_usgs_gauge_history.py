"""USGS streamgage history as a susceptibility prior for the Texas flood pilot (stage 18).

Fetch (api.waterdata.usgs.gov OGC API, anonymous): Texas stream locations with a daily-mean discharge series, then each
location's annual peak discharges. Features use only peaks before 2010 (the pilot starts in 2010, so they are known ahead):
for the nearest eligible gauge of a block group (>= 10 pre-2010 peaks, drainage area <= 1000 mi2, within 25 km):
  g_dist       km to that gauge
  g_larea      log drainage area
  g_spec       log of median annual peak per unit area (flashiness / runoff proxy, flood-prone basins are high)
  g_cv         standard deviation of log annual peaks
  g_ratio      log of (largest peak / median peak): how far the tail goes above the typical flood
Test: base model plus claim history (script 49) plus these, observed-rain arm (test 2017-2023) and forecast arm (2017-2019).

Inputs : flood_pilot_table.parquet, scripts 48/49 caches
Outputs: $WXRISK_DATA/labels_extra/flood/usgs_tx/{tx_gauges_q,tx_peaks_q}.parquet, audit/usgs_gauge_history.txt
Run    : python scripts/50_usgs_gauge_history.py --fetch      # gauges and peaks (about 900 small requests)
         python scripts/50_usgs_gauge_history.py              # test
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import fetch as dl  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import wb2  # noqa: E402

API = "https://api.waterdata.usgs.gov/ogcapi/v0"
D = wb2.cache_path("labels_extra") / "flood" / "usgs_tx"
HDR = {"User-Agent": "Mozilla/5.0"}
OUT = []


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def get_json(url: str) -> dict:
    """Anonymous use is capped at 4000 requests an hour; a 429 carries Retry-After, wait it out instead of failing."""
    import time
    import urllib.error

    for _ in range(12):
        try:
            return json.loads(dl.get_bytes(url, headers=HDR, retries=2, timeout=120))
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            wait = int(e.headers.get("Retry-After", "300")) + 5
            print(f"  rate limited, waiting {wait} s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"still rate limited: {url}")


def pages(url: str, maxp: int = 80) -> list[dict]:
    out = []
    for _ in range(maxp):
        r = get_json(url)
        out += r["features"]
        nxt = [x["href"] for x in r.get("links", []) if x.get("rel") == "next"]
        if not nxt:
            break
        url = nxt[0]
    return out


def fetch() -> None:
    D.mkdir(parents=True, exist_ok=True)
    ts = pages(f"{API}/collections/time-series-metadata/items?f=json&limit=1000&parameter_code=00060&state_name=Texas")
    q = pd.DataFrame([f["properties"] for f in ts])
    q = q[q.statistic_id == "00003"].drop_duplicates("monitoring_location_id")
    ml = pages(f"{API}/collections/monitoring-locations/items?f=json&limit=1000&state_name=Texas&site_type_code=ST")
    m = pd.DataFrame([{**f["properties"], "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
                      for f in ml if f.get("geometry")])
    m = m[m.id.isin(q.monitoring_location_id)].merge(
        q[["monitoring_location_id", "begin_utc", "end_utc"]], left_on="id", right_on="monitoring_location_id")
    m = m[["id", "monitoring_location_name", "county_code", "hydrologic_unit_code", "drainage_area", "lon", "lat",
           "begin_utc", "end_utc"]]
    m.to_parquet(D / "tx_gauges_q.parquet")
    print(f"{len(m)} gauges with daily discharge", flush=True)

    def peaks(i):
        try:
            f = pages(f"{API}/collections/peaks/items?f=json&limit=1000&monitoring_location_id={i}&parameter_code=00060")
        except Exception as e:  # noqa: BLE001 - one gauge must not stop the batch
            return i, str(e)[:80], []
        return i, None, [x["properties"] for x in f]

    part = D / "tx_peaks_q.partial.parquet"  # resumable: a gauge already in the file is not requested again
    have = pd.read_parquet(part) if part.exists() else pd.DataFrame()
    done = set(have.get("monitoring_location_id", []))
    rows, bad = ([have] if len(have) else []), []
    todo = [i for i in m.id if i not in done]
    print(f"{len(todo)} gauges to fetch ({len(done)} cached)", flush=True)
    with ThreadPoolExecutor(4) as ex:
        for k, (i, err, p) in enumerate(ex.map(peaks, todo), 1):
            if err:
                bad.append((i, err))
            elif p:
                rows.append(pd.DataFrame(p))
            if k % 100 == 0:
                pd.concat(rows, ignore_index=True).to_parquet(part)
                print(f"  {k}/{len(todo)} gauges", flush=True)
    pk = pd.concat(rows, ignore_index=True)[["monitoring_location_id", "time", "water_year", "value"]]
    pk["value"] = pd.to_numeric(pk["value"], errors="coerce")
    pk.to_parquet(D / "tx_peaks_q.parquet")
    print(f"{len(pk):,} peaks; {len(bad)} gauges failed {bad[:3]}")


def gauge_features(bgs: pd.DataFrame) -> pd.DataFrame:
    """Per block-group position (row of `fp.bg_geometry()`): features of the nearest eligible gauge."""
    m = pd.read_parquet(D / "tx_gauges_q.parquet")
    pk = pd.read_parquet(D / "tx_peaks_q.parquet")
    pk = pk[(pk.water_year <= 2009) & (pk.value > 0)]
    st = pk.assign(lv=np.log(pk.value)).groupby("monitoring_location_id").agg(
        n=("lv", "size"), med=("value", "median"), mx=("value", "max"), cv=("lv", "std"))
    m = m.merge(st, left_on="id", right_index=True)
    m = m[(m.n >= 10) & (m.drainage_area > 0) & (m.drainage_area <= 1000)].copy()
    m["spec"] = np.log(m.med / m.drainage_area)
    m["ratio"] = np.log(m.mx / m.med)
    la, lo = np.radians(bgs.lat.to_numpy()), np.radians(bgs.lon.to_numpy())
    best, arg = np.full(len(bgs), np.inf), np.zeros(len(bgs), int)
    for j, (a, b) in enumerate(zip(np.radians(m.lat), np.radians(m.lon), strict=True)):
        h = np.sin((la - a) / 2) ** 2 + np.cos(la) * np.cos(a) * np.sin((lo - b) / 2) ** 2
        d = 12742 * np.arcsin(np.sqrt(h))
        upd = d < best
        best[upd], arg[upd] = d[upd], j
    ok = best <= 25
    f = pd.DataFrame({"g_dist": np.where(ok, best, np.nan), "g_larea": np.log(m.drainage_area.to_numpy()[arg]),
                      "g_spec": m.spec.to_numpy()[arg], "g_cv": m.cv.to_numpy()[arg], "g_ratio": m.ratio.to_numpy()[arg]})
    f.loc[~ok, ["g_larea", "g_spec", "g_cv", "g_ratio"]] = np.nan
    print(f"eligible gauges {len(m)}; block groups with one within 25 km: {ok.mean():.1%}")
    return f


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
        return
    e26, e48, e49 = _load("e26", "26_flood_pilot_experiment.py"), _load("e48", "48_flood_forecast_pilot.py"), _load("e49", "49_information_audit.py")
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t = t.join(e49.history(t))
    gf = gauge_features(fp.bg_geometry())
    gcols = list(gf.columns)
    t = t.join(gf, on="bg")
    ex = {"base": [], "+ gauge history": gcols, "+ claim history": ["hist00", "prior3"],
          "+ claim history + gauge history": ["hist00", "prior3", *gcols]}
    yr = t.year.to_numpy()
    say("== USGS gauge history (peaks before 2010) added to the pilot model")
    e49.OUT.clear()
    base = e26.COUNTY + e26.EXPO + e26.CELL
    e49.say = say
    e49.fit_eval(t, yr <= 2016, yr >= 2017, base, ex, "observed rain, test 2017-2023")
    f = pd.read_parquet(e48.FEATS)
    u = t[t.year <= 2019].merge(f, on=["ep", "bg"], how="inner")
    yu = u.year.to_numpy()
    e49.fit_eval(u, yu <= 2016, yu >= 2017, e48.BASE + ["fc_f1", "fc_s", "fc_cty_max", "fc_cty_mean"], ex, "forecast rain (fcmean), test 2017-2019")
    p = wb2.cache_path("audit") / "usgs_gauge_history.txt"
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
