"""County x event table for the Earth Engine display: inputs, intermediate outputs and final result side by side (docs/GEE_PLAN.md, step 1b).

Shows the whole path of the Texas flood model at county level (block-group NFIP codes mix 2010 / 2020 vintages, so the mentor asked for county display;
county = first 5 digits of the 2010 block-group code, stable across vintages). Unit = one test event day (2017-2023, >= 1 claimed block group, no
tropical-cyclone window), scope S0, exactly the rows of scripts 85 / 94. Every column is an aggregate of the block-group rows of that county and day.

A. INPUTS, rain (gridMET 4 km observed, peak = largest daily total over S and S+1, sum = two-day total, mm; the model's own rain read):
   rain_cty_max   largest block-group-pixel peak in the county          (feature 'county max' of the boosting model)
   rain_cty_mean  mean block-group-pixel peak over the county            (feature 'county mean')
   rain_sum_mean  mean two-day sum at the block-group pixels
B. INPUTS, exposure, all from the clean NFIP policy table on day S (policies in force, wxrisk.flood_pilot.exposure_matrices), mapped to 2010 block groups:
   n_pol          policies in force                                      (sum over block groups)
   sfha_share     share of them rated in a Special Flood Hazard Area (zone A* / V*), policy-weighted
   single_share   share with single-family occupancy (codes 1, 11), policy-weighted
   mean_cov       mean building coverage per policy (script 80 unit), policy-weighted
   hist00         NFIP claims 2000-2009 in the county  (sum over block groups of expm1(log1p count))
   prior3         NFIP claims in the 3 years before S   (same)
   n_bg           block groups with a policy that day
   MODEL INPUT view (what the CNN's tabular branch receives per block group, script 89 tab_features, averaged over the county's block groups,
   unweighted, on the model's own scale; the CNN standardises them with training statistics before use):
   in_log_pol, in_sfha, in_single, in_cov, in_log_nbg (log1p block groups in the county), in_hist00 (log1p claims 2000-2009), in_prior3 (log1p claims, 3 years),
   in_month_sin, in_month_cos (calendar month of the event on the unit circle)
C. INTERMEDIATE, model outputs before ranking:
   exp_cnn        sum of the CNN claim probabilities over the block groups (expected number of claimed block groups, 5-seed mean of script 89)
   exp_boost      same for the boosting model (script 85)
   exp_pol        exposure-only baseline: n_pol (the 'policies only' ranking)
D. FINAL:
   pct_cnn / pct_boost / pct_pol   percentile of the county among the event's counties (1 = highest score; ties averaged)
GROUND TRUTH (NFIP claims in [S, S+3], S0, never a model input; arrives weeks after the event):
   n_claims, n_cbg  claims and claimed block groups; paid  building amount paid (USD); claims_per_1k  claims per 1,000 policies in force
COMPARISON per event, K = number of counties with >= 1 claim; each model's top-K counties are its "prediction":
   class_cnn / class_boost / class_pol   1 hit (predicted and claimed), 2 missed (claimed, not predicted), 3 false alarm (predicted, no claim), 0 neither
   events_index adds k_claimed_cty and hit_cnn / hit_boost / hit_pol (hits out of K)
Event index (events_index): per event the county / block-group AUC and top-k capture of CNN, boosting and policies only from audit/event_eval_tx.parquet (script 94).

Inputs : cnn_tx/e45_rows.npz, e45_pred.npz, p89_S0_own_TX_s*.npz (scripts 85, 89), audit/event_eval_tx.parquet (script 94)
Output : $WXRISK_DATA/gee_export/county_event_tx.csv (all events), county_event_tx_<year>.geojson (events with >= 10 claimed
         block groups, no geometry), events_index.csv
Run    : python scripts/112_gee_county_events.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402
from wxrisk import zstio  # noqa: E402


BIG = 10  # claimed block groups that make an event 'big' (RESULTS stages 49-52)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def wmean(df: pd.DataFrame, col: str, w: str) -> pd.Series:
    num = (df[col] * df[w]).groupby([df.ev, df.fips]).sum()
    den = df[w].groupby([df.ev, df.fips]).sum()
    return num / den.where(den > 0)


def main() -> None:
    e85 = _load("e85", "85_state_group_baseline.py")
    d = wb2.cache_path("cnn_tx")
    R = e85.features("TX")
    days = pd.DatetimeIndex(R["days"])
    yr = days.year.to_numpy()[R["day"]]
    rows = np.flatnonzero(yr >= 2017)
    tab, f = R["tab"][rows], R["f"][rows]
    T = pd.DataFrame({
        "ev": days[R["day"][rows]].strftime("%Y%m%d"), "fips": R["ctys"][R["cty"][rows]],
        "rain_pk": f[:, 0], "rain_cty_max": f[:, 3], "rain_cty_mean": f[:, 4], "rain_sum": f[:, 1],
        "n_pol": tab[:, 8], "sfha": tab[:, 1], "single": tab[:, 2], "cov": tab[:, 3],
        "hist00": np.expm1(tab[:, 5]), "prior3": np.expm1(tab[:, 6]), "n_bg": 1,
        "n": R["n0"][rows], "cbg": (R["n0"][rows] > 0).astype(np.int32), "paid": R["paid"][rows],
        "in_log_pol": tab[:, 0], "in_sfha": tab[:, 1], "in_single": tab[:, 2], "in_cov": tab[:, 3], "in_log_nbg": np.log1p(tab[:, 4]),
        "in_hist00": tab[:, 5], "in_prior3": tab[:, 6],
        "in_month_sin": np.sin(2 * np.pi * tab[:, 7] / 12), "in_month_cos": np.cos(2 * np.pi * tab[:, 7] / 12),
    })
    T["boost"] = zstio.load(d / "e45_pred.npz")["p_own"]
    ps = []
    for sd in range(5):
        fp = d / f"p89_S0_own_TX_s{sd}.npz"
        if zstio.exists(fp):
            z = zstio.load(fp)
            assert (z["bg"] == R["bg"][rows]).all()
            ps.append(z["p"])
    assert ps, "no CNN seeds found"
    T["cnn"] = np.mean(ps, axis=0)
    print(f"rows {len(T):,}; CNN seeds {len(ps)}; events {T.ev.nunique()}")

    for c in ("sfha", "single", "cov"):  # policy-weighted means need numerator = value x policies
        T[c + "_w"] = T[c] * T.n_pol
    g = T.groupby(["ev", "fips"], sort=True)
    C = g.agg(rain_cty_max=("rain_cty_max", "first"), rain_cty_mean=("rain_cty_mean", "first"), rain_sum_mean=("rain_sum", "mean"),
              n_pol=("n_pol", "sum"), sfha_w=("sfha_w", "sum"), single_w=("single_w", "sum"), cov_w=("cov_w", "sum"),
              hist00=("hist00", "sum"), prior3=("prior3", "sum"), n_bg=("n_bg", "sum"),
              exp_cnn=("cnn", "sum"), exp_boost=("boost", "sum"), n_claims=("n", "sum"), n_cbg=("cbg", "sum"), paid=("paid", "sum"),
              **{c: (c, "mean") for c in ("in_log_pol", "in_sfha", "in_single", "in_cov", "in_log_nbg", "in_hist00", "in_prior3",
                                          "in_month_sin", "in_month_cos")}).reset_index()
    for c in ("sfha", "single", "cov"):
        C["sfha_share" if c == "sfha" else "single_share" if c == "single" else "mean_cov"] = C[c + "_w"] / C.n_pol.where(C.n_pol > 0)
    C = C.drop(columns=["sfha_w", "single_w", "cov_w"])
    C["exp_pol"] = C.n_pol
    for src, dst in (("exp_cnn", "pct_cnn"), ("exp_boost", "pct_boost"), ("exp_pol", "pct_pol")):
        C[dst] = C.groupby("ev")[src].rank(pct=True, method="average")
    C["claims_per_1k"] = C.n_claims / C.n_pol.where(C.n_pol > 0) * 1000
    K = C.groupby("ev").n_claims.transform(lambda x: int((x > 0).sum()))
    truth = C.n_claims > 0
    for src, dst in (("exp_cnn", "class_cnn"), ("exp_boost", "class_boost"), ("exp_pol", "class_pol")):
        top = C.groupby("ev")[src].rank(ascending=False, method="first") <= K
        C[dst] = np.select([top & truth, ~top & truth, top & ~truth], [1, 2, 3], 0)

    # sanity: the event / county sets and totals must equal what scripts 85 / 94 evaluate
    assert abs(C.n_claims.sum() - T.n.sum()) < 1e-3 and abs(C.exp_cnn.sum() - T.cnn.sum()) < 1e-1
    assert C.fips.str.startswith("48").all() and C.fips.str.len().eq(5).all()
    print(f"county-event rows {len(C):,}; events {C.ev.nunique()}; counties {C.fips.nunique()}")
    print(C[["rain_cty_max", "rain_sum_mean", "n_pol", "sfha_share", "single_share", "mean_cov", "hist00", "prior3", "exp_cnn"]].describe().T.round(3).to_string())

    out = wb2.cache_path("gee_export")
    out.mkdir(parents=True, exist_ok=True)
    keep_full = ("exp_cnn", "exp_boost", "exp_pol")  # scores keep 9 significant digits: rounding to 4 decimals makes ties and flips the top-K flags
    num = [c for c in C.select_dtypes("float").columns if c not in keep_full]
    C[num] = C[num].astype("float32").round(4)
    for c in ("exp_cnn", "exp_boost"):
        C[c] = C[c].map(lambda v: float(f"{v:.9g}"))

    # event index: event-level metrics from script 94 (columns are (model, metric) pairs)
    ev = pd.read_parquet(wb2.cache_path("audit") / "event_eval_tx.parquet")
    # display set: events with >= BIG claimed block groups (the evaluation set of stages 49-52); the rest stay in the CSV only
    big = set(ev.index[ev[("meta", "n_claimed_bg")] >= BIG])
    C["big"] = C.ev.isin(big).astype(int)
    C.to_csv(out / "county_event_tx.csv", index=False)
    D = C[C.big == 1].drop(columns="big")
    # Earth Engine cannot export features without geometry, so each row carries its county centroid (the join key to the polygons stays fips)
    from shapely.geometry import mapping, shape

    cj = json.loads((out / "county_geometry.geojson").read_text())["features"]
    cen = {f["properties"]["GEOID"]: mapping(shape(f["geometry"]).representative_point()) for f in cj}  # inside the polygon, unlike the centroid
    for y, g in D.groupby(D.ev.str[:4]):  # one asset per year: each stays far below the client request limit
        feats = [{"type": "Feature", "geometry": cen[r["fips"]], "properties": {k: v for k, v in r.items() if pd.notna(v)}}
                 for r in g.to_dict("records")]
        p = out / f"county_event_tx_{y}.geojson"
        p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":"), default=float))
        print(f"  {p.name}: {len(g):,} rows, {g.ev.nunique()} events, {p.stat().st_size / 1e6:.1f} MB")
    flat = pd.DataFrame({f"{m}_{k}": ev[(m, k)] for m, k in ev.columns if m in ("CNN", "boosting", "policies only") and k in ("cty_auc", "cty_top10", "bg_auc", "bg_top5")})
    flat.columns = [c.replace("policies only", "pol").replace("CNN", "cnn").replace("boosting", "boost") for c in flat.columns]
    flat["n_claimed_bg"] = ev[("meta", "n_claimed_bg")]
    flat = flat.rename_axis("ev").reset_index().round(4)
    hit = C.groupby("ev").agg(k_claimed_cty=("n_claims", lambda x: int((x > 0).sum())), hit_cnn=("class_cnn", lambda x: int((x == 1).sum())),
                              hit_boost=("class_boost", lambda x: int((x == 1).sum())), hit_pol=("class_pol", lambda x: int((x == 1).sum())))
    flat = flat.merge(hit.reset_index(), on="ev", how="left")
    flat.to_csv(out / "events_index.csv", index=False)
    tx = {"type": "Point", "coordinates": [-99.3, 31.5]}  # placeholder point: an event is not a place, EE needs a geometry
    feats = [{"type": "Feature", "geometry": tx, "properties": {k: v for k, v in r.items() if pd.notna(v)}} for r in flat.to_dict("records")]
    (out / "events_index.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, separators=(",", ":"), default=float))
    assert set(flat.ev) == set(C.ev), "event sets differ between the table and the script-94 index"
    print(f"wrote {out}/county_event_tx.csv ({(out / 'county_event_tx.csv').stat().st_size / 1e6:.1f} MB), events_index.csv ({len(flat)} events)")


if __name__ == "__main__":
    main()
