"""Why does the GEFS forecast rain say almost nothing between counties? (stage 37)

Stage 36 found that, across counties inside an event, the forecast county peak ranks counties at AUC 0.59 and the forecast model adds nothing to the
exposure ranking (county AUC 0.852 against 0.847), whereas the observed-rain model reaches 0.906. Four checks:
  1 episodes     cross-county agreement of the forecast and the observed county peak on the pilot's own episodes (Spearman pooled and inside events),
                 bias ratio by longitude, and the effect of `nd` (the pilot cuts the forecast to one day when the OBSERVED episode lasted one day)
  2 alignment    pooled correlation of the GEFS ensemble mean with gridMET aggregated to the GEFS cells, for forecast day S and S+1 and gridMET shifted by -1 / 0 / +1 day,
                 on all 1,640 stored initialisation days
  3 sample       the same agreement for ALL county-days of the stored days (no condition on an observed episode): Spearman, and AUC of the forecast peak for
                 'the observed county peak reached 30 mm', pooled and inside days; against the agreement among the pilot's episodes
  4 correction   whether county-wise bias correction (scale factor, percentile within the county's own forecasts) or exposure lets the forecast peak rank
                 counties for 'any claim' inside events: county AUC on 2017-2019, scores fitted on 2010-2016
Inputs : flood_pilot_fc_features.parquet, flood_pilot_table.parquet, gefs/apcp_*.npy, gridMET (pr_tx_2000_2023.npy)
Output : $WXRISK_DATA/audit/forecast_rain_diagnosis.txt
Run    : python scripts/76_forecast_rain_diagnosis.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import zstio  # noqa: E402
from wxrisk import wb2  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e48 = _load("e48", "48_flood_forecast_pilot.py")
OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def pair_auc(score, y):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan, 0.0
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), float(n1 * n0)


def by_event_auc(df: pd.DataFrame, col: str, ycol: str = "any", min_c: int = 5) -> float:
    num = den = 0.0
    for _, g in df.groupby("sysid", sort=False):
        if len(g) < min_c:
            continue
        a, w = pair_auc(g[col].to_numpy(), g[ycol].to_numpy())
        if w:
            num += a * w
            den += w
    return num / den if den else np.nan


def episodes_part() -> pd.DataFrame:
    say("== 1. forecast against observed county peak on the pilot episodes (2010-2019)")
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "bg", "cty", "start", "year", "y", "dur", "n_pol", "lon"])
    f = pd.read_parquet(e48.FEATS)
    t = t[t.year <= e48.LAST_YEAR].merge(f, on=["ep", "bg"], how="inner")
    e = t.groupby("ep").agg(cty=("cty", "first"), start=("start", "first"), year=("year", "first"), any=("y", "max"), dur=("dur", "first"), nd=("nd", "first"),
                            fc=("fc_cty_max", "first"), oc=("oc_cty_max", "first"), o4=("o4_cty_max", "first"), pol=("n_pol", "sum"), lon=("lon", "mean")).reset_index()
    e["sysid"] = e.start.dt.strftime("%Y%m%d")
    say(f"  {len(e):,} county episodes; with a claim {e['any'].mean():.3f}")
    say(f"  Spearman of the forecast county peak with the observed county peak (same GEFS-cell aggregation): pooled {spearmanr(e.fc, e.oc)[0]:.3f}; with the 4 km value {spearmanr(e.fc, e.o4)[0]:.3f}")
    inside = [spearmanr(g.fc, g.oc)[0] for _, g in e.groupby("sysid") if len(g) >= 8]
    say(f"  inside events (same start day, >= 8 counties; {len(inside)} events): mean Spearman {np.nanmean(inside):.3f}")
    say(f"  median ratio forecast / observed county peak {np.median(e.fc / e.oc.clip(lower=1)):.2f}; forecast peak below 10 mm in {np.mean(e.fc < 10):.1%} of the episodes (observed below 10 mm: {np.mean(e.oc < 10):.1%})")
    e["lon3"] = pd.qcut(e.lon, 3, labels=["west", "middle", "east"])
    say("  by longitude tercile (median ratio forecast / observed; Spearman): " + "; ".join(
        f"{k} {np.median(g.fc / g.oc.clip(lower=1)):.2f}; {spearmanr(g.fc, g.oc)[0]:.2f}" for k, g in e.groupby("lon3", observed=True)))
    say("  by nd (1 = the observed episode lasted one day, so the pilot cut the forecast to day S): " + "; ".join(
        f"nd={k}: n {len(g)}, Spearman {spearmanr(g.fc, g.oc)[0]:.2f}, county AUC for 'any claim' {by_event_auc(g, 'fc'):.3f}" for k, g in e.groupby("nd")))
    return e


def alignment_and_sample() -> pd.DataFrame:
    say("\n== 2-3. alignment and sample (all stored initialisation days)")
    meta = zstio.load(e48.GEFS / "gefs_meta.npz", allow_pickle=True)
    glat, glon = meta["lat"], meta["lon"]
    days, lat, lon, P = fp.load_precip()
    g = fp.bg_geometry()
    py = np.abs(glat[None, :] - lat[:, None]).argmin(1)
    px = np.abs(glon[None, :] - lon[:, None]).argmin(1)
    cell_of_pixel = (py[:, None] * len(glon) + px[None, :]).ravel()
    ncell = len(glat) * len(glon)
    cnt = np.bincount(cell_of_pixel, minlength=ncell).astype(np.float32)
    cy = np.abs(glat[None, :] - g.lat.to_numpy()[:, None]).argmin(1)
    cx = np.abs(glon[None, :] - g.lon.to_numpy()[:, None]).argmin(1)
    cty_cells = {c: np.unique(cy[i] * len(glon) + cx[i]) for c, i in g.groupby("cty").indices.items()}
    ctys = sorted(cty_cells)
    tx = np.zeros(ncell, bool)
    for c in ctys:
        tx[cty_cells[c]] = True
    tx = tx.reshape(len(glat), len(glon))
    cache: dict = {}

    def obs_cell(k):
        if k not in cache:
            fl = np.nan_to_num(np.asarray(P[k]), nan=0.0).ravel()
            cache[k] = (np.bincount(cell_of_pixel, weights=fl, minlength=ncell) / np.maximum(cnt, 1)).reshape(len(glat), len(glon))
            if len(cache) > 12:
                cache.pop(next(iter(cache)))
        return cache[k]

    files = sorted(e48.GEFS.glob("apcp_*.npy"))
    acc = {(fd, sh): np.zeros(6) for fd in (1, 2) for sh in (-1, 0, 1)}
    cty_rows = []
    for f in files:
        d0 = pd.Timestamp(f.stem.split("_")[1])
        s0 = d0 + pd.Timedelta(1, "D")
        if s0 not in days:
            continue
        di = days.get_loc(s0)
        a = zstio.load(f).astype(np.float32)
        d1, d2 = a[:, :4].sum(1), a[:, 4:].sum(1)  # [member, y, x]
        fm = {1: d1.mean(0), 2: d2.mean(0)}
        for fd in (1, 2):
            x = fm[fd][tx]
            for sh in (-1, 0, 1):
                k = di + (fd - 1) + sh
                if not (0 <= k < len(days)):
                    continue
                yv = obs_cell(k)[tx]
                ok = ~np.isnan(x) & ~np.isnan(yv)
                xs, ys = x[ok].astype(np.float64), yv[ok].astype(np.float64)
                acc[(fd, sh)] += np.array([len(xs), xs.sum(), ys.sum(), (xs * xs).sum(), (ys * ys).sum(), (xs * ys).sum()])
        f1 = np.stack([d1, d2], 1).max(1).mean(0)  # per member max over the two days, then the member mean (as the pilot)
        oc = np.maximum(obs_cell(di), obs_cell(min(di + 1, len(days) - 1)))
        for c in ctys:
            cc = cty_cells[c]
            cyy, cxx = np.divmod(cc, len(glon))
            cty_rows.append((s0, c, float(np.nanmax(f1[cyy, cxx])), float(oc[cyy, cxx].max())))
    say("  pooled Pearson correlation of the ensemble mean (5 members) with gridMET aggregated to the GEFS cells, Texas cells (forecast day; gridMET day shift)")
    for (fd, sh), v in acc.items():
        n, sx, sy, sxx, syy, sxy = v
        r = (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))
        say(f"    forecast day {'S' if fd == 1 else 'S+1'}, gridMET day S{sh:+d} (relative to the intended one): r = {r:.3f}")
    c = pd.DataFrame(cty_rows, columns=["day", "cty", "fc", "oc"])
    c["sysid"] = c.day.dt.strftime("%Y%m%d")
    c["trig"] = (c.oc >= 30).astype(int)
    say(f"\n  all county-days of the stored days: {len(c):,}; observed county peak >= 30 mm in {c.trig.mean():.1%}")
    say(f"  Spearman forecast vs observed county peak: all county-days {spearmanr(c.fc, c.oc)[0]:.3f}; only county-days with an observed peak >= 30 mm (the pilot's sample) "
        f"{spearmanr(c.fc[c.trig == 1], c.oc[c.trig == 1])[0]:.3f}; with an observed peak >= 50 mm {spearmanr(c.fc[c.oc >= 50], c.oc[c.oc >= 50])[0]:.3f}")
    a_all, _ = pair_auc(c.fc.to_numpy(), c.trig.to_numpy())
    say(f"  AUC of the forecast county peak for 'the observed peak reached 30 mm': pooled {a_all:.3f}; inside days (across counties) {by_event_auc(c, 'fc', 'trig', 20):.3f}")
    say("  => where the forecast has skill: telling which county-days get >= 30 mm out of all county-days; among the days that did, its ranking of the counties is weak")
    return c


def correction_part(e: pd.DataFrame) -> None:
    say("\n== 4. does bias correction or exposure let the forecast rank counties inside events? (county AUC for 'any claim', test 2017-2019, fitted 2010-2016)")
    tr, te = e.year <= 2016, e.year >= 2017
    glob = np.median(e.oc[tr].clip(lower=1) / e.fc[tr].clip(lower=1))
    fac = {}
    for c, g in e[tr].groupby("cty"):
        fac[c] = np.median(g.oc.clip(lower=1) / g.fc.clip(lower=1)) if len(g) >= 8 else glob
    e = e.assign(fc_scaled=e.fc * e.cty.map(fac).fillna(glob))
    pct = {c: np.sort(g.fc.to_numpy()) for c, g in e[tr].groupby("cty") if len(g) >= 8}
    allfc = np.sort(e.fc[tr].to_numpy())
    e["fc_pct"] = [np.searchsorted(pct.get(c, allfc), v) / max(len(pct.get(c, allfc)), 1) for c, v in zip(e.cty, e.fc, strict=True)]
    e["lpol"] = np.log1p(e.pol)
    e["lfc"], e["loc"], e["lo4"] = np.log1p(e.fc), np.log1p(e.oc), np.log1p(e.o4)
    tt = e[te]
    say(f"  test episodes {len(tt):,}, events with >= 5 counties scored below")
    res = {"forecast county peak (raw)": by_event_auc(tt, "fc"), "forecast county peak x county scale factor": by_event_auc(tt, "fc_scaled"),
           "forecast peak, percentile within the county's own forecasts": by_event_auc(tt, "fc_pct"), "policies in the county only": by_event_auc(tt, "pol"),
           "observed county peak (GEFS cells), upper bound": by_event_auc(tt, "oc"), "observed county peak (4 km), upper bound": by_event_auc(tt, "o4")}
    for k in ("lfc", "loc"):
        for extra, cols in (("", [k]), (" + log policies", [k, "lpol"])):
            m = LogisticRegression(C=1e3, max_iter=500).fit(e.loc[tr, cols], e.loc[tr, "any"])
            res[f"logistic on {'forecast' if k == 'lfc' else 'observed'} log peak{extra}"] = by_event_auc(tt.assign(s=m.decision_function(tt[cols])), "s")
    m = LogisticRegression(C=1e3, max_iter=500).fit(e.loc[tr, ["lfc", "lpol", "fc_pct"]], e.loc[tr, "any"])
    res["logistic on forecast log peak + log policies + percentile"] = by_event_auc(tt.assign(s=m.decision_function(tt[["lfc", "lpol", "fc_pct"]])), "s")
    for k, v in res.items():
        say(f"    {k:62s} {v:.3f}")


def unconditional_ranking(c: pd.DataFrame) -> None:
    """County ranking inside a day over ALL counties (dry ones included), label = any inland non-tropical claim in the county in [S, S+3]."""
    say("\n== 5. county ranking inside a day over all counties, not only counties with an observed episode (stored days 2010-2019)")
    import polars as pl

    from wxrisk import bg_crosswalk
    from wxrisk.prep import build as prep

    cl = (prep.scan("claims_nfip").filter((pl.col("state") == "TX") & pl.col("cause").is_in(["2", "3", "4"]) & pl.col("geoid").is_not_null())
          .select(pl.col("geoid").alias("bg"), pl.col("loss_date").alias("date"), "flood_event").collect().to_pandas())
    cl["date"] = pd.to_datetime(cl.date)
    cl = cl[~cl.flood_event.fillna("").str.contains(fp.TC_PATTERN)]
    cw = bg_crosswalk.build().set_index("code")
    cl["cty"] = cl.bg.map(cw.bg10).str[:5]
    cl = cl[cl.cty.notna()]
    cdays = cl.groupby("cty").date.apply(lambda s_: np.sort(s_.to_numpy())).to_dict()

    def has_claim(cty, day):
        a = cdays.get(cty)
        if a is None:
            return 0
        lo = np.searchsorted(a, np.datetime64(day))
        hi = np.searchsorted(a, np.datetime64(day + pd.Timedelta(days=4)))
        return int(hi > lo)
    c = c.copy()
    c["any"] = [has_claim(a, b) for a, b in zip(c.cty, c.day, strict=True)]
    g = fp.bg_geometry()
    ex = fp.exposure_matrices(g.bg10.tolist())["n_pol"]
    idx = g.groupby("cty").indices
    di = (c.day - fp.EXPO_START).dt.days.to_numpy()
    cpos = {k: i for i, k in enumerate(sorted(idx))}
    pol_by = np.stack([ex[idx[k]].sum(0) for k in sorted(idx)])  # [county, day]
    c["pol"] = [pol_by[cpos[k], d] if k in cpos and 0 <= d < pol_by.shape[1] else 0 for k, d in zip(c.cty, di, strict=True)]
    c = c[c.pol > 0].copy()
    c["lpol"], c["lfc"], c["loc"] = np.log1p(c.pol), np.log1p(c.fc), np.log1p(c.oc)
    c["year"] = c.day.dt.year
    tr, te = c.year <= 2016, c.year >= 2017
    say(f"  county-days with a policy: {len(c):,}; with a claim in [S, S+3]: {c['any'].mean():.4f}; test 2017-2019: {int(te.sum()):,}")
    tt = c[te]
    res = {"forecast county peak": by_event_auc(tt, "fc", min_c=20), "policies only": by_event_auc(tt, "pol", min_c=20), "observed county peak (upper bound)": by_event_auc(tt, "oc", min_c=20)}
    for k, cols in {"forecast peak + policies": ["lfc", "lpol"], "observed peak + policies": ["loc", "lpol"]}.items():
        m = LogisticRegression(C=1e3, max_iter=500).fit(c.loc[tr, cols], c.loc[tr, "any"])
        res[f"logistic, {k}"] = by_event_auc(tt.assign(s=m.decision_function(tt[cols])), "s", min_c=20)
    say("  county AUC for 'any claim' across ALL counties inside a day:")
    for k, v in res.items():
        say(f"    {k:42s} {v:.3f}")
    sub = tt[tt.oc >= 30]
    say("  the same restricted to the counties whose observed peak reached 30 mm (the pilot's sample):")
    for k, col in (("forecast county peak", "fc"), ("policies only", "pol"), ("observed county peak", "oc")):
        say(f"    {k:42s} {by_event_auc(sub, col, min_c=5):.3f}")


def main() -> None:
    e = episodes_part()
    c = alignment_and_sample()
    correction_part(e)
    unconditional_ranking(c)
    p = wb2.cache_path("audit") / "forecast_rain_diagnosis.txt"
    p.write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
