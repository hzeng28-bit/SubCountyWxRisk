"""Two candidate features for the "insure after an incident" prior, and a leakage check on each.

CANDIDATE pool (see docs/FLOOD_PRIORS.md). Texas. Event definition = FEMA declarations (flood, hurricane,
severe/coastal/tropical storm) by county and incident period, NOT the NFIP `floodEvent` field (81% coverage,
72 events).

  known   Feature A, ex-POST label/stratifier: is a claim's loss inside a declared event (county, incident
          period + 7 d)? Compares the loss distribution inside vs outside. Not a forecast input (needs the
          outcome's own date); it can only stratify evaluation or serve as an auxiliary label.
  cohort  Feature B, ex-ANTE: is the policy a post-event buyer (originalNBDate within 24 months after the
          county's most recent declared event that ended before it)? Block-group share of such policies in
          force at Harvey (2017-08-24), non-SFHA, voluntary. Tests whether, inside the same county, block
          groups with more post-event buyers had a different Harvey claim rate (county fixed effects,
          Poisson, county-clustered bootstrap). Only events strictly BEFORE the policy's NB date are used.

Whether a block group's post-event share mixes "bought because of a past flood" with "new neighbourhood,
new stock" cannot be separated here; the result is an association, not an effect.

    python scripts/36_event_features.py | tee $WXRISK_DATA/audit/event_features.txt
    python scripts/36_event_features.py --sections known
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.prep import build as prep  # noqa: E402

warnings.filterwarnings("ignore")
FLOOD = wb2.cache_path("labels_extra") / "flood"
EVENT_TYPES = ["Flood", "Hurricane", "Severe Storm", "Coastal Storm", "Tropical Storm"]
HARVEY = (pd.Timestamp("2017-08-25"), pd.Timestamp("2017-09-05"))
WINDOW_MONTHS = 24


def events() -> pd.DataFrame:
    d = pd.read_parquet(FLOOD / "fema_declarations_tx.parquet")
    d = d[d.incidentType.isin(EVENT_TYPES) & d.declarationType.isin(["DR", "EM"])].copy()
    d["county"] = d.fipsStateCode.astype(str).str.zfill(2) + d.fipsCountyCode.astype(str).str.zfill(3)
    d["begin"] = pd.to_datetime(d.incidentBeginDate).dt.tz_localize(None).dt.normalize()
    end = pd.to_datetime(d.incidentEndDate, errors="coerce").dt.tz_localize(None).dt.normalize()
    d["end"] = end.fillna(d.begin + pd.Timedelta(days=30))
    d = d[d.county.str.len() == 5]
    return d[["disasterNumber", "county", "begin", "end"]].drop_duplicates()


def load_claims(since: int = 2010) -> pd.DataFrame:
    """Texas claims with a located block group and a building damage amount (clean table `claims_nfip`)."""
    c = (prep.scan("claims_nfip")
         .filter((pl.col("state") == "TX") & (pl.col("loss_year") >= since) & pl.col("damage_building").is_not_null()
                 & pl.col("geoid").str.starts_with("48"))
         .select(pl.col("damage_building").alias("dmg"), pl.col("loss_date").alias("date"), "geoid",
                 pl.col("geoid").str.slice(0, 5).alias("county"), pl.col("occupancy").alias("occ"))
         .collect().to_pandas())
    c["date"] = pd.to_datetime(c["date"]).astype("datetime64[ns]")  # polars gives ms; the rest of the script is ns
    return c


def sec_known() -> None:
    ev = events()
    c = load_claims()
    c = c[c.occ.isin([1, 11])].copy()
    # a claim is "in a declared event" if its date is in [begin, end + 7 d] for a declaration of its county
    m = c.reset_index(drop=True).reset_index().merge(ev, on="county", how="left")
    hit = m[(m.date >= m.begin) & (m.date <= m.end + pd.Timedelta(days=7))]
    idx = set(hit["index"])
    c = c.reset_index(drop=True)
    c["known"] = c.index.isin(idx)
    print(f"\n## known: Texas single-family claims 2010+, {len(c):,}; declared-event shares")
    print(f"share of claims inside a declared event window: {c.known.mean():.3f}")
    g = c.groupby("known").dmg
    out = pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(), "q90": g.quantile(0.9),
                        "share>50k": g.apply(lambda s: (s > 50_000).mean())}).round(3)
    print(out.to_string())
    ym = c.groupby([c.date.dt.to_period("M"), "known"]).size().unstack(fill_value=0)
    n_big = (ym[True] >= 200).sum()
    print(f"county-agnostic months with >= 200 declared-event claims: {n_big}")
    y = c.groupby([c.date.dt.year, "known"]).size().unstack(fill_value=0)
    y["known_share"] = (y[True] / (y[True] + y[False])).round(3)
    print("by year (claims outside / inside declared windows):")
    print(y.to_string())


def poisson_fe(y, x, expo, grp, boot_groups=None):
    """Poisson with group fixed effects concentrated out; returns beta for covariate matrix x."""
    codes, uniq = pd.factorize(grp)

    def nll(b):
        eta = x @ b
        mu = expo * np.exp(eta)
        s_mu = np.bincount(codes, weights=mu, minlength=len(uniq))
        s_y = np.bincount(codes, weights=y, minlength=len(uniq))
        keep = s_mu > 0
        return -(np.sum(y * eta) - np.sum(s_y[keep] * np.log(s_mu[keep])))

    return minimize(nll, np.zeros(x.shape[1]), method="BFGS").x


def harvey_policies(d0: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Voluntary non-SFHA single-family policies in force on d0 (default: the day before Harvey), with NB date, sorted by NB."""
    d0 = d0 if d0 is not None else HARVEY[0] - pd.Timedelta(days=1)
    day = d0.date()
    p = (prep.scan("policies_tx")
         .filter(pl.col("occupancy").is_in([1, 11]) & pl.col("geoid").is_not_null() & (pl.col("eff") <= day)
                 & (pl.col("coverage_end").is_null() | (pl.col("coverage_end") > day))
                 & ~pl.col("zone").str.slice(0, 1).is_in(["A", "V"]).fill_null(False) & ~pl.col("mandatory")
                 & pl.col("first_policy_date").is_not_null() & (pl.col("first_policy_date") <= day))
         .select("geoid", pl.col("n_policies").cast(pl.Float64).alias("policyCount"),
                 pl.col("first_policy_date").alias("nb"), pl.col("post_firm").cast(pl.Float64).alias("postfirm"))
         .collect().to_pandas())
    p["nb"] = pd.to_datetime(p["nb"]).astype("datetime64[ns]")
    p["county"] = p.geoid.str[:5]
    p["tract"] = p.geoid.str[:11]
    return p.sort_values("nb", kind="stable"), d0


def boot_fe(g: pd.DataFrame, cov: list[str], nboot: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Poisson (county FE) coefficients for `cov` plus a county-clustered bootstrap."""
    beta = poisson_fe(g.y.values, g[cov].values, g.n.values, g.county.values)
    rng = np.random.default_rng(seed)
    cts = g.county.unique()
    parts = {k: v for k, v in g.groupby("county")}
    bs = []
    for _ in range(nboot):
        gb = pd.concat([parts[k].assign(county=f"{k}_{i}") for i, k in enumerate(rng.choice(cts, len(cts)))])
        bs.append(poisson_fe(gb.y.values, gb[cov].values, gb.n.values, gb.county.values))
    return beta, np.array(bs)


def print_coefs(cov: list[str], beta: np.ndarray, bs: np.ndarray) -> None:
    for j, name in enumerate(cov):
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        print(f"  {name:>14}: log-rate coef {beta[j]:+.3f} [{lo:+.3f}, {hi:+.3f}]")


def sec_cohort(nboot: int) -> None:
    ev = events().sort_values("end")
    p, d0 = harvey_policies()
    # most recent declared event of the county that ENDED before the NB date (no look-ahead)
    last = pd.merge_asof(p[["nb", "county"]].reset_index(), ev[["county", "end"]].rename(columns={"end": "ev_end"}),
                         left_on="nb", right_on="ev_end", by="county", direction="backward").set_index("index")
    gap = (p.nb - last.ev_end.reindex(p.index)).dt.days
    p["post"] = (gap.notna() & (gap <= 30.44 * WINDOW_MONTHS)).astype(float)
    w = p.policyCount.astype(float)
    print(f"\n## cohort: voluntary non-SFHA single-family policies in force on {d0.date()}: {len(p):,} rows, "
          f"post-event buyer share {np.average(p.post, weights=w):.3f}")
    c = load_claims()
    c = c[c.occ.isin([1, 11]) & c.date.between(*HARVEY)]
    y_bg = c.groupby("geoid").size()
    g = pd.DataFrame({"n": w.groupby(p.geoid).sum(), "n_post": (w * p.post).groupby(p.geoid).sum(),
                      "n_pf": (w * p.postfirm).groupby(p.geoid).sum()})
    g = g[g.n >= 20].copy()
    g["y"] = y_bg.reindex(g.index).fillna(0)
    g["county"] = g.index.str[:5]
    g["share_post"] = g.n_post / g.n
    g["share_pf"] = g.n_pf / g.n
    print(f"block groups: {len(g):,}; counties: {g.county.nunique()}; Harvey claims in them: {int(g.y.sum()):,}")
    print("share_post quantiles:", g.share_post.quantile([.1, .5, .9]).round(3).to_dict())
    # keep counties that actually saw Harvey claims and have variation
    cs = g.groupby("county").y.sum()
    g = g[g.county.isin(cs[cs >= 50].index)]
    print(f"counties with >= 50 Harvey claims: {g.county.nunique()}, block groups {len(g):,}")
    cov = ["share_post", "share_pf"]
    beta, bs = boot_fe(g, cov, nboot)
    print_coefs(cov, beta, bs)
    print("  coefficient is the within-county association of the Harvey claim rate with the post-event-buyer share")


def tract_hits(ev: pd.DataFrame, d_ref: pd.Timestamp, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locally hit (tract, event) pairs before Harvey and tract-level prior intensity.

    hit = >= 3 single-family claims inside the declared window of the county AND claims / in-force policies
    (all zones, on d_ref) >= 2%. Returns (hits[tract, county, end], tracts[prior_rate, ever_hit])."""
    day = d_ref.date()
    pol = (prep.scan("policies_tx")
           .filter(pl.col("occupancy").is_in([1, 11]) & pl.col("geoid").is_not_null() & (pl.col("eff") <= day)
                   & (pl.col("coverage_end").is_null() | (pl.col("coverage_end") > day)))
           .group_by(pl.col("geoid").str.slice(0, 11).alias("tract")).agg(pl.col("n_policies").cast(pl.Float64).sum())
           .collect().to_pandas().set_index("tract").n_policies)
    c = load_claims()
    c = c[c.occ.isin([1, 11]) & (c.date < cutoff)].reset_index(drop=True)
    c["tract"] = c.geoid.str[:11]
    e = ev[(ev.end < cutoff) & (ev.begin >= d_ref - pd.Timedelta(days=365 * 3))]
    m = c[["county", "tract", "date"]].reset_index().merge(e, on="county")
    m = m[(m.date >= m.begin) & (m.date <= m.end + pd.Timedelta(days=7))]
    h = m.groupby(["disasterNumber", "tract", "county", "end"]).size().rename("n").reset_index()
    h["rate"] = h.n / h.tract.map(pol)
    hits = h[(h.n >= 3) & (h.rate >= 0.02)][["tract", "county", "end"]].sort_values("end")
    prior = (h.groupby("tract").n.sum() / pol).dropna().rename("prior_rate")
    tr = pd.DataFrame({"prior_rate": prior})
    tr["ever_hit"] = tr.index.isin(hits.tract).astype(float)
    print(f"declared events since {e.begin.min().date()}: {e.disasterNumber.nunique()}, "
          f"(tract, event) hits: {len(hits):,} in {hits.tract.nunique():,} tracts")
    return hits, tr


def post_hit_flag(p: pd.DataFrame, hits: pd.DataFrame) -> pd.Series:
    """1 if the policy's NB date is within WINDOW_MONTHS after the end of the latest hit of its own tract (no look-ahead)."""
    h = hits.rename(columns={"end": "hit_end"}).sort_values("hit_end")
    last = pd.merge_asof(p[["nb", "tract"]].reset_index(), h[["tract", "hit_end"]], left_on="nb",
                         right_on="hit_end", by="tract", direction="backward").set_index("index")
    gap = (p.nb - last.hit_end.reindex(p.index)).dt.days
    return (gap.notna() & (gap <= 30.44 * WINDOW_MONTHS)).astype(float)


def bg_table(p: pd.DataFrame, post: pd.Series, tr: pd.DataFrame, y_bg: pd.Series) -> pd.DataFrame:
    w = p.policyCount.astype(float)
    pr = p.tract.map(tr.prior_rate).fillna(0.0)
    eh = p.tract.map(tr.ever_hit).fillna(0.0)
    g = pd.DataFrame({"n": w.groupby(p.geoid).sum(), "n_post": (w * post).groupby(p.geoid).sum(),
                      "n_pf": (w * p.postfirm).groupby(p.geoid).sum(), "n_pr": (w * np.log1p(100 * pr)).groupby(p.geoid).sum(),
                      "n_eh": (w * eh).groupby(p.geoid).sum()})
    g = g[g.n >= 20].copy()
    g["y"] = y_bg.reindex(g.index).fillna(0)
    g["county"] = g.index.str[:5]
    g["share_hitbuy"], g["share_pf"] = g.n_post / g.n, g.n_pf / g.n
    g["prior_int"], g["ever_hit"] = g.n_pr / g.n, g.n_eh / g.n
    cs = g.groupby("county").y.sum()
    return g[g.county.isin(cs[cs >= 50].index)]


def sec_local(nboot: int, nperm: int, disaster: int | None) -> None:
    ev = events()
    if disaster is None:  # Harvey as originally run: statewide window, all counties
        t0, t1, counties, label = HARVEY[0], HARVEY[1], None, "Harvey"
    else:
        o = ev[ev.disasterNumber == disaster]
        t0, t1, counties, label = o.begin.min(), o.end.max() + pd.Timedelta(days=7), set(o.county), f"DR-{disaster}"
    d0 = t0 - pd.Timedelta(days=1)
    p, d0 = harvey_policies(d0)
    if counties is not None:
        p = p[p.county.isin(counties)]
    hits, tr = tract_hits(ev, d0 - pd.Timedelta(days=600), t0)
    c = load_claims()
    c = c[c.occ.isin([1, 11]) & c.date.between(t0, t1)]
    if counties is not None:
        c = c[c.county.isin(counties)]
    y_bg = c.groupby("geoid").size()
    post = post_hit_flag(p, hits)
    g = bg_table(p, post, tr, y_bg)
    w = p.policyCount.astype(float)
    print(f"\n## local [{label}]: window {t0.date()}..{t1.date()}, in force {d0.date()}: post-hit buyer share "
          f"{np.average(post, weights=w):.3f} of {len(p):,} policy rows; block groups {len(g):,}, counties {g.county.nunique()}, "
          f"claims {int(g.y.sum()):,}")
    if g.county.nunique() < 3:
        print("  too few counties for a clustered bootstrap; skipped")
        return
    print("share_hitbuy quantiles (10/50/90/99):", g.share_hitbuy.quantile([.1, .5, .9, .99]).round(3).to_dict())
    for name, cov in [("no hazard control", ["share_hitbuy", "share_pf"]),
                      ("+ tract prior intensity and ever-hit", ["share_hitbuy", "share_pf", "prior_int", "ever_hit"])]:
        beta, bs = boot_fe(g, cov, nboot)
        print(f"model: {name}")
        print_coefs(cov, beta, bs)
        if len(cov) == 4:
            real = beta[0]
    # placebo: reassign hit events to random tracts of the same county, keep the true hazard controls
    rng = np.random.default_rng(1)
    tracts_by_county = p.groupby("county").tract.unique()
    null = []
    for _ in range(nperm):
        sh = hits.copy()
        sh["tract"] = [rng.choice(tracts_by_county[k]) if k in tracts_by_county.index else t
                       for k, t in zip(sh.county, sh.tract)]
        gp = bg_table(p, post_hit_flag(p, sh), tr, y_bg)
        gp = gp[gp.county.isin(g.county.unique())]
        null.append(poisson_fe(gp.y.values, gp[["share_hitbuy", "share_pf", "prior_int", "ever_hit"]].values,
                               gp.n.values, gp.county.values)[0])
    null = np.array(null)
    print(f"placebo ({nperm} within-county reshuffles of which tracts were hit): coef 2.5/50/97.5% = "
          f"{np.percentile(null, 2.5):+.3f} / {np.median(null):+.3f} / {np.percentile(null, 97.5):+.3f}; "
          f"real {real:+.3f}, share of placebo draws >= real: {(null >= real).mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=["known", "cohort", "local"])
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--perm", type=int, default=40)
    ap.add_argument("--disaster", type=int, nargs="*", default=[None], help="outcome FEMA disaster numbers (default: Harvey)")
    a = ap.parse_args()
    if "known" in a.sections:
        sec_known()
    if "cohort" in a.sections:
        sec_cohort(a.boot)
    if "local" in a.sections:
        for d in a.disaster:
            sec_local(a.boot, a.perm, d)


if __name__ == "__main__":
    main()
