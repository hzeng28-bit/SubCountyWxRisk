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
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

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
    cols = ["state", "yearOfLoss", "dateOfLoss", "censusGeoid", "buildingDamageAmount", "occupancyType"]
    c = pd.concat([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(FLOOD / "nfip_national" / "*.parquet")))])
    c = c[(c.state == "TX") & (c.yearOfLoss >= since)].copy()
    c["dmg"] = pd.to_numeric(c.buildingDamageAmount, errors="coerce")
    c["date"] = pd.to_datetime(c.dateOfLoss).dt.tz_localize(None).dt.normalize()
    c["geoid"] = c.censusGeoid.astype(str).str.zfill(12)
    c["county"] = c.geoid.str[:5]
    c["occ"] = pd.to_numeric(c.occupancyType, errors="coerce")
    return c[c.dmg.notna() & c.geoid.str.startswith("48")]


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


def sec_cohort(nboot: int) -> None:
    ev = events().sort_values("end")
    cols = ["censusGeoid", "ratedFloodZone", "occupancyType", "policyEffectiveDate", "policyTerminationDate",
            "policyCount", "mandatoryPurchaseFlag", "originalNBDate", "postFIRMConstructionIndicator"]
    p = ds.dataset(FLOOD / "pilot" / "policies_tx.parquet").to_table(columns=cols).to_pandas()
    p["occ"] = pd.to_numeric(p.occupancyType, errors="coerce")
    d0 = HARVEY[0] - pd.Timedelta(days=1)
    eff, term = pd.to_datetime(p.policyEffectiveDate), pd.to_datetime(p.policyTerminationDate)
    z = p.ratedFloodZone.astype(str).str.upper().str[:1]
    keep = (p.occ.isin([1, 11]) & (eff <= d0) & (term.isna() | (term > d0)) & ~z.isin(["A", "V"])
            & ~p.mandatoryPurchaseFlag.astype(bool))
    p = p[keep].copy()
    p["geoid"] = p.censusGeoid.astype(str).str.zfill(12)
    p["county"] = p.geoid.str[:5]
    p["nb"] = pd.to_datetime(p.originalNBDate, errors="coerce").dt.normalize()
    p = p[p.nb.notna() & (p.nb <= d0)].sort_values("nb")
    # most recent declared event of the county that ENDED before the NB date (no look-ahead)
    last = pd.merge_asof(p[["nb", "county"]].reset_index(), ev[["county", "end"]].rename(columns={"end": "ev_end"}),
                         left_on="nb", right_on="ev_end", by="county", direction="backward").set_index("index")
    gap = (p.nb - last.ev_end.reindex(p.index)).dt.days
    p["post"] = (gap.notna() & (gap <= 30.44 * WINDOW_MONTHS)).astype(float)
    p["postfirm"] = p.postFIRMConstructionIndicator.astype(bool).astype(float)
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
    x = g[["share_post", "share_pf"]].values
    beta = poisson_fe(g.y.values, x, g.n.values, g.county.values)
    rng = np.random.default_rng(0)
    cts = g.county.unique()
    parts = {k: v for k, v in g.groupby("county")}
    bs = []
    for _ in range(nboot):
        pick = rng.choice(cts, len(cts))
        gb = pd.concat([parts[k].assign(county=f"{k}_{i}") for i, k in enumerate(pick)])
        bs.append(poisson_fe(gb.y.values, gb[["share_post", "share_pf"]].values, gb.n.values, gb.county.values))
    bs = np.array(bs)
    for j, name in enumerate(["share_post", "share_pf"]):
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        print(f"  {name}: log-rate coef {beta[j]:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"(rate ratio, 100% vs 0%: {np.exp(beta[j]):.2f}x)")
    print("  coefficient is the within-county association of the Harvey claim rate with the post-event-buyer share")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", nargs="+", default=["known", "cohort"])
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    if "known" in a.sections:
        sec_known()
    if "cohort" in a.sections:
        sec_cohort(a.boot)


if __name__ == "__main__":
    main()
