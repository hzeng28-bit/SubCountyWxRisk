"""Stock-flow model of NFIP take-up: seasonal baseline + decaying event pulse + age-dependent lapse.

CANDIDATE pool (see docs/FLOOD_PRIORS.md, RESULTS.md stage 10). Texas, single-family, non-SFHA, voluntary
(mandatoryPurchaseFlag false) policies, county x quarter, 2010Q1-2023Q4.

  flow   N[c,q]  first-term purchases (policy effective date within 45 days of originalNBDate)
         N ~ Poisson( Sref[c] * exp(alpha_c + season_q + sum_k beta_k * I[c,q-k]) ),  k = 0..6
         I[c,q] = log1p(100 * single-family claims in county c in quarter q / all-zone in-force policies)
         Sref[c] = mean stock over the training quarters (a fixed size, so projections cannot feed back on themselves)
         = the realised local hazard; the model is therefore "exposure GIVEN the hazard", not a hazard forecast.
  econ   optional extra covariates (standardised on the training quarters, lagged one quarter): year-on-year change of
         the 30-year mortgage rate (Freddie Mac PMMS, national), year-on-year log change of the Zillow home value index
         (county), log single-family permits of the previous year (Census BPS, county), log average premium of the
         in-force stock (NFIP, county). Data: scripts/fetch/fetch_econ.py, premium from scripts/24.
  lapse  surv[a]: share of a purchase cohort still in force a quarters later, pooled over training cohorts
  stock  S[c,t] rebuilt from cohorts: observed cohorts decay by conditional survival, future cohorts come from
         the flow model; legacy stock (first term before 2010) decays geometrically.

Held-out evaluation at each origin (train up to the origin quarter, test after it):
  flow:  one-step N vs (B1) season-only Poisson, (B2) same quarter last year scaled by stock; county-clustered
         bootstrap interval of the skill (1 - MSE / MSE_base on log1p N)
  stock: multi-step S vs persistence (stock at origin) and vs the season-only flow model

The pulse and retention here describe behaviour in aggregate; they do not say WHY people buy (word of mouth,
media, lenders, officials), see RESULTS.md stage 10.

    python scripts/37_stock_flow_inertia.py | tee $WXRISK_DATA/audit/stock_flow.txt
    python scripts/37_stock_flow_inertia.py --origins 2016Q4 --boot 100
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import panel, wb2  # noqa: E402

warnings.filterwarnings("ignore")
FLOOD = wb2.cache_path("labels_extra") / "flood"
NQ = 56  # 2010Q1 .. 2023Q4 (claims end in 2023)
LAGS = 7
MIN_STOCK = 300


def qidx(ts: pd.Series) -> np.ndarray:
    return ((ts.dt.year - 2010) * 4 + (ts.dt.month - 1) // 3).to_numpy()


def qlabel(q: int) -> str:
    return f"{2010 + q // 4}Q{q % 4 + 1}"


def qend(q: int) -> np.datetime64:
    return (pd.Timestamp(f"{2010 + q // 4}-{3 * (q % 4) + 3}-01") + pd.offsets.MonthEnd(0)).to_datetime64()


def load_panel():
    """Return county list and arrays: N[c,q], S[c,q], stock[c,cohort(0=legacy,1..NQ),q], S_all[c,q], claims[c,q]."""
    cols = ["censusGeoid", "ratedFloodZone", "occupancyType", "policyEffectiveDate", "policyTerminationDate",
            "policyCount", "mandatoryPurchaseFlag", "originalNBDate", "totalInsurancePremiumOfThePolicy"]
    p = ds.dataset(FLOOD / "pilot" / "policies_tx.parquet").to_table(columns=cols).to_pandas()
    p = p[pd.to_numeric(p.occupancyType, errors="coerce").isin([1, 11])].copy()
    p["cty"] = p.censusGeoid.astype(str).str.zfill(12).str[:5]
    p = p[p.cty.str.startswith("48")]
    cts = np.array(sorted(p.cty.unique()))
    ci = pd.Index(cts).get_indexer(p.cty)
    eff, term = pd.to_datetime(p.policyEffectiveDate), pd.to_datetime(p.policyTerminationDate)
    nb = pd.to_datetime(p.originalNBDate, errors="coerce")
    w = p.policyCount.astype(float).to_numpy()
    prem = pd.to_numeric(p.totalInsurancePremiumOfThePolicy, errors="coerce").fillna(0.0).to_numpy()
    sfha = p.ratedFloodZone.astype(str).str.upper().str[:1].isin(["A", "V"]).to_numpy()
    vol = ~p.mandatoryPurchaseFlag.astype(bool).to_numpy()
    focus = ~sfha & vol
    effv, termv = eff.values, term.fillna(pd.Timestamp("2100-01-01")).values
    nbq = np.where(nb.notna(), qidx(nb.fillna(pd.Timestamp("2010-01-01"))), -1)
    nbq = np.clip(nbq, -1, NQ - 1) + 1  # 0 = legacy / earlier, 1..NQ cohorts
    C = len(cts)
    stock = np.zeros((C, NQ + 1, NQ))
    s_all = np.zeros((C, NQ))
    prem_sum = np.zeros((C, NQ))
    for q in range(NQ):
        d = qend(q)
        live = (effv <= d) & (termv > d)
        m = live & focus
        stock[:, :, q] = np.bincount(ci[m] * (NQ + 1) + nbq[m], weights=w[m], minlength=C * (NQ + 1)).reshape(C, NQ + 1)
        s_all[:, q] = np.bincount(ci[live], weights=w[live], minlength=C)
        prem_sum[:, q] = np.bincount(ci[m], weights=prem[m], minlength=C)
    first = focus & (np.abs((eff - nb).dt.days.to_numpy()) <= 45)
    eq = qidx(eff)
    ok = first & (eq >= 0) & (eq < NQ)
    N = np.bincount(ci[ok] * NQ + eq[ok], weights=w[ok], minlength=C * NQ).reshape(C, NQ)
    cl = pd.concat([pd.read_parquet(f, columns=["state", "dateOfLoss", "censusGeoid", "occupancyType"])
                    for f in sorted(glob.glob(str(FLOOD / "nfip_national" / "*.parquet")))])
    cl = cl[(cl.state == "TX") & pd.to_numeric(cl.occupancyType, errors="coerce").isin([1, 11])]
    cc = pd.Index(cts).get_indexer(cl.censusGeoid.astype(str).str.zfill(12).str[:5])
    cq = qidx(pd.to_datetime(cl.dateOfLoss).dt.tz_localize(None))
    k = (cc >= 0) & (cq >= 0) & (cq < NQ)
    claims = np.bincount(cc[k] * NQ + cq[k], minlength=C * NQ).reshape(C, NQ).astype(float)
    avg_prem = prem_sum / np.clip(stock.sum(axis=1), 1, None)
    return cts, N, stock, s_all, claims, avg_prem


def intensity(claims: np.ndarray, s_all: np.ndarray) -> np.ndarray:
    """Saturating transform of claims per in-force policy: log1p(I / 1%). It keeps a Harvey-sized shock from being
    extrapolated through the exponential link (a first version without it predicted 160k purchases in 2017Q3)."""
    prev = np.concatenate([s_all[:, :1], s_all[:, :-1]], axis=1)
    return np.log1p(np.clip(claims / np.clip(prev, 1, None), 0, 1) / 0.01)


ECON = FLOOD.parent / "econ"
ECON_NAMES = ["d_mortgage_rate_4q", "d_log_home_value_4q", "log_permits_prev_year", "log_avg_premium", "log_homes_sold"]
ECON_LAG = [1, 1, 1, 1, 0]  # quarters; home sales act in the same quarter as the policy is bought (insurance at closing)
QMIN = [1]  # first training quarter (set from --econ: Redfin sales start in 2012)
ACTIVE = list(range(len(ECON_NAMES)))  # columns of Z used by the econ mode (set by --econ)


def _quarterly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly columns (datetime index) -> quarterly mean indexed 0 = 2009Q1."""
    q = df.groupby([df.index.year, (df.index.month - 1) // 3]).mean()
    q.index = (q.index.get_level_values(0) - 2009) * 4 + q.index.get_level_values(1)
    return q


def load_econ(cts: np.ndarray, avg_prem: np.ndarray) -> np.ndarray:
    """Raw (unstandardised, unlagged) economic features Z[c, q, e] for q = 2010Q1..2023Q4; NaN where unknown."""
    Z = np.full((len(cts), NQ, len(ECON_NAMES)), np.nan)
    r = pd.read_csv(ECON / "pmms_history.csv", usecols=["date", "pmms30"])
    r["date"] = pd.to_datetime(r.date, format="%m/%d/%Y")
    r = r.dropna().set_index("date").pmms30.astype(float).to_frame()
    rq = _quarterly(r).pmms30
    drate = (rq - rq.shift(4)).reindex(range(4, 4 + NQ)).to_numpy()  # quarters 2010Q1..
    Z[:, :, 0] = drate[None, :]
    z = pd.read_csv(ECON / "zillow_zhvi_county.csv", dtype={"StateCodeFIPS": str, "MunicipalCodeFIPS": str})
    z = z[z.State == "TX"].copy()
    z["fips"] = "48" + z.MunicipalCodeFIPS.str.zfill(3)
    months = [c for c in z.columns if c[:2] in ("19", "20") and c[4] == "-"]
    zz = z.set_index("fips")[months].T
    zz.index = pd.to_datetime(zz.index)
    zq = np.log(_quarterly(zz))
    dh = (zq - zq.shift(4)).reindex(range(4, 4 + NQ))
    dh = dh.reindex(columns=cts)
    Z[:, :, 1] = dh.to_numpy().T
    bp = []
    for y in range(2010, 2024):
        d = pd.read_csv(ECON / f"bps_county_{y}.txt", skiprows=3, header=None, usecols=[0, 1, 2, 6], dtype=str)
        d = d[d[1].str.strip() == "48"]
        bp.append(pd.DataFrame({"year": y, "fips": "48" + d[2].str.strip().str.zfill(3), "b": pd.to_numeric(d[6], errors="coerce")}))
    bp = pd.concat(bp).pivot_table(index="fips", columns="year", values="b").reindex(cts)
    for q in range(NQ):
        y = max(2010 + q // 4 - 1, 2010)  # the previous calendar year; 2010 uses its own year
        Z[:, q, 2] = np.log1p(bp[y].to_numpy())
    Z[:, :, 3] = np.log(np.where(avg_prem > 0, avg_prem, np.nan))
    Z[:, :, 4] = redfin_sales(cts)
    return Z


def redfin_sales(cts: np.ndarray) -> np.ndarray:
    """log of quarterly home sales (Redfin 'All Residential', not seasonally adjusted) per county; NaN before 2012."""
    cache = ECON / "redfin_tx.parquet"
    if not cache.exists():
        parts = []
        for ch in pd.read_csv(ECON / "redfin_county_market_tracker.tsv.gz", sep="\t", chunksize=200_000, low_memory=False,
                              usecols=["PERIOD_BEGIN", "REGION", "STATE_CODE", "PROPERTY_TYPE", "IS_SEASONALLY_ADJUSTED", "HOMES_SOLD"]):
            parts.append(ch[(ch.STATE_CODE == "TX") & (ch.PROPERTY_TYPE == "All Residential")
                            & (ch.IS_SEASONALLY_ADJUSTED == False)])  # noqa: E712
        pd.concat(parts).to_parquet(cache)
    d = pd.read_parquet(cache)
    z = pd.read_csv(ECON / "zillow_zhvi_county.csv", usecols=["State", "RegionName", "StateCodeFIPS", "MunicipalCodeFIPS"],
                    dtype=str)
    z = z[z.State == "TX"]
    fips = dict(zip(z.RegionName.str.strip(), "48" + z.MunicipalCodeFIPS.str.zfill(3)))  # 'Harris County' -> 48201
    d["fips"] = d.REGION.str.replace(", TX", "", regex=False).str.strip().map(fips)
    d = d.dropna(subset=["fips", "HOMES_SOLD"])
    t = pd.to_datetime(d.PERIOD_BEGIN)
    d["q"] = (t.dt.year - 2010) * 4 + (t.dt.month - 1) // 3
    sold = d.groupby(["fips", "q"]).HOMES_SOLD.sum().unstack().reindex(index=cts, columns=range(NQ))
    return np.log(sold.where(sold > 0)).to_numpy()


def econ_design(Z: np.ndarray, origin: int, qs: np.ndarray) -> np.ndarray:
    """Standardise on the training quarters (1..origin), fill unknowns with 0 (= the training mean), lag one quarter."""
    tr = Z[:, QMIN[0] : origin + 1, :]
    mu, sd = np.nanmean(tr, axis=(0, 1)), np.nanstd(tr, axis=(0, 1))
    zs = np.nan_to_num((Z - mu) / np.where(sd > 0, sd, 1), nan=0.0)
    return np.stack([zs[:, np.clip(qs - ECON_LAG[e], 0, None), e] for e in ACTIVE], axis=2)


def design(I: np.ndarray, qs: np.ndarray, mode: str, Zs: np.ndarray | None = None) -> np.ndarray:
    """x[c, q, :] for each (c, q in qs): three season dummies (Q2..Q4), LAGS lags of I (mode event/econ), and the
    lagged standardised economic features Zs (mode econ)."""
    C = I.shape[0]
    cols = [np.broadcast_to((qs % 4 == s).astype(float), (C, len(qs))) for s in (1, 2, 3)]
    if mode in ("event", "econ"):
        for k in range(LAGS):
            cols.append(np.where(qs - k >= 0, I[:, np.clip(qs - k, 0, None)], 0.0))
    x = np.stack(cols, axis=2)
    return np.concatenate([x, Zs], axis=2) if mode == "econ" else x


def fit_flow(N, S, I, Z, mode: str, origin: int, keep: np.ndarray):
    qs = np.arange(QMIN[0], origin + 1)
    x = design(I[keep], qs, mode, econ_design(Z[keep], origin, qs))
    expo = np.broadcast_to(S[keep][:, 1 : origin + 1].mean(axis=1)[:, None], (keep.sum(), len(qs)))
    y = N[keep][:, qs]
    ok = expo > 0
    g = np.broadcast_to(np.arange(keep.sum())[:, None], y.shape)
    beta = panel.poisson_fe(y[ok], x[ok], expo[ok], g[ok])
    alpha = panel.group_effects(y[ok], x[ok], expo[ok], g[ok], beta).reindex(range(keep.sum())).fillna(0).to_numpy()
    return beta, alpha


def predict_flow(beta, alpha, I, Z, keep, origin, qs, mode, s_prev):
    x = design(I[keep], qs, mode, econ_design(Z[keep], origin, qs))
    return s_prev * np.exp(alpha[:, None] + x @ beta)


def survival(stock: np.ndarray, N: np.ndarray, origin: int) -> np.ndarray:
    """Pooled cohort survival by age (quarters), from cohorts whose age-a stock is observed by the origin."""
    pooled = stock.sum(axis=0)
    n0 = N.sum(axis=0)
    ages = np.arange(0, origin)
    sv = np.zeros(len(ages))
    for a in ages:
        js = np.arange(0, origin - a + 1)
        js = js[n0[js] > 0]
        sv[a] = pooled[js + 1, js + a].sum() / n0[js].sum() if len(js) else np.nan
    # smooth pathologies: make it non-increasing from age 1 and extend with the last ratio
    sv[1:] = np.minimum.accumulate(np.nan_to_num(sv[1:], nan=sv[1:][np.isfinite(sv[1:])][-1]))
    return sv


def surv_at(sv: np.ndarray, a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    last = len(sv) - 1
    tail = sv[last] / sv[last - 4] if last >= 4 and sv[last - 4] > 0 else 0.95
    q4 = tail ** 0.25
    return np.where(a <= last, sv[np.clip(a, 0, last)], sv[last] * q4 ** (a - last))


def project_stock(stock, N, I, Z, keep, origin, horizon_end, sv, beta, alpha, mode):
    """S[c, origin+1 .. horizon_end] for kept counties, by cohort reconstruction."""
    C = keep.sum()
    st = stock[keep]
    S = st.sum(axis=1)  # [C, NQ] observed stock
    qs_f = np.arange(origin + 1, horizon_end + 1)
    out = np.zeros((C, len(qs_f)))
    legacy_ratio = (st[:, 0, origin].sum() / max(st[:, 0, origin - 4].sum(), 1)) ** 0.25
    Nhat = np.zeros((C, NQ))
    ref = S[:, 1 : origin + 1].mean(axis=1)
    for i, t in enumerate(qs_f):
        x = design(I[keep], np.array([t]), mode, econ_design(Z[keep], origin, np.array([t])))[:, 0, :]
        Nhat[:, t] = ref * np.exp(alpha + x @ beta)
        tot = st[:, 0, origin] * legacy_ratio ** (t - origin)
        for j in range(0, origin + 1):  # observed cohorts
            a0, a1 = origin - j, t - j
            tot += st[:, j + 1, origin] * surv_at(sv, a1) / max(surv_at(sv, a0), 1e-9)
        for j in range(origin + 1, t + 1):  # simulated cohorts
            tot += Nhat[:, j] * surv_at(sv, t - j)
        out[:, i] = tot
    return out, qs_f


def wape(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(np.abs(pred - obs).sum() / obs.sum())


MODES = {"econ": "event+econ", "event": "event", "season": "season-only"}


def run_origin(oq: str, boot: int, cts, N, stock, s_all, claims, Z) -> None:
    year, qn = int(oq[:4]), int(oq[-1])
    origin = (year - 2010) * 4 + qn - 1
    I = intensity(claims, s_all)
    S = stock.sum(axis=1)
    keep = (S[:, 1 : origin + 1].mean(axis=1) >= MIN_STOCK) & (S[:, origin] > 0)
    print(f"\n## origin {oq}: train 2010Q2..{oq}, test {qlabel(origin + 1)}..{qlabel(NQ - 1)}; "
          f"{keep.sum()} counties with mean stock >= {MIN_STOCK}")
    fits = {m: fit_flow(N, S, I, Z, m, origin, keep) for m in MODES}
    be, _ = fits["event"]
    print("pulse kernel, event model (log-rate per unit I, lag 0..6 quarters): " + " ".join(f"{b:+.2f}" for b in be[3:]))
    bc, _ = fits["econ"]
    print("pulse kernel, event+econ model: " + " ".join(f"{b:+.2f}" for b in bc[3 : 3 + LAGS]))
    print("econ coefficients (log-rate per training s.d., lagged one quarter): "
          + ", ".join(f"{n} {b:+.3f}" for n, b in zip([ECON_NAMES[i] for i in ACTIVE], bc[3 + LAGS :])))
    sv = survival(stock, N, origin)
    print("pooled cohort survival at age 0,1,2,4,8,12 quarters: "
          + " ".join(f"{sv[a]:.2f}" for a in (0, 1, 2, 4, 8, 12) if a < len(sv)))

    # ---- flow: one-step ahead
    qs = np.arange(origin + 1, NQ)
    s_prev = S[keep][:, qs - 1]
    ref = np.broadcast_to(S[keep][:, 1 : origin + 1].mean(axis=1)[:, None], s_prev.shape)
    y = N[keep][:, qs]
    pred = {m: predict_flow(*fits[m], I, Z, keep, origin, qs, m, ref) for m in MODES}
    b2 = np.where(S[keep][:, qs - 5] > 0, N[keep][:, qs - 4] * s_prev / np.clip(S[keep][:, qs - 5], 1, None), 0.0)
    ok = s_prev > 0
    lg = np.log1p
    rng = np.random.default_rng(0)
    idx = np.arange(keep.sum())
    draws = [rng.choice(idx, len(idx)) for _ in range(boot)]

    def skill(p, base, r=None):
        r = np.arange(len(y)) if r is None else r
        f = lambda a: np.mean(((lg(a[r]) - lg(y[r])) ** 2)[ok[r]])  # noqa: E731
        return 1 - f(p) / f(base)

    for name, p, base, bn in [("event", pred["event"], pred["season"], "season-only"),
                              ("event+econ", pred["econ"], pred["season"], "season-only"),
                              ("event+econ", pred["econ"], pred["event"], "event"),
                              ("event+econ", pred["econ"], b2, "last-year-scaled")]:
        bs = [skill(p, base, r) for r in draws]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"flow one-step skill (log1p N, county-quarter), {name} vs {bn}: {skill(p, base):+.3f} [{lo:+.3f}, {hi:+.3f}]")
    tot = pd.DataFrame({"obs": y.sum(0), "event+econ": pred["econ"].sum(0), "event": pred["event"].sum(0),
                        "season": pred["season"].sum(0), "last_year": b2.sum(0)}, index=[qlabel(q) for q in qs])
    print("state-total new purchases by quarter, first 12 test quarters:")
    print(tot.head(12).round(0).astype(int).to_string())
    print("state-total absolute % error over test quarters: "
          + ", ".join(f"{k} {wape(tot[k].values, tot.obs.values):.2f}" for k in ["event+econ", "event", "season", "last_year"]))

    # ---- stock: multi-step
    Sobs = S[keep]
    for m, name in MODES.items():
        sp, _ = project_stock(stock, N, I, Z, keep, origin, NQ - 1, sv, *fits[m], m)
        res = []
        for h in (4, 8, 12, 20):
            if origin + h <= NQ - 1:
                i = h - 1
                res.append(f"h={h}: county wAPE {wape(sp[:, i], Sobs[:, origin + h]):.3f}, state {abs(sp[:, i].sum() / Sobs[:, origin + h].sum() - 1):.3f}")
        print(f"stock {name}: " + "; ".join(res))
    res = []
    for h in (4, 8, 12, 20):
        if origin + h <= NQ - 1:
            res.append(f"h={h}: county wAPE {wape(Sobs[:, origin], Sobs[:, origin + h]):.3f}, state {abs(Sobs[:, origin].sum() / Sobs[:, origin + h].sum() - 1):.3f}")
    print("stock persistence: " + "; ".join(res))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origins", nargs="+", default=["2016Q4", "2019Q4"])
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--qmin", type=int, default=None, help="first training quarter index (0 = 2010Q1); default 1, or 8 with log_homes_sold")
    ap.add_argument("--econ", nargs="+", default=ECON_NAMES, choices=ECON_NAMES, help="economic features in the econ mode")
    a = ap.parse_args()
    ACTIVE[:] = [ECON_NAMES.index(n) for n in a.econ]
    QMIN[0] = a.qmin if a.qmin is not None else (8 if "log_homes_sold" in a.econ else 1)
    cts, N, stock, s_all, claims, avg_prem = load_panel()
    Z = load_econ(cts, avg_prem)
    print("economic features, non-missing share:", dict(zip(ECON_NAMES, np.isfinite(Z).mean(axis=(0, 1)).round(2))))
    print(f"counties {len(cts)}, first-term purchases {N.sum():,.0f}, stock check (Q2023Q4) {stock[:, :, -1].sum():,.0f}")
    for o in a.origins:
        run_origin(o, a.boot, cts, N, stock, s_all, claims, Z)


if __name__ == "__main__":
    main()
