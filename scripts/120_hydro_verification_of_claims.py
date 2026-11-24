"""Hydrologic information as a verification quantity for claim information, and how fine it has to be (stage 65).

Verification model: P(any claim | exposure, rain, flow) fitted on candidate gauge-days <= 2016 (logistic); the score of a reported
claim unit is its logit. A claim record is "verified" by hydrology when its score is high given the hydrologic state at the
reported date and place.
  V  verification power: take genuine claim gauge-days of 2017-2023, displace their reported date by k days or their location to
     another basin (near 10-60 km / far > 100 km), and measure how well the score separates genuine from displaced records (AUC).
     Compared across hydrologic information: rain only, daily-max / daily-mean flow, flow coarsened in time (3-day, 7-day windows).
  C calibration of the verification probability (reliability, Brier skill vs the base rate).
  G spatial coherence: rank correlation between gauges' daily-max flow percentiles (wet days) and between basins' claim indicators,
     by distance -> how far one gauge speaks for another place.
  R rain resolution for block-group claims: coarse-grain the 4 km gridMET field to 12 / 24 / 48 / 96 km and measure the block-group
     any-claim AUC (storm days, test 2017-2023), and shorter / longer accumulation windows.
Inputs : as scripts 116 and 118.   Run: python scripts/120_hydro_verification_of_claims.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scipy.ndimage import uniform_filter  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from wxrisk import flood_pilot as fp  # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


g116 = _load("g116", "116_gauge_outcomes_vs_claims.py")
g118 = _load("g118", "118_subcounty_completion.py")
D, DAYS, auc, km = g116.D, g116.DAYS, g116.auc, g116.km
rng = np.random.default_rng(0)


def pct_cols(M):
    return M.rank(pct=True)


def main():
    g, S, R1, R3, mem, ring, bg, pos, ex, ncl, paid = g116.build()
    fr = [f for s in g.index if (f := g116.gauge_frame(s, S, R1, R3, mem, bg, pos, ex, ncl, paid)) is not None]
    F = pd.concat(fr, ignore_index=True)
    F = F[F.pol.groupby(F.site).transform("mean") >= 20]
    sites = list(F.site.unique())
    gg = g.loc[sites]
    ns = len(sites)

    def piv(c):
        return F.pivot(index="day", columns="site", values=c).reindex(index=DAYS, columns=sites)
    QMAX, QMEAN, P3, POL, CL = piv("qmax"), piv("qmean"), piv("p3"), piv("pol"), piv("claims")
    hyd = {"rain only": None,
           "flow, daily max": pct_cols(QMAX),
           "flow, daily mean": pct_cols(QMEAN),
           "flow, 3-day max window": pct_cols(QMAX.rolling(3, center=True, min_periods=1).max()),
           "flow, 7-day max window": pct_cols(QMAX.rolling(7, center=True, min_periods=1).max())}
    RP = pct_cols(P3)
    lp = np.log1p(POL).to_numpy()
    cand = ((hyd["flow, daily max"] >= 0.9) | (P3 >= 25)).to_numpy() & (POL > 0).to_numpy() & P3.notna().to_numpy()
    yr = DAYS.year.to_numpy()[:, None] * np.ones((1, ns), int)
    y = (CL.to_numpy() > 0).astype(int)

    def feats(name, d, s, dshift=0, s_alt=None):
        """features of reported (day d, gauge s) with hydrology read at (d+dshift, s_alt or s); exposure stays with s."""
        dd = np.clip(d + dshift, 0, len(DAYS) - 1)
        ss = s if s_alt is None else s_alt
        cols = [lp[d, s], RP.to_numpy()[dd, ss]]
        if hyd[name] is not None:
            cols.append(hyd[name].to_numpy()[dd, ss])
        return np.column_stack(cols)

    di, si = np.nonzero(cand)
    tr = yr[di, si] <= 2016
    models = {}
    for nm in hyd:
        X = feats(nm, di, si)
        ok = ~np.isnan(X).any(1)
        sc = StandardScaler().fit(X[tr & ok])
        models[nm] = (sc, LogisticRegression(max_iter=500).fit(sc.transform(X[tr & ok]), y[di, si][tr & ok]))

    def score(nm, X):
        sc, m = models[nm]
        return m.decision_function(sc.transform(np.nan_to_num(X, nan=0.0)))

    # ---- C: calibration ----
    te = ~tr
    print(f"\ngauges {ns}; candidate gauge-days {len(di):,} (train {tr.sum():,}, test {te.sum():,}, test claim days {int(y[di, si][te].sum())})")
    print("=== C. Calibration of P(any claim | exposure, rain, flow), test 2017-2023 ===")
    X = feats("flow, daily max", di, si)[te]
    ok = ~np.isnan(X).any(1)
    p = 1 / (1 + np.exp(-score("flow, daily max", X)))[ok]
    yt = y[di, si][te][ok]
    base_rate = y[di, si][tr].mean()
    pr = 1 / (1 + np.exp(-score("rain only", feats("rain only", di, si)[te]))) [ok]
    print(f"  Brier skill vs train base rate {base_rate:.3f}: rain only {1 - ((pr - yt) ** 2).mean() / ((base_rate - yt) ** 2).mean():+.3f}; "
          f"+ daily-max flow {1 - ((p - yt) ** 2).mean() / ((base_rate - yt) ** 2).mean():+.3f}")
    rel = pd.DataFrame({"p": p, "y": yt}).assign(b=lambda d: pd.cut(d.p, [0, .01, .03, .1, .3, .6, 1], include_lowest=True)).groupby(
        "b", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    print(rel.round(3).to_string())

    # ---- V: displacement detection ----
    print("\n=== V. Detecting displaced claim records (genuine vs displaced; AUC of the verification score; 100 gauge bootstraps) ===")
    cd, cs = np.nonzero((y == 1) & (yr >= 2017) & (POL.to_numpy() > 0))
    print(f"  genuine claim gauge-days 2017-2023: {len(cd):,} at {len(np.unique(cs))} gauges")
    dist = km(gg.lat.to_numpy()[:, None], gg.lon.to_numpy()[:, None], gg.lat.to_numpy()[None, :], gg.lon.to_numpy()[None, :])
    far = [np.flatnonzero(dist[i] > 100) for i in range(ns)]
    near = [np.flatnonzero((dist[i] >= 10) & (dist[i] <= 60)) for i in range(ns)]
    kinds = [("date +-1 d", ("t", 1)), ("date +-2 d", ("t", 2)), ("date +-3 d", ("t", 3)), ("date +-7 d", ("t", 7)),
             ("date +-14 d", ("t", 14)), ("place: gauge 10-60 km away", ("n", 0)), ("place: gauge > 100 km away", ("f", 0))]
    rows = []
    gid = {s: np.flatnonzero(cs == s) for s in np.unique(cs)}
    ugs = np.array(list(gid))
    draws = [np.concatenate([gid[u] for u in rng.choice(ugs, len(ugs))]) for _ in range(100)]
    for kname, (kk, k) in kinds:
        if kk == "t":
            sg = rng.choice([-1, 1], len(cd))
            args = dict(dshift=sg * k, s_alt=None)
            sel = np.ones(len(cd), bool)
        else:
            alt = np.array([rng.choice((near if kk == "n" else far)[s]) if len((near if kk == "n" else far)[s]) else -1 for s in cs])
            sel = alt >= 0
            args = dict(dshift=0, s_alt=np.where(sel, alt, 0))
        row = {"corruption": kname}
        for nm in hyd:
            Xg = feats(nm, cd, cs)
            Xc = feats(nm, cd, cs, **args)
            okk = sel & ~np.isnan(Xg).any(1) & ~np.isnan(Xc).any(1)
            sg_, sc_ = score(nm, Xg), score(nm, Xc)
            # AUC of genuine vs displaced over all units
            lab = np.r_[np.ones(okk.sum()), np.zeros(okk.sum())]
            a = roc_auc_score(lab, np.r_[sg_[okk], sc_[okk]])
            row[nm] = a
        rows.append(row)
    R = pd.DataFrame(rows)
    print(R.round(3).to_string(index=False))
    # CI for two headline columns on a mid-case
    for nm in ("rain only", "flow, daily max"):
        k = 2
        sg = rng.choice([-1, 1], len(cd))
        Xg, Xc = feats(nm, cd, cs), feats(nm, cd, cs, dshift=sg * k)
        okk = ~np.isnan(Xg).any(1) & ~np.isnan(Xc).any(1)
        a_, b_ = score(nm, Xg), score(nm, Xc)
        v = []
        for ix in draws:
            ix = ix[okk[ix]]
            v.append(roc_auc_score(np.r_[np.ones(len(ix)), np.zeros(len(ix))], np.r_[a_[ix], b_[ix]]))
        print(f"  date +-2 d, {nm}: AUC {np.mean(v):.3f} [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")

    # ---- G: spatial coherence ----
    print("\n=== G. Spatial coherence between gauges (wet days: either gauge's 3-day basin rain >= 10 mm) ===")
    Qp = hyd["flow, daily max"].to_numpy()
    CR = np.log1p(1000 * CL.to_numpy() / np.where(POL.to_numpy() > 0, POL.to_numpy(), np.nan))
    wet = P3.to_numpy() >= 10
    bins = [(10, 25), (25, 50), (50, 100), (100, 200), (200, 400)]
    iu = np.triu_indices(ns, 1)
    res = {b: ([], [], []) for b in bins}
    pick = rng.choice(len(iu[0]), min(60_000, len(iu[0])), replace=False)
    for a, b in zip(iu[0][pick], iu[1][pick]):
        d = dist[a, b]
        for lo, hi in bins:
            if lo <= d < hi:
                m = (wet[:, a] | wet[:, b]) & ~np.isnan(Qp[:, a]) & ~np.isnan(Qp[:, b])
                if m.sum() >= 100:
                    res[(lo, hi)][0].append(pd.Series(Qp[m, a]).corr(pd.Series(Qp[m, b]), method="spearman"))
                    mc = m & ~np.isnan(CR[:, a]) & ~np.isnan(CR[:, b])
                    res[(lo, hi)][1].append(np.corrcoef(CR[mc, a] > 0, CR[mc, b] > 0)[0, 1] if (CR[mc, a] > 0).any() and (CR[mc, b] > 0).any() else np.nan)
                    res[(lo, hi)][2].append(pd.Series(P3.to_numpy()[m, a]).corr(pd.Series(P3.to_numpy()[m, b]), method="spearman"))
                break
    print("  distance km | gauge pairs | flow-percentile rho | basin rain rho | claim-indicator corr")
    for b, (q, c, r) in res.items():
        print(f"  {b[0]:>3}-{b[1]:<3}      | {len(q):>6}      | {np.nanmedian(q):.2f}                | {np.nanmedian(r):.2f}           | {np.nanmedian(c):.2f}")

    # ---- R: rain resolution at block-group level ----
    print("\n=== R. Rain resolution for block-group any-claim (storm days; AUC trained <= 2016, test 2017-2023) ===")
    bgm, sd, cmat, pol, _, _, _, W, z = g118.load(15)
    days, lat, lon, Pg = fp.load_precip()
    iy, ix = fp.grid_index(lat, lon, bgm.lat.to_numpy(), bgm.lon.to_numpy())
    pidx = {d: i for i, d in enumerate(days)}
    sizes = {"4 km (native)": 1, "12 km": 3, "24 km": 6, "48 km": 12, "96 km": 24}
    wins = {"day d": (0, 0), "d-1..d": (-1, 0), "d-1..d+1": (-1, 1), "d-3..d+1": (-3, 1)}
    cache = {}

    def field(k):
        if k not in cache:
            cache[k] = np.nan_to_num(np.asarray(Pg[k]), nan=0.0)
        return cache[k]
    out = {}
    for rname, m in sizes.items():
        for wname, (a, b) in wins.items():
            if rname != "4 km (native)" and wname != "d-1..d":
                continue
            if rname == "4 km (native)" or wname == "d-1..d":
                M = np.zeros((len(bgm), len(sd)), np.float32)
                for j, d in enumerate(sd):
                    k = pidx[d]
                    f = sum(field(k + o) for o in range(a, b + 1))
                    if m > 1:
                        f = uniform_filter(f, size=m, mode="nearest")
                    M[:, j] = f[iy, ix]
                out[(rname, wname)] = M
        cache.clear()
    tr_d = np.flatnonzero(sd.year <= 2016)
    te_d = np.flatnonzero(sd.year >= 2017)
    yy = (cmat > 0).astype(int)
    lpp = np.log1p(pol)
    rows = []
    for (rname, wname), M in out.items():
        Xa = np.column_stack([lpp.ravel(), np.log1p(M).ravel()]).reshape(len(bgm), len(sd), 2)
        mtr = np.zeros(W.shape, bool); mtr[:, tr_d] = True
        mte = np.zeros(W.shape, bool); mte[:, te_d] = True
        mtr, mte = mtr & W, mte & W
        sc = StandardScaler().fit(Xa[mtr])
        lr = LogisticRegression(max_iter=500).fit(sc.transform(Xa[mtr]), yy[mtr])
        s_ = lr.decision_function(sc.transform(Xa[mte]))
        pooled = auc(yy[mte], s_)
        wa = []
        for j in te_d:
            mj = W[:, j]
            yj = yy[mj, j]
            if yj.sum() >= 5:
                wa.append(auc(yj, lr.decision_function(sc.transform(Xa[mj, j]))))
        rows.append((rname, wname, pooled, np.mean(wa)))
    print(pd.DataFrame(rows, columns=["rain resolution", "accumulation window", "pooled AUC", "within-day AUC"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
