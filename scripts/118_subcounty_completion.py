"""Sub-county information completion: interpolation, residual kriging and SVD (recommender-style) on block group x storm-day claims (stage 63).

Matrix: rows = Texas block groups with policies, columns = storm days (>= 30 non-tropical inland NFIP claims statewide that day),
entry z = log1p(1000 * claims / policies in force), masked where no policy is in force. Block groups play the "users" and storm days
the "items" of a recommender; a day's observed block groups are the user's ratings, the hidden ones are to be completed.
Hidden / observed scenarios on each test day (2017-2023; factors and models fitted on storm days <= 2016):
  random reveal  only a fraction rho (5%, 20%, 50%) of block groups is observed, the rest hidden   (sparse reports)
  region block   one of 5 k-means regions is hidden, the rest observed                              (a whole area unreported)
Methods (all use only rain-field + exposure everywhere, and the observed claims of that day):
  base   HGB on rain (day, 2-day, 7-day antecedent), log policies, lat, lon   (no observed claims)
  idw    inverse-distance mean of the observed neighbours' z (8 nearest within 50 km)       = spatial interpolation
  rk     base + idw of the neighbours' residual                                              = regression kriging
  svd    rank-k SVD of the train matrix (BG factors U, per-BG mean), day vector folded in by ridge on the observed rows
  svdr   base + rank-k SVD fold-in of the residual matrix
  mix    base + average of the idw and svdr corrections
Metrics: pooled AUC of 'any claim' on hidden BG-days, mean within-day AUC, share of the day's hidden claims in the top 5% of
hidden BGs; intervals: bootstrap over test weeks (200).
Inputs : flood_pilot geometry, clean claims / policies, gridMET.   Run: python scripts/118_subcounty_completion.py [--k 10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from scipy.spatial import cKDTree  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor as HGB  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from wxrisk import flood_pilot as fp  # noqa: E402

rng = np.random.default_rng(0)
DAYS = pd.date_range("2010-01-01", "2023-12-31")


def auc(y, s):
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else np.nan


def load(min_claims=30):
    bg = fp.bg_geometry()
    cl = fp.claims_tx()
    cl = cl[cl.date.between(DAYS[0], DAYS[-1])]
    nday = cl.groupby("date").size()
    storm = nday[nday >= min_claims].index
    bgs = bg.bg10.tolist()
    ex = fp.exposure_matrices(bgs)["n_pol"]
    keep = ex.mean(1) >= 2
    bg, ex = bg[keep].reset_index(drop=True), ex[keep]
    pos = {b: i for i, b in enumerate(bg.bg10)}
    di = {d: i for i, d in enumerate(DAYS)}
    sd = pd.DatetimeIndex(storm)
    cj = sd.map(lambda d: di[d])
    cmat = np.zeros((len(bg), len(sd)), np.float32)
    cl = cl[cl.bg10.isin(pos) & cl.date.isin(sd)]
    np.add.at(cmat, (cl.bg10.map(pos).to_numpy(), pd.Index(sd).get_indexer(cl.date)), 1)
    pol = ex[:, cj]
    days, lat, lon, P = fp.load_precip()
    iy, ix = fp.grid_index(lat, lon, bg.lat.to_numpy(), bg.lon.to_numpy())
    pidx = {d: i for i, d in enumerate(days)}
    R1, R2, A7 = (np.zeros((len(bg), len(sd)), np.float32) for _ in range(3))
    for j, d in enumerate(sd):
        i = pidx[d]
        g = lambda k: np.nan_to_num(np.asarray(P[i - k])[iy, ix])
        R1[:, j] = g(0)
        R2[:, j] = R1[:, j] + g(1)
        A7[:, j] = sum(g(k) for k in range(2, 8))
    W = pol > 0
    z = np.where(W, np.log1p(1000 * cmat / np.maximum(pol, 1e-6)), 0).astype(np.float32)
    print(f"block groups {len(bg):,}; storm days {len(sd)} (train {(sd.year <= 2016).sum()}, test {(sd.year >= 2017).sum()}); "
          f"claims on them {int(cmat.sum()):,}; any-claim share of BG-days with a policy {(cmat[W] > 0).mean():.4f}")
    return bg, sd, cmat, pol, R1, R2, A7, W, z


def features(bg, pol, R1, R2, A7):
    nb, nd = R1.shape
    return np.c_[R1.ravel(), R2.ravel(), A7.ravel(), np.log1p(pol).ravel(), np.repeat(bg.lat.to_numpy(), nd),
                 np.repeat(bg.lon.to_numpy(), nd)].astype(np.float32)


def idw_fill(xy_obs, v_obs, xy_hid, k=8, rmax=50.0):
    """Inverse-distance mean of up to k observed neighbours within rmax km; 0 where none."""
    if len(xy_obs) == 0:
        return np.zeros(len(xy_hid))
    t = cKDTree(xy_obs)
    d, i = t.query(xy_hid, k=min(k, len(xy_obs)), distance_upper_bound=rmax)
    d, i = np.atleast_2d(d.T).T if d.ndim == 1 else d, np.atleast_2d(i.T).T if i.ndim == 1 else i
    ok = np.isfinite(d)
    w = np.where(ok, 1.0 / np.maximum(d, 1.0), 0)
    vv = np.where(ok, v_obs[np.where(ok, i, 0)], 0)
    s = w.sum(1)
    return np.where(s > 0, (w * vv).sum(1) / np.where(s > 0, s, 1), 0)


def foldin(U, mu, y_obs, obs, hid, lam):
    Uo = U[obs]
    v = np.linalg.solve(Uo.T @ Uo + lam * np.eye(U.shape[1]), Uo.T @ (y_obs - mu[obs]))
    return mu[hid] + U[hid] @ v


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--min-claims", type=int, default=15)
    a = ap.parse_args()
    bg, sd, cmat, pol, R1, R2, A7, W, z = load(a.min_claims)
    tr = np.flatnonzero(sd.year <= 2014)       # base model and SVD factors
    va = np.flatnonzero((sd.year >= 2015) & (sd.year <= 2016))  # stacking weights
    te = np.flatnonzero(sd.year >= 2017)
    nb = len(bg)
    print(f"fit days {len(tr)}, validation days {len(va)}, test days {len(te)}")
    X = features(bg, pol, R1, R2, A7).reshape(nb, len(sd), -1)
    idx = np.argwhere(W[:, tr])
    sub = idx[rng.choice(len(idx), min(800_000, len(idx)), replace=False)]
    m = HGB(max_iter=250, learning_rate=0.08, max_depth=5, random_state=0).fit(X[sub[:, 0], tr[sub[:, 1]]], z[sub[:, 0], tr[sub[:, 1]]])
    base = np.stack([m.predict(X[:, j]) for j in range(len(sd))], 1).astype(np.float32)
    base = np.where(W, np.maximum(base, 0), 0)
    res = np.where(W, z - base, 0)
    yt = (cmat[:, te][W[:, te]] > 0).astype(int)
    print(f"base model test AUC (all BG-days, no observed claims) {auc(yt, base[:, te][W[:, te]]):.3f}")

    def factors(M, k):
        mu = (M[:, tr] * W[:, tr]).sum(1) / np.maximum(W[:, tr].sum(1), 1)
        Mc = np.where(W[:, tr], M[:, tr] - mu[:, None], 0)
        U, s, _ = np.linalg.svd(Mc, full_matrices=False)
        ev = (s ** 2) / (s ** 2).sum()
        return mu, U[:, :k] * s[:k] ** 0.5, ev
    mu_z, Uz, evz = factors(z, a.k)
    mu_r, Ur, evr = factors(res, a.k)
    print(f"SVD rank {a.k} on fit days: explained variance z {evz[:a.k].sum():.3f} (first 3 {np.round(evz[:3], 3)}), "
          f"residual {evr[:a.k].sum():.3f} (first 3 {np.round(evr[:3], 3)})")
    xy = np.c_[bg.lon.to_numpy() * np.cos(np.radians(31)) * 111.32, bg.lat.to_numpy() * 110.57]
    region = KMeans(5, n_init=5, random_state=0).fit_predict(xy)

    def records(days, kind, rho):
        recs = []
        for jj in days:
            parts = [(m_ := rng.random(nb) < rho, ~m_)] if kind == "r" else [(region != f, region == f) for f in range(5)]
            for obs_m, hid_m in parts:
                o = np.flatnonzero(obs_m & W[:, jj])
                h = np.flatnonzero(hid_m & W[:, jj])
                if len(o) < 10 or len(h) < 10:
                    continue
                pos_o = o[cmat[o, jj] > 0]
                nn = (cKDTree(xy[pos_o]).query_ball_point(xy[h], 25.0, return_length=True) if len(pos_o) else np.zeros(len(h)))
                corr = foldin(Ur, mu_r, res[o, jj], o, h, a.lam) - mu_r[h]
                ir = idw_fill(xy[o], res[o, jj], xy[h])
                f = {"base": base[h, jj], "idw_res": ir, "corr": corr, "mu": mu_r[h], "nn": np.log1p(nn),
                     "idw": idw_fill(xy[o], z[o, jj], xy[h]), "svd": foldin(Uz, mu_z, z[o, jj], o, h, a.lam)}
                recs.append((jj, cmat[h, jj], f))
        return recs

    stack_cols = ["base", "idw_res", "corr", "mu", "nn"]
    meths = ["base", "idw", "svd", "rk", "svdr", "stack"]
    scen = [("random 5%", "r", 0.05), ("random 20%", "r", 0.2), ("random 50%", "r", 0.5), ("region block", "b", None)]
    for name, kind, rho in scen:
        rv = records(va, kind, rho)
        Xv = np.vstack([np.column_stack([r[2][c] for c in stack_cols]) for r in rv])
        yv = np.concatenate([(r[1] > 0).astype(int) for r in rv])
        sc = StandardScaler().fit(Xv)
        lr = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xv), yv)
        recs = records(te, kind, rho)
        for r in recs:
            f = r[2]
            f["rk"] = f["base"] + f["idw_res"]
            f["svdr"] = f["base"] + f["mu"] + f["corr"]
            f["stack"] = lr.decision_function(sc.transform(np.column_stack([f[c] for c in stack_cols])))
        y = np.concatenate([(r[1] > 0).astype(int) for r in recs])
        S = {k: np.concatenate([r[2][k] for r in recs]) for k in meths}
        day = np.concatenate([np.full(len(r[1]), r[0]) for r in recs])
        wid = pd.Series(sd[day].to_period("W")).astype(str).to_numpy()
        uw = np.unique(wid)
        wi = {w: np.flatnonzero(wid == w) for w in uw}
        draws = [np.concatenate([wi[w] for w in rng.choice(uw, len(uw))]) for _ in range(200)]
        print(f"\n=== {name}: {len(y):,} hidden BG-days, {int(y.sum())} with a claim; stack weights "
              f"{dict(zip(stack_cols, np.round(lr.coef_[0], 2), strict=True))} ===")
        # ablation of the stacking inputs (weights refitted on the validation days each time)
        Xte = np.vstack([np.column_stack([r[2][c] for c in stack_cols]) for r in recs])
        ab = []
        for nm, cols in (("base only", ["base"]), ("base + mu (BG claim history)", ["base", "mu"]),
                         ("base + nn (claim count within 25 km)", ["base", "nn"]), ("base + idw_res", ["base", "idw_res"]),
                         ("base + SVD fold-in (corr)", ["base", "corr"]), ("base + mu + nn", ["base", "mu", "nn"]),
                         ("base + mu + nn + idw_res", ["base", "mu", "nn", "idw_res"]), ("all (incl. SVD corr)", stack_cols)):
            ci = [stack_cols.index(c) for c in cols]
            sc2 = StandardScaler().fit(Xv[:, ci])
            l2 = LogisticRegression(max_iter=1000).fit(sc2.transform(Xv[:, ci]), yv)
            sco = l2.decision_function(sc2.transform(Xte[:, ci]))
            dd = [auc(y[ix], sco[ix]) for ix in draws]
            ab.append((nm, auc(y, sco), np.percentile(dd, 2.5), np.percentile(dd, 97.5)))
        print("  ablation (pooled AUC, test):")
        for nm, v, lo, hi in ab:
            print(f"    {nm:<40} {v:.3f} [{lo:.3f}, {hi:.3f}]")
        rows = []
        for k in meths:
            wa, cap = [], []
            for r in recs:
                c = r[1]
                if (c > 0).sum() >= 5:
                    wa.append(auc((c > 0).astype(int), r[2][k]))
                if c.sum() >= 5:
                    sc_ = r[2][k] + 1e-9 * rng.random(len(c))  # break ties (a constant score must not capture everything)
                    top = np.argsort(-sc_)[: max(1, int(0.05 * len(c)))]
                    cap.append(c[top].sum() / c.sum())
            d = [auc(y[ix], S[k][ix]) - auc(y[ix], S["base"][ix]) for ix in draws] if k != "base" else [0, 0]
            rows.append((k, auc(y, S[k]), np.nanmean(wa), np.mean(cap), np.percentile(d, 2.5), np.percentile(d, 97.5)))
        print(pd.DataFrame(rows, columns=["method", "pooled AUC", "within-day AUC", "top-5% capture", "dAUC lo", "dAUC hi"]).round(3).to_string(index=False))
    print("\n=== SVD rank sensitivity (random 20% reveal; base + mu + fold-in of the residual; pooled AUC, test) ===")
    for k in (1, 2, 3, 5, 10, 20):
        _, Uk, _ = factors(res, k)
        ys, ss = [], []
        for jj in te:
            obs_m = rng.random(nb) < 0.2
            o = np.flatnonzero(obs_m & W[:, jj]); h = np.flatnonzero(~obs_m & W[:, jj])
            ys.append((cmat[h, jj] > 0).astype(int))
            ss.append(base[h, jj] + foldin(Uk, mu_r, res[o, jj], o, h, a.lam))
        print(f"  k={k:>2}: {auc(np.concatenate(ys), np.concatenate(ss)):.3f}")


if __name__ == "__main__":
    main()
