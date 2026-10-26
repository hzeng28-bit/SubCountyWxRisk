"""Statistical diagnostics used by RAW_STATICS_DA_REPORT.md: is there cluster structure, is there regressable signal.

Clustering
  hopkins                  clustering tendency (0.5 = uniform noise, towards 1 = clustered)
  silhouette_curve         K-means silhouette / Davies-Bouldin / Calinski-Harabasz over K, against a column-permuted null
  stability                adjusted Rand index between K-means fits (other seed, other half of the rows)
  label_association        does the cluster label carry information about an outcome (mutual information vs permutation null)
  bimodality_coefficient   Sarle's coefficient for 1-D data (> 0.555 suggests more than one mode)

Regression
  cluster_corr             Spearman correlation with a cluster-bootstrap interval, optionally WITHIN groups (rank inside each group)
  vif, condition_number    collinearity of a feature set
  events_per_variable      sample-size sufficiency for a binary outcome

Everything takes an explicit `rng`/`seed` so a report can be regenerated exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

# ------------------------------------------------------------------ clustering


def hopkins(X: np.ndarray, m: int = 1000, seed: int = 0) -> float:
    """Hopkins statistic H = sum(u) / (sum(u) + sum(w)): u = distance of m uniform random points to the data, w = distance of m data points to
    their nearest other data point. Uniform noise gives about 0.5, clustered data approaches 1, a regular lattice goes below 0.5."""
    rng = np.random.default_rng(seed)
    n = len(X)
    m = min(m, n // 2)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    lo, hi = X.min(0), X.max(0)
    u = nn.kneighbors(rng.uniform(lo, hi, (m, X.shape[1])), n_neighbors=1)[0].ravel()
    w = nn.kneighbors(X[rng.choice(n, m, replace=False)], n_neighbors=2)[0][:, 1]
    return float(u.sum() / (u.sum() + w.sum()))


def permute_columns(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """The same marginals with the dependence between columns destroyed: the null for 'does the cloud have joint structure'."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])


def silhouette_curve(X: np.ndarray, ks: list[int], n_sample: int = 20000, seed: int = 0, null: bool = True) -> pd.DataFrame:
    """K-means for each K (fitted on a subsample), then silhouette on a smaller one. `null` repeats it on column-permuted data."""
    rng = np.random.default_rng(seed)
    fit = X[rng.choice(len(X), min(len(X), 4 * n_sample), replace=False)]
    sub = rng.choice(len(fit), min(len(fit), n_sample), replace=False)
    rows = []
    for name, data in (("data", fit), ("null", permute_columns(fit, seed) if null else None)):
        if data is None:
            continue
        for k in ks:
            km = KMeans(k, n_init=3, random_state=seed).fit(data)
            lab = km.labels_
            rows.append({"set": name, "K": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(data[sub], lab[sub])),
                         "davies_bouldin": float(davies_bouldin_score(data[sub], lab[sub])),
                         "calinski_harabasz": float(calinski_harabasz_score(data[sub], lab[sub]))})
    return pd.DataFrame(rows)


def silhouette_by_cluster(X: np.ndarray, lab: np.ndarray, n_sample: int = 20000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(len(X), n_sample), replace=False)
    s = silhouette_samples(X[idx], lab[idx])
    return pd.DataFrame({"cluster": lab[idx], "silhouette": s}).groupby("cluster").silhouette.agg(["mean", "size"])


def stability(X: np.ndarray, k: int, seed: int = 0, n_fit: int = 80000) -> dict[str, float]:
    """ARI between (a) two seeds on the same rows, (b) two disjoint halves, each half's model labelling ALL rows."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    a, b = idx[: min(n_fit, len(X) // 2)], idx[len(X) // 2 : len(X) // 2 + min(n_fit, len(X) // 2)]
    m1 = KMeans(k, n_init=3, random_state=seed).fit(X[a])
    m2 = KMeans(k, n_init=3, random_state=seed + 1).fit(X[a])
    m3 = KMeans(k, n_init=3, random_state=seed).fit(X[b])
    ev = rng.choice(len(X), min(len(X), 100000), replace=False)
    return {"ari_other_seed": float(adjusted_rand_score(m1.predict(X[ev]), m2.predict(X[ev]))),
            "ari_other_half": float(adjusted_rand_score(m1.predict(X[ev]), m3.predict(X[ev])))}


def label_association(lab: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int = 0) -> dict[str, float]:
    """Mutual information (nats) between cluster label and a binary outcome, with a permutation null, and eta squared of the cluster mean of y."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    k = int(lab.max()) + 1

    def mi(l, yy) -> float:
        """I(cluster; y) = sum_k sum_v p(k,v) log(p(k,v) / (p(k) p(v)))."""
        n = len(l)
        c = np.bincount(l, minlength=k).astype(float)
        c1 = np.bincount(l, weights=yy, minlength=k)
        c0 = c - c1
        p1 = yy.mean()
        out = 0.0
        for cnt, pv in ((c1, p1), (c0, 1 - p1)):
            m = cnt > 0
            out += float(np.sum(cnt[m] / n * np.log((cnt[m] / n) / ((c[m] / n) * pv))))
        return out

    obs = mi(lab, y)
    null = np.array([mi(lab, rng.permutation(y)) for _ in range(n_perm)])
    mean = pd.Series(y).groupby(lab).transform("mean").to_numpy()
    eta2 = float(1 - ((y - mean) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return {"mutual_information": obs, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)), "eta2_of_cluster_mean": eta2}


def bimodality_coefficient(x: np.ndarray) -> float:
    from scipy.stats import kurtosis, skew

    x = np.asarray(x, float)
    n = len(x)
    return float((skew(x) ** 2 + 1) / (kurtosis(x) + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def silhouette_verdict(s: float) -> str:
    """Kaufman & Rousseeuw's reading of the average silhouette."""
    if s > 0.70:
        return "strong structure"
    if s > 0.50:
        return "reasonable structure"
    if s > 0.25:
        return "weak structure"
    return "no substantial structure"


# ------------------------------------------------------------------ regression


def _pct_rank(x: np.ndarray, groups: np.ndarray | None) -> np.ndarray:
    if groups is None:
        return (rankdata(x) - 0.5) / len(x)
    s = pd.Series(x)
    return s.groupby(groups, sort=False).rank(pct=True).to_numpy() - 0.5 / s.groupby(groups, sort=False).transform("count").to_numpy()


def cluster_corr(x: np.ndarray, y: np.ndarray, groups: np.ndarray, within: bool = False, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Spearman correlation of x and y (Pearson on ranks) with a bootstrap over `groups` (clusters of dependent rows).

    within=False: ranks over all rows (between + within structure). within=True: ranks inside each group, so only the within-group co-variation
    counts (the within-event view; groups with a constant x or y contribute nothing). Per-group sufficient statistics make the bootstrap exact and fast.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, g = x[ok], y[ok], groups[ok]
    rx, ry = _pct_rank(x, g if within else None), _pct_rank(y, g if within else None)
    if within:
        rx, ry = rx - pd.Series(rx).groupby(g).transform("mean").to_numpy(), ry - pd.Series(ry).groupby(g).transform("mean").to_numpy()
    df = pd.DataFrame({"g": g, "n": 1.0, "sx": rx, "sy": ry, "sxx": rx * rx, "syy": ry * ry, "sxy": rx * ry})
    s = df.groupby("g", sort=False).sum().to_numpy()  # columns n sx sy sxx syy sxy

    def corr(t: np.ndarray) -> float:
        n, sx, sy, sxx, syy, sxy = t
        c = sxy / n - sx * sy / n**2
        v = (sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2)
        return float(c / np.sqrt(v)) if v > 0 else 0.0

    est = corr(s.sum(0))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    boots = np.array([corr(s[i].sum(0)) for i in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"rho": est, "lo": float(lo), "hi": float(hi), "n": int(len(x)), "clusters": int(len(s)),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def vif(Z: np.ndarray) -> np.ndarray:
    """Variance inflation factors from the inverse correlation matrix (columns standardised internally)."""
    C = np.corrcoef(Z, rowvar=False)
    return np.diag(np.linalg.pinv(C))


def condition_number(Z: np.ndarray) -> float:
    C = np.corrcoef(Z, rowvar=False)
    ev = np.linalg.eigvalsh(C)
    return float(np.sqrt(ev.max() / max(ev.min(), 1e-12)))


def events_per_variable(y: np.ndarray, n_features: int) -> float:
    """Rule of thumb for logistic models: the rarer class count per candidate variable; below ~10 the fit is unreliable."""
    y = np.asarray(y)
    return float(min(y.sum(), (1 - y).sum()) / n_features)


def eta2_cluster(y: np.ndarray, cat: np.ndarray, groups: np.ndarray, n_boot: int = 300, seed: int = 0) -> dict[str, float]:
    """Share of the variance of y explained by a categorical variable (eta squared), bootstrap over `groups`.

    Per-(group, category) sums make each resample a few array additions: eta2 = 1 - SSE_within / SST from the pooled sums.
    """
    ok = np.isfinite(y)
    y, cat, g = np.asarray(y, float)[ok], np.asarray(cat)[ok], np.asarray(groups)[ok]
    cats = pd.Index(pd.unique(cat))
    gi = pd.factorize(g)[0]
    ci = cats.get_indexer(cat)
    G, K = gi.max() + 1, len(cats)
    n = np.zeros((G, K))
    s = np.zeros((G, K))
    s2 = np.zeros((G, K))
    np.add.at(n, (gi, ci), 1.0)
    np.add.at(s, (gi, ci), y)
    np.add.at(s2, (gi, ci), y * y)

    def eta(idx: np.ndarray) -> float:
        nn, ss, qq = n[idx].sum(0), s[idx].sum(0), s2[idx].sum(0)
        tot_n, tot_s, tot_q = nn.sum(), ss.sum(), qq.sum()
        sst = tot_q - tot_s**2 / tot_n
        sse = (qq - np.divide(ss**2, nn, out=np.zeros_like(ss), where=nn > 0)).sum()
        return float(1 - sse / sst) if sst > 0 else 0.0

    rng = np.random.default_rng(seed)
    est = eta(np.arange(G))
    boots = np.array([eta(rng.integers(0, G, G)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"eta2": est, "lo": float(lo), "hi": float(hi), "categories": int(K), "n": int(len(y)), "clusters": int(G)}


def cluster_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int = 500, seed: int = 0) -> dict[str, float]:
    """Within-group AUC of x for a binary y: only pairs inside the same group are compared (groups with one class are dropped), pooled by pair count.

    This is the project's within-event metric; the bootstrap resamples whole groups (per-group U statistics and pair counts are summed).
    Effect-size reading (Rice & Harris, equivalent to Cohen's d of 0.2 / 0.5 / 0.8): 0.556 small, 0.638 medium, 0.714 large.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    df = pd.DataFrame({"g": np.asarray(groups)[ok], "x": np.asarray(x, float)[ok], "y": np.asarray(y, float)[ok]})
    n1 = df.groupby("g").y.transform("sum")
    n = df.groupby("g").y.transform("size")
    df = df[(n1 > 0) & (n1 < n)]
    df["r"] = df.groupby("g").x.rank()
    pos = df[df.y == 1]
    u = (pos.groupby("g").r.sum() - pos.groupby("g").size() * (pos.groupby("g").size() + 1) / 2).rename("u")
    n1g = df.groupby("g").y.sum()
    pairs = (n1g * (df.groupby("g").size() - n1g)).rename("pairs")
    s = pd.concat([u, pairs], axis=1).fillna(0.0).to_numpy()
    est = float(s[:, 0].sum() / s[:, 1].sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    boots = s[idx, 0].sum(1) / s[idx, 1].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"auc": est, "lo": float(lo), "hi": float(hi), "n": int(len(df)), "clusters": int(len(s)), "excludes_half": bool(lo > 0.5 or hi < 0.5)}


def auc_size(auc: float) -> str:
    a = max(auc, 1 - auc)
    return "negligible" if a < 0.556 else "small" if a < 0.638 else "medium" if a < 0.714 else "large"
