"""The report's diagnostics must tell structure from noise on data where the answer is known."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import diag  # noqa: E402

R = np.random.default_rng(0)


def blobs(n=3000, k=4, sep=8.0):
    c = sep * np.eye(k, 5)  # equidistant centres, so the true K is the best K
    lab = R.integers(0, k, n)
    return c[lab] + R.normal(size=(n, 5)), lab


def test_hopkins_separates_clusters_from_uniform_noise() -> None:
    X, _ = blobs()
    assert diag.hopkins(X, 500) > 0.85
    assert 0.40 < diag.hopkins(R.uniform(size=(3000, 5)), 500) < 0.60


def test_silhouette_curve_finds_k_and_the_null_has_none() -> None:
    X, _ = blobs()
    d = diag.silhouette_curve(X, [2, 4, 8], n_sample=2000)
    data = d[d.set == "data"].set_index("K")
    assert data.silhouette.idxmax() == 4 and data.silhouette[4] > 0.6
    assert diag.silhouette_verdict(float(data.silhouette[4])) in ("reasonable structure", "strong structure")
    uni = diag.silhouette_curve(R.uniform(size=(3000, 5)), [4], n_sample=2000)
    assert diag.silhouette_verdict(float(uni[uni.set == "data"].silhouette.iloc[0])) in ("no substantial structure", "weak structure")


def test_stability_is_high_for_real_clusters() -> None:
    X, _ = blobs(4000)
    s = diag.stability(X, 4, n_fit=1500)
    assert s["ari_other_seed"] > 0.95 and s["ari_other_half"] > 0.95


def test_label_association_detects_signal_and_not_noise() -> None:
    lab = R.integers(0, 10, 20000)
    y_sig = (R.random(20000) < np.where(lab < 2, 0.3, 0.02)).astype(float)
    a = diag.label_association(lab, y_sig, n_perm=100)
    assert a["p_value"] < 0.02 and a["mutual_information"] > 5 * a["null_p95"]
    b = diag.label_association(lab, (R.random(20000) < 0.05).astype(float), n_perm=100)
    assert b["p_value"] > 0.05


def test_bimodality_coefficient() -> None:
    uni = R.normal(size=5000)
    bi = np.concatenate([R.normal(-3, 1, 2500), R.normal(3, 1, 2500)])
    assert diag.bimodality_coefficient(uni) < 0.555 < diag.bimodality_coefficient(bi)


def test_cluster_corr_within_groups_removes_the_between_group_level() -> None:
    g = np.repeat(np.arange(200), 20)
    level = np.repeat(R.normal(size=200) * 5, 20)  # a group-level confounder shared by x and y
    x = level + R.normal(size=4000)
    y = level + R.normal(size=4000)
    allr = diag.cluster_corr(x, y, g, within=False, n_boot=200)
    inner = diag.cluster_corr(x, y, g, within=True, n_boot=200)
    assert allr["rho"] > 0.8  # driven by the level
    assert inner["lo"] < 0 < inner["hi"] and abs(inner["rho"]) < 0.15  # nothing inside a group
    z = x + y + R.normal(size=4000)
    assert diag.cluster_corr(x, z - level, g, within=True, n_boot=200)["excludes_zero"]


def test_vif_and_condition_number() -> None:
    a = R.normal(size=5000)
    Z = np.column_stack([a, a + 0.05 * R.normal(size=5000), R.normal(size=5000)])
    v = diag.vif(Z)
    assert v[0] > 100 and v[1] > 100 and v[2] < 1.1
    assert diag.condition_number(Z) > 10 > diag.condition_number(R.normal(size=(5000, 3)))
    assert diag.events_per_variable(np.array([1] * 30 + [0] * 970), 10) == 3.0


def test_eta2_cluster() -> None:
    g = np.repeat(np.arange(100), 30)
    cat = R.integers(0, 3, 3000)
    y = cat * 1.0 + R.normal(size=3000)
    e = diag.eta2_cluster(y, cat, g, n_boot=100)
    truth = (np.var([0, 1, 2]) / (np.var([0, 1, 2]) + 1))  # between / (between + within)
    assert abs(e["eta2"] - truth) < 0.05 and e["lo"] < e["eta2"] < e["hi"]
    null = diag.eta2_cluster(R.normal(size=3000), cat, g, n_boot=100)
    assert null["eta2"] < 0.01


def test_cluster_auc_is_within_group_and_detects_a_shift() -> None:
    g = np.repeat(np.arange(300), 30)
    level = np.repeat(R.normal(size=300) * 4, 30)
    y = (R.random(9000) < 0.15).astype(float)
    x = level + R.normal(size=9000) + 1.2 * y  # a real shift of 1.2 sd inside each group, on top of a group level
    a = diag.cluster_auc(x, y, g, n_boot=100)
    truth = 0.5 + 0.5 * (2 * __import__("scipy.stats", fromlist=["norm"]).norm.cdf(1.2 / np.sqrt(2)) - 1)
    assert abs(a["auc"] - truth) < 0.03 and a["excludes_half"] and diag.auc_size(a["auc"]) in ("medium", "large")
    noise = diag.cluster_auc(level + R.normal(size=9000), y, g, n_boot=100)  # the group level alone carries nothing inside a group
    assert not noise["excludes_half"] and diag.auc_size(noise["auc"]) == "negligible"
