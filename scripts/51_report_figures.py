"""Figures and numbers for the NFIP / sub-county progress report (docs/progress_report).

Everything is computed from the stored pilot table and the NFIP files; nothing is trained on test years.
  fig_texas_claims.png    Texas block groups coloured by NFIP claims inside the episode windows (does the outcome have spatial structure?)
  fig_county_episode.png  one county, one episode: cell rain, policies in force, outcome, simple-model score on the block-group polygons
  fig_dose_response.png   claim rate by cell-rain bin and by policy-count bin; damage ratio by interior depth (NFIP tracks hazard and exposure)
  fig_covariates.png      single-feature within-event AUC, standardised logistic coefficients, gain of each feature group over exposure + cell rain
  report_numbers.json     every number quoted in the report

Inputs : flood_pilot_table.parquet, bg_terrain.parquet, bg2010_tx.zip, nfip_national/nfip_*.parquet
Run    : uv run python scripts/51_report_figures.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
from matplotlib.collections import PolyCollection
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk.prep.scaling import FeaturePlan  # noqa: E402
from wxrisk.prep.scaling_plan import PILOT_PLAN  # noqa: E402

spec = importlib.util.spec_from_file_location("m26", ROOT / "scripts" / "26_flood_pilot_experiment.py")
m26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m26)

OUT = ROOT / "docs" / "progress_report" / "figs"
OUT.mkdir(parents=True, exist_ok=True)
NUM: dict = {}
rng = np.random.default_rng(0)


def polygons(codes: set[str] | None = None):
    """{bg10: list of rings (Nx2 arrays)} from the 2010 TIGER zip, read with pyshp."""
    z = zipfile.ZipFile(fp.PILOT / "bg2010_tx.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")),
                          dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    out = {}
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        code = rec["STATE"] + rec["COUNTY"] + rec["TRACT"] + rec["BLKGRP"]
        if codes is not None and code not in codes:
            continue
        pts = np.asarray(sr.shape.points)
        parts = list(sr.shape.parts) + [len(pts)]
        out[code] = [pts[a:b] for a, b in zip(parts[:-1], parts[1:], strict=True)]
    return out


def draw(ax, polys, codes, values, cmap, vmin=None, vmax=None, title="", cbar_label=""):
    rings, vals = [], []
    for c, v in zip(codes, values, strict=True):
        for r in polys.get(c, []):
            rings.append(r)
            vals.append(v)
    pc = PolyCollection(rings, array=np.asarray(vals, float), cmap=cmap, edgecolors="k", linewidths=0.15)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(30)))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    cb = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(cbar_label, fontsize=7)
    cb.ax.tick_params(labelsize=7)


def episode_parts(ep, y, s):
    """per-episode (pairs, pairs * AUC) for the pooled within-event AUC; episodes with < 10 rows or one class are skipped."""
    from scipy.stats import rankdata

    order = np.argsort(ep, kind="stable")
    ep, y, s = ep[order], y[order], s[order]
    cuts = np.flatnonzero(np.diff(ep)) + 1
    den, num = [], []
    for yy, ss in zip(np.split(y, cuts), np.split(s, cuts), strict=True):
        n1 = int(yy.sum())
        n0 = len(yy) - n1
        if len(yy) < 10 or n1 == 0 or n0 == 0:
            continue
        r = rankdata(ss)
        den.append(n1 * n0)
        num.append(n1 * n0 * (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return np.array(den, float), np.array(num, float)


def boot(parts_a, parts_b=None, n=1000):
    """event-clustered bootstrap of AUC_a (or AUC_a - AUC_b) from per-episode parts (same episode order)."""
    da, na = parts_a
    idx = rng.integers(0, len(da), (n, len(da)))
    a = na[idx].sum(1) / da[idx].sum(1)
    if parts_b is not None:
        db, nb = parts_b
        a = a - nb[idx].sum(1) / db[idx].sum(1)
    return np.percentile(a, [2.5, 97.5])


def main() -> None:
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t["log_area"] = np.log(t.area_km2.clip(lower=0.01))
    ter = pd.read_parquet(fp.PILOT / "bg_terrain.parquet")
    t = t.merge(ter, on="bg10", how="left")
    y = t.y.to_numpy().astype(float)
    yr = t.year.to_numpy()
    tr, te = yr <= 2016, yr >= 2017
    ep = t.ep.to_numpy()

    # ---------------------------------------------------------------- simple models (logistic regression on the scaling plan)
    GROUPS = {
        "exposure": m26.EXPO,
        "county rain": [c for c in m26.COUNTY if c != "cty_nbg"] + ["cty_nbg"],
        "cell rain": m26.CELL,
        "neighbourhood rain": m26.NEIGH,
        "location / antecedent": m26.STATIC,
        "terrain": ["elev", "rel1", "rel3", "rel10", "slope", "pos3"],
    }

    def fit_logit(cols):
        plan = FeaturePlan(tuple(r for r in PILOT_PLAN.rules if r.col in cols))
        prm = plan.fit(t, tr, "year <= 2016")
        Z = plan.apply(t, prm)
        Zv = np.nan_to_num(Z.to_numpy())
        lr = LogisticRegression(C=0.1, max_iter=300).fit(Zv[tr], y[tr])
        return lr.predict_proba(Zv[te])[:, 1], dict(zip(Z.columns, lr.coef_[0], strict=True)), lr, Zv, Z.columns

    e_te, y_te = ep[te], y[te]
    base_cols = GROUPS["exposure"] + GROUPS["county rain"] + GROUPS["cell rain"]
    p_base, coef_base, lr_base, Zb, zcols = fit_logit(base_cols)
    parts_base = episode_parts(e_te, y_te, p_base)
    NUM["L2_logistic_within_auc"] = float(parts_base[1].sum() / parts_base[0].sum())
    NUM["L2_logistic_ci"] = boot(parts_base).tolist()
    NUM["L2_coefficients_standardised"] = {k: round(float(v), 3) for k, v in coef_base.items()}

    # minimal models
    mini = {}
    for name, cols in {"policy count only": ["log_pol"], "cell rain only (f1)": ["bg_f1"],
                       "policy count + cell rain": ["log_pol", "bg_f1"],
                       "exposure (4 cols)": GROUPS["exposure"], "exposure + cell rain": GROUPS["exposure"] + GROUPS["cell rain"]}.items():
        p, _, _, _, _ = fit_logit(cols)
        pr = episode_parts(e_te, y_te, p)
        mini[name] = (float(pr[1].sum() / pr[0].sum()), boot(pr).tolist())
    NUM["minimal_models"] = mini

    # gain of each group on top of exposure + cell rain, and loss when dropped from the full model
    ref_cols = GROUPS["exposure"] + GROUPS["cell rain"]
    p_ref, _, _, _, _ = fit_logit(ref_cols)
    pr_ref = episode_parts(e_te, y_te, p_ref)
    gains = {}
    for g in ("county rain", "neighbourhood rain", "location / antecedent", "terrain"):
        p, _, _, _, _ = fit_logit(ref_cols + GROUPS[g])
        pr = episode_parts(e_te, y_te, p)
        gains[g] = (float(pr[1].sum() / pr[0].sum() - pr_ref[1].sum() / pr_ref[0].sum()), boot(pr, pr_ref).tolist())
    for g in ("exposure", "cell rain"):  # loss when dropped from exposure + cell rain
        keep = [c for c in ref_cols if c not in GROUPS[g]]
        p, _, _, _, _ = fit_logit(keep + ["cty_f1_max", "cty_f1_mean", "cty_s_mean", "cty_a14_mean", "dur", "month", "cty_nbg"])
        pr = episode_parts(e_te, y_te, p)
        gains[f"drop {g}"] = (float(pr[1].sum() / pr[0].sum() - pr_ref[1].sum() / pr_ref[0].sum()), boot(pr, pr_ref).tolist())
    NUM["group_gain_over_exposure_plus_cell"] = gains

    # single-feature scores, no training
    singles = {}
    for c in ["bg_f1", "bg_s", "bg_f3", "nb3_mean", "nb7_mean", "nep7_25", "nep15_50", "cty_f1_max", "log_pol", "sfha_share",
              "single_share", "mean_cov", "log_area", "lat", "lon", "elev", "rel3", "slope", "bg_a14"]:
        s = np.nan_to_num(t[c].to_numpy(float)[te])
        pr = episode_parts(e_te, y_te, s)
        singles[c] = (float(pr[1].sum() / pr[0].sum()), boot(pr, n=400).tolist())
    NUM["single_feature_within_auc"] = singles

    # ---------------------------------------------------------------- figure: covariates
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    names = sorted(singles, key=lambda k: singles[k][0])
    v = [singles[k][0] for k in names]
    lo = [singles[k][0] - singles[k][1][0] for k in names]
    hi = [singles[k][1][1] - singles[k][0] for k in names]
    ax[0].barh(names, np.array(v) - 0.5, left=0.5, xerr=[lo, hi], color=["#c44" if x < 0.5 else "#357" for x in v], height=0.6,
               error_kw={"lw": 0.8})
    ax[0].axvline(0.5, color="k", lw=0.8)
    ax[0].set_xlabel("within-event AUC of the raw feature (test 2017-23)", fontsize=8)
    ax[0].set_title("(a) one feature, no training", fontsize=9)
    ax[0].tick_params(labelsize=7)
    ck = [k for k in coef_base if k in zcols]
    cv = sorted(ck, key=lambda k: coef_base[k])
    ax[1].barh(cv, [coef_base[k] for k in cv], color="#357", height=0.6)
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set_xlabel("standardised coefficient (exposure + county + cell rain)", fontsize=8)
    ax[1].set_title("(b) logistic regression, train 2010-16", fontsize=9)
    ax[1].tick_params(labelsize=7)
    gk = list(gains)
    gv = [gains[k][0] for k in gk]
    ax[2].barh(gk, gv, xerr=[[gains[k][0] - gains[k][1][0] for k in gk], [gains[k][1][1] - gains[k][0] for k in gk]], color="#357",
               height=0.6, error_kw={"lw": 0.8})
    ax[2].axvline(0, color="k", lw=0.8)
    ax[2].set_xlabel("change in within-event AUC vs exposure + cell rain", fontsize=8)
    ax[2].set_title("(c) what each group adds / loses (95% CI)", fontsize=9)
    ax[2].tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "fig_covariates.png", dpi=170)
    plt.close()

    # ---------------------------------------------------------------- figure: dose response
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    tt = t.copy()
    tt["rb"] = pd.cut(tt.bg_f1, [0, 30, 50, 75, 100, 150, 1000], include_lowest=True)
    g = tt.groupby("rb", observed=True).y.agg(["mean", "size"])
    ax[0].bar(range(len(g)), g["mean"] * 100, color="#357")
    ax[0].set_xticks(range(len(g)), [str(i) for i in g.index], rotation=30, fontsize=7)
    ax[0].set_ylabel("% block groups with a claim", fontsize=8)
    ax[0].set_xlabel("cell rain F1 (mm)", fontsize=8)
    ax[0].set_title("(a) claim rate rises with rain", fontsize=9)
    tt["pb"] = pd.cut(tt.n_pol, [0, 1, 5, 20, 50, 100, 300, 1e5])
    g2 = tt.groupby("pb", observed=True).y.agg(["mean", "size"])
    ax[1].bar(range(len(g2)), g2["mean"] * 100, color="#357")
    ax[1].set_xticks(range(len(g2)), [str(i) for i in g2.index], rotation=30, fontsize=7)
    ax[1].set_xlabel("policies in force", fontsize=8)
    ax[1].set_title("(b) ... and with insured stock", fontsize=9)
    NUM["claim_rate_by_rain_pct"] = {str(k): round(float(v) * 100, 2) for k, v in g["mean"].items()}
    NUM["claim_rate_by_policies_pct"] = {str(k): round(float(v) * 100, 2) for k, v in g2["mean"].items()}

    import glob
    cl = pd.concat([pd.read_parquet(f, columns=["interiorWaterDepth", "buildingDamageAmount", "totalBuildingInsuranceCoverage", "occupancyType"])
                    for f in sorted(glob.glob(str(fp.PILOT.parent / "nfip_national" / "nfip_*.parquet")))])
    cl = cl[cl.occupancyType.isin([1, 11]) & (cl.totalBuildingInsuranceCoverage > 0) & cl.interiorWaterDepth.between(0.01, 998)]
    cl["ratio"] = (cl.buildingDamageAmount / cl.totalBuildingInsuranceCoverage).clip(upper=1.5)
    cl["db"] = pd.cut(cl.interiorWaterDepth, [0, 6, 12, 24, 48, 1000])
    g3 = cl.groupby("db", observed=True).ratio.agg(["median", "size"])
    ax[2].bar(range(len(g3)), g3["median"], color="#357")
    ax[2].set_xticks(range(len(g3)), [str(i) for i in g3.index], rotation=30, fontsize=7)
    ax[2].set_xlabel("interior water depth (in)", fontsize=8)
    ax[2].set_ylabel("median damage / building coverage", fontsize=8)
    ax[2].set_title("(c) loss ratio rises with depth (single-family)", fontsize=9)
    NUM["damage_ratio_by_interior_depth"] = {str(k): [round(float(r["median"]), 3), int(r["size"])] for k, r in g3.iterrows()}
    from scipy.stats import spearmanr
    NUM["spearman_depth_vs_ratio"] = float(spearmanr(cl.interiorWaterDepth, cl.ratio)[0])
    for a in ax:
        a.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(OUT / "fig_dose_response.png", dpi=170)
    plt.close()

    # ---------------------------------------------------------------- maps
    polys = polygons()
    tot = t.groupby("bg10").n_claims.sum()
    fig, ax = plt.subplots(figsize=(7, 6.2))
    codes = list(polys)
    vals = np.log10(tot.reindex(codes).fillna(0).to_numpy() + 1)
    draw(ax, polys, codes, vals, "YlOrRd", 0, np.percentile(vals[vals > 0], 99), "Texas block groups: NFIP claims inside rain-episode windows, 2010-2023",
         "log10(claims + 1)")
    plt.tight_layout()
    plt.savefig(OUT / "fig_texas_claims.png", dpi=170)
    plt.close()
    NUM["bg_with_any_claim"] = int((tot > 0).sum())
    NUM["bg_in_table"] = int(t.bg10.nunique())

    # county episode in the test period: many block groups, many positives, rain that varies across the county
    c = t[te].groupby("ep").agg(n=("y", "size"), pos=("y", "sum"), sd=("bg_f1", "std"), cty=("cty", "first"), start=("start", "first"))
    cand = c[(c.n >= 150) & (c.pos >= 25)].sort_values("sd", ascending=False)
    pick = cand.index[0]
    NUM["map_episode"] = {"ep": int(pick), "county": cand.loc[pick, "cty"], "start": str(cand.loc[pick, "start"].date()),
                          "block_groups": int(cand.loc[pick, "n"]), "positives": int(cand.loc[pick, "pos"]), "f1_sd_mm": float(cand.loc[pick, "sd"])}
    sel = t[ep == pick].reset_index()
    sub = np.flatnonzero(ep[te] == pick)  # positions in the test predictions
    sc = p_base[np.flatnonzero(te)[sub] - 0] if False else p_base[np.searchsorted(np.flatnonzero(te), sel["index"].to_numpy())]
    pr = episode_parts(ep[ep == pick], y[ep == pick], sc)
    NUM["map_episode"]["within_auc"] = float(pr[1].sum() / pr[0].sum()) if len(pr[0]) else None
    pol2 = {k: polys[k] for k in sel.bg10 if k in polys}
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2))
    cs = sel.bg10.tolist()
    draw(axs[0], pol2, cs, sel.bg_f1, "Blues", None, None, "(a) cell rain F1 (mm)", "mm")
    draw(axs[1], pol2, cs, np.log10(sel.n_pol), "Greens", None, None, "(b) policies in force", "log10")
    draw(axs[2], pol2, cs, sel.y, "Reds", 0, 1, f"(c) outcome: claim in window ({int(sel.y.sum())} of {len(sel)})", "0 / 1")
    draw(axs[3], pol2, cs, sc, "viridis", None, None, "(d) simple-model probability (logistic)", "p")
    fig.suptitle(f"County {NUM['map_episode']['county']}, episode starting {NUM['map_episode']['start']} (test period)", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT / "fig_county_episode.png", dpi=170)
    plt.close()

    (OUT / "report_numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
    print(json.dumps(NUM, indent=1, default=float))


if __name__ == "__main__":
    main()
