"""Covariate statistics for every relationship the project regresses, and the renderer of RAW_STATICS_DA_REPORT.md.

Imported by scripts/44_raw_report.py (`regress`, `render`). Verdict rules (written once, applied to every feature):

  established            the 95% cluster-bootstrap interval of the rank correlation excludes 0 in the full sample AND in both the fit and the
                         test period, with the same sign in both
  present, not stable    excludes 0 in the full sample but not in both periods with one sign
  not established        the interval contains 0
  size                   |rho| < 0.05 negligible, 0.05-0.10 small, 0.10-0.30 moderate, >= 0.30 large   (Spearman, within the stated grouping)

"Within" means the rank is taken inside each group (episode, county, state-month) so only co-variation inside a group counts; "between" means
group-level (episode-level) values over all groups. Bootstrap clusters are the dependent units named in each table.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
base = importlib.import_module("44_raw_report")
diag, fp, wb2 = base.diag, base.fp, base.wb2

COUNTY = ["cty_f1_max", "cty_f1_mean", "cty_s_mean", "cty_a14_mean", "dur", "month", "cty_nbg"]
EXPO = ["log_pol", "sfha_share", "single_share", "mean_cov"]
CELL = ["bg_f1", "bg_f3", "bg_s", "f1_vs_cty"]
NEIGH = ["nb3_mean", "nb7_mean", "nb15_mean", "nb7_max", "nb15_max", "nep7_25", "nep7_50", "nep15_25", "nep15_50"]
STATIC = ["lat", "lon", "log_area", "bg_a14"]
TERRAIN = ["elev", "rel1", "rel3", "pos3", "rel10", "slope"]
FEATURES = COUNTY + EXPO + CELL + NEIGH + STATIC + TERRAIN
ROLE = {**dict.fromkeys(COUNTY, "county rain / calendar"), **dict.fromkeys(EXPO, "exposure"), **dict.fromkeys(CELL, "cell rain"),
        **dict.fromkeys(NEIGH, "neighbourhood rain"), **dict.fromkeys(STATIC, "location / static"), **dict.fromkeys(TERRAIN, "terrain")}
NBOOT = 200


def size(r: float) -> str:
    a = abs(r)
    return "negligible" if a < 0.05 else "small" if a < 0.10 else "moderate" if a < 0.30 else "large"


def verdict(full: dict | None, fit: dict | None, test: dict | None, stat: str = "rho") -> str:
    """stat 'rho' (null value 0) or 'auc' (null value 0.5); the flag name differs: excludes_zero / excludes_half."""
    if full is None:
        return "n/a"
    null = 0.0 if stat == "rho" else 0.5
    flag = "excludes_zero" if stat == "rho" else "excludes_half"
    if not full[flag]:
        return "not established"
    if fit and test and fit[flag] and test[flag] and np.sign(fit[stat] - null) == np.sign(test[stat] - null):
        return "established"
    return "present, not stable"


def slim(d: dict | None) -> dict | None:
    return None if d is None else {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def collinearity(Z: pd.DataFrame, n: int = 300000) -> dict:
    """Correlation matrix, VIF and condition number of a (transformed, standardised) feature set; rows with a NaN are dropped."""
    Z = Z.dropna()
    Z = Z.sample(min(len(Z), n), random_state=base.SEED)
    keep = Z.columns[Z.std() > 0]
    Z = Z[keep]
    C = np.corrcoef(Z.to_numpy(float), rowvar=False)
    v = diag.vif(Z.to_numpy(float))
    iu = np.triu_indices(len(keep), 1)
    pairs = sorted(((abs(C[i, j]), keep[i], keep[j], C[i, j]) for i, j in zip(*iu, strict=True) if abs(C[i, j]) >= 0.9), reverse=True)
    w, vec = np.linalg.eigh(C)
    dep = []  # linear dependencies: eigenvectors of the correlation matrix whose eigenvalue is ~0 (exact) or < 0.05 (near)
    for i in range(min(6, len(w))):
        if w[i] < 0.05:
            load = {keep[j]: round(float(vec[j, i]), 3) for j in np.argsort(-np.abs(vec[:, i])) if abs(vec[j, i]) > 0.15}
            dep.append({"eigenvalue": float(w[i]), "kind": "exact" if w[i] < 1e-6 else "near", "columns": load})
    return {"columns": list(keep), "vif": {c: round(float(x), 2) for c, x in zip(keep, v, strict=True)}, "condition_number": round(diag.condition_number(Z.to_numpy(float)), 1),
            "dependencies": dep,
            "pairs_abs_ge_0.9": [(a, b, round(float(r), 3)) for _, a, b, r in pairs], "corr": np.round(C, 3).tolist(), "n_rows": int(len(Z))}


def heatmap(C: np.ndarray, names: list[str], fname: str, title: str) -> str:
    fig, ax = base.new_fig(8.2, 7.4)
    ax.grid(False)
    im = ax.imshow(C, cmap=base.DIV, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6.5, color=base.INK2)
    ax.set_yticklabels(names, fontsize=6.5, color=base.INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("correlation", fontsize=8, color=base.INK2)
    cb.ax.tick_params(colors=base.MUTED, labelsize=7)
    ax.set_title(title, fontsize=9, loc="left")
    fig.tight_layout()
    return base.save(fig, fname)


def forest(rows: list[dict], fname: str, title: str, series: list[tuple[str, str, str]], xlabel: str, stat: str = "rho", centre: float = 0.0) -> str:
    """rows: {label, <key>: {rho, lo, hi}}; series: (key, legend text, colour). Two series are offset vertically; the legend names them."""
    n = len(rows)
    fig, ax = base.new_fig(8.6, max(3.2, 0.26 * n + 1.2))
    off = np.linspace(-0.17, 0.17, len(series)) if len(series) > 1 else [0.0]
    for (key, text, col), o in zip(series, off, strict=True):
        ys, xs, lo, hi = [], [], [], []
        for i, r in enumerate(rows):
            d = r.get(key)
            if d:
                ys.append(i + o)
                xs.append(d[stat])
                lo.append(d[stat] - d["lo"])
                hi.append(d["hi"] - d[stat])
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=5, color=col, mec=base.SURF, mew=1.2, elinewidth=1.4, capsize=0, label=text)
    ax.axvline(centre, color=base.AXIS, lw=1)
    lo_x, hi_x = ax.get_xlim()
    ax.set_xlim(min(lo_x, centre - 0.01), max(hi_x, centre + 0.01))
    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=7.5, color=base.INK2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_title(title, fontsize=9, loc="left")
    if len(series) > 1:
        ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=base.INK2)
    fig.tight_layout()
    return base.save(fig, fname)


# ===================================================================================================================== flood pilot


def pilot_cov() -> dict:
    from wxrisk.prep.scaling_plan import PILOT_PLAN

    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet")
    t["log_pol"] = np.log1p(t.n_pol)
    t["log_area"] = np.log(t.area_km2.clip(lower=0.01))
    t = t.merge(pd.read_parquet(fp.PILOT / "bg_terrain.parquet"), on="bg10", how="left")
    gs = t.groupby("ep").y.agg(["size", "sum"])
    mixed = gs[(gs["size"] >= 10) & (gs["sum"] > 0) & (gs["sum"] < gs["size"])].index
    tm = t[t.ep.isin(mixed)]
    ep = t.groupby("ep").agg(rate=("y", "mean"), cty=("cty", "first"), year=("year", "first"), **{f: (f, "mean") for f in FEATURES})
    res, rows = {}, []
    for f in FEATURES:
        row = {"feature": f, "role": ROLE[f]}
        if f in COUNTY:
            row["within"] = None  # constant inside an episode
        else:
            wsets = {"all": tm, "fit": tm[tm.year <= 2016], "test": tm[tm.year >= 2017]}
            row["within"] = {k: slim(diag.cluster_auc(d[f].to_numpy(float), d.y.to_numpy(float), d.ep.to_numpy(), n_boot=NBOOT, seed=base.SEED)) for k, d in wsets.items()}
            row["within_rho"] = {k: slim(diag.cluster_corr(d[f].to_numpy(float), d.y.to_numpy(float), d.ep.to_numpy(), within=True, n_boot=NBOOT, seed=base.SEED)) for k, d in wsets.items()}
        bsets = {"all": ep, "fit": ep[ep.year <= 2016], "test": ep[ep.year >= 2017]}
        row["between"] = {k: slim(diag.cluster_corr(d[f].to_numpy(float), d.rate.to_numpy(float), d.cty.to_numpy(), within=False, n_boot=NBOOT, seed=base.SEED)) for k, d in bsets.items()}
        w, b = row["within"], row["between"]
        row["verdict_within"] = verdict(w and w["all"], w and w["fit"], w and w["test"], "auc")
        row["verdict_between"] = verdict(b["all"], b["fit"], b["test"])
        rows.append(row)
        print("pilot", f, row["verdict_within"], row["verdict_between"], flush=True)
    tr = (t.year <= 2016).to_numpy()
    params = PILOT_PLAN.fit(t, tr, "year <= 2016")
    Z = PILOT_PLAN.apply(t[tr], params)
    col = collinearity(Z)
    res.update(features=rows, collinearity=col, sample={"rows": len(t), "episodes": int(t.ep.nunique()), "mixed_episodes": int(len(mixed)), "rows_in_mixed_episodes": int(len(tm)),
                                                          "positives_fit": int(t.y[tr].sum()), "positives_test": int(t.y[~tr].sum()), "n_features": len(FEATURES),
                                                          "events_per_variable_fit": round(diag.events_per_variable(t.y[tr].to_numpy(), len(FEATURES)), 1)})
    res["figure_corr"] = heatmap(np.array(col["corr"]), col["columns"], "pilot_corr.png", "Texas flood pilot: correlation of the plan-transformed features (fit years)")
    w = [r for r in rows if r["within"]]
    w.sort(key=lambda r: -abs(r["within"]["all"]["auc"] - 0.5))
    res["figure_forest"] = forest([{"label": r["feature"], "fit": r["within"]["fit"], "test": r["within"]["test"]} for r in w], "pilot_within_forest.png",
                                  "Within-episode AUC of each feature for flood occurrence (95% interval over episodes)",
                                  [("fit", "fit years 2010-2016", base.S1), ("test", "test years 2017-2023", base.S2)], "AUC inside the episode (0.5 = no information)", stat="auc", centre=0.5)
    b = sorted(rows, key=lambda r: -abs(r["between"]["all"]["rho"]))
    res["figure_between"] = forest([{"label": r["feature"], "fit": r["between"]["fit"], "test": r["between"]["test"]} for r in b], "pilot_between_forest.png",
                                   "Between-episode Spearman correlation of the episode mean with the episode claim rate (95% interval over counties)",
                                   [("fit", "fit years 2010-2016", base.S1), ("test", "test years 2017-2023", base.S2)], "Spearman rho across episodes")
    return res


# ===================================================================================================================== SCS


def scs_cov() -> dict:
    from wxrisk.scs_eval import anomalise

    km10 = base.load_script("10_kmeans_baseline")
    df = km10.build_table(list(range(2000, 2023)))
    fit = (df.year <= 2016).to_numpy()
    test = (df.year >= 2017).to_numpy()
    cols = km10.tier_columns(df)["C (<T-3h)"]
    X = pd.DataFrame(anomalise(df, cols, pd.Series(fit)), columns=cols)
    yrs = df.year.to_numpy()
    out = {"rows_fit": int(fit.sum()), "rows_test": int(test.sum()), "positives": {}, "features": []}
    for h in ("y0", "y10k"):
        out["positives"][h] = {"fit": int(df[h][fit].sum()), "test": int(df[h][test].sum())}
    for c in cols:
        row = {"feature": c, "static": c in ("doy_s", "doy_c", "lat", "lon")}
        for h in ("y0", "y10k"):
            y = df[h].to_numpy(float)
            sets = (("all", np.ones(len(df), bool)), ("fit", fit), ("test", test))
            res = {k: slim(diag.cluster_auc(X[c].to_numpy(float)[m], y[m], yrs[m], n_boot=NBOOT, seed=base.SEED)) for k, m in sets}
            row[h] = res
            row[f"rho_{h}"] = {k: slim(diag.cluster_corr(X[c].to_numpy(float)[m], y[m], yrs[m], within=False, n_boot=NBOOT, seed=base.SEED)) for k, m in sets}
            row[f"verdict_{h}"] = verdict(res["all"], res["fit"], res["test"], "auc")
        out["features"].append(row)
        print("scs", c, row["verdict_y0"], row["verdict_y10k"], flush=True)
    col = collinearity(X[fit].reset_index(drop=True))
    out["collinearity"] = col
    rows = sorted(out["features"], key=lambda r: -abs(r["y10k"]["all"]["auc"] - 0.5))
    out["figure_corr"] = heatmap(np.array(col["corr"]), col["columns"], "scs_corr.png", "SCS tier C inputs (anomalies): correlation matrix, fit years")
    out["figure_forest"] = forest([{"label": r["feature"], "a": r["y0"]["all"], "b": r["y10k"]["all"]} for r in rows], "scs_forest.png",
                                  "Within-year AUC of each input (anomaly) for the event label, 2000-2022 (95% interval over years)",
                                  [("a", "any severe report (y0)", base.S1), ("b", ">= $10k damage (y10k)", base.S2)], "AUC among county-days of the same year (0.5 = no information)", stat="auc", centre=0.5)
    return out


# ===================================================================================================================== stock-flow


def flow_cov() -> dict:
    m37 = base.load_script("37_stock_flow_inertia")
    cts, N, stock, s_all, claims, avg_prem = m37.load_panel()
    Z = m37.load_econ(cts, avg_prem)
    S = stock.sum(axis=1)
    I = m37.intensity(claims, s_all)
    keep = S[:, 1:].mean(axis=1) >= 300
    nq = m37.NQ
    q0 = 8  # first quarter with every covariate (Redfin sales start in 2012)
    qs = np.arange(q0, nq)
    ref = S[keep][:, 1:].mean(axis=1)
    y = np.log((N[keep][:, qs] + 0.5) / ref[:, None])
    feats = {f"claim intensity, lag {k}": I[keep][:, qs - k] for k in range(m37.LAGS + 1)} if hasattr(m37, "LAGS") else {}
    for j, name in enumerate(m37.ECON_NAMES):
        feats[f"{name} (lag {m37.ECON_LAG[j]})"] = Z[keep][:, qs - m37.ECON_LAG[j], j]
    C = keep.sum()
    county = np.repeat(np.arange(C), len(qs))
    season = np.tile(qs % 4, C)

    def deseason(a: np.ndarray) -> np.ndarray:
        v = a.reshape(-1).astype(float)
        v = v - pd.Series(v).groupby(county).transform("mean").to_numpy()  # county level
        return v - pd.Series(v).groupby(season).transform("mean").to_numpy()  # common seasonal pattern

    yy = deseason(y)
    rows, mat = [], {}
    for name, a in feats.items():
        x = deseason(np.where(np.isfinite(a), a, np.nan))
        res = diag.cluster_corr(x, yy, county, within=False, n_boot=NBOOT, seed=base.SEED)
        half = qs.size // 2
        first = np.tile(np.arange(len(qs)) < half, C)
        r1 = diag.cluster_corr(x[first], yy[first], county[first], within=False, n_boot=NBOOT, seed=base.SEED)
        r2 = diag.cluster_corr(x[~first], yy[~first], county[~first], within=False, n_boot=NBOOT, seed=base.SEED)
        rows.append({"feature": name, "all": slim(res), "first_half": slim(r1), "second_half": slim(r2), "verdict": verdict(res, r1, r2)})
        mat[name] = x
        print("flow", name, rows[-1]["verdict"], flush=True)
    D = pd.DataFrame(mat).dropna()
    # season dummies belong to the design of the model; collinearity of the covariates themselves
    out = {"counties": int(C), "quarters": int(len(qs)), "rows": int(C * len(qs)), "features": rows, "collinearity": collinearity(D, 100000),
           "target": "log((N + 0.5) / mean stock): new purchases per insured policy, county-quarter, two-way demeaned (county, season)"}
    lag_rows = [r for r in rows if r["feature"].startswith("claim")]
    out["figure_ccf"] = forest([{"label": r["feature"], "a": r["all"], "b": r["first_half"], "c": r["second_half"]} for r in lag_rows], "flow_lags.png",
                               "New purchases vs claim intensity by lag (county and season removed; 95% interval over counties)",
                               [("a", "all quarters", base.S1), ("b", "2012-2017", base.S2), ("c", "2018-2023", base.S3)], "Spearman rho")
    return out


# ===================================================================================================================== damage ratio


def damage_cov() -> dict:
    import duckdb

    C = f"read_parquet('{wb2.DATA_DIR / 'clean' / 'claims_nfip.parquet'}')"
    d = duckdb.sql(f"""select state || '_' || strftime(loss_date, '%Y-%m') as ev, loss_year, depth_ext, damage_building / cov_building as dr, foundation, zone, post_firm, elevated, occupancy
        from {C} where occupancy in (1, 11) and cov_building > 0 and damage_building is not null and depth_ext > 0 and loss_year >= 2016 and foundation is not null""").df()
    d["y"] = np.log(d.dr.clip(1e-4, 1.5))
    d["ld"] = np.log(d.depth_ext)
    d["found"] = d.foundation.map({"1": "slab", "2": "basement", "3": "crawl"}).fillna("elevated")
    d["sfha"] = d.zone.fillna("").str[:1].isin(["A", "V"]).astype(int)
    g = d.ev.to_numpy()
    kn = duckdb.sql(f"""select loss_year, occupancy, count(*) as n, avg((foundation is not null)::int) as known from {C}
        where occupancy in (1, 11) and loss_year >= 2010 group by 1, 2 order by 1, 2""").df()
    out = {"foundation_known_by_occupancy": {str(y): {int(o): [int(n), round(float(k), 3)] for o, n, k in zip(g.occupancy, g.n, g.known, strict=True)} for y, g in kn.groupby("loss_year")},
           "rows": len(d), "clusters_state_month": int(d.ev.nunique()), "target": "log(building damage / building coverage), clipped to [1e-4, 1.5]; claims 2016+ with exterior depth > 0"}
    out["depth"] = {"between_and_within": slim(diag.cluster_corr(d.ld.to_numpy(), d.y.to_numpy(), g, within=False, n_boot=NBOOT, seed=base.SEED)),
                    "within_state_month": slim(diag.cluster_corr(d.ld.to_numpy(), d.y.to_numpy(), g, within=True, n_boot=NBOOT, seed=base.SEED))}
    cats = {"foundation group": d.found, "special flood hazard area": d.sfha, "post-FIRM": d.post_firm.astype(int), "elevated indicator": d.elevated.astype(int),
            "occupancy (1 vs 11)": d.occupancy, "loss year": d.loss_year}
    out["eta2"] = {k: slim(diag.eta2_cluster(d.y.to_numpy(), v.to_numpy(), g, n_boot=NBOOT, seed=base.SEED)) for k, v in cats.items()}
    X = pd.get_dummies(d.found, drop_first=True).astype(float)
    X["log_depth"], X["sfha"], X["post_firm"], X["elevated"] = d.ld, d.sfha, d.post_firm.astype(float), d.elevated.astype(float)
    out["collinearity"] = collinearity(X, 100000)
    fig, ax = base.new_fig(7.4, 3.0)
    ks = sorted((k for k in out["eta2"] if out["eta2"][k]["categories"] > 1), key=lambda k: out["eta2"][k]["eta2"])
    ax.barh(range(len(ks)), [out["eta2"][k]["eta2"] for k in ks], height=0.5, color=base.S1)
    ax.errorbar([out["eta2"][k]["eta2"] for k in ks], range(len(ks)), xerr=[[out["eta2"][k]["eta2"] - out["eta2"][k]["lo"] for k in ks], [out["eta2"][k]["hi"] - out["eta2"][k]["eta2"] for k in ks]],
                fmt="none", ecolor=base.INK2, elinewidth=1.2, capsize=0)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=8, color=base.INK2)
    ax.set_xlabel("eta squared: share of the variance of log damage ratio (95% interval over state-months)", fontsize=8)
    ax.set_title("Damage ratio: variance explained by each categorical covariate", fontsize=9, loc="left")
    fig.tight_layout()
    out["figure_eta2"] = base.save(fig, "damage_eta2.png")
    return out


def regress() -> None:
    R = {"flood_pilot": pilot_cov(), "scs": scs_cov(), "stock_flow": flow_cov(), "damage_ratio": damage_cov()}
    base.dump("regress", R)


# ===================================================================================================================== render


def render() -> None:
    importlib.import_module("44c_raw_report_render").build()  # the Markdown writer lives in its own file


if __name__ == "__main__":
    regress()
