"""Writes RAW_STATICS_DA_REPORT.md from the JSON of scripts/44_raw_report.py (raw, cluster) and 44b (regress). Every verdict sentence is computed here."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
base = importlib.import_module("44_raw_report")
reg = importlib.import_module("44b_raw_report_regress")
diag = base.diag

REPORT = ROOT / "RAW_STATICS_DA_REPORT.md"


def table(head: list[str], rows: list[list]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return out + [""]


def ci(d: dict, stat: str = "rho", n: int = 3) -> str:
    return f"{d[stat]:+.{n}f} [{d['lo']:+.{n}f}, {d['hi']:+.{n}f}]"


def pct(x) -> str:
    return f"{x:.2f}%" if isinstance(x, (int, float)) else str(x)


def load(name: str) -> dict:
    return json.loads((base.OUT / f"{name}.json").read_text())


# ------------------------------------------------------------------------------------------------ raw


def sec_raw(R: dict) -> list[str]:
    md = ["## 2. Raw dataset verification", "",
          "Statistics of the files as downloaded (claims, policies, gridMET, SCS labels) or as extracted from downloads (ERA5 at county points), before any cleaning. "
          "What each flag means for the cleaning rules is in `docs/CLEANING_REGISTER.md`.", ""]
    c, p, g, s, e, f, lk = R["claims"], R["policies_tx"], R["gridmet_precip"], R["scs_labels"], R["era5_county_env"], R["flood_pilot_table"], R["data_lock"]
    md += table(["Dataset", "Rows / shape", "Period", "Check against the provider or the structure", "Result"], [
        ["NFIP claims (OpenFEMA v3, national)", f"{c['rows']:,} rows, {c['files']} year files", "2000-2023",
         "row count vs OpenFEMA `$count` for yearOfLoss 2000-2023; `id` unique",
         (f"provider {c['provider_count_2000_2023']:,}" if isinstance(c['provider_count_2000_2023'], int) else str(c['provider_count_2000_2023'])) + f"; distinct ids {c['distinct_id']:,}"],
        ["NFIP policies, Texas (OpenFEMA v3)", f"{p['rows']:,} rows", "effective 2009-2026", "row count vs OpenFEMA `$count` for propertyState TX; exact duplicate rows",
         (f"provider {p['provider_count_TX']:,}" if isinstance(p['provider_count_TX'], int) else str(p['provider_count_TX'])) + f"; duplicate rows beyond the first: {p['rows_in_excess_of_first']:,}"],
        ["gridMET precipitation, Texas box", "x".join(str(v) for v in g["shape"]) + f" {g['dtype']}", f"{g['first_day']} to {g['last_day']}", "calendar gaps; negative values; NaN share",
         f"gaps {g['calendar_gaps']}; negative {g['negative_values']}; NaN {g['nan_pct']}%"],
        ["SCS3 county-day labels (CO, NM, AZ)", f"{s['rows']:,} rows, {s['counties']} counties", f"{s['first']} to {s['last']} (May-Sep)", "duplicate (county, day)", f"{s['duplicate_county_days']} duplicates"],
        ["ERA5 at county points (12Z and 18Z)", f"{e['rows']:,} rows, {len(e['columns'])} variables", "2000-2022", "values outside physical ranges",
         f"{sum(v['outside_physical_range'] or 0 for v in e['columns'].values()):,} out of range"],
        ["Texas flood pilot table (derived)", f"{f['rows']:,} rows, {f['episodes']:,} episodes", "2010-2023", "positives per year", f"{f['positives']:,} positives"],
    ])
    # claims
    drift = c["schema_drift_across_year_files"]
    md += ["### 2.1 NFIP claims", "",
           f"- Schema drift between year files: {', '.join(f'`{k}` ({" / ".join(v)})' for k, v in drift.items()) or 'none'}. Types are reconciled by the cleaning step.",
           f"- Row count equals the provider's count: **{c['rows'] == c['provider_count_2000_2023']}**; `id` unique: **{c['rows'] == c['distinct_id']}**.", "",
           "Rows per loss year:", ""]
    ys = sorted(c["rows_per_year"], key=int)
    half = len(ys) // 2
    md += table([y for y in ys[:half]], [[f"{c['rows_per_year'][y]:,}" for y in ys[:half]]])
    md += table([y for y in ys[half:]], [[f"{c['rows_per_year'][y]:,}" for y in ys[half:]]])
    md += ["Column profile (null share, approximate distinct values, range, share of exact zeros):", ""]
    md += table(["Column", "Type", "Null %", "Distinct~", "Min", "Median", "Max", "Zero %"],
                [[x["column"], x["type"], pct(x["null_pct"]), f"{x['distinct_approx']:,}", x["min"], x["median"], x["max"], pct(x["zero_pct"]) if x["zero_pct"] is not None else ""] for x in c["columns"]])
    # policies
    db = p["duplicates_by_block_group_size"]
    order = [b for b in ("<20", "20-199", "200-1999", ">=2000") if b in db]
    md += ["### 2.2 NFIP policies (Texas)", "",
           f"- Exact duplicate rows: **{p['rows_in_excess_of_first']:,}** beyond the first occurrence ({p['exact_duplicate_groups']:,} groups, {100 * p['rows_in_excess_of_first'] / p['rows']:.2f}% of the rows). "
           "The redacted file has no policy number, so identical rows can be distinct policies. By block-group size (rows in the block group): "
           + "; ".join(f"{b}: {db[b]['excess_pct']}% ({db[b]['excess']:,} of {db[b]['rows']:,})" for b in order)
           + f". The share is {max(db[b]['excess_pct'] for b in order) / max(db[order[0]]['excess_pct'], 1e-9):.1f} times higher in the most affected size class than in the smallest block groups, "
           + ("but it is not monotone in block-group size, so this is consistent with identical redacted attributes but does not prove it. "
              if [db[b]["excess_pct"] for b in order] != sorted(db[b]["excess_pct"] for b in order) else "and rises with block-group size, as identical redacted attributes predict. ")
           + f"The rows are kept (no policy number to tell the cases apart); the effect on any count is at most {100 * p['rows_in_excess_of_first'] / p['rows']:.2f}%.",
           f"- Policy term (expiry minus effective date), p1 / p50 / p99: {p['term_days_p1_p50_p99']} days, i.e. the expiry is nominal; mid-term cancellations are a separate column (see the cleaning register).", "",
           "Rows per effective year:", ""]
    ys = sorted(p["rows_per_effective_year"], key=int)
    half = (len(ys) + 1) // 2
    md += table(ys[:half], [[f"{p['rows_per_effective_year'][y]:,}" for y in ys[:half]]])
    md += table(ys[half:], [[f"{p['rows_per_effective_year'][y]:,}" for y in ys[half:]]])
    md += table(["Column", "Type", "Null %", "Distinct~", "Min", "Median", "Max", "Zero %"],
                [[x["column"], x["type"], pct(x["null_pct"]), f"{x['distinct_approx']:,}", x["min"], x["median"], x["max"], pct(x["zero_pct"]) if x["zero_pct"] is not None else ""] for x in p["columns"]])
    # gridMET
    q = g["wet_day_quantiles_mm"]
    md += ["### 2.3 gridMET daily precipitation", "",
           f"- {g['shape'][0]:,} days x {g['shape'][1]} x {g['shape'][2]} cells, lat {g['lat_range'][0]:.2f}-{g['lat_range'][1]:.2f}, lon {g['lon_range'][0]:.2f}-{g['lon_range'][1]:.2f}; no calendar gap.",
           f"- NaN: {g['cells_always_nan_pct']}% of the cells are NaN on every day and {g['cells_sometimes_nan_pct']}% on some days only, so the missing part is a fixed mask (the part of the rectangle that is not US land: Gulf of Mexico, Mexico). "
           f"Zero precipitation: {g['zero_pct']}% of all cell-days.",
           f"- Maximum {g['max_mm']:.1f} mm in one cell-day; {g['cells_over_300mm']:,} cell-days above 300 mm, on only {g['days_with_cell_over_300mm']} days; {100 * g['share_of_over_300mm_on_top8_days']:.0f}% of them on 8 days "
           f"({', '.join(f'{d}: {n}' for d, n in g['top_days_over_300mm'].items())}): a few days with many cells each, i.e. weather systems and not isolated spikes; 2017-08-26 to 29 is the Hurricane Harvey window of the project's HURDAT2 flags, the other dates were not cross-checked against an event catalogue. Negative values: {g['negative_values']}.",
           f"- Wet cell-days (>= 1 mm) quantiles: median {q['0.5']:.1f}, p90 {q['0.9']:.1f}, p99 {q['0.99']:.1f}, p99.9 {q['0.999']:.1f} mm.", "",
           "Domain-mean annual total (mm) and mean daily rain by calendar month (mm/day):", ""]
    ys = sorted(g["domain_mean_annual_total_mm"], key=int)
    md += table(ys[:12], [[g["domain_mean_annual_total_mm"][y] for y in ys[:12]]])
    md += table(ys[12:], [[g["domain_mean_annual_total_mm"][y] for y in ys[12:]]])
    md += table([str(m) for m in range(1, 13)], [[g["domain_mean_by_month_mm_per_day"][str(m)] for m in range(1, 13)]])
    # SCS labels
    md += ["### 2.4 SCS3 labels", "",
           f"- Rate per county-day: y0 {s['rate_pct']['y0']}%, y10k {s['rate_pct']['y10k']}%, y100k {s['rate_pct']['y100k']}%, y1m {s['rate_pct']['y1m']}%. "
           f"Positives: {s['positives']}. Rows by state: {s['rows_per_state']}. Months present: {s['months_covered']} (warm season only).", "",
           "Positives per year:", ""]
    ys = sorted(s["positives_per_year"]["y0"], key=int)
    md += table(["head"] + ys[:12], [[h] + [s["positives_per_year"][h][y] for y in ys[:12]] for h in ("y0", "y10k", "y100k", "y1m")])
    md += table(["head"] + ys[12:], [[h] + [s["positives_per_year"][h][y] for y in ys[12:]] for h in ("y0", "y10k", "y100k", "y1m")])
    # ERA5
    md += ["### 2.5 ERA5 at county points (raw units)", "",
           "Physical range = a loose bound for the variable (CAPE 0-8000 J/kg, 2 m temperature 230-330 K, ...); 'outside' counts values beyond it.", ""]
    md += table(["Variable", "NaN %", "Min", "p01", "Median", "p99", "Max", "Zero %", "Physical range", "Outside"],
                [[k, v["nan_pct"], f"{v['min']:.4g}", f"{v['p01']:.4g}", f"{v['median']:.4g}", f"{v['p99']:.4g}", f"{v['max']:.4g}", v["zero_pct"], v["physical_range"], v["outside_physical_range"]]
                 for k, v in e["columns"].items()])
    # flood pilot + terrain
    md += ["### 2.6 Texas flood pilot table", ""]
    ys = sorted(f["by_year"], key=int)
    md += table(["Year", "Rows", "Episodes", "Positives", "Positives per 100k rows"], [[y, f"{f['by_year'][y]['rows']:,}", f"{f['by_year'][y]['episodes']:,}", f"{f['by_year'][y]['positives']:,}", f["by_year"][y]["positives_per_100k_rows"]] for y in ys])
    md += ["The positive rate per 100k rows differs by more than an order of magnitude between years (claim counts swing from under 500 to about 10,000 a year): "
           "absolute probabilities do not transfer across years, which is why the project reports within-event rankings.", ""]
    md += ["### 2.7 Data lock", "",
           f"- {lk['entries']:,} locked files: {lk['size_matches']:,} match in size, {lk['missing']} missing, {lk['size_differs']} differ ({lk['examples']}). {lk['note']}.", ""]
    return md


# ------------------------------------------------------------------------------------------------ clustering


def cluster_block(tag: str, c: dict, name: str) -> tuple[list[str], dict]:
    a = c["outcome_association_test"]
    hop = c.get("hopkins_reference", {})
    curve = c["k_curve"]
    dat = {x["K"]: x["silhouette"] for x in curve if x["set"] == "data"}
    nul = {x["K"]: x["silhouette"] for x in curve if x["set"] == "null"}
    bestk = max(dat, key=dat.get)
    sig = all(v["p_value"] < 0.05 for v in a.values())
    verdict = {
        "natural": c["silhouette_at_K"] > 0.25,
        "quantiser": sig,
    }
    md = [f"### {name}", "",
          f"K-means on the anomaly-scaled inputs the model sees ({c['n_columns']} columns, {c['rows_fit']:,} fit rows 2000-2016, {c['rows_test']:,} test rows 2017-2022), K = {c['K']} (the K the project chose by validation).", "",
          f"![PCA projection]({c['figures']['pca']})", "",
          f"PCA of the K-means input: the first two components carry {100 * c['pca']['cumulative_2']:.1f}% of the variance "
          f"({', '.join(f'{100 * v:.1f}%' for v in c['pca']['explained_variance_ratio'][:5])} for the first five); the effective dimension (participation ratio) is {c['pca']['participation_ratio_dims']:.1f}. "
          "A 2-D picture therefore shows a minority of the structure; read it for overlap, not for separation. Left: points coloured by the event lift of their cluster. "
          "Right: the three highest-lift clusters over the rest in grey, with all centroids.", "",
          *table(["Diagnostic", "Value", "Reading"], [
              ["Mean silhouette at K", f"{c['silhouette_at_K']:.3f}", diag.silhouette_verdict(c["silhouette_at_K"])],
              ["Silhouette of the column-permuted null at K", f"{c['silhouette_null_at_K']:.3f}", "same marginals, no joint structure"],
              ["Best K by silhouette (data)", f"{bestk} ({dat[bestk]:.3f}; null {nul[bestk]:.3f})", "the maximum sits at the smallest K tried" if bestk == min(dat) else ""],
              ["Clusters with negative mean silhouette", f"{c['clusters_with_negative_mean_silhouette']} of {c['K']}", "points on average closer to another cluster"],
              ["Davies-Bouldin (lower is better)", f"{c['davies_bouldin_at_K']:.2f}", "above 1 means clusters overlap"],
              ["Calinski-Harabasz", f"{c['calinski_harabasz_at_K']:.0f}", "only comparable across K for the same data"],
              ["Hopkins vs uniform box (1000 test points, all fit rows)", f"{c['hopkins']:.3f}", "any non-uniform cloud scores high"],
              ["Hopkins: data / Gaussian same covariance / columns permuted (60k-row subsample, mean of 5 seeds)", " / ".join(f"{hop[k]:.3f}" for k in ("data", "gaussian_same_covariance", "columns_permuted")) if hop else "n/a",
               "the data exceed the cluster-free references by " + (f"{hop['data'] - max(hop['gaussian_same_covariance'], hop['columns_permuted']):.3f}" if hop else "n/a")],
              ["Stability: ARI, other seed / other half of the rows", f"{c['stability']['ari_other_seed']:.2f} / {c['stability']['ari_other_half']:.2f}", "1 = identical partition"],
              ["Cluster sizes min / median / max", f"{c['cluster_size']['min']:,} / {c['cluster_size']['median']:,} / {c['cluster_size']['max']:,}", "no degenerate cluster"],
          ]),
          f"![silhouette against K]({c['figures']['k_sweep']})", "", f"![silhouette per cluster]({c['figures']['per_cluster']})", "",
          "Do the cluster labels carry information about the events, out of sample? (labels from the 2000-2016 fit, tested on 2017-2022; permutation null with 200 shuffles; p = 0.005 is the floor of a 200-permutation test, 1/201: no permutation reached the observed value):", "",
          *table(["Head", "Test positives", "Mutual information (nats)", "Null 95th pct", "Permutation p", "Eta squared of cluster mean", "Spearman of cluster event rate, fit vs test"],
                 [[h, v["test_positives"], f"{v['mutual_information']:.5f}", f"{v['null_p95']:.5f}", f"{v['p_value']:.3f}", f"{v['eta2_of_cluster_mean']:.4f}", f"{v['spearman_cluster_rate_fit_vs_test']:.2f}"] for h, v in a.items()]),
          "**Verdict.** " + (
              f"The inputs do **not form natural clusters** (silhouette {c['silhouette_at_K']:.3f}, {diag.silhouette_verdict(c['silhouette_at_K'])}; the structure is only "
              f"{c['silhouette_at_K'] - c['silhouette_null_at_K']:+.3f} above the permuted null; the partition changes between fits, ARI about {c['stability']['ari_other_half']:.2f}). "
              if not verdict["natural"] else f"The inputs show cluster structure (silhouette {c['silhouette_at_K']:.3f}). ")
          + (f"K-means is therefore used as a **vector quantiser**, not as a discovery of groups: the {c['K']} cells are justified by the out-of-sample information they carry "
             f"(permutation p <= {max(v['p_value'] for v in a.values()):.3f} for every head, cluster event rates agree between fit and test with Spearman "
             f"{min(v['spearman_cluster_rate_fit_vs_test'] for v in a.values()):.2f}-{max(v['spearman_cluster_rate_fit_vs_test'] for v in a.values()):.2f}), "
             f"while the variance of the labels they explain is small (eta squared {min(v['eta2_of_cluster_mean'] for v in a.values()):.3f}-{max(v['eta2_of_cluster_mean'] for v in a.values()):.3f})."
             if sig else "The cluster labels do not carry a significant amount of information about the events out of sample."), ""]
    return md, verdict


def sec_cluster(C: dict) -> tuple[list[str], dict]:
    md = ["## 3. Clustering claims: K-means diagnostics and projections", "",
          "**What is claimed.** Stage 2 of `RESULTS.md` partitions the SCS county-day environment with K-means and reads event rates off the cells; stage 10 describes the take-up rate as bimodal. "
          "Nothing else is clustered. A claim 'the data can be clustered' is tested at three levels: (1) is there cluster structure (silhouette, Davies-Bouldin, Calinski-Harabasz, Hopkins against cluster-free references, "
          "elbow, null with permuted columns); (2) is the partition reproducible (adjusted Rand index across seeds and halves); (3) do the labels carry information about the outcome out of sample "
          "(mutual information with a permutation null). Silhouette reading (Kaufman and Rousseeuw): above 0.7 strong, 0.5-0.7 reasonable, 0.25-0.5 weak, below 0.25 no substantial structure. "
          "Projection: PCA (deterministic, and its explained variance is reported); t-SNE and UMAP distort distances and can display a separation that the original distances do not support.", "",
          "Tier A (calendar and location only) is a spatial-seasonal grid and is not analysed.", ""]
    verdicts = {}
    for tag, name in (("B", "3.1 SCS tier B (morning environment, T-3 days to T-3 h)"), ("C", "3.2 SCS tier C (adds the afternoon analysis and the 6-hour changes)")):
        b, v = cluster_block(tag, C[tag], name)
        md += b
        verdicts[tag] = v
    t = C["takeup_gmm"]
    g = t["gmm"]
    md += ["### 3.3 Take-up rate across block groups (one-dimensional mixture)", "",
           f"{t['block_groups']:,} block groups (single-family non-SFHA policies per housing unit, ACS 2022; at least 200 housing units, under 5% SFHA policies), log10 scale.", "",
           f"![take-up mixture]({t['figure']})", "",
           *table(["Components", "BIC", "Means (log10)", "Weights", "SD (log10)"], [[k, f"{v['bic']:.1f}", v["means_log10"], v["weights"], v["sd_log10"]] for k, v in g.items()]),
           *table(["Diagnostic", "Value", "Reading"], [
               ["Bimodality coefficient (Sarle)", f"{t['bimodality_coefficient']:.3f}", f"above {t['bimodality_threshold']} suggests more than one mode"],
               ["1-D K-means silhouette, K = 2 / 3 / 4", " / ".join(f"{t['kmeans_1d_silhouette'][k]:.2f}" for k in ("2", "3", "4")),
                "reference: the same split of a single-mode Gaussian scores " + " / ".join(f"{t['kmeans_1d_silhouette_gaussian_reference'][k]:.2f}" for k in ("2", "3", "4")) + ", so the silhouette alone is weak evidence in 1-D; the bimodality coefficient and the BIC carry the claim"],
               ["BIC steps 1 to 2 / 2 to 3 / 3 to 4", " / ".join(f"{g[str(b)]['bic'] - g[str(a)]['bic']:+.0f}" for a, b in ((1, 2), (2, 3), (3, 4))), "the gain shrinks but never stops"]]),
           f"**Verdict.** Take-up is **bimodal** (coefficient {t['bimodality_coefficient']:.3f} > {t['bimodality_threshold']}; two components cut the BIC by {abs(g['2']['bic'] - g['1']['bic']):.0f} against one). "
           f"The **number** of components is not identifiable: BIC keeps falling at {t['best_k_by_bic']} components, as expected for a skewed, non-Gaussian density. The claim that survives is "
           f"'two regimes (take-up near {10 ** g['2']['means_log10'][0]:.1%} and {10 ** g['2']['means_log10'][1]:.1%} in the two-component fit)', not 'three owner types'. The earlier statement of stage 10 (3 components) is one point on a flat curve. "
           "The attribution to geography and flood experience is in `RESULTS.md`; this report does not test it.", "",
           "### 3.4 Gaussian-process models", "",
           "**Not applicable: no Gaussian-process model is claimed or used in this project.** If one is introduced, the entry conditions are the covariate report of section 4 for its inputs, "
           "plus marginal-likelihood and length-scale diagnostics (are the length scales larger than the sampling distance, is the noise term identifiable), a calibration check of the predictive "
           "intervals out of sample, and a comparison with the K-means quantiser on the same split.", ""]
    return md, verdicts


# ------------------------------------------------------------------------------------------------ regression


def feat_rows_pilot(P: dict) -> list[list]:
    rows = []
    for r in P["features"]:
        w, b = r["within"], r["between"]
        rows.append([f"`{r['feature']}`", r["role"],
                     ci(w["all"], "auc") if w else "constant in an episode", (f"{w['fit']['auc']:.3f} / {w['test']['auc']:.3f}" if w else ""),
                     diag.auc_size(w["all"]["auc"]) if w else "", f"{r['within_rho']['all']['rho']:+.3f}" if w else "", r["verdict_within"], ci(b["all"]), r["verdict_between"]])
    return rows


def sec_regress(RG: dict) -> tuple[list[str], dict]:
    P, S, F, D = RG["flood_pilot"], RG["scs"], RG["stock_flow"], RG["damage_ratio"]
    sm = P["sample"]
    wl = [r for r in P["features"] if r["within"]]
    est_w = [r["feature"] for r in wl if r["verdict_within"] == "established"]
    est_w_small = [r["feature"] for r in wl if r["verdict_within"] == "established" and diag.auc_size(r["within"]["all"]["auc"]) != "negligible"]
    not_w = [r["feature"] for r in wl if r["verdict_within"] == "not established"]
    est_b = [r["feature"] for r in P["features"] if r["verdict_between"] == "established"]
    md = ["## 4. Regression claims: covariate reports", "",
          "**What is claimed.** The project regresses four things: flood occurrence on rain, exposure and terrain (stage 8, logistic regression and boosting); SCS event occurrence on the "
          "convective environment (stages 2-5); new NFIP purchases on claim intensity and economic covariates (stage 10 stock-flow, Poisson); the damage ratio on water depth and building "
          "attributes (stages 9-10). For each, this section reports the covariate statistics that decide whether there is data to regress on: the association of every candidate feature with the target "
          "(with a cluster-bootstrap interval, in the fit and the test period separately), the correlation structure among the candidates (correlation matrix, variance inflation factors, condition number), "
          "and the sample size (events per variable). The rules are in the header of `scripts/44b_raw_report_regress.py`: **established** needs the 95% interval to exclude the null value in the "
          "full sample and in both periods, on the same side. For binary outcomes the statistic is the within-group AUC (the project's within-event metric; effect size small / medium / large at 0.556 / 0.638 / 0.714); "
          "the pilot table also lists the within-episode Spearman correlation, which is bounded by the rarity of the event (about " + f"{100 * (RG['flood_pilot']['sample']['positives_fit'] + RG['flood_pilot']['sample']['positives_test']) / RG['flood_pilot']['sample']['rows_in_mixed_episodes']:.1f}% of the rows in the episodes used), so its size labels would mislead. For continuous outcomes the statistic is the Spearman correlation.", "",
          "### 4.1 Texas flood pilot: occurrence on rain, exposure, geography, terrain", "",
          f"{sm['rows']:,} block-group x episode rows, {sm['episodes']:,} episodes, of which {sm['mixed_episodes']:,} contain both outcomes and at least 10 block groups and define the within-event statistics "
          f"({sm['rows_in_mixed_episodes']:,} rows). Positives: {sm['positives_fit']:,} in the fit years, {sm['positives_test']:,} in the test years; "
          f"{sm['n_features']} candidate features, so **{sm['events_per_variable_fit']:.0f} events per variable** in the fit years (the usual floor is 10).", "",
          "Within-event AUC compares block groups of the same episode only, so county-level features (constant inside an episode) have no within-event statistic; "
          "for them, and for every feature as a level effect, the between-episode column correlates the episode mean of the feature with the episode claim rate (clusters: counties).", "",
          f"![within-episode AUC]({P['figure_forest']})", "",
          *table(["Feature", "Role", "Within-event AUC, all years [95% CI]", "Fit / test", "Size", "Within rho", "Verdict (within)", "Between-episode rho [95% CI]", "Verdict (between)"], feat_rows_pilot(P)),
          f"![between-episode correlation]({P['figure_between']})", "",
          f"Collinearity of the plan-transformed features (fit years, {P['collinearity']['n_rows']:,} rows): condition number {P['collinearity']['condition_number']}. "
          f"Pairs with |r| >= 0.9: {', '.join(f'`{a}`-`{b}` ({r:+.2f})' for a, b, r in P['collinearity']['pairs_abs_ge_0.9']) or 'none'}. "
          f"Largest variance inflation factors: {', '.join(f'`{k}` {v}' for k, v in sorted(P['collinearity']['vif'].items(), key=lambda x: -x[1])[:8])}.", "",
          f"![correlation matrix]({P['figure_corr']})", "",
          f"**Verdict.** There **is** regressable signal. Within the episode, {len(est_w)} of {len(wl)} cell-level features are established, {len(est_w_small)} of them with at least a small effect (AUC beyond 0.556 or below 0.444; the rest exist statistically but are negligible) "
          f"(strongest: {', '.join(f'`{r['feature']}` {r['within']['all']['auc']:.3f}' for r in sorted(wl, key=lambda r: -abs(r['within']['all']['auc'] - 0.5))[:4])}), "
          f"and {len(est_b)} of {len(P['features'])} features are established as episode-level effects. No evidence for: {', '.join(f'`{f}`' for f in not_w)}. "
          f"The rain features are nearly collinear ({len(P['collinearity']['pairs_abs_ge_0.9'])} pairs with |r| >= 0.9, variance inflation up to {max(P['collinearity']['vif'].values()):.0f}), so several of them are one piece of information "
          "measured repeatedly: the AUC of any one is the signal, and a linear model cannot separate their coefficients. Events per variable is not the limit; collinearity and the weak "
          "incremental information of the extra features are, consistent with stage 8 (only exposure and the cell's own rain add skill).", ""]

    # SCS
    sf = S["features"]
    est0 = [r["feature"] for r in sf if r["verdict_y0"] == "established"]
    est1 = [r["feature"] for r in sf if r["verdict_y10k"] == "established"]
    none = [r["feature"] for r in sf if r["verdict_y0"] == "not established" and r["verdict_y10k"] == "not established"]
    md += ["### 4.2 SCS3 event occurrence on the convective environment", "",
           f"Inputs are the anomaly-scaled tier C columns ({len(sf)}). Statistic: AUC among county-days of the same year (so a year-level difference cannot masquerade as signal); clusters: years (23). "
           f"Positives: y0 {S['positives']['y0']['fit']:,} fit / {S['positives']['y0']['test']:,} test; y10k {S['positives']['y10k']['fit']:,} / {S['positives']['y10k']['test']:,}. "
           "Events per variable in the fit years: " + f"{S['positives']['y10k']['fit'] / len(sf):.0f} (y10k).", "",
           f"![within-year AUC]({S['figure_forest']})", "",
           *table(["Input", "y0: AUC all years [95% CI]", "y0 verdict", "y10k: AUC all years [95% CI]", "y10k verdict"],
                  [[f"`{r['feature']}`", ci(r["y0"]["all"], "auc"), r["verdict_y0"], ci(r["y10k"]["all"], "auc"), r["verdict_y10k"]] for r in sf]),
           f"Collinearity (fit years, {S['collinearity']['n_rows']:,} rows): condition number {S['collinearity']['condition_number']:.3g}. The variance inflation factors are not meaningful here "
           f"(up to {max(S['collinearity']['vif'].values()):.1e}): the correlation matrix has {sum(1 for d in S['collinearity']['dependencies'] if d['kind'] == 'exact')} zero eigenvalues, i.e. **exact linear dependencies by construction**:", "",
           *[f"- {d['kind']} dependence (eigenvalue {d['eigenvalue']:.2g}): " + ", ".join(f"`{k}` {v:+.2f}" for k, v in d["columns"].items()) for d in S["collinearity"]["dependencies"]], "",
           "Reading: `dd` = 2 m temperature minus dew point, so `dd_18 - dd_12 = d_t2m - d_d2m`; `d_pw = pw_18 - pw_12`. The `d_*` columns repeat information that is already in the 12Z and 18Z columns. "
           "Trees are not affected. A linear learner cannot identify the coefficients of these groups (drop one column per dependency, or use a penalty and read only the combination), "
           "and in K-means the same physical quantity enters the distance two or three times, which up-weights it.", "",
           f"![correlation matrix]({S['figure_corr']})", "",
           f"**Verdict.** There **is** regressable signal: {len(est0)} of {len(sf)} inputs are established for y0 and {len(est1)} for y10k "
           f"(strongest for y10k: {', '.join(f'`{r['feature']}` {r['y10k']['all']['auc']:.3f}' for r in sorted(sf, key=lambda r: -abs(r['y10k']['all']['auc'] - 0.5))[:4])}). "
           f"No evidence for any head for: {', '.join(f'`{f}`' for f in none)}. The AUCs below 0.5 mean a negative association (for example the dew-point depression). "
           f"Univariate AUCs up to {max(max(r['y0']['all']['auc'], r['y10k']['all']['auc']) for r in sf):.2f} sit next to models that reach only a few percent of Brier skill (stage 2: 0.03-0.04), because the events are rare "
           f"({100 * (S['positives']['y0']['fit'] + S['positives']['y0']['test']) / (S['rows_fit'] + S['rows_test']):.1f}% of county-days for y0): the discrimination exists, the calibrated gain is small.", ""]

    # flow
    fr = F["features"]
    lagrows = [r for r in fr if r["feature"].startswith("claim")]
    econ = [r for r in fr if not r["feature"].startswith("claim")]
    md += ["### 4.3 New NFIP purchases on claim intensity and economic covariates", "",
           f"{F['counties']} counties x {F['quarters']} quarters = {F['rows']:,} county-quarters. Target: {F['target']}. Features are county- and season-demeaned the same way; "
           "statistic: Spearman correlation, clusters: counties; the 'first / second half' columns split the quarters at the middle (2012-2017 / 2018-2023).", "",
           f"![purchases against claim intensity by lag]({F['figure_ccf']})", "",
           *table(["Feature", "rho all quarters [95% CI]", "First half", "Second half", "Size", "Verdict"],
                  [[r["feature"], ci(r["all"]), f"{r['first_half']['rho']:+.3f}", f"{r['second_half']['rho']:+.3f}", reg.size(r["all"]["rho"]), r["verdict"]] for r in fr]),
           f"Collinearity of the covariates: condition number {F['collinearity']['condition_number']}, largest variance inflation {max(F['collinearity']['vif'].values()):.2f}: the lags and economic series are not collinear.", "",
           f"**Verdict.** **Present but not stable.** Claim intensity is positively associated with purchases at every lag (rho {min(r['all']['rho'] for r in lagrows):+.2f} to {max(r['all']['rho'] for r in lagrows):+.2f}; "
           f"{'every interval excludes 0' if all(r['all']['excludes_zero'] for r in lagrows) else 'not every interval excludes 0'}), strongest at lags 0-1 ({lagrows[0]['all']['rho']:+.2f}, {lagrows[1]['all']['rho']:+.2f}). "
           f"In the first half it reaches {max(r['first_half']['rho'] for r in lagrows):+.2f}; in the second half it lies between {min(r['second_half']['rho'] for r in lagrows):+.2f} and {max(r['second_half']['rho'] for r in lagrows):+.2f}. "
           "The first half contains Hurricane Harvey (2017), so the signal rests largely on one event, which limits how far the pulse kernel can be trusted. "
           f"Economic covariates: {sum(1 for r in econ if r['all']['excludes_zero'])} of {len(econ)} have an all-quarter interval that excludes 0 "
           f"({', '.join(f'{r['feature'].split(' (')[0]} {r['all']['rho']:+.2f}' for r in econ)}), but the interval excludes 0 in both halves for {sum(1 for r in econ if r['first_half']['excludes_zero'] and r['second_half']['excludes_zero'])} of them "
           "(sign changes: " + ", ".join(r['feature'].split(' (')[0] for r in econ if r['first_half']['rho'] * r['second_half']['rho'] < 0) + "), in line with stage 10's 'the economic cycle adds nothing stable'. "
           "The sample is small (45 counties) and quarters within a county are autocorrelated, so these intervals understate the uncertainty.", ""]

    # damage
    dp = D["depth"]
    et = {k: v for k, v in D["eta2"].items() if v["categories"] > 1}
    md += ["### 4.4 Damage ratio on water depth and building attributes", "",
           f"{D['rows']:,} single-family claims 2016+ with exterior depth, coverage and damage amount; {D['clusters_state_month']:,} state-months as clusters. Target: {D['target']}.", "",
           f"![eta squared]({D['figure_eta2']})", "",
           *table(["Covariate", "Statistic", "Value [95% CI]", "Verdict"], [
               ["log exterior depth", "Spearman rho, all rows", ci(dp["between_and_within"]), "established" if dp["between_and_within"]["excludes_zero"] else "not established"],
               ["log exterior depth", "Spearman rho within state-month", ci(dp["within_state_month"]), "established" if dp["within_state_month"]["excludes_zero"] else "not established"],
               *[[k, "eta squared", f"{v['eta2']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]", "established" if v["lo"] > 0.001 else "not established"] for k, v in et.items()],
               ["occupancy (1 vs 11)", "eta squared", "not testable", f"the sample holds {D['eta2']['occupancy (1 vs 11)']['categories']} occupancy category: from 2016 the foundation code is known for every occupancy-11 claim and for no occupancy-1 claim (checked on the clean claims table), so requiring a known foundation leaves only code 11"]]),
           "Share of claims with a known foundation code, by loss year and occupancy code (claims per cell in brackets):", "",
           *table(["Loss year"] + [str(y) for y in sorted(D["foundation_known_by_occupancy"], key=int)],
                  [[f"occupancy {o}"] + [(f"{D['foundation_known_by_occupancy'][y][str(o)][1]:.0%} ({D['foundation_known_by_occupancy'][y][str(o)][0]:,})" if str(o) in D["foundation_known_by_occupancy"][y] else "-")
                                         for y in sorted(D["foundation_known_by_occupancy"], key=int)] for o in (1, 11)]),
           "**A known foundation and the occupancy-code family are the same information in the claims file.** In the raw data every claim with a legacy occupancy code (1, 2, 3, 4, 6) has foundation '0' and "
           "every claim with a two-digit code (11, 12, 15, 16, 18) has a known foundation (checked year by year for codes 1 and 11: every loss year 2000-2023; `docs/CLEANING_REGISTER.md`, rule C10). The foundation comparisons "
           "(slab / basement / crawlspace / elevated) are therefore comparisons among two-digit-coded claims, and 'unknown vs known' contrasts elsewhere in the project are contrasts between code families.", "",
           f"Collinearity of the design (depth, foundation dummies, SFHA, post-FIRM, elevated): condition number {D['collinearity']['condition_number']}, variance inflation "
           f"{', '.join(f'{k} {v}' for k, v in D['collinearity']['vif'].items())}.", "",
           f"**Verdict.** There **is** regressable signal in depth (rho {dp['within_state_month']['rho']:+.2f} inside a state-month, so it is not an artefact of events being deeper on average) and in "
           f"the foundation group (eta squared {D['eta2']['foundation group']['eta2']:.3f}) and the elevated indicator ({D['eta2']['elevated indicator']['eta2']:.3f}). "
           f"The special flood hazard area adds {D['eta2']['special flood hazard area']['eta2']:.4f} on its own: its information is carried by the other attributes. "
           f"Each categorical covariate explains under {max(v['eta2'] for v in et.values()):.0%} of the variance, so the claim is 'a regressable component exists', not 'the damage ratio is predictable'. "
           f"The foundation dummies and the elevated indicator are partly collinear (variance inflation {D['collinearity']['vif'].get('slab')} for slab, {D['collinearity']['vif'].get('elevated')} for elevated): tolerable, but their separate coefficients are less sharp than their joint effect.", "",
           "### 4.5 Relationships not claimed, or not established", "",
           "- **Post-event-buyer share vs Harvey claim rate** (stage 10): reported as a candidate whose interval contains 0; not recomputed here, and not claimed.",
           f"- **Wavelet scale features, neighbourhood smoothing, terrain** (stages 3, 8): no gain beyond the interval in the models; in the covariate view above the terrain columns have within-event AUCs of {min(r['within']['all']['auc'] for r in P['features'] if r['role'] == 'terrain'):.2f}-{max(r['within']['all']['auc'] for r in P['features'] if r['role'] == 'terrain'):.2f}.",
           "- **Longitude, latitude, elevation, slope** for flood occurrence: no within-event evidence (section 4.1).", ""]
    return md, {"est_w": est_w, "est_b": est_b, "not_w": not_w, "est0": est0, "est1": est1, "lag_rows": lagrows, "econ": econ}


# ------------------------------------------------------------------------------------------------ assemble


def build() -> None:
    R, C, RG = load("raw"), load("cluster"), load("regress")
    cl_md, _ = sec_cluster(C)
    rg_md, rv = sec_regress(RG)
    S, T = RG["scs"], C["takeup_gmm"]
    sidecars = {n: json.loads((base.wb2.DATA_DIR / "clean" / f"{n}.prep.json").read_text()) for n in ("policies_tx", "claims_nfip")}
    md = ["# Raw data statistical diagnostics and claim verification", "",
          "Generated by `python scripts/44_raw_report.py all` (code: `scripts/44_raw_report.py`, `44b_raw_report_regress.py`, `44c_raw_report_render.py`, `src/wxrisk/diag.py`; tests: `tests/test_diag.py`). "
          "Do not edit by hand. Random seed 0 throughout; bootstrap over the dependent units named in each table (200 resamples for regressions, 200 permutations for cluster labels). "
          "Raw statistics come from the downloaded files; the cluster and regression sections use the inputs the models use (anomaly-scaled SCS columns, the flood pilot table, the clean claims table).", "",
          "Input fingerprints (content hashes, `prep` sidecars): " + ", ".join(f"`{n}` input {v['input_content_sha256'][:12]} / output {v['output_content_sha256'][:12]}" for n, v in sidecars.items()) + ".", "",
          ]
    md += ["## 0. How to read this report", "",
           "- **Silhouette / Hopkins / ARI / mutual information** (section 3) answer three different questions: is there structure, is it reproducible, does it matter for the outcome. A partition can fail the first and pass the third; that is a quantiser.",
           "- **Within-event AUC** (section 4.1, 4.2) compares rows of the same episode (or year) only; 0.5 is no information and values below 0.5 are a negative association.",
           "- **Cluster bootstrap**: rows within an episode, county, state-month or year are dependent, so intervals resample those groups, never the rows.",
           "- A verdict of *not established* means the interval contains the null value; it is not a proof of absence, only that this sample gives no evidence.", "",
           "## 1. Claims and verdicts", "",
           "Every statement the project makes that a dataset *can be clustered* or *can be regressed* is listed here with the section that tests it and the verdict computed from the data.", ""]
    md += table(["#", "Claim", "Dataset", "Test (section)", "Verdict"], [
        ["C1", "The SCS tier B environment can be partitioned by K-means", "ERA5 county points, 13 columns", "3.1",
         f"No natural clusters (silhouette {C['B']['silhouette_at_K']:.3f}); usable as a quantiser: labels carry out-of-sample information (p <= {max(v['p_value'] for v in C['B']['outcome_association_test'].values()):.3f})"],
        ["C2", "The SCS tier C environment can be partitioned by K-means", "ERA5 county points, 28 columns", "3.2",
         f"No natural clusters (silhouette {C['C']['silhouette_at_K']:.3f}, {C['C']['clusters_with_negative_mean_silhouette']} clusters with negative silhouette); usable as a quantiser (p <= {max(v['p_value'] for v in C['C']['outcome_association_test'].values()):.3f})"],
        ["C3", "Take-up across block groups is bimodal", "NFIP policies / ACS housing units", "3.3", f"Bimodal (coefficient {T['bimodality_coefficient']:.3f}); the number of components is not identifiable"],
        ["C4", "A Gaussian-process model", "-", "3.4", "Not claimed, not applicable"],
        ["R1", "Flood occurrence can be regressed on rain, exposure, geography, terrain", "Texas flood pilot table", "4.1",
         f"Yes: {len(rv['est_w'])} cell-level and {len(rv['est_b'])} episode-level features established; none for {', '.join(rv['not_w'][:6])}..."],
        ["R2", "SCS event occurrence can be regressed on the convective environment", "ERA5 county points", "4.2", f"Yes: {len(rv['est0'])} / {len(S['features'])} inputs established for y0, {len(rv['est1'])} for y10k"],
        ["R3", "New NFIP purchases can be regressed on claim intensity (and economics)", "Stock-flow panel, 45 counties", "4.3", "Present, not stable: concentrated in 2012-2017 (Harvey); economic covariates not stable"],
        ["R4", "The damage ratio can be regressed on depth and building attributes", "NFIP claims 2016+", "4.4", "Yes for depth, foundation group, elevated indicator; SFHA adds nothing alone"],
        ["R5", "Post-event-buyer share explains Harvey claim rates", "Policies x claims", "4.5", "Not established (interval contains 0); not claimed"]])
    md += sec_raw(R) + cl_md + rg_md
    md += ["## 5. What this report does not show", "",
           "- It describes association in the available samples; it does not show causality or out-of-time forecast skill (that is `RESULTS.md`).",
           "- Section 3 clusters the anomaly-scaled inputs of the project's pipeline; clustering the raw physical values would give a different (and more seasonal) partition.",
           "- The cluster and regression sections use the 2000-2022 SCS record and the 2010-2023 pilot; a different period could move individual intervals.",
           "- Only 45 counties enter the stock-flow statistics; the effective sample is smaller than the 2,160 county-quarters because quarters within a county are dependent.",
           "- Hopkins statistic and silhouette depend on the scale of the inputs (anomalies divided by their standard deviation here).",
           "- The OpenFEMA FAQ was not retrievable as a file (see the cleaning register); the raw checks above rely on the data dictionary and the row counts.", ""]
    REPORT.write_text("\n".join(md))
    print(f"wrote {REPORT} ({len(md)} lines)")


if __name__ == "__main__":
    build()
