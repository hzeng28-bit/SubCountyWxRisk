"""RAW_STATICS_DA_REPORT.md: raw-data verification, cluster diagnostics, and covariate reports for the claims the project makes.

    python scripts/44_raw_report.py raw        # statistics of the raw datasets (claims, policies, gridMET, SCS labels, ERA5 county env, ...)
    python scripts/44_raw_report.py cluster    # K-means / mixture diagnostics + PCA projections for every dataset we cluster (+ the Hopkins references)
    python scripts/44_raw_report.py redraw     # redraw the K-sweep and take-up figures from the stored numbers (no refit)
    python scripts/44_raw_report.py regress    # covariate statistics for every relationship we regress
    python scripts/44_raw_report.py render     # RAW_STATICS_DA_REPORT.md from the JSON in $WXRISK_DATA/audit/raw_report and the figures
    python scripts/44_raw_report.py all

Figures go to docs/figures/raw_report/ (PNG, light surface; categorical hues limited to the first three validated slots, other groups grey;
continuous encodings use the single-hue blue ramp or the blue-red diverging pair of the dataviz reference palette).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wxrisk import diag, integrity, wb2  # noqa: E402
from wxrisk import flood_pilot as fp  # noqa: E402

warnings.filterwarnings("ignore")
OUT = wb2.DATA_DIR / "audit" / "raw_report"
FIG = ROOT / "docs" / "figures" / "raw_report"
FLOOD = wb2.cache_path("labels_extra") / "flood"
SEED = 0

# ---- dataviz reference palette (light surface)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"  # categorical slots 1-3 (validated all-pairs) and "other"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])


def style(ax, grid: bool = True) -> None:
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)
    if grid:
        ax.grid(color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)


def new_fig(w: float, h: float, **kw):
    fig, axes = plt.subplots(figsize=(w, h), facecolor=SURF, **kw)
    for ax in np.atleast_1d(axes).ravel():
        style(ax)
    return fig, axes


def save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)
    return f"docs/figures/raw_report/{name}"


def dump(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
    print(f"wrote {OUT / (name + '.json')}", flush=True)


def load_script(num_name: str):
    spec = importlib.util.spec_from_file_location(f"s_{num_name[:2]}", ROOT / "scripts" / f"{num_name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# =====================================================================================================================
# RAW: statistics of the raw datasets
# =====================================================================================================================

NUM = ("DOUBLE", "BIGINT", "INTEGER", "SMALLINT", "TINYINT", "FLOAT", "HUGEINT")


def q(sql: str) -> list:
    return duckdb.sql(sql).fetchall()


def profile(src: str) -> list[dict]:
    """One row per column of a (possibly multi-file) parquet source: type, null share, distinct values, range, zero share."""
    cols = duckdb.sql(f"describe select * from {src}").fetchall()
    aggs = []
    for name, typ, *_ in cols:
        c = f'"{name}"'
        aggs += [f"count({c})", f"approx_count_distinct({c})"]
        if typ in NUM or typ.startswith("DECIMAL"):
            aggs += [f"min({c})", f"quantile_cont({c}, 0.5)", f"max({c})", f"count(*) filter (where {c} = 0)"]
        elif typ == "BOOLEAN":
            aggs += ["NULL", "NULL", "NULL", f"count(*) filter (where not {c})"]
        else:
            aggs += [f"min({c})", "NULL", f"max({c})", "NULL"]
    n = q(f"select count(*) from {src}")[0][0]
    r = q("select " + ", ".join(aggs) + f" from {src}")[0]
    out = []
    for i, (name, typ, *_) in enumerate(cols):
        nn, nd, lo, med, hi, z = r[6 * i : 6 * i + 6]
        out.append({"column": name, "type": typ, "null_pct": round(100 * (1 - nn / n), 3), "distinct_approx": int(nd), "min": str(lo)[:24], "median": None if med is None else str(med)[:24],
                    "max": str(hi)[:24], "zero_pct": None if z is None else round(100 * z / n, 3)})
    return out


def raw() -> None:
    R: dict = {}
    claims = f"read_parquet('{FLOOD / 'nfip_national' / '*.parquet'}', union_by_name=true)"
    pol = f"read_parquet('{FLOOD / 'pilot' / 'policies_tx.parquet'}')"

    # ---- NFIP claims
    import pyarrow.parquet as pq

    files = sorted((FLOOD / "nfip_national").glob("nfip_*.parquet"))
    schemas = {f.stem: {fl.name: str(fl.type) for fl in pq.read_schema(f)} for f in files}
    cols = sorted({c for s in schemas.values() for c in s})
    drift = {c: sorted({s.get(c, "ABSENT") for s in schemas.values()}) for c in cols}
    drift = {c: v for c, v in drift.items() if len(v) > 1}
    n_rows, n_ids = q(f"select count(*), count(distinct id) from {claims}")[0]
    from wxrisk import fetch

    try:
        provider = fetch.openfema_count("NfipClaims", "v3", "yearOfLoss ge 2000 and yearOfLoss le 2023")
    except Exception as e:  # noqa: BLE001 - network is a check, not a requirement
        provider = f"unavailable ({type(e).__name__})"
    R["claims"] = {"files": len(files), "rows": n_rows, "distinct_id": n_ids, "provider_count_2000_2023": provider, "schema_drift_across_year_files": drift,
                   "rows_per_year": dict(q(f"select yearOfLoss, count(*) from {claims} group by 1 order by 1")), "columns": profile(claims)}
    print("claims done", flush=True)

    # ---- NFIP policies (Texas extract)
    try:
        prov_pol = fetch.openfema_count("NfipPolicies", "v3", "propertyState eq 'TX'")
    except Exception as e:  # noqa: BLE001
        prov_pol = f"unavailable ({type(e).__name__})"
    allcols = ", ".join(f'"{r[0]}"' for r in duckdb.sql(f"describe select * from {pol}").fetchall())  # explicit: GROUP BY ALL would group by nothing here
    dup = q(f"select count(*), coalesce(sum(c - 1), 0) from (select count(*) c from {pol} group by {allcols} having count(*) > 1)")[0]
    # duplicates by the size of the block group: identical redacted rows are expected where many similar policies exist; double loading would not depend on it
    bucket = q(f"""with g as (select any_value(censusGeoid) as geoid, count(*) as c from {pol} group by {allcols}),
        bg as (select censusGeoid as geoid, count(*) as n from {pol} group by 1)
        select case when n < 20 then '<20' when n < 200 then '20-199' when n < 2000 then '200-1999' else '>=2000' end as b, sum(c) as rows, sum(c - 1) as excess
        from g join bg using (geoid) group by 1""")
    R["policies_tx"] = {"rows": q(f"select count(*) from {pol}")[0][0], "provider_count_TX": prov_pol, "exact_duplicate_groups": dup[0], "rows_in_excess_of_first": int(dup[1]),
                        "duplicates_by_block_group_size": {b: {"rows": int(r), "excess": int(x), "excess_pct": round(100 * x / r, 3)} for b, r, x in bucket},
                        "rows_per_effective_year": dict(q(f"select year(policyEffectiveDate), count(*) from {pol} group by 1 order by 1")),
                        "term_days_p1_p50_p99": q(f"select quantile_cont(date_diff('day', policyEffectiveDate, policyTerminationDate), [0.01, 0.5, 0.99]) from {pol}")[0][0],
                        "columns": profile(pol)}
    print("policies done", flush=True)

    # ---- gridMET daily precipitation (Texas)
    days, lat, lon, P = fp.load_precip()
    n = nan = zero = neg = 0
    mx = 0.0
    daily_mean = np.empty(len(days))
    big = 0
    wet_sample = []
    nan_per_cell = np.zeros(P.shape[1:], np.int32)
    big_by_day = np.zeros(len(days), np.int32)
    rng = np.random.default_rng(SEED)
    for i in range(0, len(days), 365):
        c = np.asarray(P[i : i + 365])
        fin = np.isfinite(c)
        nan += int((~fin).sum())
        n += c.size
        zero += int((c == 0).sum())
        neg += int((c < 0).sum())
        big += int((c > 300).sum())
        nan_per_cell += np.isnan(c).sum(0)
        big_by_day[i : i + len(c)] = (c > 300).sum((1, 2))
        mx = max(mx, float(np.nanmax(c)))
        daily_mean[i : i + len(c)] = np.nanmean(c, axis=(1, 2))
        w = c[fin & (c >= 1.0)]
        wet_sample.append(rng.choice(w, min(len(w), 20000), replace=False))
    dm = pd.Series(daily_mean, index=days)
    wet = np.concatenate(wet_sample)
    top = pd.Series(big_by_day, index=days)
    top = top[top > 0].sort_values(ascending=False)
    gaps = int((np.diff(days.values).astype("timedelta64[D]").astype(int) != 1).sum())
    R["gridmet_precip"] = {"shape": list(P.shape), "dtype": str(P.dtype), "lat_range": [float(lat.min()), float(lat.max())], "lon_range": [float(lon.min()), float(lon.max())],
                           "first_day": str(days[0].date()), "last_day": str(days[-1].date()), "calendar_gaps": gaps, "nan_pct": round(100 * nan / n, 4), "zero_pct": round(100 * zero / n, 2),
                           "negative_values": neg, "cells_over_300mm": big, "max_mm": mx,
                           "cells_always_nan_pct": round(100 * float((nan_per_cell == len(days)).mean()), 2), "cells_sometimes_nan_pct": round(100 * float(((nan_per_cell > 0) & (nan_per_cell < len(days))).mean()), 3),
                           "days_with_cell_over_300mm": int(len(top)), "top_days_over_300mm": {str(d.date()): int(v) for d, v in top.head(8).items()},
                           "share_of_over_300mm_on_top8_days": round(float(top.head(8).sum() / top.sum()), 3),
                           "wet_day_quantiles_mm": {str(p): float(np.quantile(wet, p)) for p in (0.5, 0.9, 0.99, 0.999)},
                           "domain_mean_annual_total_mm": {str(y): round(float(v), 1) for y, v in dm.groupby(dm.index.year).sum().items()},
                           "domain_mean_by_month_mm_per_day": {str(m): round(float(v), 3) for m, v in dm.groupby(dm.index.month).mean().items()}}
    print("gridmet done", flush=True)

    # ---- SCS3 labels (county-day)
    lab = pd.read_parquet(wb2.cache_path("labels_scs3_conm_az.parquet"))
    lab["year"] = lab.date.dt.year
    heads = ["y0", "y10k", "y100k", "y1m"]
    R["scs_labels"] = {"rows": len(lab), "counties": int(lab.cty.nunique()), "first": str(lab.date.min().date()), "last": str(lab.date.max().date()),
                       "duplicate_county_days": int(lab.duplicated(["cty", "date"]).sum()),
                       "positives": {h: int(lab[h].sum()) for h in heads}, "rate_pct": {h: round(100 * float(lab[h].mean()), 3) for h in heads},
                       "positives_per_year": {h: {str(y): int(v) for y, v in lab.groupby("year")[h].sum().items()} for h in heads},
                       "rows_per_state": lab.cty.str[:2].map({"08": "CO", "35": "NM", "04": "AZ"}).value_counts().to_dict(),
                       "months_covered": sorted(lab.date.dt.month.unique().tolist())}

    # ---- ERA5 county environment (raw units, before log / anomalies)
    from wxrisk import county_env

    cty = lab.drop_duplicates("cty")[["cty", "lat", "lon"]].reset_index(drop=True)
    env = county_env.load(list(range(2000, 2023)), cty)
    # loose physical bounds in the units county_env.py builds: lapse75 = T700 - T500 (K), w700 = wind SPEED at 700 hPa (m/s), dd = T2m - Td2m (K, a
    # few thousandths below 0 occur from rounding), shear = |wind500 - wind10m| (m/s), q850 in kg/kg, pw = total column water (kg/m2), msl in Pa
    phys = {"cape": (0, 8000), "cin": (0, 1500), "dd": (-0.5, 60), "t2m": (230, 330), "d2m": (220, 310), "pw": (0, 80), "q850": (0, 0.03), "lapse75": (-5, 35),
            "shear": (0, 80), "w700": (0, 100), "msl": (9.0e4, 1.1e5)}
    rows = {}
    for c in env.columns:
        if c in ("cty", "date"):
            continue
        v = env[c].astype(float)
        base = c.split("_")[0]
        lo_hi = phys.get(base)
        rows[c] = {"nan_pct": round(100 * float(v.isna().mean()), 3), "min": float(v.min()), "p01": float(v.quantile(0.01)), "median": float(v.median()),
                   "p99": float(v.quantile(0.99)), "max": float(v.max()), "zero_pct": round(100 * float((v == 0).mean()), 2),
                   "outside_physical_range": None if lo_hi is None else int(((v < lo_hi[0]) | (v > lo_hi[1])).sum()), "physical_range": lo_hi}
    R["era5_county_env"] = {"rows": len(env), "columns": rows}
    print("scs/era5 done", flush=True)

    # ---- flood pilot table (derived backbone) and static layers
    t = pd.read_parquet(fp.PILOT / "flood_pilot_table.parquet", columns=["ep", "bg", "year", "y", "cty", "n_pol", "dur"])
    g = t.groupby("year").agg(rows=("y", "size"), positives=("y", "sum"), episodes=("ep", "nunique"))
    g["positives_per_100k_rows"] = (g.positives / g.rows * 1e5).round(1)
    terr = pd.read_parquet(fp.PILOT / "bg_terrain.parquet")
    R["flood_pilot_table"] = {"rows": len(t), "episodes": int(t.ep.nunique()), "block_groups": int(t.bg.nunique()), "positives": int(t.y.sum()), "by_year": g.to_dict("index"),
                              "terrain_rows": len(terr), "terrain_nan_pct": {c: round(100 * float(terr[c].isna().mean()), 3) for c in terr.columns if c != "bg10"}}

    # ---- the data lock
    lock = json.loads((ROOT / "data.lock.json").read_text())["entries"]
    ok = miss = diff = 0
    bad = []
    for rel, e in lock.items():
        p = wb2.DATA_DIR / rel
        if not p.exists():
            miss += 1
            bad.append((rel, "missing"))
        elif p.stat().st_size != e["bytes"]:
            diff += 1
            bad.append((rel, f"size {p.stat().st_size} != locked {e['bytes']}"))
        else:
            ok += 1
    R["data_lock"] = {"entries": len(lock), "size_matches": ok, "missing": miss, "size_differs": diff, "examples": bad[:12],
                      "note": "size check only; the content-hash check is `python scripts/32_data_lock.py verify`", "algo": integrity.__name__}
    dump("raw", R)


# =====================================================================================================================
# CLUSTER
# =====================================================================================================================


def pca_scatter(name: str, tier: str, X, fit, lab, rate_lift, centres_idx, k: int) -> str:
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(SEED)
    Xf = X[fit]
    pca = PCA(n_components=min(10, X.shape[1]), random_state=SEED).fit(Xf)
    idx = rng.choice(len(Xf), min(25000, len(Xf)), replace=False)
    P2 = pca.transform(Xf[idx])[:, :2]
    labs = lab[fit][idx]
    C2 = pca.transform(pca_centres(Xf, lab[fit], k))[:, :2]
    lift = rate_lift[labs]
    fig, ax = new_fig(11.2, 4.7, ncols=2)
    sc = ax[0].scatter(P2[:, 0], P2[:, 1], c=lift, s=4, cmap=SEQ, alpha=0.6, linewidths=0, vmin=np.quantile(rate_lift, 0.05), vmax=np.quantile(rate_lift, 0.95))
    cb = fig.colorbar(sc, ax=ax[0], fraction=0.046, pad=0.02)
    cb.set_label("cluster event rate / climatology (y10k, fit years)", fontsize=8, color=INK2)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    ax[0].set_title(f"{tier}: PCA of the K-means input, coloured by cluster lift", fontsize=9, loc="left")
    ax[1].scatter(P2[:, 0], P2[:, 1], s=3, c=GREY, alpha=0.25, linewidths=0)
    top = np.argsort(rate_lift)[::-1][:3]
    for c, col in zip(top, (S1, S2, S3), strict=True):
        m = labs == c
        ax[1].scatter(P2[m, 0], P2[m, 1], s=5, c=col, alpha=0.85, linewidths=0, label=f"cluster {c}: lift {rate_lift[c]:.1f}, n={int((lab[fit] == c).sum()):,}")
        ax[1].scatter(C2[c, 0], C2[c, 1], s=70, c=col, edgecolors=SURF, linewidths=2, zorder=5)
        ax[1].annotate(str(c), C2[c], xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK)
    ax[1].scatter(C2[:, 0], C2[:, 1], s=14, c=INK2, alpha=0.7, linewidths=0, zorder=4)
    ax[1].set_title(f"the 3 highest-lift clusters of {k}; grey = other clusters; dots = centroids", fontsize=9, loc="left")
    ax[1].legend(fontsize=7.5, frameon=False, loc="upper right", labelcolor=INK2)
    for a in ax:
        a.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% of variance)", fontsize=8)
        a.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}%)", fontsize=8)
    fig.tight_layout()
    return save(fig, name), pca


def pca_centres(Xf, lab, k) -> np.ndarray:
    return np.vstack([Xf[lab == c].mean(0) if (lab == c).any() else np.zeros(Xf.shape[1]) for c in range(k)])


def k_sweep_fig(prefix: str, tier: str, curve: pd.DataFrame) -> str:
    d, nl = curve[curve.set == "data"], curve[curve.set == "null"]
    fig, ax = new_fig(9.6, 3.7, ncols=2)
    for df_, col in ((d, S1), (nl, S2)):
        ax[0].plot(df_.K, df_.silhouette, color=col, lw=2, marker="o", ms=5, mfc=col, mec=SURF, mew=1.5)
    ax[0].annotate("data", (d.K.iloc[0], d.silhouette.iloc[0]), xytext=(6, 6), textcoords="offset points", fontsize=8, color=INK2)
    ax[0].annotate("same data, columns permuted (null)", (nl.K.iloc[0], nl.silhouette.iloc[0]), xytext=(6, -15), textcoords="offset points", fontsize=8, color=INK2)
    ax[0].axhline(0.25, color=AXIS, lw=1)
    ax[0].text(d.K.iloc[0], 0.235, "0.25: below this, no substantial structure", fontsize=7.5, color=MUTED, va="top")
    ax[0].set_ylim(0, max(0.27, float(d.silhouette.max()) * 1.1))
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("K", fontsize=8)
    ax[0].set_ylabel("mean silhouette", fontsize=8)
    ax[0].set_title(f"{tier}: silhouette against K", fontsize=9, loc="left")
    ax[1].plot(d.K, d.inertia, color=S1, lw=2, marker="o", ms=5, mfc=S1, mec=SURF, mew=1.5)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].minorticks_off()
    ax[1].set_xlabel("K", fontsize=8)
    ax[1].set_ylabel("inertia (within-cluster sum of squares)", fontsize=8)
    ax[1].set_title("elbow plot (a kink would mark a natural K)", fontsize=9, loc="left")
    fig.tight_layout()
    return save(fig, f"{prefix}_k_sweep.png")


def cluster_figs(prefix: str, tier: str, curve: pd.DataFrame, per_cluster: pd.DataFrame, k: int) -> dict:
    f1 = k_sweep_fig(prefix, tier, curve)
    pc = per_cluster.sort_values("mean")
    fig, ax = new_fig(9.6, 3.2)
    wmean = float((pc["mean"] * pc["size"]).sum() / pc["size"].sum())
    ax.bar(np.arange(len(pc)), pc["mean"], width=0.7, color=S1)
    ax.axhline(0, color=AXIS, lw=1)
    ax.axhline(wmean, color=INK2, lw=1)
    ax.text(0, wmean + 0.004, "size-weighted mean", fontsize=7.5, color=INK2)
    ax.set_xlabel(f"cluster (sorted by silhouette), K = {k}", fontsize=8)
    ax.set_ylabel("mean silhouette", fontsize=8)
    ax.set_title(f"{tier}: silhouette of each cluster", fontsize=9, loc="left")
    fig.tight_layout()
    return {"k_sweep": f1, "per_cluster": save(fig, f"{prefix}_silhouette.png")}


def redraw() -> None:
    """Redraw the figures that need only stored numbers (K sweeps) and the take-up mixture, without refitting the K-means models."""
    R = json.loads((OUT / "cluster.json").read_text())
    for tag in ("B", "C"):
        k_sweep_fig(f"scs_{tag}", f"SCS tier {tag}", pd.DataFrame(R[tag]["k_curve"]))
    R["takeup_gmm"] = takeup()
    dump("cluster", R)


def cluster() -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    from wxrisk.scs_eval import anomalise, climatology, cluster_stats

    km10 = load_script("10_kmeans_baseline")
    df = km10.build_table(list(range(2000, 2023)))
    fit = (df.year <= 2016).to_numpy()
    test = (df.year >= 2017).to_numpy()
    tiers = km10.tier_columns(df)
    p0 = climatology(df, pd.Series(fit))
    kres = pd.read_csv(wb2.cache_path("kmeans_results.csv"))
    bestk = kres.groupby("tier").K.first().to_dict()
    R: dict = {"note_tier_A": "tier A has only calendar and location columns (doy_s, doy_c, lat, lon); its clusters are a spatial-seasonal grid, not weather states: not analysed here"}
    rng = np.random.default_rng(SEED)
    ks = [2, 4, 8, 16, 32, 64, 128]
    for tier, tag in (("B (T-3D..T-3h)", "B"), ("C (<T-3h)", "C")):
        cols, k = tiers[tier], int(bestk[tier])
        X = anomalise(df, cols, pd.Series(fit))
        Xf = X[fit]
        hop = diag.hopkins(Xf, 1000, SEED)
        curve = diag.silhouette_curve(Xf, ks, n_sample=15000, seed=SEED)
        sub = rng.choice(len(Xf), min(150000, len(Xf)), replace=False)
        km = KMeans(k, n_init=3, random_state=SEED).fit(Xf[sub])
        lab = km.predict(X)
        idx = rng.choice(fit.sum(), 20000, replace=False)
        sil = float(silhouette_score(Xf[idx], lab[fit][idx]))
        per = diag.silhouette_by_cluster(Xf, lab[fit], 20000, SEED)
        row = {"K": k, "n_columns": len(cols), "rows_fit": int(fit.sum()), "rows_test": int(test.sum()), "hopkins": hop,
               "silhouette_at_K": sil, "silhouette_verdict": diag.silhouette_verdict(sil),
               "davies_bouldin_at_K": float(davies_bouldin_score(Xf[idx], lab[fit][idx])), "calinski_harabasz_at_K": float(calinski_harabasz_score(Xf[idx], lab[fit][idx])),
               "silhouette_null_at_K": float(curve[(curve.set == "null") & (curve.K == k)].silhouette.iloc[0]),
               "clusters_with_negative_mean_silhouette": int((per["mean"] < 0).sum()), "stability": diag.stability(Xf, k, SEED),
               "cluster_size": {"min": int(np.bincount(lab[fit]).min()), "median": int(np.median(np.bincount(lab[fit]))), "max": int(np.bincount(lab[fit]).max())},
               "k_curve": curve.to_dict("records")}
        # the outcome side: do the labels carry information about the events, out of sample?
        assoc = {}
        for h in ("y0", "y10k", "y100k"):
            y = df[h].to_numpy(float)
            assoc[h] = diag.label_association(lab[test], y[test], n_perm=200, seed=SEED)
            rate, lift = cluster_stats(lab, fit, p0[h], k, y)
            m = np.bincount(lab[test], minlength=k) > 20
            from scipy.stats import spearmanr

            tr_rate = pd.Series(y[test]).groupby(lab[test]).mean().reindex(range(k))
            assoc[h]["spearman_cluster_rate_fit_vs_test"] = float(spearmanr(rate[m], tr_rate.to_numpy()[m]).statistic)
            assoc[h]["test_positives"] = int(y[test].sum())
            if h == "y10k":
                lift10 = lift
        row["outcome_association_test"] = assoc
        figs = cluster_figs(f"scs_{tag}", f"SCS tier {tag}", curve, per, k)
        scat, pca = pca_scatter(f"scs_{tag}_pca.png", f"SCS tier {tag}", X, fit, lab, lift10, None, k)
        figs["pca"] = scat
        ev = pca.explained_variance_ratio_
        row["pca"] = {"explained_variance_ratio": [float(v) for v in ev], "cumulative_2": float(ev[:2].sum()), "participation_ratio_dims": float(ev.sum() ** 2 / (ev**2).sum()),
                      "components_for_80pct": int(np.searchsorted(np.cumsum(ev), 0.8) + 1) if ev.sum() >= 0.8 else None}
        row["figures"] = figs
        R[tag] = row
        print(f"cluster {tag} done: silhouette {sil:.3f}, hopkins {hop:.3f}", flush=True)

    # ---- 1-D mixture: take-up rate across block groups
    R["takeup_gmm"] = takeup()
    dump("cluster", R)


def takeup() -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    m34 = load_script("34_takeup_rate")
    hu = m34.housing_units()
    p = m34.snapshot_policies("2022-12-31")
    w = p.policyCount.astype(float)
    g = pd.DataFrame({"n_x": (w * ~p.sfha).groupby(p.geoid).sum(), "n_sfha": (w * p.sfha).groupby(p.geoid).sum()})
    g["hu"] = hu.reindex(g.index)
    g = g.dropna(subset=["hu"])
    g = g[(g.hu >= 200) & (g.n_sfha / (g.n_x + g.n_sfha).clip(lower=1) < 0.05)].copy()
    x = np.log10((g.n_x / g.hu).clip(lower=1e-4)).to_numpy()
    out = {"block_groups": len(x), "bimodality_coefficient": diag.bimodality_coefficient(x), "bimodality_threshold": 0.555}
    gm = {}
    for k in (1, 2, 3, 4):
        m = GaussianMixture(k, n_init=5, random_state=SEED).fit(x[:, None])
        gm[k] = {"bic": float(m.bic(x[:, None])), "weights": [round(float(v), 3) for v in m.weights_], "means_log10": [round(float(v), 3) for v in m.means_.ravel()],
                 "sd_log10": [round(float(v), 3) for v in np.sqrt(m.covariances_.ravel())]}
    out["gmm"] = gm
    out["best_k_by_bic"] = min(gm, key=lambda k: gm[k]["bic"])
    rng = np.random.default_rng(SEED)
    sub = rng.choice(len(x), min(len(x), 6000), replace=False)
    out["kmeans_1d_silhouette"] = {k: float(silhouette_score(x[sub, None], KMeans(k, n_init=5, random_state=SEED).fit(x[sub, None]).labels_)) for k in (2, 3, 4)}
    # what a split of a single-mode Gaussian scores in 1-D: the reference for the silhouette above
    gref = rng.normal(x.mean(), x.std(), 6000)[:, None]
    out["kmeans_1d_silhouette_gaussian_reference"] = {k: float(silhouette_score(gref, KMeans(k, n_init=5, random_state=SEED).fit(gref).labels_)) for k in (2, 3, 4)}
    kk = out["best_k_by_bic"]
    fig, ax = new_fig(7.4, 3.7)
    ax.hist(x, bins=60, density=True, color="#cde2fb", edgecolor=SURF, linewidth=0.8)
    xs = np.linspace(x.min(), x.max(), 400)
    for k, col in ((1, S2), (2, S3), (kk, S1)):
        if k == kk and kk in (1, 2):
            continue
        m = GaussianMixture(k, n_init=5, random_state=SEED).fit(x[:, None])
        dens = np.exp(m.score_samples(xs[:, None]))
        ax.plot(xs, dens, color=col, lw=2)
        right = xs > -1.5 if k == 2 else np.ones_like(xs, bool)  # label the two-component curve at its upper mode on the right
        i = int(np.argmax(np.where(right, dens, -1)))
        ax.annotate(f"{k} component{'s' if k > 1 else ''}", (xs[i], dens[i]), xytext=(8, 5), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("log10 take-up (non-SFHA single-family policies / housing units)", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.set_title(f"Take-up across {len(x):,} block groups: mixture fits", fontsize=9, loc="left")
    fig.tight_layout()
    out["figure"] = save(fig, "takeup_gmm.png")
    return out


def cluster_ref() -> None:
    """Hopkins against references that have no clusters: a Gaussian cloud with the same mean and covariance, and the same data with the columns permuted.
    Against a uniform box any non-uniform single-mode cloud scores high, so the uniform reference alone says nothing about clusters."""
    from wxrisk.scs_eval import anomalise

    km10 = load_script("10_kmeans_baseline")
    df = km10.build_table(list(range(2000, 2023)))
    fit = (df.year <= 2016).to_numpy()
    tiers = km10.tier_columns(df)
    R = json.loads((OUT / "cluster.json").read_text())
    rng = np.random.default_rng(SEED)
    for tier, tag in (("B (T-3D..T-3h)", "B"), ("C (<T-3h)", "C")):
        Xf = anomalise(df, tiers[tier], pd.Series(fit))[fit]
        sub = Xf[rng.choice(len(Xf), 60000, replace=False)]
        gauss = rng.multivariate_normal(sub.mean(0), np.cov(sub, rowvar=False), size=len(sub)).astype(np.float32)
        ref = {"data": sub, "gaussian_same_covariance": gauss, "columns_permuted": diag.permute_columns(sub, SEED)}
        R[tag]["hopkins_reference"] = {k: float(np.mean([diag.hopkins(v, 1000, SEED + i) for i in range(5)])) for k, v in ref.items()}
        print(tag, R[tag]["hopkins_reference"], flush=True)
    dump("cluster", R)


# =====================================================================================================================
# main
# =====================================================================================================================


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("raw", "all"):
        raw()
    if cmd in ("cluster", "all"):
        cluster()
        cluster_ref()
    if cmd == "redraw":
        redraw()
    if cmd == "cluster-ref":
        cluster_ref()
    if cmd in ("regress", "all"):
        from importlib import (
            import_module,  # regression + render live in 44b to keep this file readable
        )

        sys.path.insert(0, str(ROOT / "scripts"))
        m = import_module("44b_raw_report_regress")
        m.regress()
    if cmd in ("render", "all"):
        from importlib import import_module

        sys.path.insert(0, str(ROOT / "scripts"))
        import_module("44b_raw_report_regress").render()


if __name__ == "__main__":
    main()
