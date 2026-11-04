"""Per-event hierarchical evaluation against a blind guess (audit follow-up, 2026-11-02): county ranking and block-group probabilities, Texas, scope S0.

State choice: Texas has the most S0 claims (140 k against 70-169 k), the most rows and the most test event days of the three pipelines (stages 44-45).
Unit = one test event day (a day in 2017-2023 with >= 1 claimed block group, no tropical-cyclone window), as stages 45 / 49.
Scores: blind guess (random, 20 draws), policies only (exposure), boosting own fit (script 85), CNN own fit (script 89, mean of the saved seeds).
Per event: county AUC and top-10% county capture; block-group AUC and top-5% capture ("win rate" = AUC, the probability that a claimed unit is ranked above an
unclaimed one; a blind guess is 0.5 / capture = k). Probability part: the CNN probability rescaled inside the event so its sum equals the event's claimed
block-group count (a level oracle: the level is NOT forecast, see MODELING_DECISIONS "level vs shape"), Brier skill against the uniform probability with the same sum.
Then worst / mode / mean / best summary over events and four example events (the event nearest each statistic) with a county map and a block-group map.
Inputs : cnn_tx/e45_rows.npz, e45_pred.npz, p89_S0_own_TX_s*.npz, static.npz, labels_extra/census/cb_2020_us_county_20m.zip
Output : audit/event_eval_tx.txt, audit/event_eval_tx.parquet, docs/figures/event_eval_tx_*.png
Run    : python scripts/94_event_hierarchy_eval.py
"""

from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402
from wxrisk import zstio  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT: list[str] = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mode_of(x: np.ndarray, bins: int = 10) -> float:
    h, e = np.histogram(x, bins=bins, range=(0, 1))
    i = int(h.argmax())
    return float((e[i] + e[i + 1]) / 2)


def main() -> None:
    e75, e85 = _load("e75", "75_task_metrics.py"), _load("e85", "85_state_group_baseline.py")
    d = wb2.cache_path("cnn_tx")
    R = e85.features("TX")
    yr = pd.DatetimeIndex(R["days"]).year.to_numpy()[R["day"]]
    rows = np.flatnonzero(yr >= 2017)
    T = e85.table(R, rows)
    T["bg"] = R["bg"][rows]
    base = zstio.load(d / "e45_pred.npz")
    T["boost"] = base["p_own"]
    ps = []
    for sd in range(5):
        f = d / f"p89_S0_own_TX_s{sd}.npz"
        if zstio.exists(f):
            z = zstio.load(f)
            assert (z["bg"] == T.bg.to_numpy()).all()
            ps.append(z["p"])
    say(f"CNN seeds: {len(ps)}; rows {len(T):,}; event days {T.sysid.nunique()}")
    T["cnn"] = np.mean(ps, axis=0)
    rng = np.random.default_rng(0)
    T["rand"] = rng.random(len(T))
    scores = {"blind guess": ("rand", "mean"), "policies only": ("n_pol", "sum"), "boosting": ("boost", "sum"), "CNN": ("cnn", "sum")}
    tabs = e75.event_table(T, scores)

    cols = {"cty_auc": ("c_auc", 0.5), "cty_top10": ("c_cap_n_0.1", 0.1), "bg_auc": ("bg_auc", 0.5), "bg_top5": ("bg_cap_n_0.05", 0.05)}
    res = {}
    for name, t in tabs.items():
        res[name] = t[[c for c, _ in cols.values()]].rename(columns={v[0]: k for k, v in cols.items()})
    ev = pd.concat(res, axis=1)

    # probability part: within-event rescaled CNN probability against the uniform probability with the same sum
    bs, num, den = {}, 0.0, 0.0
    for sid, g in T.groupby("sysid", sort=False):
        y = (g.n.to_numpy() > 0).astype(float)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        q = g.cnn.to_numpy().astype(float)
        p = np.clip(q * y.sum() / q.sum(), 1e-6, 1 - 1e-6)
        u = np.full(len(y), y.sum() / len(y))
        bs[sid] = 1 - ((p - y) ** 2).sum() / ((u - y) ** 2).sum()
        num += ((p - y) ** 2).sum()
        den += ((u - y) ** 2).sum()
    ev[("CNN", "bss_shape")] = pd.Series(bs)
    nclaim = T.assign(y=T.n > 0).groupby("sysid").y.sum()
    ev[("meta", "n_claimed_bg")] = nclaim.reindex(ev.index)
    ev.to_parquet(wb2.cache_path("audit") / "event_eval_tx.parquet")

    def summary(x: pd.Series, chance: float) -> str:
        x = x.dropna().to_numpy()
        return (f"worst {x.min():.3f} | p10 {np.percentile(x, 10):.3f} | mode {mode_of(x) if chance else np.nan:.3f} | median {np.median(x):.3f} | "
                f"mean {x.mean():.3f} | p90 {np.percentile(x, 90):.3f} | best {x.max():.3f} | wins>chance {np.mean(x > chance):.1%} (n={len(x)})")

    big = ev[("meta", "n_claimed_bg")] >= 10
    say(f"events with >= 10 claimed block groups: {int(big.sum())} of {len(ev)}; these hold {T[T.sysid.isin(ev.index[big])].n.sum() / T.n.sum():.1%} of test claims")
    for lab, m in (("ALL events", slice(None)), (">= 10 claimed block groups", big)):
        say(f"\n######## {lab}")
        for k, (src, chance) in cols.items():
            say(f"\n== {k} (blind guess = {chance})")
            for name in scores:
                say(f"  {name:14s} " + summary(ev.loc[m, (name, k)], chance))
    say("\n== within-event Brier skill of the rescaled CNN probability vs uniform probability (level given; chance = 0)")
    x = ev[("CNN", "bss_shape")].dropna()
    say("  CNN            " + f"worst {x.min():.3f} | p10 {np.percentile(x, 10):.3f} | median {x.median():.3f} | mean {x.mean():.3f} | "
        f"p90 {np.percentile(x, 90):.3f} | best {x.max():.3f} | wins>0 {np.mean(x > 0):.1%} (n={len(x)})")
    say(f"  pooled over all events (long-run steady state): BSS {1 - num / den:+.4f}")
    say("\n== head to head per event (CNN vs the comparison, block-group AUC)")
    for other in ("blind guess", "policies only", "boosting"):
        dlt = (ev[("CNN", "bg_auc")] - ev[(other, "bg_auc")]).dropna()
        say(f"  CNN beats {other:14s} in {np.mean(dlt > 0):.1%} of events (mean AUC diff {dlt.mean():+.3f})")
    (wb2.cache_path("audit") / "event_eval_tx.txt").write_text("\n".join(OUT) + "\n")
    plots(T, ev, big, "big", "events with >= 10 claimed block groups")
    plots(T, ev, pd.Series(True, index=ev.index), "all", "all events")


def counties_xy():
    import shapefile
    z = zipfile.ZipFile(wb2.cache_path("labels_extra") / "census" / "cb_2020_us_county_20m.zip")
    stem = [n for n in z.namelist() if n.endswith(".shp")][0][:-4]
    rd = shapefile.Reader(shp=io.BytesIO(z.read(stem + ".shp")), shx=io.BytesIO(z.read(stem + ".shx")), dbf=io.BytesIO(z.read(stem + ".dbf")))
    names = [f[0] for f in rd.fields[1:]]
    out = {}
    for sr in rd.iterShapeRecords():
        rec = dict(zip(names, sr.record, strict=True))
        if rec["STATEFP"] != "48":
            continue
        pts = np.array(sr.shape.points)
        parts = list(sr.shape.parts) + [len(pts)]
        out[rec["STATEFP"] + rec["COUNTYFP"]] = [pts[a:b] for a, b in zip(parts[:-1], parts[1:], strict=True)]
    return out


def plots(T: pd.DataFrame, ev: pd.DataFrame, m: pd.Series, tag: str, what: str) -> None:
    ev = ev[m]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    FIG.mkdir(parents=True, exist_ok=True)
    cols = {"blind guess": "#9aa0a6", "policies only": "#e8a33d", "boosting": "#4c78a8", "CNN": "#d1495b"}
    labels = {"cty_auc": "County ranking: win rate (AUC)", "bg_auc": "Block-group ranking: win rate (AUC)",
              "cty_top10": "Share of claims in the top-10% counties", "bg_top5": "Share of claims in the top-5% block groups"}
    chance = {"cty_auc": 0.5, "bg_auc": 0.5, "cty_top10": 0.1, "bg_top5": 0.05}

    # figure 1: distributions over events with worst / mode / mean / best
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, k in zip(axs.ravel(), labels, strict=True):
        for name in ("blind guess", "policies only", "CNN"):
            x = ev[(name, k)].dropna().to_numpy()
            ax.hist(x, bins=np.linspace(0, 1, 41), alpha=0.55, color=cols[name], label=name, density=True)
        x = ev[("CNN", k)].dropna().to_numpy()
        for lab, v, ls in (("worst", x.min(), ":"), ("mode", mode_of(x), "-"), ("mean", x.mean(), "--"), ("best", x.max(), ":")):
            ax.axvline(v, color="k", ls=ls, lw=1)
            ax.text(v, ax.get_ylim()[1] * 0.97, f" {lab}\n {v:.2f}", fontsize=8, va="top", rotation=0)
        ax.axvline(chance[k], color=cols["blind guess"], lw=2)
        ax.set_title(labels[k] + f"   (CNN beats chance in {np.mean(x > chance[k]):.0%} of {len(x)} events)", fontsize=10)
        ax.set_xlim(0, 1)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Texas, 2017-2023, {what} ({len(ev)}): CNN own fit against blind guess and exposure-only ranking\n"
                 "(thin black lines: CNN worst / mode / mean / best event; grey line = chance level)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / f"event_eval_tx_distribution_{tag}.png", dpi=140)
    plt.close(fig)

    # figure 2: win rate per event, sorted (CNN vs baselines)
    fig, ax = plt.subplots(figsize=(10, 4))
    for name in ("blind guess", "policies only", "boosting", "CNN"):
        x = np.sort(ev[(name, "bg_auc")].dropna().to_numpy())
        ax.plot(np.linspace(0, 100, len(x)), x, color=cols[name], label=name, lw=2 if name == "CNN" else 1.2)
    ax.axhline(0.5, color="grey", lw=1)
    ax.set_xlabel("event percentile (events sorted by their own block-group AUC)")
    ax.set_ylabel("block-group AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / f"event_eval_tx_sorted_{tag}.png", dpi=140)
    plt.close(fig)

    # figure 3: example events (nearest to worst / mode / mean / best of the CNN block-group AUC)
    S = zstio.load(wb2.cache_path("cnn_tx") / "static.npz", allow_pickle=True)
    lat, lon = S["lat4"][S["b4_y"]], S["lon4"][S["b4_x"]]
    bg_ll = {int(i): (lon[i], lat[i]) for i in range(len(lat))}
    cty = counties_xy()
    x = ev[("CNN", "bg_auc")].dropna()
    pick = {"worst": x.idxmin(), "mode": (x - mode_of(x.to_numpy())).abs().idxmin(), "mean": (x - x.mean()).abs().idxmin(), "best": x.idxmax()}
    fig, axs = plt.subplots(2, 4, figsize=(20, 9))
    for j, (lab, sid) in enumerate(pick.items()):
        g = T[T.sysid == sid]
        c = g.groupby("cty").agg(n=("n", "sum"), s=("cnn", "sum"))
        cn = c.n.rank(pct=True)
        ax = axs[0, j]
        sc = c.s.rank(pct=True)
        polys, colr = [], []
        for code, parts in cty.items():
            for p in parts:
                polys.append(p)
                colr.append(sc.get(code, np.nan))
        pc = PolyCollection(polys, array=np.array(colr, float), cmap="YlOrRd", edgecolors="#555", linewidths=0.3)
        ax.add_collection(pc)
        for code, parts in cty.items():
            if code in c.index and c.n[code] > 0:
                p = max(parts, key=len)
                ax.plot(*p.mean(0), "o", mfc="none", mec="#1a5fb4", ms=3 + 3 * np.log1p(c.n[code]), mew=1.4)
        ax.autoscale()
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{lab}: {sid}\ncounty rank by predicted claims (colour); blue rings = counties with claims", fontsize=9)
        ax = axs[1, j]
        ax.add_collection(PolyCollection([p for ps_ in cty.values() for p in ps_], facecolors="none", edgecolors="#8a8a8a", linewidths=0.4, zorder=0))
        xy = np.array([bg_ll[b] for b in g.bg.to_numpy()])
        q = g.cnn.to_numpy()
        y = g.n.to_numpy() > 0
        p = np.clip(q * y.sum() / q.sum(), 0, 1)
        o = np.argsort(p)
        ax.scatter(xy[o, 0], xy[o, 1], c=p[o], s=3, zorder=2, cmap="viridis", norm=matplotlib.colors.LogNorm(1e-4, 1))
        ax.scatter(xy[y, 0], xy[y, 1], s=30, facecolors="none", edgecolors="red", lw=1.0, zorder=3)
        ax.autoscale()
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"AUC {x[sid]:.2f}, {int(y.sum())} claimed block groups of {len(y):,}\ncolour: CNN probability (level scaled to the count), red = claimed", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / f"event_eval_tx_examples_{tag}.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
