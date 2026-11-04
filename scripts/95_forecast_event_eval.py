"""Forecast-mode per-event evaluation against a blind guess, Texas (follow-up of script 94; RESULTS stage 54).

Same events and metrics as script 94, but the scores are those available at issue time: the multi-source network `fc + expo` (GEFSv12 ensemble fields + exposure,
script 91, 2 seeds; no observed rain) and, as the best-case reference on the same rows, `rain + expo` (observed gridMET). Rows are those of script 81's eval: all days
2017-2019 (the GEFS reforecast ends in 2019), scope S0, all block groups with a policy, so days WITHOUT claims are included in the rows.
Event-level metrics (as 94) are conditional on a day having >= 1 claimed block group. Extra, all-days check: can the forecast tell the days on which
>= 10 block groups claim from the other days (day score = sum of block-group probabilities; AUC over all 1,091 days).
Inputs : cnn/ rows (script 80), ms_fc+expo_s*.npy, ms_rain+expo_s*.npy (script 91), static.npz
Output : audit/forecast_event_eval_tx.txt, docs/figures/forecast_eval_tx_*.png
Run    : python scripts/95_forecast_event_eval.py
"""

from __future__ import annotations

import importlib.util
import sys
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


def main() -> None:
    from sklearn.metrics import roc_auc_score

    e75, e94 = _load("e75", "75_task_metrics.py"), _load("e94", "94_event_hierarchy_eval.py")
    D = wb2.cache_path("cnn")
    S = zstio.load(D / "static.npz", allow_pickle=True)
    days = pd.DatetimeIndex(S["days"])
    ptr = zstio.load(D / "rows_ptr.npy")
    test = np.flatnonzero(days.year >= 2017)
    lo, hi = ptr[test[0]], ptr[test[-1] + 1]
    day_of_row = np.repeat(np.arange(len(days)), np.diff(ptr))[lo:hi]
    bg = zstio.load(D / "rows_bg.npy")[lo:hi]
    ctys = np.array([str(c) for c in S["ctys"]])
    T = pd.DataFrame({"sysid": days[day_of_row].strftime("%Y%m%d"), "cty": ctys[S["cty_idx"].astype(np.int64)[bg]], "bg": bg,
                      "n": zstio.load(D / "rows_n.npy")[lo:hi], "paid": zstio.load(D / "rows_paid.npy")[lo:hi],
                      "n_pol": np.asarray(zstio.load(D / "rows_tab.npy", mmap_mode="r")[lo:hi, 8], np.float32)})
    ref = pd.read_parquet(D / "e1_test_rows.parquet", columns=["sysid", "cty", "n", "n_pol"])
    assert len(ref) == len(T) and (ref.n.to_numpy() == T.n.to_numpy()).all() and (ref.n_pol.to_numpy() == T.n_pol.to_numpy()).all(), "row alignment with e1_test_rows"
    for key, tag in (("fc", "fc+expo"), ("obs", "rain+expo")):
        fs = sorted(D.glob(f"ms_{tag}_s[0-9]*.npy"))
        T[key] = np.mean([zstio.load(f) for f in fs], axis=0)
        say(f"{tag}: {len(fs)} seeds")
    T["rand"] = np.random.default_rng(0).random(len(T))
    scores = {"blind guess": ("rand", "mean"), "policies only": ("n_pol", "sum"), "forecast (fc+expo)": ("fc", "sum"), "observed (rain+expo)": ("obs", "sum")}
    tabs = e75.event_table(T, scores)
    cols = {"cty_auc": "c_auc", "cty_top10": "c_cap_n_0.1", "bg_auc": "bg_auc", "bg_top5": "bg_cap_n_0.05"}
    chance = {"cty_auc": 0.5, "cty_top10": 0.1, "bg_auc": 0.5, "bg_top5": 0.05}
    ev = pd.concat({n: t[list(cols.values())].rename(columns={v: k for k, v in cols.items()}) for n, t in tabs.items()}, axis=1)
    g = T.assign(y=T.n > 0).groupby("sysid")
    nclaim = g.y.sum()
    ev[("meta", "n")] = nclaim.reindex(ev.index)

    # level-given Brier skill of the forecast probability inside the event
    bs, num, den = {}, 0.0, 0.0
    for sid, gg in T.groupby("sysid", sort=False):
        y = (gg.n.to_numpy() > 0).astype(float)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        q = gg.fc.to_numpy().astype(float)
        p = np.clip(q * y.sum() / q.sum(), 1e-6, 1 - 1e-6)
        u = np.full(len(y), y.sum() / len(y))
        bs[sid] = 1 - ((p - y) ** 2).sum() / ((u - y) ** 2).sum()
        num += ((p - y) ** 2).sum()
        den += ((u - y) ** 2).sum()
    ev[("forecast (fc+expo)", "bss_shape")] = pd.Series(bs)
    ev.to_parquet(wb2.cache_path("audit") / "forecast_event_eval_tx.parquet")

    big = ev[("meta", "n")] >= 10
    say(f"\nrows {len(T):,}; days {T.sysid.nunique()}; events (>= 1 claimed BG and not all) {len(ev)}; with >= 10 claimed BGs {int(big.sum())} "
        f"({T[T.sysid.isin(ev.index[big])].n.sum() / T.n.sum():.1%} of the claims)")

    def summ(x, ch):
        x = x.dropna().to_numpy()
        return (f"worst {x.min():.3f} | p10 {np.percentile(x, 10):.3f} | mode {e94.mode_of(x):.3f} | median {np.median(x):.3f} | mean {x.mean():.3f} | "
                f"p90 {np.percentile(x, 90):.3f} | best {x.max():.3f} | wins>chance {np.mean(x > ch):.1%} (n={len(x)})")

    for lab, m in (("ALL events", slice(None)), (">= 10 claimed block groups", big)):
        say(f"\n######## {lab}")
        for k in cols:
            say(f"\n== {k} (blind guess = {chance[k]})")
            for n in scores:
                say(f"  {n:22s}" + summ(ev.loc[m, (n, k)], chance[k]))
    x = ev.loc[big, ("forecast (fc+expo)", "bss_shape")].dropna()
    say(f"\n== forecast probability, level given: within-event Brier skill vs uniform (>= 10 claims): median {x.median():.4f} mean {x.mean():+.4f} "
        f"worst {x.min():.3f} best {x.max():.3f} wins>0 {np.mean(x > 0):.1%}; pooled over all events {1 - num / den:+.4f}")
    say("\n== head to head per event, block-group AUC (>= 10 claims)")
    for other in ("blind guess", "policies only", "observed (rain+expo)"):
        dlt = (ev.loc[big, ("forecast (fc+expo)", "bg_auc")] - ev.loc[big, (other, "bg_auc")]).dropna()
        say(f"  forecast beats {other:22s} in {np.mean(dlt > 0):.1%} of events (mean AUC diff {dlt.mean():+.3f})")

    # all-days check: can the forecast flag the days with >= 10 claimed block groups?
    sc = T.groupby("sysid").agg(fc=("fc", "sum"), obs=("obs", "sum"), pol=("n_pol", "sum"), rand=("rand", "sum"))
    yday = (nclaim.reindex(sc.index) >= 10).to_numpy()
    say(f"\n== all {len(sc)} days: AUC for flagging the {int(yday.sum())} days with >= 10 claimed block groups (day score = sum of block-group probabilities; chance 0.5)")
    for c, lab in (("rand", "blind guess"), ("pol", "policies only"), ("fc", "forecast"), ("obs", "observed")):
        say(f"  {lab:14s} {roc_auc_score(yday, sc[c]):.3f}")
    (wb2.cache_path("audit") / "forecast_event_eval_tx.txt").write_text("\n".join(OUT) + "\n")
    plots(T, ev, big, e94)


def plots(T, ev, big, e94) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    FIG.mkdir(parents=True, exist_ok=True)
    colr = {"blind guess": "#9aa0a6", "policies only": "#e8a33d", "forecast (fc+expo)": "#2a9d8f", "observed (rain+expo)": "#d1495b"}
    labels = {"cty_auc": "County ranking: win rate (AUC)", "bg_auc": "Block-group ranking: win rate (AUC)",
              "cty_top10": "Share of claims in the top-10% counties", "bg_top5": "Share of claims in the top-5% block groups"}
    chance = {"cty_auc": 0.5, "bg_auc": 0.5, "cty_top10": 0.1, "bg_top5": 0.05}
    F = "forecast (fc+expo)"
    e = ev[big]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    for ax, k in zip(axs.ravel(), labels, strict=True):
        bins = np.linspace(0, 1, 41)
        for n in ("blind guess", "policies only", F):
            ax.hist(e[(n, k)].dropna(), bins=bins, alpha=0.55, color=colr[n], label=n, density=True)
        ax.hist(e[("observed (rain+expo)", k)].dropna(), bins=bins, histtype="step", color=colr["observed (rain+expo)"], lw=1.6, label="observed rain (best case)", density=True)
        x = e[(F, k)].dropna().to_numpy()
        for lab, v, ls in (("worst", x.min(), ":"), ("mode", e94.mode_of(x), "-"), ("mean", x.mean(), "--"), ("best", x.max(), ":")):
            ax.axvline(v, color="k", ls=ls, lw=1)
            ax.text(v, ax.get_ylim()[1] * 0.97, f" {lab}\n {v:.2f}", fontsize=8, va="top")
        ax.axvline(chance[k], color=colr["blind guess"], lw=2)
        ax.set_title(labels[k] + f"   (forecast beats chance in {np.mean(x > chance[k]):.0%} of {len(x)} events)", fontsize=10)
        ax.set_xlim(0, 1)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"Texas 2017-2019, FORECAST mode (GEFS + exposure), events with >= 10 claimed block groups ({len(e)})\n"
                 "thin black lines: forecast worst / mode / mean / best event; grey line = chance; red outline = observed-rain model on the same events", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_eval_tx_distribution_big.png", dpi=140)
    plt.close(fig)

    S = zstio.load(wb2.cache_path("cnn") / "static.npz", allow_pickle=True)
    lat, lon = S["lat4"][S["b4_y"]], S["lon4"][S["b4_x"]]
    cty = e94.counties_xy()
    outline = [p for ps_ in cty.values() for p in ps_]
    x = e[(F, "bg_auc")].dropna()
    pick = {"worst": x.idxmin(), "mode": (x - e94.mode_of(x.to_numpy())).abs().idxmin(), "mean": (x - x.mean()).abs().idxmin(), "best": x.idxmax()}
    fig, axs = plt.subplots(3, 4, figsize=(20, 13.5))
    for j, (lab, sid) in enumerate(pick.items()):
        g = T[T.sysid == sid]
        c = g.groupby("cty").agg(n=("n", "sum"), s=("fc", "sum"))
        sc = c.s.rank(pct=True)
        ax = axs[0, j]
        polys = [p for code, parts in cty.items() for p in parts]
        cc = [sc.get(code, np.nan) for code, parts in cty.items() for _ in parts]
        ax.add_collection(PolyCollection(polys, array=np.array(cc, float), cmap="YlOrRd", edgecolors="#555", linewidths=0.3))
        for code, parts in cty.items():
            if code in c.index and c.n[code] > 0:
                ax.plot(*max(parts, key=len).mean(0), "o", mfc="none", mec="#1a5fb4", ms=3 + 3 * np.log1p(c.n[code]), mew=1.4)
        ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{lab}: {sid}  (forecast)\ncounty rank by forecast claims (colour); blue rings = counties with claims", fontsize=9)
        y = g.n.to_numpy() > 0
        xy = np.column_stack([lon[g.bg.to_numpy()], lat[g.bg.to_numpy()]])
        for r, (col, ttl) in enumerate((("fc", "FORECAST"), ("obs", "observed rain (best case)")), start=1):
            ax = axs[r, j]
            q = g[col].to_numpy()
            p = np.clip(q * y.sum() / q.sum(), 0, 1)
            o = np.argsort(p)
            ax.add_collection(PolyCollection(outline, facecolors="none", edgecolors="#8a8a8a", linewidths=0.4, zorder=0))
            ax.scatter(xy[o, 0], xy[o, 1], c=p[o], s=3, cmap="viridis", norm=matplotlib.colors.LogNorm(1e-4, 1), zorder=2)
            ax.scatter(xy[y, 0], xy[y, 1], s=30, facecolors="none", edgecolors="red", lw=1.0, zorder=3)
            ax.autoscale(); ax.set_aspect("equal"); ax.axis("off")
            auc = e.loc[sid, (F if col == "fc" else "observed (rain+expo)", "bg_auc")]
            ax.set_title(f"{ttl}: AUC {auc:.2f}, {int(y.sum())} claimed BGs of {len(y):,}\ncolour = probability scaled to the count, red = claimed", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_eval_tx_examples_big.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
