"""Task scorecard for the Texas CNN product, observed and forecast mode (RESULTS stage 55): the target-metric specification of docs/MODEL_AUDIT.md section 5.

Task (Impact Forecasting.pdf): inside an event rank units (block groups, counties) by forecast impact so that limited resources go to the highest; the probability
is an auxiliary product. Unit = block group x day (stage 38 decision), event = test day with >= 1 claimed block group; counties = sum of block-group scores.
Runs: (A) observed rain, own-state fit of script 89 (2 seeds), boosting of script 85, policies only, rain only (4 km pixel peak), blind guess; test 2017-2023, S0.
      (B) forecast mode `fc + expo` of script 91 (2 seeds), policies only, blind guess, observed `rain + expo` as reference; test 2017-2019, S0, all days.
Primary (ranking): block-group AUC (pair weighted), capture of claim count / paid amount in the top 1 / 5 / 10 / 20% block groups, NDCG at 5% (paid), county AUC,
capture in the top 5 / 10 / 20% counties; 95% intervals by a bootstrap over events (script 75); paired difference to policies only.
Auxiliary (probability): expected / observed claimed block groups per event, calibration slope and reliability, share of claims outside the top 5%, Brier skill
(pooled test-period rate as baseline; re-centred with the trailing three test years where they exist; same-year re-centring = oracle).
Inputs : cnn_tx/e45_rows.npz, e45_pred.npz, p89_S0_own_TX_s*.npz; cnn/ rows and ms_fc+expo / ms_rain+expo (script 91)
Output : audit/task_scorecard.txt, audit/task_scorecard.csv, docs/figures/scorecard_tx.png
Run    : python scripts/96_task_scorecard.py
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
KGRID = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_observed(e85):
    d = wb2.cache_path("cnn_tx")
    R = e85.features("TX")
    yr = pd.DatetimeIndex(R["days"]).year.to_numpy()[R["day"]]
    rows = np.flatnonzero(yr >= 2017)
    T = e85.table(R, rows)
    T["bg"] = R["bg"][rows]
    T["rain"] = R["f"][rows, 0]
    T["boost"] = zstio.load(d / "e45_pred.npz")["p_own"]
    ps = [zstio.load(f)["p"] for f in zstio.glob(d, "p89_S0_own_TX_s[0-9]*.npz")]
    T["cnn"] = np.mean(ps, axis=0)
    T["rand"] = np.random.default_rng(0).random(len(T))
    scores = {"blind guess": ("rand", "mean"), "policies only": ("n_pol", "sum"), "rain only": ("rain", "sum"), "boosting": ("boost", "sum"), "CNN (observed)": ("cnn", "sum")}
    return T, scores, "CNN (observed)"


def build_forecast():
    D = wb2.cache_path("cnn")
    S = zstio.load(D / "static.npz", allow_pickle=True)
    days = pd.DatetimeIndex(S["days"])
    ptr = zstio.load(D / "rows_ptr.npy")
    test = np.flatnonzero(days.year >= 2017)
    lo, hi = ptr[test[0]], ptr[test[-1] + 1]
    dr = np.repeat(np.arange(len(days)), np.diff(ptr))[lo:hi]
    bg = zstio.load(D / "rows_bg.npy")[lo:hi]
    ctys = np.array([str(c) for c in S["ctys"]])
    T = pd.DataFrame({"sysid": days[dr].strftime("%Y%m%d"), "cty": ctys[S["cty_idx"].astype(np.int64)[bg]], "bg": bg, "n": zstio.load(D / "rows_n.npy")[lo:hi],
                      "paid": zstio.load(D / "rows_paid.npy")[lo:hi], "n_pol": np.asarray(zstio.load(D / "rows_tab.npy", mmap_mode="r")[lo:hi, 8], np.float32)})
    T["year"] = T.sysid.str[:4].astype(int)
    for key, tag in (("fc", "fc+expo"), ("obs", "rain+expo")):
        T[key] = np.mean([zstio.load(f) for f in sorted(D.glob(f"ms_{tag}_s[0-9]*.npy"))], axis=0)
    T["rand"] = np.random.default_rng(0).random(len(T))
    scores = {"blind guess": ("rand", "mean"), "policies only": ("n_pol", "sum"), "CNN (forecast)": ("fc", "sum"), "CNN (observed, ref.)": ("obs", "sum")}
    return T, scores, "CNN (forecast)"


def gain_curves(T, scores):
    """Mean over events of the share of the event's claim count inside the top k of the block groups (k grid)."""
    out = {n: [] for n in scores}
    rng = np.random.default_rng(1)
    for _, g in T.groupby("sysid", sort=False):
        n = g.n.to_numpy()
        if n.sum() <= 0 or g.cty.nunique() < 2 or (n > 0).all():
            continue
        m = np.maximum(np.ceil(KGRID * len(n)).astype(int), 1)
        for name, (col, _) in scores.items():
            s = g[col].to_numpy().astype(float)
            o = np.argsort(-(s + rng.random(len(s)) * 1e-12))
            cs = np.cumsum(n[o]) / n.sum()
            out[name].append(cs[m - 1])
    return {k: np.nanmean(v, axis=0) for k, v in out.items()}


def calibration(y, p, nb=40):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    lg = np.log(p / (1 - p))
    q = np.unique(np.quantile(lg[:: max(len(lg) // 2_000_000, 1)], np.linspace(0, 1, nb + 1)))
    b = np.clip(np.digitize(lg, q[1:-1]), 0, len(q) - 2)
    n = np.bincount(b, minlength=len(q) - 1).astype(float)
    sy = np.bincount(b, weights=y, minlength=len(q) - 1)
    sp = np.bincount(b, weights=p, minlength=len(q) - 1)
    sl = np.bincount(b, weights=lg, minlength=len(q) - 1)
    ok = n > 0
    from scipy.optimize import minimize

    x, k, m = sl[ok] / n[ok], sy[ok], n[ok]

    def nll(w):
        z = w[0] + w[1] * x
        return float((m * np.logaddexp(0, z) - k * z).sum())

    w = minimize(nll, [0.0, 1.0], method="Nelder-Mead").x
    return {"slope": float(w[1]), "intercept": float(w[0]), "mean_p": sp[ok] / n[ok], "obs": k / m, "n": m}


def brier_block(T, col, years):
    y = (T.n.to_numpy() > 0).astype(np.float64)
    p = np.clip(T[col].to_numpy().astype(np.float64), 1e-7, 1 - 1e-7)
    yr = T.year.to_numpy()
    r_all = y.mean()
    res = {"pred": 1 - ((p - y) ** 2).mean() / ((r_all - y) ** 2).mean()}
    rows = []
    for Y in years:
        m = yr == Y
        # same-year oracle: scale p so that the mean matches the year
        po = np.clip(p[m] * y[m].mean() / p[m].mean(), 1e-7, 1 - 1e-7)
        bs_o = 1 - ((po - y[m]) ** 2).mean() / ((y[m].mean() - y[m]) ** 2).mean()
        t = np.isin(yr, [Y - 3, Y - 2, Y - 1])
        if t.sum() > 0 and all((yr == k).any() for k in (Y - 3, Y - 2, Y - 1)):
            r_t = y[t].mean()
            pt = np.clip(p[m] * r_t / p[t].mean(), 1e-7, 1 - 1e-7)
            bs_t = 1 - ((pt - y[m]) ** 2).mean() / ((r_t - y[m]) ** 2).mean()
        else:
            bs_t = np.nan
        rows.append((Y, bs_t, bs_o))
    res["years"] = rows
    return res


def run(label, T, scores, model, e75, rep):
    say(f"\n################ {label}: rows {len(T):,}, days {T.sysid.nunique()}")
    tabs = e75.event_table(T, scores)
    names = list(scores)
    base = "policies only"
    n_bg, n_c = int(tabs[model].bg_auc.notna().sum()), int(tabs[model].c_auc.notna().sum())
    say(f"events scored: block-group metrics {n_bg}, county metrics {n_c}")
    metrics = [("BG AUC (pair-weighted)", "bg_auc", "bg_w")] + [(f"BG capture of claim count, top {int(k * 100)}%", f"bg_cap_n_{k}", None) for k in (0.01, 0.05, 0.1, 0.2)] + [
        ("BG capture of paid amount, top 5%", "bg_cap_p_0.05", None), ("BG NDCG@5% (paid)", "bg_ndcg_p", None), ("county AUC (pair-weighted)", "c_auc", "c_w")] + [
        (f"county capture of claim count, top {int(k * 100)}%", f"c_cap_n_{k}", None) for k in (0.05, 0.1, 0.2)] + [("county Spearman with claim count", "c_rho", None)]
    rows = []
    for lab, col, w in metrics:
        res, dd = e75.boot({n: t for n, t in tabs.items()}, col, w, ref=base)
        for n in names:
            rows.append({"run": label, "metric": lab, "score": n, "point": res[n][0], "lo": res[n][1], "hi": res[n][2],
                         "diff_vs_policies": dd[n][0] if n in dd else np.nan, "diff_lo": dd[n][1] if n in dd else np.nan, "diff_hi": dd[n][2] if n in dd else np.nan})
        say(f"  {lab}:")
        for n in names:
            r = rows[-len(names) + names.index(n)]
            extra = f"   vs policies {r['diff_vs_policies']:+.3f} [{r['diff_lo']:+.3f}, {r['diff_hi']:+.3f}]" if n != base and n in dd else ""
            say(f"      {n:22s} {r['point']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]{extra}")
    # by year
    say("  by year (model: BG AUC | BG top-5% capture; policies only in brackets)")
    by_year = []
    for yv in sorted(T.year.unique()):
        ev = [i for i in tabs[model].index if i[:4] == str(yv)]
        a = e75.pooled(tabs[model].loc[ev], "bg_auc", "bg_w")
        c = e75.pooled(tabs[model].loc[ev], "bg_cap_n_0.05", None)
        a0 = e75.pooled(tabs[base].loc[ev], "bg_auc", "bg_w")
        c0 = e75.pooled(tabs[base].loc[ev], "bg_cap_n_0.05", None)
        by_year.append((yv, a, c, a0, c0, len(ev)))
        say(f"    {yv}: {a:.3f} | {c:.3f}   ({a0:.3f} | {c0:.3f}); events {len(ev)}")
    # auxiliary
    col = scores[model][0]
    y = (T.n.to_numpy() > 0).astype(np.float64)
    p = T[col].to_numpy()
    cal = calibration(y, p)
    exp_obs = T.assign(p=p, y=y).groupby("sysid").agg(exp=("p", "sum"), obs=("y", "sum"))
    exp_obs = exp_obs[exp_obs.obs > 0]
    ratio = exp_obs.exp / exp_obs.obs
    from scipy.stats import spearmanr

    say("  auxiliary (probability):")
    say(f"    calibration slope {cal['slope']:.3f} (intercept {cal['intercept']:+.2f}); mean p {p.mean():.2e}, observed rate {y.mean():.2e} (ratio {p.mean() / y.mean():.2f})")
    say(f"    expected / observed claimed block groups per event: median {ratio.median():.2f}, p10 {ratio.quantile(.1):.2f}, p90 {ratio.quantile(.9):.2f}; "
        f"Spearman(expected, observed) {spearmanr(exp_obs.exp, exp_obs.obs)[0]:.3f} over {len(exp_obs)} events")
    for lo_, hi_ in ((1, 2), (3, 9), (10, 29), (30, 10 ** 9)):
        s = exp_obs[(exp_obs.obs >= lo_) & (exp_obs.obs <= hi_)]
        if len(s) > 3:
            say(f"      observed {lo_}-{hi_ if hi_ < 10 ** 9 else '+'}: events {len(s)}, median ratio {(s.exp / s.obs).median():.2f}, Spearman {spearmanr(s.exp, s.obs)[0]:.3f}")
    outside = 1 - tabs[model]["bg_cap_n_0.05"].dropna()
    say(f"    share of the event's claims outside the top-5% block groups: mean {outside.mean():.3f}, median {outside.median():.3f} (blind guess would be 0.95)")
    bs = brier_block(T, col, sorted(T.year.unique()))
    say(f"    Brier skill, as predicted (baseline = test-period rate): {bs['pred']:+.4f}")
    for Y, bt, bo in bs["years"]:
        say(f"      {Y}: trailing-3y re-centred {bt:+.4f}   same-year oracle {bo:+.4f}")
    rep[label] = {"tabs": tabs, "model": model, "cal": cal, "ratio": exp_obs.assign(r=ratio), "gain": gain_curves(T, scores), "by_year": by_year, "scores": scores}
    return rows


def plot(rep):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colr = {"blind guess": "#9aa0a6", "policies only": "#e8a33d", "rain only": "#8d6a9f", "boosting": "#4c78a8", "CNN (observed)": "#d1495b",
            "CNN (forecast)": "#2a9d8f", "CNN (observed, ref.)": "#d1495b"}
    fig, axs = plt.subplots(3, 3, figsize=(20, 16))
    for r, (label, R) in enumerate(rep.items()):
        ax = axs[r, 0]
        for n, v in R["gain"].items():
            ax.plot(KGRID * 100, v * 100, marker="o", ms=3, color=colr[n], label=n, lw=2.2 if n == R["model"] else 1.3, ls="--" if n == "blind guess" else "-")
        ax.set_xscale("log")
        ax.set_xlabel("share of block groups reviewed (%)")
        ax.set_ylabel("share of the event's claims found (%)")
        ax.set_title(f"{label}: capture curve (mean over events)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axs[r, 1]
        names = list(R["scores"])
        mets = [("bg_auc", "bg_w", "BG AUC"), ("bg_cap_n_0.05", None, "BG top-5%\ncapture"), ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10%\ncapture")]
        e75 = _load("e75", "75_task_metrics.py")
        w = 0.8 / len(names)
        for i, n in enumerate(names):
            pts, los, his = [], [], []
            for col, wc, _ in mets:
                res, _ = e75.boot({n: R["tabs"][n]}, col, wc)
                pts.append(res[n][0]); los.append(res[n][0] - res[n][1]); his.append(res[n][2] - res[n][0])
            ax.bar(np.arange(len(mets)) + i * w, pts, w, yerr=[los, his], color=colr[n], label=n, capsize=2)
        ax.set_xticks(np.arange(len(mets)) + 0.4 - w / 2)
        ax.set_xticklabels([m[2] for m in mets], fontsize=9)
        ax.axhline(0.5, color="#9aa0a6", lw=0.8)
        ax.set_title(f"{label}: primary metrics with 95% intervals over events", fontsize=10)
        ax.set_ylim(0, 1)
        ax = axs[r, 2]
        yrs = [b[0] for b in R["by_year"]]
        ax.plot(yrs, [b[1] for b in R["by_year"]], "-o", color=colr[R["model"]], label=f"{R['model']} BG AUC")
        ax.plot(yrs, [b[3] for b in R["by_year"]], "-o", color=colr["policies only"], label="policies only BG AUC")
        ax.plot(yrs, [b[2] for b in R["by_year"]], "--s", color=colr[R["model"]], label=f"{R['model']} top-5% capture")
        ax.plot(yrs, [b[4] for b in R["by_year"]], "--s", color=colr["policies only"], label="policies only top-5% capture")
        ax.axhline(0.5, color="#9aa0a6", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xticks(yrs)
        ax.set_title(f"{label}: by year", fontsize=10)
        ax.legend(fontsize=8)
    # probability row: reliability, expected vs observed for each run
    for r, (label, R) in enumerate(list(rep.items())):
        pass
    ax = axs[2, 0]
    for label, R in rep.items():
        c = R["cal"]
        ax.loglog(c["mean_p"], np.maximum(c["obs"], 1e-7), "-o", ms=3, label=f"{label} (slope {c['slope']:.2f})", color=colr[R["model"]])
    ax.plot([1e-6, 1], [1e-6, 1], color="k", lw=0.8)
    ax.set_xlabel("predicted probability (bin mean)")
    ax.set_ylabel("observed claim rate")
    ax.set_title("Reliability (all rows; the diagonal is perfect)", fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[2, 1]
    for label, R in rep.items():
        eo = R["ratio"]
        ax.scatter(eo.obs, eo.exp, s=8, alpha=0.4, color=colr[R["model"]], label=label)
    lim = [0.8, 1e4]
    ax.plot(lim, lim, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("observed claimed block groups in the event")
    ax.set_ylabel("expected (sum of p)")
    ax.set_title("Expected vs observed claimed block groups per event", fontsize=10)
    ax.legend(fontsize=8)
    ax = axs[2, 2]
    ax.axis("off")
    txt = "Reading the figure\n\n- capture curve: the further above the diagonal-like blind guess,\n  the more claims a fixed review budget finds\n- bars: model vs baselines, 95% intervals by event bootstrap\n- by year: stability across test years\n- reliability: row-level probabilities sit close to the\n  diagonal (slope 0.92 observed, 1.13 forecast)\n- expected vs observed: the per-event total is too low for\n  the big events (median ratio 0.36 / 0.09 at 30+ claimed\n  block groups), so the level is not a forecast of event size\n  (numbers in audit/task_scorecard.txt)"
    ax.text(0, 1, txt, va="top", fontsize=11)
    fig.suptitle("Texas flood impact ranking: task scorecard (rows 1-2: observed rain 2017-2023, forecast mode 2017-2019)", fontsize=13)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "scorecard_tx.png", dpi=110)
    plt.close(fig)


def main() -> None:
    e75, e85 = _load("e75", "75_task_metrics.py"), _load("e85", "85_state_group_baseline.py")
    rep, rows = {}, []
    for label, builder in (("A observed rain 2017-2023", lambda: build_observed(e85)), ("B forecast mode 2017-2019", build_forecast)):
        T, scores, model = builder()
        if "year" not in T:
            T["year"] = T.sysid.str[:4].astype(int)
        rows += run(label, T, scores, model, e75, rep)
        del T
    pd.DataFrame(rows).to_csv(wb2.cache_path("audit") / "task_scorecard.csv", index=False)
    (wb2.cache_path("audit") / "task_scorecard.txt").write_text("\n".join(OUT) + "\n")
    plot(rep)


if __name__ == "__main__":
    main()
