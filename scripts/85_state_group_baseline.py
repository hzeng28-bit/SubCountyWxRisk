"""Boosting baseline on the observed-rain rows of the state group, with the state-held-out check (RESULTS stage 45).

Rows of `cnn_<st>/` (script 84), scope S0 (rows inside a tropical-cyclone window dropped, label = claims without a tropical-named event),
train 2010-2016, test 2017-2023 (gridMET reaches 2023, so the test period is longer than the 2017-2019 of the forecast line).
Features: the 9 tabular columns of script 80 + the observed rain at the block group's 4 km pixel (peak, 2-day sum), the 5 x 5 pixel maximum of the peak
(about 20 km) and the county maximum / mean of the block-group pixel peaks of the day. Negatives of the training days are subsampled 1 in 10 (weight 10);
the iteration count is chosen on blocked validation months (1 in 7) of the training years of the SAME fit, then refitted on train + validation.
Test rows keep only days with >= 1 claimed block group (an event without a claim has no ranking metric).

Fits (rows = block group x day; test each on TX, LA, FL separately; events = days):
  own      trained on the test state alone
  tx       trained on Texas only (transfer of the pilot's state)
  pool     trained on TX + LA + FL
  loso     trained on the two OTHER states (state-held-out)
Metrics as scripts 75 / 81: cross-county block-group AUC, top-5% capture of claim count, NDCG5 with paid amount, county AUC, top-10% county capture, with a
bootstrap over days; baseline "policies only".
Inputs : cnn_tx / cnn_la / cnn_fl (script 84)
Output : $WXRISK_DATA/audit/state_group_baseline.txt, cnn_<st>/e45_rows.npz
Run    : python scripts/85_state_group_baseline.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk import zstio  # noqa: E402

STATES = ["TX", "LA", "FL"]
LOG: list[str] = []


def say(s=""):
    print(s, flush=True)
    LOG.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def features(st: str) -> dict:
    """Rows kept for the baseline of one state, with their features; cached in cnn_<st>/e45_rows.npz."""
    d = wb2.cache_path(f"cnn_{st.lower()}")
    cache = d / "e45_rows.npz"
    if zstio.exists(cache):
        return dict(zstio.load(cache, allow_pickle=True))
    S = zstio.load(d / "static.npz", allow_pickle=True)
    days = pd.DatetimeIndex(S["days"])
    ptr, rbg = zstio.load(d / "rows_ptr.npy"), zstio.load(d / "rows_bg.npy")
    tab, tc = zstio.load(d / "rows_tab.npy", mmap_mode="r"), zstio.load(d / "rows_tc.npy")
    n0, nall, paid = zstio.load(d / "rows_n0.npy"), zstio.load(d / "rows_nall.npy"), zstio.load(d / "rows_paid0.npy")
    O = zstio.load(d / "obs4k_day_maps.npy", mmap_mode="r")
    by, bx, cty = S["b4_y"].astype(np.int64), S["b4_x"].astype(np.int64), S["cty_idx"].astype(np.int64)
    nc = len(S["ctys"])
    rng = np.random.default_rng(0)
    out = {k: [] for k in ("day", "bg", "tab", "f", "n0", "nall", "paid", "tc")}
    for n in range(len(days)):
        sl = slice(ptr[n], ptr[n + 1])
        keep_day = (n0[sl][~tc[sl]] > 0).any() if days[n].year >= 2017 else True
        if not keep_day:
            continue
        ok = ~tc[sl]
        y = n0[sl] > 0
        if days[n].year < 2017:
            ok &= y | (rng.random(sl.stop - sl.start) < 0.1)
        if not ok.any():
            continue
        bg = rbg[sl][ok]
        o = np.asarray(O[n], np.float32)
        pk, sm = o[0], o[1]
        nb = maximum_filter(pk, size=5)
        # county maximum / mean over the block groups that are rows of the day (all rows, not only the kept ones)
        allbg = rbg[sl][~tc[sl]]
        pv = pk[by[allbg], bx[allbg]]
        c_all = cty[allbg]
        cmax = np.full(nc, 0.0, np.float32)
        np.maximum.at(cmax, c_all, pv)
        cmean = np.bincount(c_all, weights=pv, minlength=nc) / np.maximum(np.bincount(c_all, minlength=nc), 1)
        c = cty[bg]
        f = np.column_stack([pk[by[bg], bx[bg]], sm[by[bg], bx[bg]], nb[by[bg], bx[bg]], cmax[c], cmean[c]]).astype(np.float32)
        out["day"].append(np.full(len(bg), n, np.int32))
        out["bg"].append(bg)
        out["tab"].append(np.asarray(tab[sl][ok]))
        out["f"].append(f)
        for k, a in (("n0", n0), ("nall", nall), ("paid", paid), ("tc", tc)):
            out[k].append(a[sl][ok])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["days"] = days.values
    res["cty"] = cty[res["bg"]].astype(np.int16)
    res["ctys"] = S["ctys"]
    np.savez(cache, **res)
    return res


def table(R: dict, rows: np.ndarray) -> pd.DataFrame:
    days = pd.DatetimeIndex(R["days"])
    di = R["day"][rows]
    return pd.DataFrame({"sysid": days[di].strftime("%Y%m%d"), "cty": R["ctys"][R["cty"][rows]], "n": R["n0"][rows], "paid": R["paid"][rows],
                         "n_pol": R["tab"][rows, 8], "year": days.year.to_numpy()[di]})


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    e75 = _load("e75", "75_task_metrics.py")
    R = {}
    for st in STATES:
        R[st] = features(st)
        r = R[st]
        yr = pd.DatetimeIndex(r["days"]).year.to_numpy()[r["day"]]
        say(f"{st}: kept rows {len(r['day']):,} (train years {int((yr < 2017).sum()):,}, test years {int((yr >= 2017).sum()):,}); claimed rows in test "
            f"{int(((r['n0'] > 0) & (yr >= 2017)).sum()):,}")

    def split(st):
        r = R[st]
        days = pd.DatetimeIndex(r["days"])
        yr = days.year.to_numpy()[r["day"]]
        ym = (days.year * 12 + days.month).to_numpy()[r["day"]]
        return yr < 2017, (yr < 2017) & (ym % 7 == 0), yr >= 2017

    def XY(st, sel):
        r = R[st]
        X = np.column_stack([r["tab"][sel], r["f"][sel]])
        return X, (r["n0"][sel] > 0).astype(np.float32)

    def fit(train_states):
        Xs, ys, vs = [], [], []
        for st in train_states:
            tr, va, _ = split(st)
            X, y = XY(st, tr)
            Xs.append(X)
            ys.append(y)
            vs.append(va[tr])
        X, y, v = np.concatenate(Xs), np.concatenate(ys), np.concatenate(vs)
        w = np.where(y == 1, 1.0, 10.0).astype(np.float32)

        def mk(n):
            return HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300, l2_regularization=1.0,
                                                  early_stopping=False, random_state=0)
        m = mk(300).fit(X[~v], y[~v], sample_weight=w[~v])
        sc = [log_loss(y[v], np.clip(p[:, 1], 1e-7, 1 - 1e-7), sample_weight=w[v], labels=[0, 1]) for p in m.staged_predict_proba(X[v])]
        n_it = int(np.argmin(sc)) + 1
        return mk(n_it).fit(X, y, sample_weight=w), n_it

    fits = {"own": lambda st: [st], "tx": lambda st: ["TX"], "pool": lambda st: STATES, "loso": lambda st: [s for s in STATES if s != st]}
    models, iters = {}, {}
    for name, f in fits.items():
        for key in {tuple(f(st)) for st in STATES}:
            models[key], iters[key] = fit(list(key))
            say(f"fit on {'+'.join(key)}: {iters[key]} iterations")
    metrics = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"),
               ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))
    for st in STATES:
        _, _, te = split(st)
        rows = np.flatnonzero(te)
        T = table(R[st], rows)
        X = np.column_stack([R[st]["tab"][rows], R[st]["f"][rows]])
        scores = {"policies only": ("n_pol", "sum")}
        for name, f in fits.items():
            if name == "tx" and st == "TX":
                continue
            if name == "own" and st == "TX":
                pass
            T[f"p_{name}"] = models[tuple(f(st))].predict_proba(X)[:, 1]
            scores[name] = (f"p_{name}", "sum")
        # test-row predictions, for the paired comparison with the CNN of script 89
        np.savez(wb2.cache_path(f"cnn_{st.lower()}") / "e45_pred.npz", **{c: T[c].to_numpy(np.float32) for c in T.columns if c.startswith("p_")})
        say(f"\n== test {st}: rows {len(T):,}, days {T.sysid.nunique()}, claimed block groups {int((T.n > 0).sum()):,}")
        tabs = e75.event_table(T, scores)
        say("    score order: " + " / ".join(scores))
        for col, w, lab in metrics:
            pt = {k: e75.pooled(t, col, w) for k, t in tabs.items()}
            line = f"    {lab:24s} " + " / ".join(f"{pt[k]:.3f}" for k in scores)
            ref = "own"
            for other in ("pool", "loso", "tx"):
                if other in tabs and other != ref:
                    _, dd = e75.boot({other: tabs[other], ref: tabs[ref]}, col, w, ref=ref)
                    line += f"   [{other} - own] {dd[other][0]:+.3f} [{dd[other][1]:+.3f}, {dd[other][2]:+.3f}]"
            say(line)
    (wb2.cache_path("audit") / "state_group_baseline.txt").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
