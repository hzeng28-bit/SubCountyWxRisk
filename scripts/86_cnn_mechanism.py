"""Why does the 4 km CNN beat boosting on summaries, with so little data? (RESULTS stage 46)

The CNN of stage 42 has 55 k parameters, its best validation epoch is the 2nd-3rd, and a quarter of the training months already put it above the boosting baseline.
Boosting got: tabular rows + observed rain at the 4 km pixel and the 0.25 degree cell + county peak / mean. The CNN additionally gets coordinate and exposure maps, sees a
neighbourhood (receptive field about 65 pixels), is trained with a listwise loss, and reads the whole day. Four mechanisms are tested one at a time:
  ladder  boosting with, in steps, the information the CNN has and boosting lacks: `geo` (pixel latitude / longitude), `ms` (multi-scale neighbourhood mean / max of the
          peak and mean of the sum, 3-33 pixels = 12-130 km), `ctx` (day context: domain maximum, rank of the pixel peak among the day's rows, peak minus county mean,
          number of rows above 25 mm) and all three. If boosting + `ms` reaches the CNN, "precise structure" is neighbourhood context, not sub-pixel detail.
  probe   test-time perturbations of the rain channels of the trained full CNN (weights from `81 train --variant full`): shift by k pixels, Gaussian blur, 6 x 6 block
          mean (a 0.25 degree product on the 4 km grid), rain removed. A model that uses pixel-exact structure collapses at a 1-2 pixel shift; one that uses a smooth
          neighbourhood degrades slowly.
  eval    the one-change retrainings of script 81 (`--variant`): coarse_rain (training and test rain replaced by 6 x 6 block means), shuffle_rain (rain of another day of the
          same month: only the static maps and the tabular features can work), nocoord, nostatic (exposure maps zeroed), bce (no within-day listwise term), against full,
          and the ladder, with a bootstrap over days.
Metrics (scripts 75 / 81) on the test days with >= 1 claimed block group: cross-county block-group AUC, top-5% capture, NDCG5 paid, county AUC, top-10% county capture.
Inputs : cnn/* (script 80), cnn/e1_hgb_features.npz, cnn/v_<variant>_s<seed>.npy and w_*.pt (script 81)
Output : $WXRISK_DATA/cnn/ladder_*.npy, probe_*.npy, $WXRISK_DATA/audit/cnn_mechanism.txt
Run    : python scripts/86_cnn_mechanism.py ladder ; python scripts/86_cnn_mechanism.py probe ; python scripts/86_cnn_mechanism.py eval
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk import zstio  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e81 = _load("e81", "81_cnn_e1.py")
D = e81.D
OUT: list[str] = []
ONLY: list[str] = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ ladder

def extra_features(d, rid: np.ndarray) -> dict[str, np.ndarray]:
    """Columns of the three feature groups for the rows `rid` (indices into the rows of script 80)."""
    from scipy.stats import rankdata

    geo = np.column_stack([d.lat4[d.st["b4_y"].astype(int)], d.lon4[d.st["b4_x"].astype(int)]]).astype(np.float32)
    ms = np.full((len(rid), 8), np.nan, np.float32)
    ctx = np.full((len(rid), 4), np.nan, np.float32)
    day_of = np.searchsorted(d.ptr, rid, side="right") - 1
    pos = 0
    for n in np.unique(day_of):
        k = np.flatnonzero(day_of == n)
        bgs, _, _, _ = d.rows(int(n))
        sel = rid[k] - d.ptr[n]
        o = np.asarray(d.O[n], np.float32)
        pk, sm = o[0], o[1]
        maps = [uniform_filter(pk, 3), uniform_filter(pk, 9), uniform_filter(pk, 33), maximum_filter(pk, 3), maximum_filter(pk, 9), maximum_filter(pk, 33),
                uniform_filter(sm, 9), uniform_filter(sm, 33)]
        fl = d.bpix4[bgs]
        for j, m in enumerate(maps):
            ms[k, j] = m.ravel()[fl[sel]]
        pv = pk.ravel()[fl]
        cty = d.cty_idx[bgs]
        cmean = np.bincount(cty, weights=pv, minlength=len(d.ctys)) / np.maximum(np.bincount(cty, minlength=len(d.ctys)), 1)
        ctx[k, 0] = pk[d.valid4].max()
        ctx[k, 1] = (rankdata(pv) / len(pv))[sel]
        ctx[k, 2] = (pv - cmean[cty])[sel]
        ctx[k, 3] = (pv > 25).sum()
        pos += len(k)
    day_rows_bg = d.rbg[rid]
    return {"geo": geo[day_rows_bg], "ms": ms, "ctx": ctx}


def run_ladder() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    d = e81.Data()
    f = zstio.load(D / "e1_hgb_features.npz")
    rid, X0 = f["rid"], f["X"]
    say(f"rows {len(rid):,}; computing the extra feature groups ...")
    ex = extra_features(d, rid)
    day_of_row = np.repeat(np.arange(d.D), np.diff(d.ptr))[rid]
    tr, va, te = d.is_train[day_of_row], d.is_val[day_of_row], d.is_test[day_of_row]
    y = (d.rn[rid] > 0).astype(np.float32)
    w = np.where(y == 1, 1.0, 10.0).astype(np.float32)
    sets = {"base": [], "geo": ["geo"], "ms": ["ms"], "ctx": ["ctx"], "geo+ms+ctx": ["geo", "ms", "ctx"],
            # day-normalised weights (each day weighs the same, positives of a day share its weight): the binary analogue of the listwise loss
            "base+dayw": [], "geo+ms+ctx+dayw": ["geo", "ms", "ctx"],
            # no rain at all: the static knowledge of boosting (tabular rows; + pixel coordinates) against the CNN with shuffled rain
            "norain_tab": [], "norain_tab+geo": ["geo"]}
    only = set(ONLY) if ONLY else set(sets)
    npos = np.bincount(day_of_row[y == 1], minlength=d.D)
    wd = (w / np.maximum(npos[day_of_row], 1)).astype(np.float32)
    wd *= w.sum() / wd.sum()
    for name, groups in sets.items():
        if name not in only:
            continue
        w_use = wd if name.endswith("+dayw") else w
        base_cols = X0[:, :9] if name.startswith("norain") else X0
        X = np.column_stack([base_cols] + [ex[g] for g in groups]) if groups else base_cols

        def mk(n):
            return HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
        m = mk(600).fit(X[tr], y[tr], sample_weight=w_use[tr])
        sc = [log_loss(y[va], np.clip(p[:, 1], 1e-7, 1 - 1e-7), sample_weight=w_use[va], labels=[0, 1]) for p in m.staged_predict_proba(X[va])]
        n_it = int(np.argmin(sc)) + 1
        mdl = mk(n_it).fit(X[tr | va], y[tr | va], sample_weight=w_use[tr | va])
        p = mdl.predict_proba(X[te])[:, 1].astype(np.float32)
        np.save(D / f"ladder_{name}.npy", p)
        say(f"ladder {name}: {X.shape[1]} features, {n_it} iterations (cap 600), validation loss {min(sc):.4f}")
    (wb2.cache_path("audit") / "cnn_mechanism_ladder.txt").write_text("\n".join(OUT) + "\n")


# ------------------------------------------------------------------ probe

def claim_days(d) -> np.ndarray:
    """Test days that contain a claimed block group."""
    return np.array([n for n in np.flatnonzero(d.is_test) if (d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any()])


def run_probe(seed: int) -> None:
    import torch

    from wxrisk import cnn, compute

    d = e81.Data()
    dev = compute.device()
    rng = np.random.default_rng(seed)
    sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in rng.choice(np.flatnonzero(d.is_train), 300, replace=False)])
    tf = e81.tab_features(sample)
    mu, sd = tf.mean(0), tf.std(0) + 1e-6
    idx_np, ok_np = d.pool("4km")
    pool_idx, pool_ok = torch.from_numpy(idx_np).to(dev), torch.from_numpy(ok_np).to(dev)
    model = cnn.Hybrid(d.image("4km", 0).shape[0], 9, dilations=(1, 2, 4, 8, 16)).to(dev)
    model.load_state_dict(torch.load(D / f"w_full_s{seed}.pt", map_location=dev))
    model.eval()
    days = claim_days(d)
    v = d.valid4.astype(np.float32)

    def shift(a, dy, dx):
        out = np.zeros_like(a)
        H, W = a.shape
        out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = a[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
        return out

    def raw(img):  # the two rain channels are log1p(x) / 3
        return np.expm1(img[:2] * 3)

    def put(img, r):
        img = img.copy()
        img[:2] = np.log1p(np.maximum(r, 0)) / 3 * v
        return img

    conds = {"none": lambda r: r, "no_rain": lambda r: r * 0}
    for k in (1, 2, 4, 8, 16):
        conds[f"shift_x{k}"] = lambda r, k=k: np.stack([shift(r[0], 0, k), shift(r[1], 0, k)])
    conds["shift_y4"] = lambda r: np.stack([shift(r[0], 4, 0), shift(r[1], 4, 0)])
    for sg in (1, 2, 4):
        conds[f"blur{sg}"] = lambda r, sg=sg: np.stack([gaussian_filter(r[0] * v, sg), gaussian_filter(r[1] * v, sg)])
    conds["coarse6"] = lambda r: np.stack([e81.coarse(r[0], v), e81.coarse(r[1], v)])
    for name, fn in conds.items():
        preds = []
        with torch.no_grad():
            for n in days:
                bgs, tab, _, _ = d.rows(int(n))
                img = put(d.image("4km", int(n)), fn(raw(d.image("4km", int(n)))))
                tt = torch.from_numpy(((e81.tab_features(tab) - mu) / sd).astype(np.float32)).to(dev)
                lg = model(torch.from_numpy(img[None]).to(dev), torch.from_numpy(d.bpix4[bgs]).to(dev), torch.from_numpy(d.cty_idx[bgs]).to(dev), tt, pool_idx, pool_ok)
                preds.append(torch.sigmoid(lg).cpu().numpy())
        np.save(D / f"probe_{name}_s{seed}.npy", np.concatenate(preds).astype(np.float32))
        say(f"probe {name}: {sum(len(p) for p in preds):,} rows")


# ------------------------------------------------------------------ mech: what the trained network reads

def run_mech(seed: int) -> None:
    """(1) head-input ablation: which of the four inputs of the head (own-pixel embedding, county mean, county max of the embedding, tabular branch) carry the
    skill; (2) input-gradient footprint: where, relative to a claimed block group, the rain pixels that move its logit lie (|gradient| and signed, by distance)."""
    import torch

    from wxrisk import cnn, compute

    d = e81.Data()
    dev = compute.device()
    rng = np.random.default_rng(seed)
    sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in rng.choice(np.flatnonzero(d.is_train), 300, replace=False)])
    tf = e81.tab_features(sample)
    mu, sd = tf.mean(0), tf.std(0) + 1e-6
    idx_np, ok_np = d.pool("4km")
    pool_idx, pool_ok = torch.from_numpy(idx_np).to(dev), torch.from_numpy(ok_np).to(dev)
    model = cnn.Hybrid(d.image("4km", 0).shape[0], 9, dilations=(1, 2, 4, 8, 16)).to(dev)
    model.load_state_dict(torch.load(D / f"w_full_s{seed}.pt", map_location=dev))
    model.eval()
    days = claim_days(d)

    def head(img, bgs, tab, drop: str):
        e = model.emb
        flat = model.embed(img)[0].reshape(e, -1).T
        own = flat[torch.from_numpy(d.bpix4[bgs]).to(dev)]
        g = flat[pool_idx.clamp(min=0)]
        ok = pool_ok.unsqueeze(-1)
        mean = (g * ok).sum(1) / ok.sum(1).clamp(min=1)
        mx = g.masked_fill(~ok, float("-inf")).amax(1)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        c = torch.from_numpy(d.cty_idx[bgs]).to(dev)
        tb = model.tab(torch.from_numpy(((e81.tab_features(tab) - mu) / sd).astype(np.float32)).to(dev))
        parts = {"own": own, "county_mean": mean[c], "county_max": mx[c], "tab": tb}
        for k in drop.split("+"):
            if k in parts:
                parts[k] = torch.zeros_like(parts[k])
        return model.head(torch.cat([parts["own"], parts["county_mean"], parts["county_max"], parts["tab"]], 1)).squeeze(-1)

    conds = ["none", "own", "county_mean", "county_max", "county_mean+county_max", "tab", "own+county_mean+county_max"]
    for name in conds:
        preds = []
        with torch.no_grad():
            for n in days:
                bgs, tab, _, _ = d.rows(int(n))
                img = torch.from_numpy(d.image("4km", int(n))[None]).to(dev)
                preds.append(torch.sigmoid(head(img, bgs, tab, name)).cpu().numpy())
        np.save(D / f"head_drop_{name}_s{seed}.npy", np.concatenate(preds).astype(np.float32))
        say(f"head ablation: dropped {name}")
    # footprint: gradient of the logit of a claimed block group with respect to the two rain channels
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 10 ** 6]
    absm, signm, n_s = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1), 0
    yy, xx = np.mgrid[0:d.H4, 0:d.W4]
    for n in rng.choice(days, 120, replace=False):
        bgs, tab, cnt, _ = d.rows(int(n))
        pos = np.flatnonzero(cnt > 0)
        for r in rng.choice(pos, min(3, len(pos)), replace=False):
            img = torch.from_numpy(d.image("4km", int(n))[None]).to(dev).requires_grad_(True)
            lg = head(img, bgs[r:r + 1], tab[r:r + 1], "none")
            lg.sum().backward()
            g = img.grad[0, :2].sum(0).cpu().numpy()
            ra = np.hypot(yy - d.st["b4_y"][bgs[r]], xx - d.st["b4_x"][bgs[r]])
            tot = np.abs(g).sum()
            if tot <= 0:
                continue
            for j in range(len(edges) - 1):
                m = (ra >= edges[j]) & (ra < edges[j + 1]) if j else (ra < edges[1])
                absm[j] += np.abs(g[m]).sum() / tot
                signm[j] += g[m].sum() / tot
            n_s += 1
    say(f"gradient footprint over {n_s} claimed block groups (1 px = 4 km): share of |gradient| / signed share, by distance ring")
    for j in range(len(edges) - 1):
        say(f"    {edges[j]:>3d} <= r < {edges[j + 1] if edges[j + 1] < 10 ** 5 else 'inf':>4}: {absm[j] / n_s:.3f} / {signm[j] / n_s:+.3f}")
    (wb2.cache_path("audit") / "cnn_mechanism_mech.txt").write_text("\n".join(OUT) + "\n")


# ------------------------------------------------------------------ eval

METRICS = (("bg_auc", "bg_w", "BG AUC"), ("bg_cap_n_0.05", None, "BG top-5%"), ("bg_ndcg_p", None, "BG NDCG5"), ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "cty top-10%"))


def test_frame(d, days: np.ndarray, full_rows: bool):
    """Table of the test rows of `days`; full_rows: the prediction arrays hold ALL test rows (select the claim days from them)."""
    e75 = e81._load("e75", "75_task_metrics.py")
    test = np.flatnonzero(d.is_test)
    lo = d.ptr[test[0]]
    sel_rows, sel_in_test = [], []
    for n in days:
        sel_rows.append(np.arange(d.ptr[n], d.ptr[n + 1]))
        sel_in_test.append(np.arange(d.ptr[n], d.ptr[n + 1]) - lo)
    rows, in_test = np.concatenate(sel_rows), np.concatenate(sel_in_test)
    day_of = np.repeat(np.arange(d.D), np.diff(d.ptr))[rows]
    bg = d.rbg[rows]
    T = pd.DataFrame({"sysid": d.days[day_of].strftime("%Y%m%d"), "cty": np.array(d.ctys)[d.cty_idx[bg]], "n": d.rn[rows], "paid": d.rpaid[rows],
                      "n_pol": np.asarray(d.rtab[rows, 8], np.float32)})
    return e75, T, in_test


def report(e75, T: pd.DataFrame, models: list[str], ref: str, title: str) -> None:
    scores = {"policies only": ("n_pol", "sum")}
    scores.update({m: (f"p_{m}", "sum") for m in models})
    tabs = e75.event_table(T, scores)
    say(f"\n== {title}: {T.sysid.nunique()} days, claimed block groups {int((T.n > 0).sum()):,}")
    say("    " + f"{'':30s}" + "".join(f"{lab:>34s}" for _, _, lab in METRICS))
    for m in scores:
        line = f"    {m:30s}"
        for col, w, lab in METRICS:
            pt = e75.pooled(tabs[m], col, w)
            if m in (ref, "policies only"):
                line += f"{pt:>34.3f}"
            else:
                _, dd = e75.boot({m: tabs[m], ref: tabs[ref]}, col, w, ref=ref)
                line += f"{pt:>9.3f} {dd[m][0]:+.3f} [{dd[m][1]:+.3f},{dd[m][2]:+.3f}]".rjust(34)
        say(line)


def within_county_auc(T: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    """Pair-weighted AUC of the score INSIDE each county-day (block groups of the same county compared with each other) and, for contrast, the share of
    the AUC pairs that are between counties is not needed here: the cross-county AUC of the event table already covers it."""
    key = T.sysid + "_" + T.cty
    y = (T.n.to_numpy() > 0)
    g = pd.Series(key).astype("category").cat.codes.to_numpy()
    n1 = np.bincount(g, weights=y)
    n0 = np.bincount(g) - n1
    ok = (n1 > 0) & (n0 > 0)
    out = {}
    for c in cols:
        r = pd.Series(T[c].to_numpy()).groupby(g).rank().to_numpy()  # average ranks inside the county-day
        s1 = np.bincount(g, weights=np.where(y, r, 0))
        auc = (s1 - n1 * (n1 + 1) / 2) / np.maximum(n1 * n0, 1)
        w = (n1 * n0)[ok]
        out[c] = float((auc[ok] * w).sum() / w.sum())
    return out


def run_eval() -> None:
    d = e81.Data()
    days = claim_days(d)
    e75, T, in_test = test_frame(d, days, True)
    models = []

    def add(name, arr):
        T[f"p_{name}"] = arr[in_test]
        models.append(name)
    seeds = {}
    for v in e81.VARIANTS:
        fs = sorted(D.glob(f"v_{v}_s*.npy"))
        if fs:
            seeds[v] = [zstio.load(f) for f in fs]
            add(f"{v} (mean of {len(fs)} seeds)", np.mean(seeds[v], axis=0))
    for name in ("base", "geo", "ms", "ctx", "geo+ms+ctx", "base+dayw", "geo+ms+ctx+dayw", "norain_tab", "norain_tab+geo"):
        if (D / f"ladder_{name}.npy").exists():
            add(f"boosting {name}", zstio.load(D / f"ladder_{name}.npy"))
    old = [zstio.load(D / f"e1_pred_cnn4km_s{s}.npy") for s in range(3)]
    add("stage-42 CNN 4 km (3 seeds)", np.mean(old, axis=0))
    ref = next(m for m in models if m.startswith("full"))
    report(e75, T, models, ref, f"ablations and ladder against {ref}")
    wc = within_county_auc(T, [f"p_{m}" for m in models] + ["n_pol"])
    say("\n== AUC inside a county-day (block groups of one county compared with each other; pair weighted)")
    for m, v in wc.items():
        say(f"    {m.replace('p_', ''):40s} {v:.3f}")
    # seed spread of the full model and of each variant, so differences can be read against the noise
    say("\n== seed spread (per-seed pooled metric on the same days)")
    for v, arrs in seeds.items():
        if len(arrs) < 2:
            continue
        T2 = T[["sysid", "cty", "n", "paid", "n_pol"]].copy()
        for i, a in enumerate(arrs):
            T2[f"s{i}"] = a[in_test]
        tabs = e75.event_table(T2, {f"s{i}": (f"s{i}", "sum") for i in range(len(arrs))})
        say(f"    {v:14s} " + " | ".join(f"{lab} " + "/".join(f"{e75.pooled(t, col, w):.3f}" for t in tabs.values()) for col, w, lab in METRICS))
    pr = sorted(D.glob("probe_*_s0.npy"))
    if pr:
        T3 = T[["sysid", "cty", "n", "paid", "n_pol"]].copy()
        names = []
        for f in pr:
            nm = f.stem.replace("probe_", "").replace("_s0", "")
            T3[f"p_{nm}"] = zstio.load(f)
            names.append(nm)
        order = ["none"] + [n for n in ("shift_x1", "shift_x2", "shift_x4", "shift_x8", "shift_x16", "shift_y4", "blur1", "blur2", "blur4", "coarse6", "no_rain") if n in names]
        report(e75, T3, order, "none", "test-time perturbation of the rain channels of the full CNN (seed 0)")
    hd = sorted(D.glob("head_drop_*_s0.npy"))
    if hd:
        T4 = T[["sysid", "cty", "n", "paid", "n_pol"]].copy()
        names = []
        for f in hd:
            nm = f.stem.replace("head_drop_", "").replace("_s0", "")
            T4[f"p_{nm}"] = zstio.load(f)
            names.append(nm)
        order = ["none"] + [n for n in ("own", "county_mean", "county_max", "county_mean+county_max", "tab", "own+county_mean+county_max") if n in names]
        report(e75, T4, order, "none", "head inputs of the full CNN set to zero at test time (seed 0)")
    (wb2.cache_path("audit") / "cnn_mechanism.txt").write_text("\n".join(OUT) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ladder", "probe", "mech", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=[], help="ladder: fit only these sets")
    a = ap.parse_args()
    ONLY.extend(a.only)
    {"ladder": run_ladder, "probe": lambda: run_probe(a.seed), "mech": lambda: run_mech(a.seed), "eval": run_eval}[a.mode]()


if __name__ == "__main__":
    main()
