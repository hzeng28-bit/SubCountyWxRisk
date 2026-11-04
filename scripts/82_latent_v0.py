"""V0/V1 of docs/CNN_HYBRID_DESIGN.md section 10: is a pretrained latent code of the observed 4 km field feasible, and does it carry what the labelled sample cannot teach?

Modes
  build   gridMET 4 km two-day window maps (peak, sum over S and S+1, as in script 80) for every start day 2000-2019 -> cnn/gridmet_win.npy (float16), cnn/gridmet_win_days.npy
  train   convolutional VAE (stride 6 = the GEFS cell, C latent channels per cell, small KL), log rain, tail-weighted loss; pretraining windows 2000-2016 (--from-year 2000) or 2010-2016,
          blocked validation months as in script 81, test years never enter
  probe   V0 gate: reconstruction fidelity on the extreme pixels and county peaks (test years), scale probes (does the cell latent keep the sub-cell peak and spread beyond the
          cell mean?), then the latent means for every day tensor -> cnn/z_<tag>.npy
  v1      V1: boosting with the frozen latent (own cell + county mean) added to the 17 summaries of script 81, against the summaries alone and the supervised 4 km CNN; learning curve in
          the number of training months for both boosting versions
  curve   data-sufficiency diagnostics: training storms and claimed block-group days, learning curve of the supervised 4 km CNN (runs of script 81 --frac)

Inputs : cnn/* (script 80), cnn/e1_hgb_features.npz and e1_pred_* (script 81), gridMET (script 22)
Output : cnn/gridmet_win*.npy, cnn/vae_<tag>.pt, cnn/z_<tag>.npy, cnn/v1_pred_*.npy, $WXRISK_DATA/audit/latent_v0_*.txt
Run    : python scripts/82_latent_v0.py build ; train --tag c8_2000 --from-year 2000 --c 8 ; probe --tag c8_2000 ; v1 --tag c8_2000 ; curve
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk import zstio  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = wb2.cache_path("cnn")
OUT: list[str] = []
PAD = 4  # 260 x 320 -> 264 x 324, divisible by 6
S = 6


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def save_log(name):
    wb2.cache_path("audit").mkdir(parents=True, exist_ok=True)
    (wb2.cache_path("audit") / name).write_text("\n".join(OUT) + "\n")


# ------------------------------------------------------------------ data

def run_build() -> None:
    from wxrisk import flood_pilot as fp
    days, lat, lon, P = fp.load_precip()
    keep = np.flatnonzero((days.year <= 2019))
    n = len(keep) - 1
    W = np.lib.format.open_memmap(D / "gridmet_win.npy", mode="w+", dtype=np.float16, shape=(n, 2, *P.shape[1:]))
    for i in range(n):
        x = np.nan_to_num(np.asarray(P[i:i + 2]), nan=0.0)
        W[i, 0], W[i, 1] = x.max(0), x.sum(0)
    W.flush()
    np.save(D / "gridmet_win_days.npy", days[:n].to_numpy())
    O = zstio.load(D / "obs4k_day_maps.npy", mmap_mode="r")
    st = zstio.load(D / "static.npz", allow_pickle=True)
    tdays = pd.DatetimeIndex(st["days"])
    ix = days[:n].get_indexer(tdays)
    pick = np.random.default_rng(0).choice(len(tdays), 20, replace=False)
    err = max(float(np.abs(np.asarray(W[ix[k]], np.float32) - np.asarray(O[k], np.float32)).max()) for k in pick)
    say(f"windows {n} ({days[0].date()} .. {days[n - 1].date()}); max abs difference to obs4k_day_maps on 20 days: {err:.4f} mm")
    save_log("latent_v0_build.txt")


class Wins:
    def __init__(self):
        self.W = zstio.load(D / "gridmet_win.npy", mmap_mode="r")
        self.days = pd.DatetimeIndex(zstio.load(D / "gridmet_win_days.npy"))
        st = zstio.load(D / "static.npz", allow_pickle=True)
        self.valid = st["obs4_valid"]
        yr = self.days.year.to_numpy()
        ym = yr * 12 + self.days.month.to_numpy()
        self.is_test = yr >= 2017
        self.is_val = (~self.is_test) & ((ym % 7) == 0)
        self.is_train = (~self.is_test) & ~self.is_val
        self.year = yr

    def batch(self, idx):
        w = np.asarray(self.W[np.sort(idx)], np.float32)  # [B, 2, 260, 320]
        v = np.broadcast_to(self.valid[None].astype(np.float32), (len(w), *self.valid.shape))
        x = np.stack([np.log1p(w[:, 0]) / 3, np.log1p(w[:, 1]) / 3, v], 1)
        x = np.pad(x, ((0, 0), (0, 0), (0, PAD), (0, PAD)))
        peak = np.pad(w[:, 0], ((0, 0), (0, PAD), (0, PAD)))
        return x.astype(np.float32), peak.astype(np.float32), np.sort(idx)


# ------------------------------------------------------------------ model

def make_model(c: int):
    import torch
    from torch import nn

    class Res(nn.Module):
        def __init__(self, ch, dil=1):
            super().__init__()
            self.conv = nn.Conv2d(ch, ch, 3, padding=dil, dilation=dil)
            self.norm = nn.GroupNorm(8, ch)

        def forward(self, x):
            return x + nn.functional.gelu(self.norm(self.conv(x)))

    def seq(ch, n, dils=None):
        return nn.Sequential(*[Res(ch, (dils or [1] * n)[i]) for i in range(n)])

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.c = c
            self.enc = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(), seq(32, 2),
                                     nn.Conv2d(32, 64, 3, stride=3), nn.GroupNorm(8, 64), nn.GELU(), seq(64, 2),
                                     nn.Conv2d(64, 96, 2, stride=2), nn.GroupNorm(8, 96), nn.GELU(), seq(96, 2, [1, 2]))
            self.head = nn.Conv2d(96, 2 * c, 1)
            self.dec = nn.Sequential(nn.Conv2d(c, 96, 1), nn.GELU(), seq(96, 2, [1, 2]),
                                     nn.ConvTranspose2d(96, 64, 2, stride=2), nn.GroupNorm(8, 64), nn.GELU(), seq(64, 2),
                                     nn.ConvTranspose2d(64, 32, 3, stride=3), nn.GroupNorm(8, 32), nn.GELU(), seq(32, 2),
                                     nn.Conv2d(32, 2, 1))

        def encode(self, x):
            mu, lv = self.head(self.enc(x)).chunk(2, 1)
            return mu, lv.clamp(-12, 4)

        def forward(self, x, sample=True):
            mu, lv = self.encode(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
            return self.dec(z), mu, lv

    return VAE()


def run_train(tag: str, c: int, from_year: int, epochs: int, beta: float, seed: int, smoke: int = 0, ckpt_dir: str = "") -> None:
    import torch

    from wxrisk import compute

    w = Wins()
    dev = compute.device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(c).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    tr = np.flatnonzero(w.is_train & (w.year >= from_year))
    va = np.flatnonzero(w.is_val & (w.year >= from_year))
    if smoke:
        tr, va = tr[:smoke], va[:smoke // 4]
    say(f"VAE {tag}: C={c}, beta={beta}, parameters {npar:,}; windows train {len(tr)} (from {from_year}), validation {len(va)}; {compute.describe()}")
    B = 8
    steps = epochs * (len(tr) // B)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps + 5, pct_start=0.15)

    def loss_fn(idx, sample):
        x, peak, _ = w.batch(idx)
        xt = torch.from_numpy(x).to(dev)
        pk = torch.from_numpy(peak).to(dev)
        rec, mu, lv = model(xt, sample)
        tgt = xt[:, :2]
        wt = xt[:, 2:3] * (1 + 3 * torch.clamp(pk[:, None] / 25, 0, 4))  # valid pixels, tail-weighted
        recon = (wt * (rec - tgt) ** 2).sum() / (wt.sum() * 2)
        kl = 0.5 * (mu ** 2 + torch.exp(lv) - lv - 1).mean()
        return recon + beta * kl, recon, kl

    best, best_state, ep0 = np.inf, None, 0
    CK = Path(ckpt_dir) if ckpt_dir else D  # on Colab: a Drive folder, so a disconnect loses at most one epoch
    CK.mkdir(parents=True, exist_ok=True)
    ck_path = CK / f"vae_{tag}.ckpt"
    if ck_path.exists():  # free Colab sessions disconnect: resume from the last finished epoch
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best, best_state, ep0 = ck["best"], ck["best_state"], ck["epoch"]
        rng = np.random.default_rng(seed + 1000 * ep0)
        say(f"  resumed from {ck_path.name} after epoch {ep0}")
    for ep in range(ep0, epochs):
        t0 = time.time()
        model.train()
        order = rng.permutation(tr)
        tot = 0.0
        k = 0
        for i in range(0, len(order) - B + 1, B):
            loss, _, _ = loss_fn(order[i:i + B], True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += float(loss.detach())
            k += 1
        model.eval()
        with torch.no_grad():
            vs = [loss_fn(va[j:j + B], False) for j in range(0, min(len(va), 240), B)]
        vr = float(np.mean([float(v[1]) for v in vs]))
        vk = float(np.mean([float(v[2]) for v in vs]))
        say(f"  epoch {ep + 1}: train loss {tot / k:.5f}, validation reconstruction {vr:.5f}, KL/element {vk:.3f} ({time.time() - t0:.0f} s)")
        if vr < best:
            best, best_state = vr, {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "best": best, "best_state": best_state, "epoch": ep + 1}, ck_path.with_suffix(".tmp"))
        ck_path.with_suffix(".tmp").replace(ck_path)
    torch.save({"state": best_state, "c": c}, CK / f"vae_{tag}.pt")
    say(f"saved vae_{tag}.pt, best validation reconstruction {best:.5f}")
    save_log(f"latent_v0_train_{tag}.txt")


def load_model(tag, dev):
    import torch

    ck = torch.load(D / f"vae_{tag}.pt", map_location="cpu", weights_only=False)
    m = make_model(ck["c"])
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck["c"]


# ------------------------------------------------------------------ probe

def blocks(a: np.ndarray) -> np.ndarray:
    """[B, H, W] -> [B, H/6, W/6, 36]."""
    b, h, wd = a.shape
    return a.reshape(b, h // S, S, wd // S, S).transpose(0, 1, 3, 2, 4).reshape(b, h // S, wd // S, S * S)


def ridge_r2(Xtr, ytr, Xte, yte, lam=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = np.c_[(Xtr - mu) / sd, np.ones(len(Xtr))]
    coef = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ ytr)
    pred = np.c_[(Xte - mu) / sd, np.ones(len(Xte))] @ coef
    return 1 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()


def run_probe(tag: str) -> None:
    import torch

    from wxrisk import compute

    e81 = _load("e81", "81_cnn_e1.py")
    w = Wins()
    dev = compute.device()
    model, c = load_model(tag, dev)
    say(f"V0 probe {tag}: C={c}")
    te = np.flatnonzero(w.is_test)
    tr_all = np.flatnonzero(w.is_train)
    d = e81.Data()
    idx4, ok4 = d.pool("4km")  # county pixels at 4 km (flat indices into 260 x 320)
    ys, xs = np.divmod(idx4, d.W4)
    cpix = np.where(ok4, ys * (d.W4 + PAD) + xs, 0)  # the same pixels in the padded 264 x 324 image
    ratio = {th: [] for th in (10, 25, 50, 100)}
    rows = []
    cty_obs, cty_rec = [], []
    sub = {"m": [], "M": [], "s": [], "z": [], "w": []}
    with torch.no_grad():
        for j in range(0, len(te), 8):
            x, peak, ids = w.batch(te[j:j + 8])
            rec, mu, _ = model(torch.from_numpy(x).to(dev), False)
            rec = rec.cpu().numpy()
            rp = np.expm1(3 * np.clip(rec[:, 0], 0, None))
            vm = x[:, 2] > 0
            for th in ratio:
                m = vm & (peak >= th)
                if m.any():
                    ratio[th].append((peak[m].sum(), rp[m].sum(), m.sum(), np.abs(rp[m] - peak[m]).sum()))
            m = vm
            rows.append((((rec[:, 0] - x[:, 0]) ** 2 * m).sum(), m.sum()))
            for b in range(len(ids)):
                po, pr_ = peak[b].ravel()[cpix], rp[b].ravel()[cpix]
                cty_obs.append(np.where(ok4, po, 0).max(1))
                cty_rec.append(np.where(ok4, pr_, 0).max(1))
            lp = np.log1p(peak)
            v6 = blocks(vm.astype(np.float32)).min(-1) > 0
            bl = blocks(lp)
            z = mu.cpu().numpy().transpose(0, 2, 3, 1)
            sub["m"].append(bl.mean(-1)[v6]); sub["M"].append(bl.max(-1)[v6]); sub["s"].append(bl.std(-1)[v6]); sub["z"].append(z[v6]); sub["w"].append(np.broadcast_to(ids[:, None, None], v6.shape)[v6])
    say(f"  reconstruction RMSE of log1p(peak)/3 on test windows: {np.sqrt(sum(r[0] for r in rows) / sum(r[1] for r in rows)):.4f}")
    say("  extreme pixels (test windows): observed peak >= X mm -> mean reconstructed / mean observed, mean absolute error (mm)")
    for th, v in ratio.items():
        o, r, n, e = (sum(q[i] for q in v) for i in range(4))
        say(f"    >= {th:3d} mm: {n:,.0f} pixels, reconstructed / observed {r / o:.3f}, mean absolute error {e / n:.1f} mm")
    co, cr = np.concatenate(cty_obs), np.concatenate(cty_rec)
    for th in (25, 50, 100):
        m = co >= th
        say(f"  county peak >= {th} mm ({int(m.sum())} county-days): reconstructed / observed {cr[m].sum() / co[m].sum():.3f}, correlation {np.corrcoef(co[m], cr[m])[0, 1]:.3f}")
    # scale probes: per forecast cell, what the latent knows beyond the cell mean (the forecast's information). Split by test window (first half / second half of the
    # windows), linear ridge and gradient boosting on a subsample
    from sklearn.ensemble import HistGradientBoostingRegressor

    cat = {k: np.concatenate(v) for k, v in sub.items()}
    n = len(cat["m"])
    half = cat["w"] < np.median(cat["w"])
    rs = np.random.default_rng(0)
    pick = rs.choice(n, 400_000, replace=False)
    htr, hte = pick[half[pick]], pick[~half[pick]]
    say(f"  scale probes on {n:,} test cells, windows split in half (R2 on the second half; sub-cell targets are in log rain)")
    say(f"    {'target':36s} {'linear: mean':>13s} {'latent':>8s} {'mean + latent':>14s}   {'boosting: mean':>15s} {'mean + latent':>14s}")
    for tgt, lab in (("M", "sub-cell maximum"), ("s", "sub-cell standard deviation")):
        m_, z_, y_ = cat["m"][:, None], cat["z"], cat[tgt]
        mz = np.c_[m_, z_]
        lin = [ridge_r2(f[half], y_[half], f[~half], y_[~half]) for f in (m_, z_, mz)]
        gb = []
        for f in (m_, mz):
            r = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=0).fit(f[htr], y_[htr])
            pr = r.predict(f[hte])
            gb.append(1 - ((y_[hte] - pr) ** 2).sum() / ((y_[hte] - y_[hte].mean()) ** 2).sum())
        say(f"    {lab:36s} {lin[0]:13.3f} {lin[1]:8.3f} {lin[2]:14.3f}   {gb[0]:15.3f} {gb[1]:14.3f}")
    a_ = ridge_r2(cat["z"][half], cat["m"][half], cat["z"][~half], cat["m"][~half])
    say(f"    cell mean of log rain read linearly from the latent: R2 {a_:.3f}")
    # latent means for every day tensor
    ix = w.days.get_indexer(d.days)
    Z = np.zeros((d.D, c, d.H4 // S + 1 if False else (d.H4 + PAD) // S, (d.W4 + PAD) // S), np.float16)
    with torch.no_grad():
        for j in range(0, d.D, 16):
            x, _, ids = w.batch(ix[j:j + 16])
            mu, _ = model.encode(torch.from_numpy(x).to(dev))
            Z[j:j + len(ids)] = mu.cpu().numpy()
    np.save(D / f"z_{tag}.npy", Z)
    say(f"  saved z_{tag}.npy {Z.shape}")
    save_log(f"latent_v0_probe_{tag}.txt")


# ------------------------------------------------------------------ v1

def event_scores(T, cols, e75):
    scores = {lab: (col, "sum") for lab, col in cols.items()}
    return e75.event_table(T, scores), scores


def run_v1(tags: list[str]) -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    e81 = _load("e81", "81_cnn_e1.py")
    e75 = _load("e75", "75_task_metrics.py")
    e75.OUT = OUT
    d = e81.Data()
    feats = zstio.load(D / "e1_hgb_features.npz")
    rid, X = feats["rid"], feats["X"]
    day_of_row = np.repeat(np.arange(d.D), np.diff(d.ptr))
    dr = day_of_row[rid]
    yk = (d.rn[rid] > 0).astype(np.float32)
    wt = np.where(yk == 1, 1.0, 10.0).astype(np.float32)
    tr, va, te = d.is_train[dr], d.is_val[dr], d.is_test[dr]
    bg = d.rbg[rid]
    cty = d.cty_idx[bg]
    cell = (d.st["b4_y"].astype(np.int64)[bg] // S) * ((d.W4 + PAD) // S) + d.st["b4_x"].astype(np.int64)[bg] // S

    def zfeat(tag):
        Z = zstio.load(D / f"z_{tag}.npy", mmap_mode="r")
        c = Z.shape[1]
        ZZ = np.asarray(Z, np.float32).reshape(d.D, c, -1)
        own = ZZ[dr, :, cell]
        # county mean of the latent over the cells of the county's block groups
        cmean = np.zeros((d.D, len(d.ctys), c), np.float32)
        for k in range(len(d.ctys)):
            cells = np.unique((d.st["b4_y"].astype(np.int64)[d.cty_idx == k] // S) * ((d.W4 + PAD) // S) + d.st["b4_x"].astype(np.int64)[d.cty_idx == k] // S)
            if len(cells):
                cmean[:, k] = ZZ[:, :, cells].mean(2)
        return np.c_[own, cmean[dr, cty]]

    def fit(Xm, rows_train, n_it=150):
        m = HistGradientBoostingClassifier(max_iter=n_it, learning_rate=0.05, max_depth=5, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
        return m.fit(Xm[rows_train], yk[rows_train], sample_weight=wt[rows_train])

    sets = {"summaries": X}
    for tag in tags:
        sets[f"summaries + latent {tag}"] = np.c_[X, zfeat(tag)]
    preds = {}
    for name, Xm in sets.items():
        m = fit(Xm, tr | va)
        p = m.predict_proba(Xm[te])[:, 1].astype(np.float32)
        preds[name] = p
        np.save(D / f"v1_pred_{name.replace(' ', '_')}.npy", p)
        say(f"fitted '{name}': {Xm.shape[1]} features")
    # test table (same rows as e1_hgb_features test rows)
    test_days = np.flatnonzero(d.is_test)
    lo, hi = d.ptr[test_days[0]], d.ptr[test_days[-1] + 1]
    keep_te = (rid >= lo) & (rid < hi)
    assert keep_te.sum() == te.sum()
    rr = rid[te]
    T = pd.DataFrame({"sysid": d.days[day_of_row[rr]].strftime("%Y%m%d"), "cty": np.array(d.ctys)[d.cty_idx[d.rbg[rr]]], "n": d.rn[rr], "paid": d.rpaid[rr],
                      "n_pol": np.asarray(d.rtab[rr, 8], np.float32)})
    # the stored e1 predictions cover ALL test rows; the boosting features keep only a negative sample for training but all test rows
    full = np.arange(lo, hi)
    assert len(full) == te.sum()
    cols = {}
    for name, p in preds.items():
        T["p_" + name] = p
        cols[f"boosting {name}"] = "p_" + name
    cs = sorted(D.glob("e1_pred_cnn4km_s*.npy"))
    T["p_cnn4km"] = np.mean([zstio.load(f) for f in cs], axis=0)
    cols["CNN 4 km (seed mean)"] = "p_cnn4km"
    tabs, scores = event_scores(T, {**cols, "policies only": "n_pol"}, e75)
    metrics = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"), ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))
    say("\n== V1: frozen latent in the boosting model, all test days (point values in the order: " + " / ".join(scores) + ")")
    ref = "boosting summaries"
    others = [k for k in scores if k.startswith("boosting summaries + latent")] + ["CNN 4 km (seed mean)"]
    for col, wcol, lab in metrics:
        pt = {k: e75.pooled(t, col, wcol) for k, t in tabs.items()}
        line = f"  {lab:24s} " + " / ".join(f"{pt[k]:.3f}" for k in scores)
        for o in others:
            _, dd = e75.boot({o: tabs[o], ref: tabs[ref]}, col, wcol, ref=ref)
            line += f"   [{o[:30]} - summaries] {dd[o][0]:+.3f} [{dd[o][1]:+.3f}, {dd[o][2]:+.3f}]"
        say(line)
    # learning curve in the number of training months, boosting with and without the latent
    say("\n== learning curve in training months (boosting; county AUC | BG top-5% capture | BG NDCG5 paid)")
    for frac in (0.25, 0.5, 1.0):
        sel = e81.month_subset(d.days, frac)[dr] | ~d.is_train[dr]
        line = f"  {int(frac * 100):3d}% of the training months, {int((tr & sel).sum()):,} training rows:"
        cc = {}
        for name, Xm in sets.items():
            m = fit(Xm, (tr & sel) | (va & sel))
            T["q"] = m.predict_proba(Xm[te])[:, 1].astype(np.float32)
            t2, _ = event_scores(T, {"q": "q"}, e75)
            t2 = t2["q"]
            line += f"  {name}: {e75.pooled(t2, 'c_auc', 'c_w'):.3f} | {e75.pooled(t2, 'bg_cap_n_0.05', None):.3f} | {e75.pooled(t2, 'bg_ndcg_p', None):.3f};"
        say(line)
    save_log("latent_v0_v1.txt")


# ------------------------------------------------------------------ curve

def run_curve() -> None:
    e81 = _load("e81", "81_cnn_e1.py")
    e75 = _load("e75", "75_task_metrics.py")
    e75.OUT = OUT
    d = e81.Data()
    has = np.array([(d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any() for n in range(d.D)])
    tr = d.is_train
    # storms: claimed days separated by gaps of at least 3 days start a new episode
    cd = d.days[tr & has]
    ep = int((np.diff(cd.to_numpy()).astype("timedelta64[D]").astype(int) >= 3).sum() + 1)
    cb = int(sum((d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).sum() for n in np.flatnonzero(tr)))
    say(f"training sample: {int(tr.sum())} days, {int((tr & has).sum())} with a claim, about {ep} claim episodes (gap >= 3 days), {cb:,} claimed block-group days (overlapping 4-day windows)")
    big = int(sum(((d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).sum() >= 30) for n in np.flatnonzero(tr)))
    say(f"  days with >= 30 claimed block groups in training: {big}; the evaluation's big-event stratum holds 77.6% of the test claims")
    test = np.flatnonzero(d.is_test)
    lo, hi = d.ptr[test[0]], d.ptr[test[-1] + 1]
    day_of_row = np.repeat(np.arange(d.D), np.diff(d.ptr))[lo:hi]
    bg = d.rbg[lo:hi]
    T = pd.DataFrame({"sysid": d.days[day_of_row].strftime("%Y%m%d"), "cty": np.array(d.ctys)[d.cty_idx[bg]], "n": d.rn[lo:hi], "paid": d.rpaid[lo:hi], "n_pol": np.asarray(d.rtab[lo:hi, 8], np.float32)})
    say("\nsupervised 4 km CNN, learning curve in training months (mean over the seeds available; county AUC | BG top-5% capture | BG NDCG5 paid | BG AUC)")
    for lab, pat in (("25%", "lc25_pred_cnn4km_s*.npy"), ("50%", "lc50_pred_cnn4km_s*.npy"), ("100%", "e1_pred_cnn4km_s[01].npy")):
        fs = sorted(D.glob(pat))
        if not fs:
            continue
        res = []
        for f in fs:
            T["q"] = zstio.load(f)
            t, _ = event_scores(T, {"q": "q"}, e75)
            t = t["q"]
            res.append([e75.pooled(t, "c_auc", "c_w"), e75.pooled(t, "bg_cap_n_0.05", None), e75.pooled(t, "bg_ndcg_p", None), e75.pooled(t, "bg_auc", "bg_w")])
        r = np.array(res)
        say(f"  {lab:>4s} ({len(fs)} seeds): " + " | ".join(f"{m:.3f}" for m in r.mean(0)) + "   seed range of county AUC " + f"{r[:, 0].min():.3f}-{r[:, 0].max():.3f}")
    save_log("latent_v0_curve.txt")


def run_zcmp(tag: str) -> None:
    """V1c: the 0.25 degree CNN with the frozen latent channels against the same CNN without them, the 4 km CNN and boosting; negative control with shuffled latents."""
    e81 = _load("e81", "81_cnn_e1.py")
    e75 = _load("e75", "75_task_metrics.py")
    e75.OUT = OUT
    d = e81.Data()
    test = np.flatnonzero(d.is_test)
    lo, hi = d.ptr[test[0]], d.ptr[test[-1] + 1]
    day_of_row = np.repeat(np.arange(d.D), np.diff(d.ptr))[lo:hi]
    bg = d.rbg[lo:hi]
    T = pd.DataFrame({"sysid": d.days[day_of_row].strftime("%Y%m%d"), "cty": np.array(d.ctys)[d.cty_idx[bg]], "n": d.rn[lo:hi], "paid": d.rpaid[lo:hi], "n_pol": np.asarray(d.rtab[lo:hi, 8], np.float32)})
    groups = {"CNN 0.25 deg": "e1_pred_cnn025_s[0-2].npy", f"CNN 0.25 deg + latent {tag}": f"z1_pred_cnn025_{tag}_s*.npy", f"CNN 0.25 deg + shuffled latent {tag}": f"z1_pred_cnn025_{tag}shuf_s*.npy",
              "CNN 4 km": "e1_pred_cnn4km_s*.npy", "boosting summaries": "e1_pred_hgb.npy"}
    cols = {}
    for lab, pat in groups.items():
        fs = sorted(D.glob(pat))
        if fs:
            T["p_" + lab] = np.mean([zstio.load(f) for f in fs], axis=0)
            cols[lab] = "p_" + lab
            say(f"  {lab}: {len(fs)} runs")
    tabs, scores = event_scores(T, {**cols, "policies only": "n_pol"}, e75)
    metrics = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"), ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))
    say("\npoint values in the order: " + " / ".join(scores))
    base = "CNN 0.25 deg"
    for col, wcol, lab in metrics:
        pt = {k: e75.pooled(t, col, wcol) for k, t in tabs.items()}
        line = f"  {lab:24s} " + " / ".join(f"{pt[k]:.3f}" for k in scores)
        for o in scores:
            if o in (base, "policies only") or o.startswith("boosting"):
                continue
            _, dd = e75.boot({o: tabs[o], base: tabs[base]}, col, wcol, ref=base)
            line += f"   [{o[:34]} - CNN 0.25] {dd[o][0]:+.3f} [{dd[o][1]:+.3f}, {dd[o][2]:+.3f}]"
        say(line)
    save_log("latent_v0_zcmp.txt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "train", "probe", "v1", "curve", "zcmp"])
    ap.add_argument("--tag", default="c8_2000")
    ap.add_argument("--c", type=int, default=8)
    ap.add_argument("--from-year", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="", help="where checkpoints and the final model are written (default cnn/)")
    ap.add_argument("--smoke", type=int, default=0, help="train on this many windows only (test of the pipeline)")
    ap.add_argument("--tags", default="c8_2000")
    a = ap.parse_args()
    if a.mode == "build":
        run_build()
    elif a.mode == "train":
        run_train(a.tag, a.c, a.from_year, a.epochs, a.beta, a.seed, a.smoke, a.ckpt_dir)
    elif a.mode == "probe":
        run_probe(a.tag)
    elif a.mode == "v1":
        run_v1(a.tags.split(","))
    elif a.mode == "zcmp":
        run_zcmp(a.tag)
    else:
        run_curve()


if __name__ == "__main__":
    main()
