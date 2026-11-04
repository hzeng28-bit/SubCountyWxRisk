"""E1 of docs/CNN_HYBRID_DESIGN.md: does two-dimensional reading beat point and county summaries when the rain field is OBSERVED? (the gate)

Everything is on the all-days unit of scripts 77-80 (the day tensors and rows of `cnn/`, scope S0, train 2010-2016, test 2017-2019), observed rain only.
Modes:
  hgb    boosting baseline with the strongest summaries of the observed field: the tabular rows (log policies, SFHA share, single-family share, mean
         coverage, block groups in the county, hist00, prior3, month, policies) + observed rain at the 0.25 deg cell and its county peak / mean + observed rain at the
         4 km pixel and its county peak / mean. Two versions: iterations chosen on blocked validation months, and 200 iterations. Negatives subsampled 1 in 10, weight 10.
  train  CNN hybrid (src/wxrisk/cnn.py) on the observed field: --res 025 (observed peak and sum on the GEFS grid + exposure + coordinates + masks) or --res 4km (the
         4 km maps + exposure upsampled + coordinates + mask). One day per batch (the image and a sample of its block groups: all claimed rows and up to 2,000
         others, weights restoring the normaliser), listwise + 0.3 x binary cross entropy loss, validation on blocked months (about 1 in 7), best epoch kept.
  eval   the metrics of scripts 75-79 (event = a day; cross-county AUC, top-k capture, NDCG with paid amount, county AUC, top 10% of counties), paired differences with a
         bootstrap over days, overall and by observed size of the day and by the day's largest observed county peak; the spread over seeds.
Gate: the hybrid must beat the boosting baseline with an interval above zero (county AUC and top-10% county capture, block-group top-5% capture and NDCG).
Inputs : cnn/* (script 80)
Output : $WXRISK_DATA/cnn/e1_pred_*.npy, e1_hgb_features.npz, $WXRISK_DATA/audit/e1_cnn_observed.txt
Run    : python scripts/81_cnn_e1.py hgb ; python scripts/81_cnn_e1.py train --res 025 --seed 0 ; python scripts/81_cnn_e1.py eval
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

from wxrisk import flood_pilot as fp  # noqa: E402
from wxrisk import zstio  # noqa: E402
from wxrisk import wb2  # noqa: E402
from wxrisk.tblog import RunLog  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = wb2.cache_path("cnn")
OUT: list[str] = []
NNEG, LAM, ACCUM = 2000, 0.3, 4
# ablation variants of the 4 km input / loss (script 86): "" = the model of stage 42; the others change ONE thing
VARIANT = ""
VARIANTS = ("full", "nocoord", "nostatic", "shuffle_rain", "coarse_rain", "bce")


def coarse(a: np.ndarray, v: np.ndarray, k: int = 6) -> np.ndarray:
    """Mean of the valid pixels over k x k blocks, put back on the 4 km grid (what a 0.25 degree product would show)."""
    H, W = a.shape
    pad = ((0, -H % k), (0, -W % k))
    x, w = np.pad(a * v, pad), np.pad(v, pad)
    bs = x.reshape(x.shape[0] // k, k, x.shape[1] // k, k).sum((1, 3))
    bw = w.reshape(w.shape[0] // k, k, w.shape[1] // k, k).sum((1, 3))
    m = np.repeat(np.repeat(bs / np.maximum(bw, 1), k, 0), k, 1)[:H, :W]
    return (m * v).astype(np.float32)


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


class Data:
    """The arrays of script 80, the split and the pooling indices."""

    def __init__(self):
        self.st = zstio.load(D / "static.npz", allow_pickle=True)
        self.G = zstio.load(D / "gefs_day_tensors.npy")  # [day, 18, 53, 61] float16, in memory (424 MB)
        self.O = zstio.load(D / "obs4k_day_maps.npy", mmap_mode="r")
        self.ptr = zstio.load(D / "rows_ptr.npy")
        self.rbg = zstio.load(D / "rows_bg.npy")
        self.rtab = zstio.load(D / "rows_tab.npy", mmap_mode="r")
        self.rn = zstio.load(D / "rows_n.npy")
        self.rpaid = zstio.load(D / "rows_paid.npy")
        self.days = pd.DatetimeIndex(self.st["days"])
        self.D = len(self.days)
        yr = self.days.year.to_numpy()
        ym = yr * 12 + self.days.month.to_numpy()
        self.is_test = yr >= 2017
        self.is_val = (~self.is_test) & ((ym % 7) == 0)  # blocked validation months
        self.is_train = (~self.is_test) & ~self.is_val
        self.H, self.W = self.G.shape[2:]
        cov = self.st["obs_cover"]
        self.cov = cov
        self.obs_ok = cov >= 0.5
        self.ctys = [str(c) for c in self.st["ctys"]]
        self.cty_idx = self.st["cty_idx"].astype(np.int64)
        self.bpix025 = (self.st["bcell_y"].astype(np.int64) * self.W + self.st["bcell_x"].astype(np.int64))
        self.lat4, self.lon4 = self.st["lat4"], self.st["lon4"]
        self.H4, self.W4 = len(self.lat4), len(self.lon4)
        self.bpix4 = (self.st["b4_y"].astype(np.int64) * self.W4 + self.st["b4_x"].astype(np.int64))
        glat, glon = self.st["glat"], self.st["glon"]
        py = np.abs(glat[None, :] - self.lat4[:, None]).argmin(1)
        px = np.abs(glon[None, :] - self.lon4[:, None]).argmin(1)
        self.cell4 = (py[:, None] * self.W + px[None, :]).ravel()  # GEFS cell of every 4 km pixel
        self.valid4 = self.st["obs4_valid"]
        self.pools = {}
        self.z = None
        self.rain_perm = np.arange(self.D)
        if VARIANT == "shuffle_rain":  # every day gets the rain map of another day of the same calendar month
            rng = np.random.default_rng(0)
            mo = self.days.month.to_numpy()
            for m in range(1, 13):
                ix = np.flatnonzero(mo == m)
                self.rain_perm[ix] = rng.permutation(ix)

    def attach_latent(self, tag: str, shuffle: bool = False) -> None:
        """Frozen VAE latent (script 82) as extra channels on the GEFS grid: each cell takes the latent cell (6 x 6 gridMET pixels) that holds its centre.
        shuffle=True pairs every day with the latent of another day of the same calendar month (negative control)."""
        Z = zstio.load(D / f"z_{tag}.npy").astype(np.float32)  # [day, C, 44, 54]
        r = np.abs(self.st["glat"][:, None] - self.lat4[None, :]).argmin(1) // 6
        c = np.abs(self.st["glon"][:, None] - self.lon4[None, :]).argmin(1) // 6
        r, c = np.minimum(r, Z.shape[2] - 1), np.minimum(c, Z.shape[3] - 1)
        inside = (np.abs(self.st["glat"][:, None] - self.lat4[None, :]).min(1) < 0.15)[:, None] & (np.abs(self.st["glon"][:, None] - self.lon4[None, :]).min(1) < 0.15)[None, :]
        G = Z[:, :, r][:, :, :, c]  # [day, C, 53, 61]
        G = G * inside[None, None]
        mu = G[self.is_train].mean((0, 2, 3), keepdims=True)
        sd = G[self.is_train].std((0, 2, 3), keepdims=True) + 1e-6
        G = ((G - mu) / sd) * inside[None, None]
        if shuffle:
            rng = np.random.default_rng(0)
            perm = np.arange(self.D)
            mo = self.days.month.to_numpy()
            for m in range(1, 13):
                ix = np.flatnonzero(mo == m)
                perm[ix] = rng.permutation(ix)
            G = G[perm]
        self.z = G.astype(np.float32)

    def pool(self, res: str):
        """Padded county pooling indices at the resolution of the encoder input."""
        from wxrisk import cnn

        if res not in self.pools:
            if res == "025":
                groups = [np.unique(self.bpix025[self.cty_idx == k]) for k in range(len(self.ctys))]
            else:
                masks = fp.county_masks(self.lat4, self.lon4)
                groups = []
                for k, c in enumerate(self.ctys):
                    g = np.flatnonzero(masks[c].ravel()) if c in masks else np.array([], np.int64)
                    groups.append(g if len(g) else np.unique(self.bpix4[self.cty_idx == k]))
            self.pools[res] = cnn.pad_index(groups)
        return self.pools[res]

    def rows(self, n: int):
        sl = slice(self.ptr[n], self.ptr[n + 1])
        return self.rbg[sl], np.asarray(self.rtab[sl], np.float32), self.rn[sl], self.rpaid[sl]

    def obs_cell(self, n: int):
        """Observed peak and sum on the GEFS cells, corrected for partial gridMET coverage (cells below 50% coverage are missing)."""
        g = self.G[n]
        c = np.maximum(self.cov, 1e-6)
        return [np.where(self.obs_ok, np.asarray(g[i], np.float32) / c, np.nan) for i in (10, 11)]

    def image(self, res: str, n: int) -> np.ndarray:
        g = self.G[n]
        if res == "025":
            peak, ssum = self.obs_cell(n)
            ok = self.obs_ok
            ch = [np.log1p(np.nan_to_num(peak)) / 3, np.log1p(np.nan_to_num(ssum)) / 3, np.asarray(g[7], np.float32) / 12, np.asarray(g[8], np.float32),
                  np.asarray(g[9], np.float32) / 9, self.lat_map025, self.lon_map025, self.st["valid_tx"].astype(np.float32), ok.astype(np.float32)]
            if self.z is not None:
                ch = ch + list(self.z[n])
            return np.stack(ch).astype(np.float32)
        o = np.asarray(self.O[self.rain_perm[n]], np.float32)
        v = self.valid4.astype(np.float32)
        if VARIANT == "coarse_rain":
            o = np.stack([coarse(o[0], v), coarse(o[1], v)])
        exp = [np.asarray(g[i], np.float32).ravel()[self.cell4].reshape(self.H4, self.W4) for i in (7, 8, 9)]
        if VARIANT == "nostatic":
            exp = [np.zeros_like(e) for e in exp]
        lat, lon = (np.zeros_like(self.lat_map4), np.zeros_like(self.lon_map4)) if VARIANT == "nocoord" else (self.lat_map4, self.lon_map4)
        ch = [np.log1p(o[0]) / 3, np.log1p(o[1]) / 3, v, exp[0] / 12, exp[1], exp[2] / 9, lat, lon]
        return np.stack(ch).astype(np.float32)

    @property
    def lat_map025(self):
        glat, glon = self.st["glat"], self.st["glon"]
        return np.repeat(((glat - 31) / 7)[:, None], len(glon), 1).astype(np.float32)

    @property
    def lon_map025(self):
        glat, glon = self.st["glat"], self.st["glon"]
        return np.repeat(((glon + 100) / 8)[None, :], len(glat), 0).astype(np.float32)

    @property
    def lat_map4(self):
        return np.repeat(((self.lat4 - 31) / 7)[:, None], self.W4, 1).astype(np.float32)

    @property
    def lon_map4(self):
        return np.repeat(((self.lon4 + 100) / 8)[None, :], self.H4, 0).astype(np.float32)


def tab_features(tab: np.ndarray) -> np.ndarray:
    """Standardisation-ready tabular features of the CNN: log policies, SFHA, single-family, coverage, log block groups, hist00, prior3, month sin / cos."""
    m = tab[:, 7]
    return np.column_stack([tab[:, 0], tab[:, 1], tab[:, 2], tab[:, 3], np.log1p(tab[:, 4]), tab[:, 5], tab[:, 6],
                            np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]).astype(np.float32)


# ------------------------------------------------------------------ hgb

def obs_features(d: Data, n: int, bgs: np.ndarray) -> np.ndarray:
    """Observed-rain summaries of the rows of day n: cell peak and sum, county peak and mean (0.25 deg); 4 km pixel peak and sum, county peak and mean."""
    out = np.full((len(bgs), 8), np.nan, np.float32)
    peak, ssum = d.obs_cell(n)
    pix = d.bpix025[bgs]
    out[:, 0], out[:, 1] = peak.ravel()[pix], ssum.ravel()[pix]
    idx, ok = d.pool("025")
    v = peak.ravel()[idx]
    good = ok & np.isfinite(v)
    cmax = np.where(good, v, -np.inf).max(1)
    cmean = np.where(good, v, 0).sum(1) / np.maximum(good.sum(1), 1)
    cmax = np.where(np.isfinite(cmax), cmax, np.nan)
    k = d.cty_idx[bgs]
    out[:, 2], out[:, 3] = cmax[k], cmean[k]
    o = np.asarray(d.O[n], np.float32)
    p4, s4 = o[0].ravel(), o[1].ravel()
    out[:, 4], out[:, 5] = p4[d.bpix4[bgs]], s4[d.bpix4[bgs]]
    idx4, ok4 = d.pool("4km")
    v4 = p4[idx4]
    m4 = np.where(ok4, v4, -np.inf).max(1)
    a4 = np.where(ok4, v4, 0).sum(1) / np.maximum(ok4.sum(1), 1)
    out[:, 6], out[:, 7] = m4[k], a4[k]
    return out


def run_hgb() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    d = Data()
    rng = np.random.default_rng(0)
    R = len(d.rbg)
    y = (d.rn > 0)
    day_of_row = np.repeat(np.arange(d.D), np.diff(d.ptr))
    train_day = d.is_train[day_of_row]
    val_day = d.is_val[day_of_row]
    test_day = d.is_test[day_of_row]
    keep = (train_day | val_day) & (y | (rng.random(R) < 0.1))
    keep |= test_day
    say(f"rows {R:,}; kept for the baseline: train/val sample {int((keep & ~test_day).sum()):,}, test {int(test_day.sum()):,}")
    X = np.full((int(keep.sum()), 17), np.nan, np.float32)
    pos = 0
    t0 = time.time()
    for n in range(d.D):
        bgs, tab, _, _ = d.rows(n)
        sl = np.arange(d.ptr[n], d.ptr[n + 1])
        k = keep[sl]
        if not k.any():
            continue
        f = obs_features(d, n, bgs)
        X[pos:pos + int(k.sum()), :9] = tab[k]
        X[pos:pos + int(k.sum()), 9:] = f[k]
        pos += int(k.sum())
        if n % 700 == 0:
            say(f"  features day {n} / {d.D} ({time.time() - t0:.0f} s)")
    rid = np.flatnonzero(keep)
    yk = y[rid].astype(np.float32)
    w = np.where(yk == 1, 1.0, 10.0).astype(np.float32)
    tr, va, te = train_day[rid], val_day[rid], test_day[rid]
    np.savez(D / "e1_hgb_features.npz", rid=rid, X=X)

    def mk(n):
        return HistGradientBoostingClassifier(max_iter=n, learning_rate=0.05, max_depth=5, min_samples_leaf=300, l2_regularization=1.0, early_stopping=False, random_state=0)
    m = mk(300).fit(X[tr], yk[tr], sample_weight=w[tr])
    sc = [log_loss(yk[va], np.clip(p[:, 1], 1e-7, 1 - 1e-7), sample_weight=w[va], labels=[0, 1]) for p in m.staged_predict_proba(X[va])]  # Brier picks 2 iterations on rare events
    n_it = int(np.argmin(sc)) + 1
    say(f"validation (blocked months) picks {n_it} iterations")
    fit_rows = tr | va
    for name, n_fit in (("hgb", n_it), ("hgb200", 200)):
        mdl = mk(n_fit).fit(X[fit_rows], yk[fit_rows], sample_weight=w[fit_rows])
        p = mdl.predict_proba(X[te])[:, 1].astype(np.float32)
        np.save(D / f"e1_pred_{name}.npy", p)
        say(f"fitted {name} ({n_fit} iterations), test rows {len(p):,}, mean prediction {p.mean() * 100:.3f}%")


# ------------------------------------------------------------------ cnn

def month_subset(days: pd.DatetimeIndex, frac: float) -> np.ndarray:
    """Deterministic subset of calendar months (storms stay whole) for the learning curves: month ym is kept when its hash falls below frac."""
    ym = days.year.to_numpy().astype(np.int64) * 12 + days.month.to_numpy()
    return ((ym * 2654435761) % 1000) / 1000.0 < frac


def day_loss(cnn, logit, b):
    """The listwise + binary loss of stage 41; variant "bce" keeps only the weighted binary cross entropy (no within-day ranking term)."""
    if VARIANT == "bce":
        import torch

        pos = b[4] > 0
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, pos.to(logit.dtype), reduction="none")
        return (b[5] * bce).sum() / pos.sum().clamp(min=1)
    return cnn.day_loss(logit, b[4], b[5], LAM)


def run_train(res: str, seed: int, epochs: int, frac: float = 1.0, z: str = "", z_shuffle: bool = False) -> None:
    import torch

    from wxrisk import cnn, compute

    d = Data()
    if z:
        d.attach_latent(z, z_shuffle)
    dev = compute.device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    say(f"E1 CNN {res} seed {seed}: {compute.describe()}; days train {int(d.is_train.sum())}, val {int(d.is_val.sum())}, test {int(d.is_test.sum())}")
    sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in rng.choice(np.flatnonzero(d.is_train), 300, replace=False)])
    tf = tab_features(sample)
    mu, sd = tf.mean(0), tf.std(0) + 1e-6
    idx_np, ok_np = d.pool(res)
    pool_idx, pool_ok = torch.from_numpy(idx_np).to(dev), torch.from_numpy(ok_np).to(dev)
    in_ch = d.image(res, 0).shape[0]
    model = cnn.Hybrid(in_ch, 9, dilations=(1, 2, 4, 1) if res == "025" else (1, 2, 4, 8, 16)).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    pixmap = d.bpix025 if res == "025" else d.bpix4
    W = d.W if res == "025" else d.W4

    def day_batch(n, sampled: bool, r: np.random.Generator):
        bgs, tab, cnt, _ = d.rows(n)
        pos = np.flatnonzero(cnt > 0)
        neg = np.flatnonzero(cnt <= 0)
        w = np.ones(len(bgs), np.float32)
        if sampled and len(neg) > NNEG:
            sel = r.choice(neg, NNEG, replace=False)
            w[sel] = len(neg) / NNEG
            rows = np.sort(np.concatenate([pos, sel]))
        else:
            rows = np.arange(len(bgs))
        img = torch.from_numpy(d.image(res, n)[None]).to(dev)
        tt = torch.from_numpy(((tab_features(tab[rows]) - mu) / sd).astype(np.float32)).to(dev)
        return (img, torch.from_numpy(pixmap[bgs[rows]]).to(dev), torch.from_numpy(d.cty_idx[bgs[rows]]).to(dev), tt,
                torch.from_numpy(cnt[rows]).to(dev), torch.from_numpy(w[rows]).to(dev), rows)

    def day_logit(b):
        return model(b[0], b[1], b[2], b[3], pool_idx, pool_ok)

    has_pos = np.array([(d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any() for n in range(d.D)])
    train_days = np.flatnonzero(d.is_train & month_subset(d.days, frac))
    say(f"training days after the month subset ({frac:.0%}): {len(train_days)}")
    val_days = np.flatnonzero(d.is_val & (has_pos | (np.random.default_rng(1).random(d.D) < 0.2)))
    steps = epochs * int(((has_pos[train_days]).sum() + 0.25 * (~has_pos[train_days]).sum()) // ACCUM)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10), pct_start=0.15)
    tb = RunLog(f"e1/{res}_{VARIANT or 'base'}_s{seed}" + (f"_f{frac:g}" if frac < 1 else "") + (f"_{z}" if z else ""), {"res": res, "seed": seed, "epochs": epochs, "frac": frac, "variant": VARIANT or "full", "parameters": npar})
    gstep = 0
    best, best_state = np.inf, None
    say(f"parameters {npar:,}; input channels {in_ch}; epochs {epochs}")
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = np.concatenate([train_days[has_pos[train_days]], rng.choice(train_days[~has_pos[train_days]], int(0.25 * (~has_pos[train_days]).sum()), replace=False)])
        rng.shuffle(order)
        tot, k, last = 0.0, 0, 0.0
        for i, n in enumerate(order):
            b = day_batch(int(n), True, rng)
            loss = day_loss(cnn, day_logit(b), b) / ACCUM
            loss.backward()
            tot += float(loss) * ACCUM
            k += 1
            if (i + 1) % ACCUM == 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                gstep += 1
                tb.scalar("train/loss_step", (tot - last) / ACCUM, gstep)
                tb.scalar("train/grad_norm", float(gn), gstep)
                tb.scalar("train/lr", opt.param_groups[0]["lr"], gstep)
                last = tot
                opt.step()
                opt.zero_grad()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
        model.eval()
        vr = np.random.default_rng(123)
        with torch.no_grad():
            vl = np.mean([float(day_loss(cnn, day_logit(b), b)) for b in (day_batch(int(n), True, vr) for n in val_days)])
        say(f"  epoch {ep + 1}: train loss {tot / k:.4f}, validation loss {vl:.4f} ({time.time() - t0:.0f} s)")
        tb.scalars("epoch/loss", {"train": tot / k, "val": vl}, ep + 1)
        tb.scalar("epoch/seconds", time.time() - t0, ep + 1)
        tb.scalar("epoch/days_per_s", k / max(time.time() - t0, 1e-6), ep + 1)
        tb.scalar("epoch/best_val", min(best, vl), ep + 1)
        tb.weights(model, ep + 1)
        if vl < best:
            best, best_state = vl, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    tb.close()
    model.eval()
    preds = []
    with torch.no_grad():
        for n in np.flatnonzero(d.is_test):
            b = day_batch(int(n), False, rng)
            preds.append(torch.sigmoid(day_logit(b)).cpu().numpy())
    p = np.concatenate(preds).astype(np.float32)
    if VARIANT:
        torch.save(best_state, D / f"w_{VARIANT}_s{seed}.pt")
        np.save(D / f"v_{VARIANT}_s{seed}.npy", p)
        say(f"saved {VARIANT} predictions ({len(p):,} rows), best validation loss {best:.4f}")
        (wb2.cache_path("audit") / f"v_{VARIANT}_train_s{seed}.txt").write_text("\n".join(OUT) + "\n")
        return
    np.save(D / (f"z1_pred_cnn{res}_{z}{'shuf' if z_shuffle else ''}_s{seed}.npy" if z else f"e1_pred_cnn{res}_s{seed}.npy" if frac >= 1 else f"lc{int(round(frac * 100))}_pred_cnn{res}_s{seed}.npy"), p)
    say(f"saved test predictions ({len(p):,} rows, mean {p.mean() * 100:.3f}%), best validation loss {best:.4f}")
    (wb2.cache_path("audit") / (f"z1_train_{res}_{z}{'shuf' if z_shuffle else ''}_s{seed}.txt" if z else f"e1_train_{res}_s{seed}.txt" if frac >= 1 else f"lc{int(round(frac * 100))}_train_{res}_s{seed}.txt")).write_text("\n".join(OUT) + "\n")


# ------------------------------------------------------------------ eval

def run_eval() -> None:
    e75 = _load("e75", "75_task_metrics.py")
    e75.OUT = OUT
    d = Data()
    test = np.flatnonzero(d.is_test)
    lo, hi = d.ptr[test[0]], d.ptr[test[-1] + 1]
    rows = np.arange(lo, hi)
    day_of_row = np.repeat(np.arange(d.D), np.diff(d.ptr))[lo:hi]
    bg = d.rbg[lo:hi]
    T = pd.DataFrame({"sysid": d.days[day_of_row].strftime("%Y%m%d"), "cty": np.array(d.ctys)[d.cty_idx[bg]], "n": d.rn[lo:hi], "paid": d.rpaid[lo:hi],
                      "n_pol": np.asarray(d.rtab[lo:hi, 8], np.float32), "year": d.days.year.to_numpy()[day_of_row]})
    feats = zstio.load(D / "e1_hgb_features.npz")
    pos = np.searchsorted(feats["rid"], rows)
    T["obs_cty_max"] = feats["X"][pos, 11]  # columns 9-16: cell peak, cell sum, county peak, county mean (0.25 deg); 4 km pixel peak, sum, county peak, mean
    names = {}
    for f in sorted(D.glob("e1_pred_*.npy")):
        col = f.stem.replace("e1_pred_", "p_")
        T[col] = zstio.load(f)
        names[col] = f.stem
    cnn_cols = [c for c in T.columns if c.startswith("p_cnn")]
    for res in ("025", "4km"):
        cs = [c for c in cnn_cols if c.startswith(f"p_cnn{res}_s")]
        if cs:
            T[f"p_cnn{res}"] = T[cs].mean(axis=1)
    say(f"test rows {len(T):,}, days {T.sysid.nunique()}, claimed block groups {int((T.n > 0).sum()):,}; predictions: {sorted(names)}")
    scores = {}
    for lab, col in (("CNN 0.25 deg (seed mean)", "p_cnn025"), ("CNN 4 km (seed mean)", "p_cnn4km"), ("boosting, observed summaries (validation iterations)", "p_hgb"),
                     ("boosting, observed summaries (200 iterations)", "p_hgb200")):
        if col in T:
            scores[lab] = (col, "sum")
    scores["policies only"] = ("n_pol", "sum")
    metrics = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"),
               ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))

    def stratum(name, days):
        sub = T if days is None else T[T.sysid.isin(days)]
        tabs = e75.event_table(sub, scores)
        first = next(iter(tabs))
        say(f"\n  {name}: {sub.sysid.nunique()} days, {int(tabs[first].bg_auc.notna().sum())} scored, {sub.n.sum() / T.n.sum():.1%} of the test claim count")
        refs = [k for k in scores if k.startswith("boosting")]
        cnns = [k for k in scores if k.startswith("CNN")]
        for col, w, lab in metrics:
            pt = {k: e75.pooled(t, col, w) for k, t in tabs.items()}
            line = f"    {lab:24s} " + " / ".join(f"{pt[k]:.3f}" for k in scores)
            for c in cnns:
                for r in refs[:1]:
                    _, dd = e75.boot({c: tabs[c], r: tabs[r]}, col, w, ref=r)
                    line += f"   [{c.split(' (')[0]} - boosting] {dd[c][0]:+.3f} [{dd[c][1]:+.3f}, {dd[c][2]:+.3f}]"
            say(line)
    say("score order in the point values: " + " / ".join(scores))
    say("\n== overall")
    stratum("all test days", None)
    g = T.groupby("sysid")
    day = pd.DataFrame({"claimed": g.n.apply(lambda s: int((s > 0).sum())), "obsmax": g.obs_cty_max.max()})
    say("\n== by observed size of the day (claimed block groups)")
    for lab, lo_, hi_ in (("1-2", 1, 2), ("3-9", 3, 9), ("10-29", 10, 29), (">= 30", 30, 10 ** 9)):
        stratum(lab, day.index[(day.claimed >= lo_) & (day.claimed <= hi_)])
    say("\n== by the day's largest observed county peak (mm)")
    for lab, lo_, hi_ in (("< 25", 0, 25), ("25-50", 25, 50), ("50-100", 50, 100), (">= 100", 100, 1e9)):
        stratum(lab, day.index[(day.obsmax >= lo_) & (day.obsmax < hi_)])
    say("\n== seed spread (all test days; county AUC | block-group top-5% capture)")
    for res in ("025", "4km"):
        cs = [c for c in cnn_cols if c.startswith(f"p_cnn{res}_s")]
        if len(cs) > 1:
            tabs = e75.event_table(T, {c: (c, "sum") for c in cs})
            ca = [e75.pooled(t, "c_auc", "c_w") for t in tabs.values()]
            bc = [e75.pooled(t, "bg_cap_n_0.05", None) for t in tabs.values()]
            say(f"  CNN {res}: county AUC {min(ca):.3f}-{max(ca):.3f} (sd {np.std(ca):.4f}); top-5% capture {min(bc):.3f}-{max(bc):.3f}; seeds {len(cs)}")
    T[["sysid", "cty", "n", "paid", "n_pol"] + [c for c in T.columns if c.startswith("p_")]].to_parquet(D / "e1_test_rows.parquet")
    (wb2.cache_path("audit") / "e1_cnn_observed.txt").write_text("\n".join(OUT) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["hgb", "train", "eval"])
    ap.add_argument("--res", choices=["025", "4km"], default="025")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--z", default="", help="tag of a frozen VAE latent (script 82) added as channels (0.25 deg only)")
    ap.add_argument("--z-shuffle", action="store_true", help="negative control: latent of another day of the same month")
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training months (learning curve)")
    ap.add_argument("--variant", choices=VARIANTS, default="", help="one-change ablation of the 4 km model (script 86)")
    a = ap.parse_args()
    global VARIANT
    VARIANT = a.variant
    if a.mode == "hgb":
        run_hgb()
        (wb2.cache_path("audit") / "e1_hgb.txt").write_text("\n".join(OUT) + "\n")
    elif a.mode == "train":
        run_train(a.res, a.seed, a.epochs, a.frac, a.z, a.z_shuffle)
    else:
        run_eval()


if __name__ == "__main__":
    main()
