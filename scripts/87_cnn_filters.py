"""Which neighbourhood patterns did the 4 km CNN learn? Filter groups and activation diagnostics (RESULTS stage 47).

Trained model: `81 train --variant full` (w_full_s<seed>.pt; stem 8 -> 32, five dilated residual blocks, 1x1 to a 16-dim embedding). Input channels:
0 rain peak, 1 rain 2-day sum (both log1p / 3), 2 valid mask, 3-5 exposure maps, 6-7 latitude / longitude.
  extract  (1) stem filters: 3 x 3 kernels of the two rain channels split into mean (DC), gradient and centre-minus-surround parts, input-channel norms;
           (2) activation statistics of the stem, the five blocks and the embedding on 100 training days with claims: spread, dead channels, correlation of every
           channel with the rain at 1 / 3 / 9 / 33 pixel scale and with centre-minus-surround (single pixel minus the 9-pixel mean);
           (3) embedding footprints: mean signed and absolute input gradient of every embedding channel at the pixel of a claimed block group (+-40 pixels);
           (4) synthetic rain patterns (point, blobs, ring, hole, bands, edges, uniform +- spike) placed at two busy block groups, on a dry and a wet background:
               response of the 16 own-pixel embedding channels and of the logit; and the response MAP of the logit / embedding to a 50 mm blob moved over +-40 pixels;
           (5) the pixels where each embedding channel is most / least active on the test days: mean rain crop around them;
           (6) own-pixel embedding of every test row of the claim days (cache for a later 'boosting on the embedding' test).
  groups   k-means (silhouette-chosen k) of the stem filters and of the 16 embedding channels with PCA projections; per-channel and per-group replacement of the channel by
           its training mean in the embedding map (before the county pooling) and the ranking metrics of the test days.
  plot     figures to docs/figures/cnn_filters/, summary text to $WXRISK_DATA/audit/cnn_filters_s<seed>.txt
Inputs : cnn/* (script 80), cnn/w_full_s<seed>.pt (script 81)
Output : $WXRISK_DATA/cnn/filters_s<seed>.npz, own_emb_s<seed>.npy, chan_abl_s<seed>.npz
Run    : python scripts/87_cnn_filters.py extract ; ... groups ; ... plot       (--seed 0 / 1)
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

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
FIG = Path(__file__).resolve().parents[1] / "docs" / "figures" / "cnn_filters"
OUT: list[str] = []
HALF = 40  # footprint half window (pixels); receptive field is +-32
RINGS = [0, 1, 2, 4, 8, 16, 32, 10 ** 6]
CH = ["peak", "sum", "valid", "exp0", "exp1", "exp2", "lat", "lon"]
LAYERS = ["stem"] + [f"block{i}" for i in range(5)] + ["emb"]
SCALES = {"r1": 1, "r3": 3, "r9": 9, "r33": 33}


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


# ------------------------------------------------------------------ model and data

class Ctx:
    """Data, trained model, tabular standardisation and the county pooling of one seed."""

    def __init__(self, seed: int):
        import torch

        from wxrisk import cnn, compute

        self.torch, self.seed = torch, seed
        self.d = d = e81.Data()
        self.dev = compute.device()
        rng = np.random.default_rng(seed)
        sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in rng.choice(np.flatnonzero(d.is_train), 300, replace=False)])
        tf = e81.tab_features(sample)
        self.mu, self.sd = tf.mean(0), tf.std(0) + 1e-6
        idx, ok = d.pool("4km")
        self.pool_idx, self.pool_ok = torch.from_numpy(idx).to(self.dev), torch.from_numpy(ok).to(self.dev)
        self.model = cnn.Hybrid(d.image("4km", 0).shape[0], 9, dilations=(1, 2, 4, 8, 16)).to(self.dev)
        self.model.load_state_dict(torch.load(D / f"w_full_s{seed}.pt", map_location=self.dev))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.valid = torch.from_numpy(d.valid4.astype(np.float32)).to(self.dev)
        self.claim_days = np.array([n for n in np.flatnonzero(d.is_test) if (d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any()])
        self.train_claim_days = np.array([n for n in np.flatnonzero(d.is_train) if (d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any()])

    def img(self, n: int):
        return self.torch.from_numpy(self.d.image("4km", int(n))[None]).to(self.dev)

    def tab_t(self, tab: np.ndarray):
        return self.torch.from_numpy(((e81.tab_features(tab) - self.mu) / self.sd).astype(np.float32)).to(self.dev)

    def pooled(self, emb):
        """emb [B, E, H, W] -> county mean and max maps [B, K, E] (K counties)."""
        torch = self.torch
        flat = emb.reshape(emb.shape[0], emb.shape[1], -1)
        g = flat[:, :, self.pool_idx.clamp(min=0)]  # [B, E, K, L]
        ok = self.pool_ok[None, None]
        mean = (g * ok).sum(-1) / ok.sum(-1).clamp(min=1)
        mx = g.masked_fill(~ok, float("-inf")).amax(-1)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return mean, mx  # [B, E, K]

    def logit_rows(self, emb, bgs: np.ndarray, tab_t):
        """Logit of block-group rows from one embedding map [1, E, H, W]."""
        torch, d = self.torch, self.d
        flat = emb[0].reshape(emb.shape[1], -1).T
        own = flat[torch.from_numpy(d.bpix4[bgs]).to(self.dev)]
        mean, mx = self.pooled(emb)
        c = torch.from_numpy(d.cty_idx[bgs]).to(self.dev)
        z = torch.cat([own, mean[0].T[c], mx[0].T[c], self.model.tab(tab_t)], 1)
        return self.model.head(z).squeeze(-1), own


# ------------------------------------------------------------------ (1) stem filters

def stem_descriptors(W: np.ndarray) -> dict[str, np.ndarray]:
    """W [32, 8, 3, 3]. For the two rain channels: DC = sum of the kernel, gradient vector (east - west, south - north), centre minus the mean of the 8 neighbours."""
    out = {}
    for c, nm in ((0, "peak"), (1, "sum")):
        K = W[:, c]
        out[f"{nm}_dc"] = K.sum((1, 2))
        out[f"{nm}_gx"] = (K[:, :, 2] - K[:, :, 0]).sum(1)
        out[f"{nm}_gy"] = (K[:, 2, :] - K[:, 0, :]).sum(1)
        out[f"{nm}_cs"] = K[:, 1, 1] - (K.sum((1, 2)) - K[:, 1, 1]) / 8
    out["norm"] = np.sqrt((W ** 2).sum((2, 3)))  # [32, 8]
    return out


# ------------------------------------------------------------------ (2) activation statistics

def run_activations(c: Ctx, rng, n_days: int = 100) -> dict:
    torch = c.torch
    F = torch.nn.functional
    m = c.model
    days = rng.choice(c.train_claim_days, min(n_days, len(c.train_claim_days)), replace=False)
    names = list(SCALES) + ["contrast"]
    acc = {l: {k: 0.0 for k in ("n", "sx", "sxx", "smax", "sact")} | {f"sy_{r}": 0.0 for r in names} | {f"syy_{r}": 0.0 for r in names} | {f"sxy_{r}": 0.0 for r in names}
           for l in LAYERS}
    ch_sizes = {l: (16 if l == "emb" else 32) for l in LAYERS}
    for l in LAYERS:
        for k in list(acc[l]):
            acc[l][k] = torch.zeros(ch_sizes[l], dtype=torch.float64, device="cpu") if k not in ("n",) else 0.0
    v = c.valid[None, None]
    with torch.no_grad():
        for n in days:
            x = c.img(n)
            h = m.stem(x)
            acts = {"stem": h}
            for i, b in enumerate(m.blocks):
                h = b(h)
                acts[f"block{i}"] = h
            acts["emb"] = m.out(h)
            r0 = x[:, :1]
            reg = {nm: F.avg_pool2d(r0, k, 1, k // 2, count_include_pad=False)[..., :r0.shape[2], :r0.shape[3]] if k > 1 else r0 for nm, k in SCALES.items()}
            reg["contrast"] = r0 - reg["r9"]
            for l, a in acts.items():
                a = a[0]  # [C, H, W]; float32 on device, accumulated in float64 on the host
                mk = v[0, 0].bool()
                A = a[:, mk]  # [C, N]
                s = acc[l]
                s["n"] += A.shape[1]
                s["sx"] += A.sum(1).cpu().double()
                s["sxx"] += (A ** 2).sum(1).cpu().double()
                for nm, y in reg.items():
                    Y = y[0, 0][mk]
                    s[f"sy_{nm}"] += Y.sum().cpu().double()
                    s[f"syy_{nm}"] += (Y ** 2).sum().cpu().double()
                    s[f"sxy_{nm}"] += (A * Y[None]).sum(1).cpu().double()
    res = {}
    for l in LAYERS:
        s = acc[l]
        n = s["n"]
        mean = (s["sx"] / n).numpy()
        std = np.sqrt(np.maximum((s["sxx"] / n).numpy() - mean ** 2, 0))
        corr = []
        for nm in names:
            my = (s[f"sy_{nm}"][0] / n).item()
            vy = (s[f"syy_{nm}"][0] / n).item() - my ** 2
            cov = (s[f"sxy_{nm}"] / n).numpy() - mean * my
            corr.append(cov / np.sqrt(np.maximum(std ** 2 * vy, 1e-18)))
        res[f"act_{l}_mean"], res[f"act_{l}_std"], res[f"act_{l}_corr"] = mean, std, np.array(corr).T  # corr [C, 5]
    res["act_corr_names"] = np.array(names)
    return res


# ------------------------------------------------------------------ (3) embedding footprints

def run_footprints(c: Ctx, rng, n_days: int = 80, per_day: int = 3) -> dict:
    torch, d = c.torch, c.d
    H, W = d.H4, d.W4
    G = np.zeros((16, 2, 2 * HALF + 1, 2 * HALF + 1))
    GA = np.zeros_like(G)
    n_s = 0
    inp = None
    for n in rng.choice(c.claim_days, n_days, replace=False):
        bgs, tab, cnt, _ = d.rows(int(n))
        pos = np.flatnonzero(cnt > 0)
        x = c.img(n).requires_grad_(True)
        e = c.model.embed(x)
        for r in rng.choice(pos, min(per_day, len(pos)), replace=False):
            y0, x0 = int(d.st["b4_y"][bgs[r]]), int(d.st["b4_x"][bgs[r]])
            for k in range(16):
                (g,) = torch.autograd.grad(e[0, k, y0, x0], x, retain_graph=True)
                g = g[0, :2].cpu().numpy()
                pad = np.pad(g, ((0, 0), (HALF, HALF), (HALF, HALF)))
                crop = pad[:, y0:y0 + 2 * HALF + 1, x0:x0 + 2 * HALF + 1]
                G[k] += crop
                GA[k] += np.abs(crop)
            n_s += 1
    return {"fp_signed": G / n_s, "fp_abs": GA / n_s, "fp_n": np.array(n_s)}


# ------------------------------------------------------------------ (4) synthetic patterns

def sites(c: Ctx):
    """Two busy block groups: most claimed training days overall, and the most claimed one at least 2 degrees away."""
    d = c.d
    row_day = np.repeat(np.arange(d.D), np.diff(d.ptr))
    tr = d.is_train[row_day]
    cnt = np.bincount(d.rbg[tr], weights=(d.rn[tr] > 0), minlength=len(d.st["b4_y"]))
    order = np.argsort(-cnt)
    lat = d.lat4[d.st["b4_y"]]
    lon = d.lon4[d.st["b4_x"]]
    a = int(order[0])
    b = next(int(o) for o in order if np.hypot(lat[o] - lat[a], lon[o] - lon[a]) > 2.0)
    return [a, b]


def patterns(y0, x0, H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    dy, dx = yy - y0, xx - x0
    r = np.hypot(dy, dx)
    P = 50.0
    out = {"point": np.where(r < 0.5, P, 0.0)}
    for s in (1, 2, 4, 8):
        out[f"blob_s{s}"] = P * np.exp(-r ** 2 / (2 * s ** 2))
    out["ring_6-8"] = np.where((r >= 6) & (r <= 8), P, 0.0)
    out["hole_r12"] = np.where((r <= 12) & (r > 1.5), P, 0.0)
    for ang in (0, 45, 90):
        t = np.deg2rad(ang)
        u, w = dx * np.cos(t) + dy * np.sin(t), -dx * np.sin(t) + dy * np.cos(t)
        out[f"band_{ang}"] = np.where(np.abs(u) <= 20, P * np.exp(-w ** 2 / (2 * 1.5 ** 2)), 0.0)
        out[f"band_{ang}_off8"] = np.where(np.abs(u) <= 20, P * np.exp(-(w - 8) ** 2 / (2 * 1.5 ** 2)), 0.0)
    out["uniform_25"] = np.full((H, W), 25.0)
    out["uniform_25+spike"] = np.full((H, W), 25.0) + np.where(r < 0.5, 25.0, 0.0)
    out["uniform_25-dry_centre"] = np.where(r <= 1.5, 0.0, 25.0)
    out["edge_wet_W"] = np.where(dx <= 0, P, 0.0)
    out["edge_wet_E"] = np.where(dx >= 0, P, 0.0)
    out["edge_wet_N"] = np.where(dy <= 0, P, 0.0)
    out["edge_wet_S"] = np.where(dy >= 0, P, 0.0)
    return {k: v.astype(np.float32) for k, v in out.items()}


def with_rain(c: Ctx, base: np.ndarray, r: np.ndarray, bg_mm: float) -> np.ndarray:
    """Replace the two rain channels of the base image: peak r (mm, plus a uniform bg_mm), 2-day sum 1.4 x that."""
    v = c.d.valid4.astype(np.float32)
    img = base.copy()
    pk = r + bg_mm
    img[0] = np.log1p(pk) / 3 * v
    img[1] = np.log1p(1.4 * pk) / 3 * v
    return img


def run_synthetic(c: Ctx) -> dict:
    torch, d = c.torch, c.d
    H, W = d.H4, d.W4
    res = {}
    row_day = np.repeat(np.arange(d.D), np.diff(d.ptr))
    for si, b in enumerate(sites(c)):
        y0, x0 = int(d.st["b4_y"][b]), int(d.st["b4_x"][b])
        rows = np.flatnonzero(d.rbg == b)
        rows = rows[np.isin(row_day[rows], c.claim_days)]
        tab = np.median(e81.tab_features(np.asarray(d.rtab[rows], np.float32)), axis=0)
        tab_t = torch.from_numpy(((tab - c.mu) / c.sd).astype(np.float32))[None].to(c.dev)
        nday = int(c.claim_days[np.abs(c.claim_days - c.claim_days[len(c.claim_days) // 2]).argmin()])
        base = d.image("4km", nday)
        pats = patterns(y0, x0, H, W)
        res[f"site{si}_bg"] = np.array(b)
        res[f"site{si}_yx"] = np.array([y0, x0])
        res[f"site{si}_names"] = np.array(list(pats))

        def run(imgs):
            """imgs [B, 8, H, W] -> logit [B], own-pixel embedding [B, 16], rain at the pixel."""
            lg, ow = [], []
            with torch.no_grad():
                for i in range(0, len(imgs), 24):
                    x = torch.from_numpy(np.stack(imgs[i:i + 24])).to(c.dev)
                    e = c.model.embed(x)
                    mean, mx = c.pooled(e)
                    own = e[:, :, y0, x0]
                    cc = int(d.cty_idx[b])
                    z = torch.cat([own, mean[:, :, cc], mx[:, :, cc], c.model.tab(tab_t).expand(len(x), -1)], 1)
                    lg.append(c.model.head(z).squeeze(-1).cpu().numpy())
                    ow.append(own.cpu().numpy())
            return np.concatenate(lg), np.concatenate(ow)

        for bgname, bgmm in (("dry", 0.0), ("wet", 15.0)):
            names = ["none"] + list(pats)
            imgs = [with_rain(c, base, np.zeros((H, W), np.float32), bgmm)] + [with_rain(c, base, p, bgmm) for p in pats.values()]
            lg, ow = run(imgs)
            res[f"site{si}_{bgname}_logit"] = lg
            res[f"site{si}_{bgname}_emb"] = ow
            # response MAP of a 50 mm sigma-1.5 blob moved over +-HALF pixels
            offs = np.arange(-HALF, HALF + 1, 2)
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            imgs2 = [with_rain(c, base, 50 * np.exp(-((yy - y0 - oy) ** 2 + (xx - x0 - ox) ** 2) / (2 * 1.5 ** 2)).astype(np.float32), bgmm) for oy in offs for ox in offs]
            lg2, ow2 = run(imgs2)
            res[f"site{si}_{bgname}_map_logit"] = lg2.reshape(len(offs), len(offs))
            res[f"site{si}_{bgname}_map_emb"] = ow2.reshape(len(offs), len(offs), 16)
            say(f"synthetic site {si} ({bgname}): logit none {lg[0]:+.2f}, blob_s2 {lg[1 + list(pats).index('blob_s2')]:+.2f}")
        res["map_offsets"] = offs
    return res


# ------------------------------------------------------------------ (5) top activations and (6) own embedding cache

def run_test_pass(c: Ctx, top: int = 200, crop: int = 20) -> dict:
    """One pass over the claim test days: own-pixel embedding of every row (cache), top / bottom pixels per channel, mean rain crop of the extremes."""
    torch, d = c.torch, c.d
    H, W = d.H4, d.W4
    own_all, hits_hi, hits_lo = [], [], []
    valid = c.valid.bool()
    with torch.no_grad():
        for n in c.claim_days:
            x = c.img(n)
            e = c.model.embed(x)[0]  # [16, H, W]
            bgs, _, _, _ = d.rows(int(n))
            own_all.append(e.reshape(16, -1)[:, torch.from_numpy(d.bpix4[bgs]).to(c.dev)].T.cpu().numpy())
            f = e.masked_fill(~valid[None], float("nan")).reshape(16, -1)
            hi = torch.nan_to_num(f, nan=-1e9).topk(5, dim=1)
            lo = torch.nan_to_num(f, nan=1e9).topk(5, dim=1, largest=False)
            hits_hi.append((hi.values.cpu().numpy(), hi.indices.cpu().numpy(), n))
            hits_lo.append((lo.values.cpu().numpy(), lo.indices.cpu().numpy(), n))
    np.save(D / f"own_emb_s{c.seed}.npy", np.concatenate(own_all).astype(np.float16))
    say(f"own-pixel embedding cache: {sum(len(a) for a in own_all):,} rows")
    out = {}
    img_cache = {}

    # sort selections by day so the one-image cache is reused
    for tag, hits, largest in (("hi", hits_hi, True), ("lo", hits_lo, False)):
        res = np.zeros((16, 2, 2 * crop + 1, 2 * crop + 1))
        for k in range(16):
            vals = np.concatenate([h[0][k] for h in hits])
            idx = np.concatenate([h[1][k] for h in hits])
            dd = np.concatenate([np.full(5, h[2]) for h in hits])
            o = np.argsort(-vals if largest else vals)[:top]
            o = o[np.argsort(dd[o], kind="stable")]
            for n, p in zip(dd[o], idx[o]):
                if n not in img_cache:
                    img_cache.clear()
                    img_cache[n] = d.image("4km", int(n))[:2]
                im = np.pad(img_cache[n], ((0, 0), (crop, crop), (crop, crop)))
                y, xx = divmod(int(p), W)
                res[k] += im[:, y:y + 2 * crop + 1, xx:xx + 2 * crop + 1]
            res[k] /= top
        out[f"top_{tag}"] = res
    # unconditional mean crop at claimed-block-group pixels (reference)
    ref = np.zeros((2, 2 * crop + 1, 2 * crop + 1))
    k_ref = 0
    rng = np.random.default_rng(0)
    for n in rng.choice(c.claim_days, 100, replace=False):
        bgs, _, cnt, _ = d.rows(int(n))
        im = np.pad(d.image("4km", int(n))[:2], ((0, 0), (crop, crop), (crop, crop)))
        for r in np.flatnonzero(cnt > 0)[:5]:
            y0, x0 = int(d.st["b4_y"][bgs[r]]), int(d.st["b4_x"][bgs[r]])
            ref += im[:, y0:y0 + 2 * crop + 1, x0:x0 + 2 * crop + 1]
            k_ref += 1
    out["claimed_crop"] = ref / k_ref
    return out


def run_extract(seed: int) -> None:
    c = Ctx(seed)
    rng = np.random.default_rng(seed)
    sd = {k: v.cpu().numpy() for k, v in c.model.state_dict().items()}
    res = {}
    for k, v in stem_descriptors(sd["stem.0.weight"]).items():
        res[f"stem_{k}"] = v
    res["stem_W"] = sd["stem.0.weight"]
    res["head_own_w"] = sd["head.0.weight"][:, :16]
    say("(2) activation statistics")
    res.update(run_activations(c, rng))
    say("(3) embedding footprints")
    res.update(run_footprints(c, rng))
    say("(4) synthetic patterns")
    res.update(run_synthetic(c))
    say("(5) test pass")
    res.update(run_test_pass(c))
    np.savez(D / f"filters_s{seed}.npz", **res)
    say(f"saved {D / f'filters_s{seed}.npz'}")


# ------------------------------------------------------------------ groups and per-channel / per-group ablation

def kmeans_pick(X: np.ndarray, ks=(2, 3, 4, 5), seed=0):
    """Standardise, k-means for each k, pick the best silhouette; returns labels, k, the silhouette table and the 2-D PCA projection."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score

    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    tab, best = {}, None
    for k in ks:
        if k >= len(Z):
            continue
        lab = KMeans(k, n_init=20, random_state=seed).fit_predict(Z)
        s = silhouette_score(Z, lab)
        tab[k] = float(s)
        if best is None or s > best[0]:
            best = (s, k, lab)
    pca = PCA(2).fit(Z)
    return best[2], best[1], tab, pca.transform(Z), pca.explained_variance_ratio_


def emb_features(F: dict) -> np.ndarray:
    """Per embedding channel: signed and absolute share of the input gradient by distance ring (rain peak + sum), correlation with the rain at 4 scales / contrast."""
    sg, ab = F["fp_signed"].sum(1), F["fp_abs"].sum(1)  # [16, 81, 81]
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    r = np.hypot(yy, xx)
    rows = []
    for k in range(16):
        tot = ab[k].sum() + 1e-12
        sgn = [sg[k][(r >= RINGS[j]) & (r < RINGS[j + 1])].sum() / tot if j else sg[k][r < 1].sum() / tot for j in range(len(RINGS) - 1)]
        abs_ = [ab[k][(r >= RINGS[j]) & (r < RINGS[j + 1])].sum() / tot if j else ab[k][r < 1].sum() / tot for j in range(len(RINGS) - 1)]
        # the sign of an embedding channel is arbitrary: orient every channel so that its signed gradient at the pixel and its 1-pixel ring is positive
        sgn_k = 1.0 if sg[k][r < 2].sum() >= 0 else -1.0
        rows.append([sgn_k * v for v in sgn] + abs_ + [sgn_k * v for v in F["act_emb_corr"][k]])
    return np.array(rows)


def stem_features(F: dict) -> np.ndarray:
    cols = [F[f"stem_{p}"] for p in ("peak_dc", "peak_gx", "peak_gy", "peak_cs", "sum_dc", "sum_gx", "sum_gy", "sum_cs")]
    nrm = F["stem_norm"]
    return np.column_stack(cols + [nrm[:, 0] + nrm[:, 1], nrm[:, 3:6].sum(1), nrm[:, 6:8].sum(1)])


def run_groups(seed: int) -> None:
    c = Ctx(seed)
    torch, d = c.torch, c.d
    F = dict(zstio.load(D / f"filters_s{seed}.npz", allow_pickle=True))
    e86 = _load("e86", "86_cnn_mechanism.py")
    e75, T, _ = e86.test_frame(d, c.claim_days, True)
    lab_e, k_e, sil_e, pc_e, ev_e = kmeans_pick(emb_features(F))
    lab_s, k_s, sil_s, pc_s, ev_s = kmeans_pick(stem_features(F))
    say(f"embedding channels: k = {k_e}, silhouette {sil_e}, groups {[int((lab_e == g).sum()) for g in range(k_e)]}")
    say(f"stem filters: k = {k_s}, silhouette {sil_s}, groups {[int((lab_s == g).sum()) for g in range(k_s)]}")
    # training mean of each embedding channel over valid pixels of 60 training days (replacement value of the ablation)
    rng = np.random.default_rng(seed)
    mu = np.zeros(16)
    nn_ = 0
    with torch.no_grad():
        for n in rng.choice(c.train_claim_days, 60, replace=False):
            e = c.model.embed(c.img(n))[0]
            m = c.valid.bool()
            mu += e[:, m].sum(1).cpu().double().numpy()
            nn_ += int(m.sum())
    mu /= nn_
    variants = {"none": []} | {f"ch{k}": [k] for k in range(16)} | {f"group{g}": list(np.flatnonzero(lab_e == g)) for g in range(k_e)}
    preds = {k: [] for k in variants}
    with torch.no_grad():
        for n in c.claim_days:
            bgs, tab, _, _ = d.rows(int(n))
            tt = c.tab_t(tab)
            e = c.model.embed(c.img(n))
            for name, chs in variants.items():
                e2 = e.clone()
                for k in chs:
                    e2[0, k] = float(mu[k])
                lg, _ = c.logit_rows(e2, bgs, tt)
                preds[name].append(torch.sigmoid(lg).cpu().numpy())
    arr = {k: np.concatenate(v).astype(np.float32) for k, v in preds.items()}
    np.savez(D / f"chan_abl_s{seed}.npz", lab_e=lab_e, lab_s=lab_s, pc_e=pc_e, pc_s=pc_s, ev_e=ev_e, ev_s=ev_s, mu=mu, sil_e=np.array(list(sil_e.items())),
             sil_s=np.array(list(sil_s.items())), **{f"p_{k}": v for k, v in arr.items()})
    # metrics
    for k, v in arr.items():
        T[f"p_{k}"] = v
    scores = {k: (f"p_{k}", "sum") for k in variants}
    tabs = e75.event_table(T, scores)
    metrics = (("bg_auc", "bg_w", "BG AUC"), ("bg_cap_n_0.05", None, "BG top-5%"), ("bg_ndcg_p", None, "BG NDCG5"), ("c_auc", "c_w", "county AUC"))
    say("\n== replace channel(s) of the embedding map by the training mean, test days with claims")
    pooled = np.zeros((len(scores), len(metrics)))
    delta = np.zeros((len(scores), len(metrics), 3))
    for i, name in enumerate(scores):
        line = f"    {name:8s} " + ("members " + ",".join(map(str, variants[name])) if variants[name] else "reference").ljust(34)
        for j, (col, w, lab) in enumerate(metrics):
            pt = pooled[i, j] = e75.pooled(tabs[name], col, w)
            if name == "none":
                line += f"  {lab} {pt:.3f}"
            else:
                _, dd = e75.boot({name: tabs[name], "none": tabs["none"]}, col, w, ref="none")
                delta[i, j] = dd[name]
                line += f"  {lab} {pt:.3f} ({dd[name][0]:+.3f} [{dd[name][1]:+.3f},{dd[name][2]:+.3f}])"
        say(line)
    np.savez(D / f"chan_abl_metrics_s{seed}.npz", variants=np.array(list(scores)), metrics=np.array([m[2] for m in metrics]), pooled=pooled, delta=delta)
    (wb2.cache_path("audit") / f"cnn_filters_groups_s{seed}.txt").write_text("\n".join(OUT) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "groups", "plot"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.mode == "plot":
        plot = _load("p87", "87_cnn_filters_plot.py")
        plot.main(a.seed)
        return
    {"extract": run_extract, "groups": run_groups}[a.mode](a.seed)


if __name__ == "__main__":
    main()
