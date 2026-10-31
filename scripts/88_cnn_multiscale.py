"""Coarse-to-fine multi-scale CNN with a learned vote (RESULTS stage 48).

Stage 47: the full-resolution CNN reads a local, smooth rain intensity (2-3 px reach) and a weak west-sided tail; no long-range structure is used. This tests whether giving
the model explicit coarse scales helps. `cnn.MultiScale`: one small CNN per scale s in {1, 2, 4, 8, 16} px (the day's stack average-pooled by s), a head per scale, a gate that
weights the scales per block group from the tabular branch and the embeddings; the output is the weighted sum of the per-scale logits. Loss = listwise + binary loss of the vote
+ 0.5 x the mean of the same loss on every scale's own logit, so each scale is also a stand-alone predictor. No county pooling. Parameters matched to the 55 k of stage 46.
Models (tags): `ms` (scales 1 2 4 8 16), `single` (scale 1 only, same widths, 12 k parameters), `single_wide` (scale 1 only, width 40, parameters matched to `ms`). Same data, split, sampling, optimiser, 5 epochs, blocked validation as script 81.
  train  one tag and seed -> cnn/w_<tag>_s<seed>.pt, v_<tag>_s<seed>.npy (vote), vs_<tag>_s<seed>.npy (per-scale probabilities, rows x S), vw_<tag>_s<seed>.npy (weights)
  eval   ranking metrics of the vote, of every scale alone, equal-weight and coarse-to-fine cumulative votes, against `full` (stage 46) with a bootstrap over days; weights by rain class
Inputs : cnn/* (script 80), cnn/v_full_s*.npy (script 81)
Output : $WXRISK_DATA/cnn/*, $WXRISK_DATA/audit/cnn_multiscale.txt
Run    : python scripts/88_cnn_multiscale.py train --tag ms --seed 0 ; ... eval
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


e81 = _load("e81", "81_cnn_e1.py")
D = e81.D
OUT: list[str] = []
SCALES = {"ms": (1, 2, 4, 8, 16), "single": (1,), "single_wide": (1,)}
WIDTH = {"ms": 16, "single": 16, "single_wide": 40}  # single_wide: one scale with the width that matches the parameter count of `ms`
AUX = 0.5


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def run_train(tag: str, seed: int, epochs: int) -> None:
    import torch

    from wxrisk import cnn, compute

    d = e81.Data()
    dev = compute.device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    scales = SCALES[tag]
    say(f"multi-scale {tag} {scales} seed {seed}: {compute.describe()}")
    sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in rng.choice(np.flatnonzero(d.is_train), 300, replace=False)])
    tf = e81.tab_features(sample)
    mu, sd = tf.mean(0), tf.std(0) + 1e-6
    model = cnn.MultiScale(d.image("4km", 0).shape[0], 9, scales=scales, ch=WIDTH[tag]).to(dev)
    say(f"parameters {sum(p.numel() for p in model.parameters()):,}; epochs {epochs}")

    def day_batch(n, sampled: bool, r: np.random.Generator):
        bgs, tab, cnt, _ = d.rows(n)
        pos = np.flatnonzero(cnt > 0)
        neg = np.flatnonzero(cnt <= 0)
        w = np.ones(len(bgs), np.float32)
        if sampled and len(neg) > e81.NNEG:
            sel = r.choice(neg, e81.NNEG, replace=False)
            w[sel] = len(neg) / e81.NNEG
            rows = np.sort(np.concatenate([pos, sel]))
        else:
            rows = np.arange(len(bgs))
        img = torch.from_numpy(d.image("4km", n)[None]).to(dev)
        tt = torch.from_numpy(((e81.tab_features(tab[rows]) - mu) / sd).astype(np.float32)).to(dev)
        return img, torch.from_numpy(d.bpix4[bgs[rows]]).to(dev), tt, torch.from_numpy(cnt[rows]).to(dev), torch.from_numpy(w[rows]).to(dev)

    def loss_of(b):
        vote, per, _ = model(b[0], b[1], b[2])
        loss = cnn.day_loss(vote, b[3], b[4], e81.LAM)
        if len(scales) > 1:
            loss = loss + AUX * sum(cnn.day_loss(per[:, j], b[3], b[4], e81.LAM) for j in range(len(scales))) / len(scales)
        return loss

    has_pos = np.array([(d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any() for n in range(d.D)])
    train_days = np.flatnonzero(d.is_train)
    val_days = np.flatnonzero(d.is_val & (has_pos | (np.random.default_rng(1).random(d.D) < 0.2)))
    steps = epochs * int((has_pos[train_days].sum() + 0.25 * (~has_pos[train_days]).sum()) // e81.ACCUM)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10), pct_start=0.15)
    best, best_state = np.inf, None
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = np.concatenate([train_days[has_pos[train_days]], rng.choice(train_days[~has_pos[train_days]], int(0.25 * (~has_pos[train_days]).sum()), replace=False)])
        rng.shuffle(order)
        tot, k = 0.0, 0
        for i, n in enumerate(order):
            loss = loss_of(day_batch(int(n), True, rng)) / e81.ACCUM
            loss.backward()
            tot += float(loss) * e81.ACCUM
            k += 1
            if (i + 1) % e81.ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
        model.eval()
        vr = np.random.default_rng(123)
        with torch.no_grad():
            vl = np.mean([float(loss_of(day_batch(int(n), True, vr))) for n in val_days])
        say(f"  epoch {ep + 1}: train loss {tot / k:.4f}, validation loss {vl:.4f} ({time.time() - t0:.0f} s)")
        if vl < best:
            best, best_state = vl, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    pv, ps, pw = [], [], []
    with torch.no_grad():
        for n in np.flatnonzero(d.is_test):
            b = day_batch(int(n), False, rng)
            vote, per, w = model(b[0], b[1], b[2])
            pv.append(torch.sigmoid(vote).cpu().numpy())
            ps.append(torch.sigmoid(per).cpu().numpy())
            pw.append(w.cpu().numpy())
    torch.save(best_state, D / f"w_{tag}_s{seed}.pt")
    np.save(D / f"v_{tag}_s{seed}.npy", np.concatenate(pv).astype(np.float32))
    np.save(D / f"vs_{tag}_s{seed}.npy", np.concatenate(ps).astype(np.float32))
    np.save(D / f"vw_{tag}_s{seed}.npy", np.concatenate(pw).astype(np.float32))
    say(f"saved {tag} seed {seed}: {sum(len(p) for p in pv):,} test rows, best validation loss {best:.4f}")
    (wb2.cache_path("audit") / f"v_{tag}_train_s{seed}.txt").write_text("\n".join(OUT) + "\n")


def run_eval() -> None:
    e86 = _load("e86", "86_cnn_mechanism.py")
    d = e81.Data()
    days = e86.claim_days(d)
    e75, T, in_test = e86.test_frame(d, days, True)
    models = []

    def add(name, arr):
        T[f"p_{name}"] = arr[in_test]
        models.append(name)
    add("full (stage 46, mean of 2 seeds)", np.mean([np.load(f) for f in sorted(D.glob("v_full_s*.npy"))], axis=0))
    seeds = {}
    for tag in SCALES:
        fs = sorted(D.glob(f"v_{tag}_s*.npy"))
        if fs:
            seeds[tag] = [np.load(f) for f in fs]
            add(f"{tag} vote (mean of {len(fs)} seeds)", np.mean(seeds[tag], axis=0))
    S = SCALES["ms"]
    if "ms" in seeds:
        per = [np.load(f) for f in sorted(D.glob("vs_ms_s*.npy"))]
        wts = [np.load(f) for f in sorted(D.glob("vw_ms_s*.npy"))]
        pm, wm = np.mean(per, axis=0), np.mean(wts, axis=0)
        for j, s in enumerate(S):
            add(f"ms scale {s} px alone", pm[:, j])
        lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
        L = np.mean([lg(p) for p in per], axis=0)
        add("ms equal-weight vote of the scales", 1 / (1 + np.exp(-L.mean(1))))
        for k in range(1, len(S)):  # coarse-to-fine: the coarsest k scales with the learned weights renormalised
            sel = list(range(len(S) - k, len(S)))
            wr = wm[:, sel] / wm[:, sel].sum(1, keepdims=True)
            add(f"ms coarse-to-fine: scales {[S[j] for j in sel]}", 1 / (1 + np.exp(-(wr * L[:, sel]).sum(1))))
    e86.OUT.clear()
    e86.say = say
    e86.report(e75, T, models, models[0], "multi-scale vote against the stage-46 full CNN")
    wc = e86.within_county_auc(T, [f"p_{m}" for m in models])
    say("\n== AUC inside a county-day")
    for m, v in wc.items():
        say(f"    {m.replace('p_', ''):50s} {v:.3f}")
    say("\n== per-seed pooled metric on the same days")
    for tag, arrs in seeds.items():
        T2 = T[["sysid", "cty", "n", "paid", "n_pol"]].copy()
        for i, a in enumerate(arrs):
            T2[f"s{i}"] = a[in_test]
        tabs = e75.event_table(T2, {f"s{i}": (f"s{i}", "sum") for i in range(len(arrs))})
        say(f"    {tag:8s} " + " | ".join(f"{lab} " + "/".join(f"{e75.pooled(t, col, w):.3f}" for t in tabs.values()) for col, w, lab in e86.METRICS))
    if "ms" in seeds:
        # learned weights by the observed peak rain at the block group's pixel (test rows of the claim days)
        test = np.flatnonzero(d.is_test)
        lo = d.ptr[test[0]]
        rain = np.concatenate([np.asarray(d.O[n][0], np.float32)[d.st["b4_y"][d.rbg[d.ptr[n]:d.ptr[n + 1]]], d.st["b4_x"][d.rbg[d.ptr[n]:d.ptr[n + 1]]]] for n in test])
        r = rain[in_test]
        w = wm[in_test]
        cls = np.digitize(r, [1, 10, 25, 50])
        say("\n== mean learned weight of each scale by observed peak rain at the pixel (mm)  [seed mean]")
        say("    class (mm)      rows      " + "  ".join(f"{s:>5d}px" for s in S))
        for c, nme in enumerate(["< 1", "1-10", "10-25", "25-50", ">= 50"]):
            m = cls == c
            say(f"    {nme:12s} {int(m.sum()):9,d}   " + "  ".join(f"{w[m, j].mean():7.3f}" for j in range(len(S))) if m.any() else f"    {nme:12s} 0")
        claimed = T.n.to_numpy() > 0
        say("    claimed rows  " + f"{int(claimed.sum()):9,d}   " + "  ".join(f"{w[claimed, j].mean():7.3f}" for j in range(len(S))))
        say("    all rows      " + f"{len(w):9,d}   " + "  ".join(f"{w[:, j].mean():7.3f}" for j in range(len(S))))
    (wb2.cache_path("audit") / "cnn_multiscale.txt").write_text("\n".join(OUT) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--tag", choices=list(SCALES), default="ms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    run_train(a.tag, a.seed, a.epochs) if a.mode == "train" else run_eval()


if __name__ == "__main__":
    main()
