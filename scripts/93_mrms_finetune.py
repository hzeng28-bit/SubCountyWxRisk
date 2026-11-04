"""Pretrain on gridMET, fine-tune with the hourly MRMS channels (RESULTS stage 52).

MRMS hourly rain exists only from 2015-03 (script 92), so a model trained on MRMS alone sees about 22 of the 84 training months of the E1 split. Here the gridMET
encoder trained on 2010-2016 (script 91 `--sources rain`, files `ms_rain_s<seed>.pt`) is the pretrained network; a second fine source `mrms` (6 channels of script 92) is added
by function-preserving expansion (zero weights for the new input slice of the stem, so the expanded network starts as the pretrained one) and the result is fine-tuned on the
training and validation months from 2015-03 on. Test = the E1 test days 2017-2019, so the numbers are directly comparable with stages 42 and 51 (the 2020-10 change of the MRMS
precipitation product lies outside the test years).
Variants (--variant):
  ft_rain       control: the pretrained gridMET network fine-tuned on the same window with no MRMS (what fine-tuning alone does)
  ft_mrms       pretrained network + MRMS channels, fine-tuned
  ft_mrms_shuf  negative control: the MRMS maps of another day of the same calendar month
  scratch_rain  gridMET only, trained from random weights on the window only
  scratch_mrms  gridMET + MRMS from random weights on the window only (what the short MRMS record gives without pretraining)
Same loss, sampling and validation as script 81 (listwise + 0.3 x BCE, one day per batch, blocked validation months, best epoch kept).
Inputs : cnn/* (script 80), cnn/mrms4k_* (script 92), cnn/ms_rain_s<seed>.pt (script 91)
Output : cnn/ms_<variant>_s<seed>.npy (test predictions, read by `eval`), audit/mrms_finetune.txt
Run    : python scripts/93_mrms_finetune.py train --variant ft_mrms --seed 0 ; python scripts/93_mrms_finetune.py eval
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
from wxrisk.tblog import RunLog  # noqa: E402

NNEG, LAM, ACCUM = 2000, 0.3, 4
OUT: list[str] = []
VARIANTS = ("ft_rain", "ft_mrms", "ft_mrms_shuf", "scratch_rain", "scratch_mrms")


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def expand_state(old: dict, new_model) -> dict:
    """Pretrained state dict -> state dict of the network with one more fine source: the stem's new input slice is zero, everything else is copied."""
    sd = new_model.state_dict()
    for k, v in old.items():
        if v.shape == sd[k].shape:
            sd[k] = v
        else:  # stem.0.weight [ch, 8 * n_fine, 3, 3]: the pretrained slice comes first
            assert k == "stem.0.weight" and v.shape[0] == sd[k].shape[0] and v.shape[1] < sd[k].shape[1], (k, v.shape, sd[k].shape)
            w = torch_zeros_like(sd[k])
            w[:, : v.shape[1]] = v
            sd[k] = w
    return sd


def torch_zeros_like(t):
    import torch

    return torch.zeros_like(t)


def run_train(variant: str, seed: int, epochs: int) -> None:
    import torch

    from wxrisk import compute
    from wxrisk.cnn import day_loss
    from wxrisk.encoders import GridSpec, MultiSourceNet

    e81 = _load("e81", "81_cnn_e1.py")
    d = e81.Data()
    dev = compute.device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use_mrms = "mrms" in variant
    pre = variant.startswith("ft_")
    fine = [GridSpec("rain", 3)] + ([GridSpec("mrms", 6)] if use_mrms else [])
    model = MultiSourceNet(fine, None, 9, proj=8, ch=32, dilations=(1, 2, 4, 8, 16)).to(dev)
    if pre:
        old = torch.load(e81.D / f"ms_rain_s{seed}.pt", map_location="cpu")
        model.load_state_dict(expand_state(old, model) if use_mrms else old)
    npar = sum(p.numel() for p in model.parameters())
    # MRMS maps
    M = zstio.load(e81.D / "mrms4k_day_maps.npy", mmap_mode="r")
    mdays = pd.DatetimeIndex(zstio.load(e81.D / "mrms4k_days.npy"))
    first = int(np.flatnonzero(d.days >= mdays[0])[0])
    assert d.days[first] == mdays[0] and len(M) == len(d.days) - first, "MRMS maps do not line up with the E1 days"
    valid = d.valid4.astype(np.float32)
    perm = np.arange(d.D)
    if variant == "ft_mrms_shuf":
        mo = d.days.month.to_numpy()
        for m in range(1, 13):
            ix = np.flatnonzero((mo == m) & (np.arange(d.D) >= first))
            perm[ix] = np.random.default_rng(0).permutation(ix)

    def mrms_img(n):
        m = np.asarray(M[perm[n] - first], np.float32)
        ch = [np.log1p(m[0]) / 3, np.log1p(m[1]) / 3, np.log1p(m[2]) / 3, m[3], m[4], m[5]]
        return np.stack(ch) * valid[None]

    say(f"{variant} seed {seed}: {compute.describe()}; parameters {npar:,}; pretrained {pre}")
    window = np.arange(d.D) >= first
    train_idx = np.flatnonzero(d.is_train & window)
    sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in rng.choice(np.flatnonzero(d.is_train), 300, replace=False)])
    tf = e81.tab_features(sample)
    mu, sd_ = tf.mean(0), tf.std(0) + 1e-6
    pix4 = d.bpix4

    def day_batch(n, sampled, r):
        bgs, tab, cnt, _ = d.rows(n)
        pos, neg = np.flatnonzero(cnt > 0), np.flatnonzero(cnt <= 0)
        w = np.ones(len(bgs), np.float32)
        if sampled and len(neg) > NNEG:
            sel = r.choice(neg, NNEG, replace=False)
            w[sel] = len(neg) / NNEG
            rows = np.sort(np.concatenate([pos, sel]))
        else:
            rows = np.arange(len(bgs))
        f = {"rain": torch.from_numpy(d.image("4km", n)[0:3][None]).to(dev)}
        if use_mrms:
            f["mrms"] = torch.from_numpy(mrms_img(n)[None].astype(np.float32)).to(dev)
        num = torch.from_numpy(((e81.tab_features(tab[rows]) - mu) / sd_).astype(np.float32)).to(dev)
        return (f, torch.from_numpy(pix4[bgs[rows]]).to(dev), num, torch.from_numpy(cnt[rows]).to(dev), torch.from_numpy(w[rows]).to(dev))

    def logit(b):
        return model(b[0], {}, None, b[1], b[2], torch.zeros(len(b[1]), 0, dtype=torch.long, device=dev))

    has_pos = np.array([(d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any() for n in range(d.D)])
    val_days = np.flatnonzero(d.is_val & window & (has_pos | (np.random.default_rng(1).random(d.D) < 0.2)))
    say(f"window from {d.days[first].date()}: train days {len(train_idx)} ({int(has_pos[train_idx].sum())} with a claim), validation days {len(val_days)}")
    steps = epochs * int((has_pos[train_idx].sum() + 0.25 * (~has_pos[train_idx]).sum()) // ACCUM)
    lr = 1e-3 if pre else 2e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 10), pct_start=0.15)
    tb = RunLog(f"mrms_finetune/{variant}_s{seed}", {"variant": variant, "seed": seed, "epochs": epochs, "pretrained": pre, "lr": lr, "parameters": npar})
    gstep = 0
    best, best_state = np.inf, None
    if pre:  # validation loss of the starting point, to see whether fine-tuning improves on it
        model.eval()
        vr = np.random.default_rng(123)
        with torch.no_grad():
            v0 = float(np.mean([float(day_loss(logit(b), b[3], b[4], LAM)) for b in (day_batch(int(n), True, vr) for n in val_days)]))
        say(f"  validation loss of the starting network on the window: {v0:.4f}")
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = np.concatenate([train_idx[has_pos[train_idx]], rng.choice(train_idx[~has_pos[train_idx]], int(0.25 * (~has_pos[train_idx]).sum()), replace=False)])
        rng.shuffle(order)
        tot, k, last = 0.0, 0, 0.0
        for i, n in enumerate(order):
            b = day_batch(int(n), True, rng)
            loss = day_loss(logit(b), b[3], b[4], LAM) / ACCUM
            loss.backward()
            tot += loss.item() * ACCUM
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
            vl = float(np.mean([float(day_loss(logit(b), b[3], b[4], LAM)) for b in (day_batch(int(n), True, vr) for n in val_days)]))
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
            preds.append(torch.sigmoid(logit(day_batch(int(n), False, rng))).cpu().numpy())
    p = np.concatenate(preds).astype(np.float32)
    np.save(e81.D / f"ms_{variant}_s{seed}.npy", p)
    say(f"saved {len(p):,} test predictions, best validation loss {best:.4f}")
    (wb2.cache_path("audit") / f"mrms_finetune_train_{variant}_s{seed}.txt").write_text("\n".join(OUT) + "\n")


def run_eval(tags: list[str]) -> None:
    e75 = _load("e75", "75_task_metrics.py")
    e81 = _load("e81", "81_cnn_e1.py")
    T = pd.read_parquet(e81.D / "e1_test_rows.parquet")
    scores = {}
    for t in tags:
        fs = sorted(e81.D.glob(f"ms_{t}_s[0-9]*.npy"))
        if fs:
            T[f"p_{t}"] = np.mean([zstio.load(f) for f in fs], axis=0)
            scores[f"{t} ({len(fs)} seed{'s' if len(fs) > 1 else ''})"] = (f"p_{t}", "sum")
    T["p_cnn4km"] = T[[c for c in T.columns if c.startswith("p_cnn4km_s")]].mean(axis=1)
    scores["CNN 4 km, stage 42 (3 seeds)"] = ("p_cnn4km", "sum")
    scores["policies only"] = ("n_pol", "sum")
    metrics = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"),
               ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))
    tabs = e75.event_table(T, scores)
    names = list(scores)
    say(f"test rows {len(T):,}, days {T.sysid.nunique()}")
    for col, w, lab in metrics:
        pt = {k: e75.pooled(t, col, w) for k, t in tabs.items()}
        say(f"  {lab:24s} " + " / ".join(f"{k.split(' (')[0]} {pt[k]:.3f}" for k in names))
    refs = [n for n in names if n.startswith(tags[0] + " ")]
    ref = refs[0]
    say(f"  paired differences against {ref} (point [95% over days]; BG AUC, top-5%, NDCG5 paid, county AUC, county top-10%):")
    for n in names:
        if n == ref or n.startswith("policies"):
            continue
        parts = []
        for col, w, lab in metrics:
            _, dd = e75.boot({n: tabs[n], ref: tabs[ref]}, col, w, ref=ref)
            parts.append(f"{dd[n][0]:+.3f} [{dd[n][1]:+.3f}, {dd[n][2]:+.3f}]")
        say(f"    {n}: " + "; ".join(parts))
    (wb2.cache_path("audit") / "mrms_finetune.txt").write_text("\n".join(OUT) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--variant", choices=VARIANTS, default="ft_mrms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--tags", nargs="+", default=["rain", "ft_rain", "ft_mrms", "ft_mrms_shuf", "scratch_rain", "scratch_mrms"],
                    help="first tag is the reference of the paired differences")
    a = ap.parse_args()
    if a.mode == "train":
        run_train(a.variant, a.seed, a.epochs)
    else:
        run_eval(a.tags)


if __name__ == "__main__":
    main()
