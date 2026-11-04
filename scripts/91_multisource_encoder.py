"""Multi-source neural encoder on the Texas E1 data (RESULTS stage 51): sources are declared, not hand-assembled (src/wxrisk/encoders.py).

Same rows, split, loss and schedule as script 81 (all-days unit, scope S0, train 2010-2016 with blocked validation months, test 2017-2019, one day per batch,
listwise + 0.3 x BCE, 5 epochs, best validation epoch kept); the network is `encoders.MultiSourceNet` (no county pooling). Sources, chosen with --sources:
  rain   fine 4 km  log1p observed peak and 2-day sum of gridMET, validity mask (3 channels)                   OBSERVED
  expo   fine 4 km  log policies, SFHA share, log claims 2000-2009 of the 0.25 degree cell put on the 4 km grid  (issue time)
  fc     coarse 0.25 deg  GEFS ensemble mean peak and sum, five sorted member peaks (log1p / 3), Texas mask      (forecast, issue time)
  era5   coarse 0.25 deg  ERA5 tcwv, q850, u850, v850, u500, v500 (train-standardised), coverage mask            (valid time: ceiling only)
  geo    rows       NFHL shares (SFHA, shaded X, mapped) and HAND statistics of the block group (script 58), log1p of the metre-valued ones
The row branch always carries the 9 tabular columns of script 81. `rain` + nothing else is the stage-46 `nostatic` model with another code path.
Forecast mode = no `rain`: e.g. `fc,expo` (what is known at issue time).
Inputs : cnn/* (script 80), labels_extra/flood/pilot/bg_flood_geometry.parquet (script 58), cnn/e1_test_rows.parquet (script 81 eval)
Output : $WXRISK_DATA/cnn/ms_<tag>_s<seed>.npy (test predictions), audit/multisource_train_<tag>_s<seed>.txt, audit/multisource_encoder.txt
Run    : python scripts/91_multisource_encoder.py train --sources rain,expo --seed 0 ; python scripts/91_multisource_encoder.py eval --tags rain rain+expo ...
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
GEO = ["nfhl_sfha", "nfhl_x02", "nfhl_mapped", "h1_med", "h1_p10", "h1_le2", "h1_le5", "h10_med", "h10_le5"]
LOGGED = {"h1_med", "h1_p10", "h10_med"}


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def geo_table(d) -> np.ndarray:
    """[block group, 9] geometry features in the order of the day tensors' block groups (standardised later on training rows)."""
    g = pd.read_parquet(wb2.cache_path("labels_extra") / "flood" / "pilot" / "bg_flood_geometry.parquet").set_index("bg10").reindex(d.st["bg10"])
    x = g[GEO].copy()
    for c in LOGGED:
        x[c] = np.log1p(x[c].clip(lower=0))
    return x.fillna(0).to_numpy(np.float32)


def run_train(sources: list[str], seed: int, epochs: int, tag: str, p_drop: float) -> None:
    import torch

    from wxrisk import compute
    from wxrisk.cnn import day_loss
    from wxrisk.encoders import GridSpec, MultiSourceNet

    e81 = _load("e81", "81_cnn_e1.py")
    d = e81.Data()
    dev = compute.device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    use = set(sources)
    fine = ([GridSpec("rain", 3)] if "rain" in use else []) + ([GridSpec("expo", 3)] if "expo" in use else [])
    coarse = ([GridSpec("fc", 8)] if "fc" in use else []) + ([GridSpec("era5", 7)] if "era5" in use else [])
    if not fine:  # the trunk needs a fine-grid input; forecast mode uses expo
        raise SystemExit("at least one fine source (rain or expo) is needed")
    geo = geo_table(d) if "geo" in use else None
    n_num = 9 + (len(GEO) if geo is not None else 0)
    model = MultiSourceNet(fine, coarse, n_num, proj=8, ch=32, dilations=(1, 2, 4, 8, 16), modality_dropout=p_drop).to(dev)
    npar = sum(p.numel() for p in model.parameters())
    say(f"multi-source encoder {tag} seed {seed}: sources {sorted(use)} + tab{' + geo rows' if geo is not None else ''}; {compute.describe()}; parameters {npar:,}")
    # standardisation of the row features and of ERA5 on training days only
    train_idx = np.flatnonzero(d.is_train)
    sample_days = rng.choice(train_idx, 300, replace=False)
    sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in sample_days])
    tf = e81.tab_features(sample)
    mu, sd = tf.mean(0), tf.std(0) + 1e-6
    if geo is not None:
        bgs_s = np.concatenate([d.rbg[d.ptr[n]:d.ptr[n + 1]] for n in sample_days])
        gmu, gsd = geo[bgs_s].mean(0), geo[bgs_s].std(0) + 1e-6
    if "era5" in use:
        e5 = np.stack([np.asarray(d.G[n][12:18], np.float32) for n in rng.choice(train_idx, 150, replace=False)])
        emu, esd = np.nanmean(e5, (0, 2, 3), keepdims=True)[0], np.nanstd(e5, (0, 2, 3), keepdims=True)[0] + 1e-6
    valid_tx = d.st["valid_tx"].astype(np.float32)[None]
    era_valid = d.st["era_valid"].astype(np.float32)[None]
    up = torch.from_numpy(d.cell4).to(dev)
    pix4 = d.bpix4

    def grids(n):
        img = d.image("4km", n)  # [8, H, W]: rain 0-2, expo 3-5, lat / lon 6-7 (not used here)
        f, c = {}, {}
        if "rain" in use:
            f["rain"] = torch.from_numpy(img[0:3][None]).to(dev)
        if "expo" in use:
            f["expo"] = torch.from_numpy(img[3:6][None]).to(dev)
        g = np.asarray(d.G[n], np.float32)
        if "fc" in use:
            c["fc"] = torch.from_numpy(np.concatenate([np.log1p(np.clip(np.nan_to_num(g[0:7]), 0, None)) / 3, valid_tx])[None]).to(dev)
        if "era5" in use:
            e = (g[12:18] - emu) / esd
            c["era5"] = torch.from_numpy(np.concatenate([np.nan_to_num(e), era_valid])[None].astype(np.float32)).to(dev)
        return f, c

    def day_batch(n, sampled: bool, r: np.random.Generator):
        bgs, tab, cnt, _ = d.rows(n)
        pos, neg = np.flatnonzero(cnt > 0), np.flatnonzero(cnt <= 0)
        w = np.ones(len(bgs), np.float32)
        if sampled and len(neg) > NNEG:
            sel = r.choice(neg, NNEG, replace=False)
            w[sel] = len(neg) / NNEG
            rows = np.sort(np.concatenate([pos, sel]))
        else:
            rows = np.arange(len(bgs))
        num = (e81.tab_features(tab[rows]) - mu) / sd
        if geo is not None:
            num = np.concatenate([num, (geo[bgs[rows]] - gmu) / gsd], axis=1)
        f, c = grids(n)
        return (f, c, torch.from_numpy(pix4[bgs[rows]]).to(dev), torch.from_numpy(num.astype(np.float32)).to(dev),
                torch.from_numpy(cnt[rows]).to(dev), torch.from_numpy(w[rows]).to(dev))

    nocat = torch.zeros(0, 0, dtype=torch.long)

    def logit(b):
        return model(b[0], b[1], up, b[2], b[3], nocat.new_zeros((len(b[2]), 0)))

    has_pos = np.array([(d.rn[d.ptr[n]:d.ptr[n + 1]] > 0).any() for n in range(d.D)])
    train_days = np.flatnonzero(d.is_train)
    val_days = np.flatnonzero(d.is_val & (has_pos | (np.random.default_rng(1).random(d.D) < 0.2)))
    steps = epochs * int(((has_pos[train_days]).sum() + 0.25 * (~has_pos[train_days]).sum()) // ACCUM)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10), pct_start=0.15)
    tb = RunLog(f"multisource/{tag}_s{seed}", {"sources": ",".join(sorted(use)), "seed": seed, "epochs": epochs, "modality_dropout": p_drop, "parameters": npar})
    gstep = 0
    best, best_state = np.inf, None
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = np.concatenate([train_days[has_pos[train_days]], rng.choice(train_days[~has_pos[train_days]], int(0.25 * (~has_pos[train_days]).sum()), replace=False)])
        rng.shuffle(order)
        tot, k, last = 0.0, 0, 0.0
        for i, n in enumerate(order):
            b = day_batch(int(n), True, rng)
            loss = day_loss(logit(b), b[4], b[5], LAM) / ACCUM
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
            vl = float(np.mean([float(day_loss(logit(b), b[4], b[5], LAM)) for b in (day_batch(int(n), True, vr) for n in val_days)]))
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
    np.save(e81.D / f"ms_{tag}_s{seed}.npy", p)
    torch.save(best_state, e81.D / f"ms_{tag}_s{seed}.pt")
    say(f"saved {len(p):,} test predictions, best validation loss {best:.4f}")
    (wb2.cache_path("audit") / f"multisource_train_{tag}_s{seed}.txt").write_text("\n".join(OUT) + "\n")


def run_eval(tags: list[str]) -> None:
    e75 = _load("e75", "75_task_metrics.py")
    e81 = _load("e81", "81_cnn_e1.py")
    T = pd.read_parquet(e81.D / "e1_test_rows.parquet")
    scores = {}
    cs = [c for c in T.columns if c.startswith("p_cnn4km_s")]
    T["p_cnn4km"] = T[cs].mean(axis=1)
    scores["CNN 4 km, stage 42 (3 seeds)"] = ("p_cnn4km", "sum")
    scores["boosting, observed summaries"] = ("p_hgb", "sum")
    for t in tags:
        fs = sorted(e81.D.glob(f"ms_{t}_s[0-9]*.npy"))
        if not fs:
            continue
        T[f"p_ms_{t}"] = np.mean([zstio.load(f) for f in fs], axis=0)
        scores[f"MS {t} ({len(fs)} seed{'s' if len(fs) > 1 else ''})"] = (f"p_ms_{t}", "sum")
    scores["policies only"] = ("n_pol", "sum")
    metrics = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"),
               ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))
    tabs = e75.event_table(T, scores)
    names = list(scores)
    say(f"test rows {len(T):,}, days {T.sysid.nunique()}; score order: " + " / ".join(names))
    for col, w, lab in metrics:
        pt = {k: e75.pooled(t, col, w) for k, t in tabs.items()}
        say(f"  {lab:24s} " + " / ".join(f"{pt[k]:.3f}" for k in names))
    ref = names[0]
    say("  paired differences against " + ref + " (point [95% over days]):")
    for n in names:
        if not n.startswith("MS"):
            continue
        parts = []
        for col, w, lab in metrics:
            _, dd = e75.boot({n: tabs[n], ref: tabs[ref]}, col, w, ref=ref)
            parts.append(f"{lab.split()[-1]} {dd[n][0]:+.3f} [{dd[n][1]:+.3f}, {dd[n][2]:+.3f}]")
        say(f"    {n}: " + "; ".join(parts))
    (wb2.cache_path("audit") / "multisource_encoder.txt").write_text("\n".join(OUT) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--sources", default="rain", help="comma list of rain, expo, fc, era5, geo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--p-drop", type=float, default=0.0, help="modality dropout probability")
    ap.add_argument("--tags", nargs="+", default=[])
    a = ap.parse_args()
    if a.mode == "train":
        run_train(a.sources.split(","), a.seed, a.epochs, a.sources.replace(",", "+") + (f"_pd{a.p_drop:g}" if a.p_drop else ""), a.p_drop)
    else:
        run_eval(a.tags)


if __name__ == "__main__":
    main()
