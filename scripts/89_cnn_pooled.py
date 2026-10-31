"""Pooled multi-state 4 km CNN (RESULTS stage 49): one network on the rain fields of TX, LA and FL, against fits on one state and on the other two.

The question (user): can a convolutional encoder replace the hand-built rain features (pixel value, 5 x 5 maximum, county maximum / mean, ...) and map
different states' inputs into one shared decision space? Stage 46 showed that coordinates and exposure maps add nothing to the 4 km CNN, so the shared input is
the rain alone: log1p peak and log1p two-day sum of gridMET at 4 km, plus the validity mask (3 channels, any grid size). The tabular branch keeps the 9 columns of
script 80 (per-state rows of script 84). Architecture, loss, schedule and validation protocol are those of script 81 (`cnn.Hybrid`, 55 k parameters, listwise + 0.3 x BCE,
one day per batch, up to 2,000 sampled negatives, 5 epochs, best validation epoch kept); county pooling indices come from the block-group pixels of each county.
Train 2010-2016 (blocked validation months, 1 in 7), test 2017-2023 days with >= 1 claimed block group, rows as the boosting baseline of script 85
(S0: rows inside a tropical-cyclone window dropped and tropical-named claims not counted; S2: all rows, all claims).

Fits (--train): pool = TX+LA+FL; own_<ST> = the test state alone; loso_<ST> = the two other states. A fit is scored on all three states' test rows (predictions are saved);
`eval` pairs them as scripts 75 / 85: cross-county block-group AUC, top-5% capture, NDCG5 with paid amount, county AUC, top-10% county capture, bootstrap over days,
against the boosting predictions of script 85 (`cnn_<st>/e45_pred.npz`, S0 only).
Inputs : cnn_<st>/ (script 84), cnn_<st>/e45_rows.npz and e45_pred.npz (script 85)
Output : $WXRISK_DATA/cnn_<st>/p89_<scope>_<tag>_s<seed>.npz, $WXRISK_DATA/cnn_pool/w_<scope>_<tag>_s<seed>.pt, audit/cnn_pooled_<scope>.txt
Run    : python scripts/89_cnn_pooled.py train --train TX+LA+FL --tag pool --seed 0 ; python scripts/89_cnn_pooled.py eval --seeds 0 1
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

STATES = ["TX", "LA", "FL"]
NNEG, LAM, ACCUM = 2000, 0.3, 4
OUT: list[str] = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def month_subset(days: pd.DatetimeIndex, frac: float) -> np.ndarray:
    ym = days.year.to_numpy().astype(np.int64) * 12 + days.month.to_numpy()
    return ((ym * 2654435761) % 1000) / 1000.0 < frac


def tab_features(tab: np.ndarray) -> np.ndarray:
    """Script 81's tabular input: log policies, SFHA, single-family, coverage, log block groups, hist00, prior3, month sin / cos."""
    m = tab[:, 7]
    return np.column_stack([tab[:, 0], tab[:, 1], tab[:, 2], tab[:, 3], np.log1p(tab[:, 4]), tab[:, 5], tab[:, 6],
                            np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]).astype(np.float32)


class State:
    """The arrays of one state (script 84), its split and the rows kept under the scope."""

    def __init__(self, st: str, scope: str):
        d = wb2.cache_path(f"cnn_{st.lower()}")
        self.st, self.dir, self.scope = st, d, scope
        S = np.load(d / "static.npz", allow_pickle=True)
        self.days = pd.DatetimeIndex(S["days"])
        self.ptr = np.load(d / "rows_ptr.npy")
        self.rbg = np.load(d / "rows_bg.npy")
        self.tab = np.load(d / "rows_tab.npy", mmap_mode="r")
        self.cnt = np.load(d / ("rows_n0.npy" if scope == "S0" else "rows_nall.npy"))
        self.paid = np.load(d / "rows_paid0.npy")
        self.ok = ~np.load(d / "rows_tc.npy") if scope == "S0" else np.ones(len(self.rbg), bool)
        self.O = np.load(d / "obs4k_day_maps.npy", mmap_mode="r")
        self.valid = S["obs4_valid"].astype(np.float32)
        self.H, self.W = self.valid.shape
        self.pix = (S["b4_y"].astype(np.int64) * self.W + S["b4_x"].astype(np.int64))
        self.cty = S["cty_idx"].astype(np.int64)
        self.D = len(self.days)
        self.n_cty = len(S["ctys"])
        from wxrisk import cnn

        self.pool = cnn.pad_index([np.unique(self.pix[self.cty == k]) for k in range(self.n_cty)])
        cs = np.concatenate([[0], np.cumsum((self.cnt > 0) & self.ok)])
        self.has_pos = cs[self.ptr[1:]] - cs[self.ptr[:-1]] > 0
        yr = self.days.year.to_numpy()
        ym = yr * 12 + self.days.month.to_numpy()
        self.is_test = yr >= 2017
        self.is_val = (~self.is_test) & (ym % 7 == 0)
        self.is_train = (~self.is_test) & ~self.is_val
        self.test_days = np.flatnonzero(self.is_test & self.has_pos)

    def rows(self, n: int):
        sl = slice(self.ptr[n], self.ptr[n + 1])
        k = self.ok[sl]
        return self.rbg[sl][k], np.asarray(self.tab[sl][k], np.float32), self.cnt[sl][k]

    def image(self, n: int) -> np.ndarray:
        o = np.asarray(self.O[n], np.float32)
        return np.stack([np.log1p(o[0]) / 3, np.log1p(o[1]) / 3, self.valid]).astype(np.float32)


def run_train(train: list[str], tag: str, seed: int, epochs: int, frac: float, scope: str) -> None:
    import torch

    from wxrisk import cnn, compute

    dev = compute.device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    S = {s: State(s, scope) for s in STATES}
    say(f"pooled CNN {tag} seed {seed} scope {scope}: trained on {'+'.join(train)}; {compute.describe()}")
    sample = np.concatenate([S[s].rows(int(n))[1] for s in train
                             for n in rng.choice(np.flatnonzero(S[s].is_train), 100, replace=False)])
    tf = tab_features(sample)
    mu, sd = tf.mean(0), tf.std(0) + 1e-6
    pools = {s: tuple(torch.from_numpy(a).to(dev) for a in S[s].pool) for s in STATES}
    model = cnn.Hybrid(3, 9, dilations=(1, 2, 4, 8, 16)).to(dev)
    npar = sum(p.numel() for p in model.parameters())

    def day_batch(s: str, n: int, sampled: bool, r: np.random.Generator):
        x = S[s]
        bgs, tab, cnt = x.rows(n)
        pos, neg = np.flatnonzero(cnt > 0), np.flatnonzero(cnt <= 0)
        w = np.ones(len(bgs), np.float32)
        if sampled and len(neg) > NNEG:
            sel = r.choice(neg, NNEG, replace=False)
            w[sel] = len(neg) / NNEG
            rows = np.sort(np.concatenate([pos, sel]))
        else:
            rows = np.arange(len(bgs))
        tt = torch.from_numpy(((tab_features(tab[rows]) - mu) / sd).astype(np.float32)).to(dev)
        return (torch.from_numpy(x.image(n)[None]).to(dev), torch.from_numpy(x.pix[bgs[rows]]).to(dev), torch.from_numpy(x.cty[bgs[rows]]).to(dev), tt,
                torch.from_numpy(cnt[rows]).to(dev), torch.from_numpy(w[rows]).to(dev), s)

    def logit(b):
        return model(b[0], b[1], b[2], b[3], *pools[b[6]])

    pos_days, neg_days, val_days = [], [], []
    for s in train:
        x = S[s]
        tr = np.flatnonzero(x.is_train & month_subset(x.days, frac))
        pos_days += [(s, int(n)) for n in tr[x.has_pos[tr]]]
        neg_days += [(s, int(n)) for n in tr[~x.has_pos[tr]]]
        vr = np.random.default_rng(1).random(x.D)
        val_days += [(s, int(n)) for n in np.flatnonzero(x.is_val & (x.has_pos | (vr < 0.2)))]
    n_epoch = len(pos_days) + int(0.25 * len(neg_days))
    steps = epochs * (n_epoch // ACCUM)
    say(f"train days {len(pos_days)} with a claim + {len(neg_days)} without (a quarter per epoch), validation {len(val_days)}; parameters {npar:,}; epochs {epochs}")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 10), pct_start=0.15)
    best, best_state = np.inf, None
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        pick = rng.choice(len(neg_days), int(0.25 * len(neg_days)), replace=False)
        order = pos_days + [neg_days[i] for i in pick]
        rng.shuffle(order)
        tot, k = 0.0, 0
        for i, (s, n) in enumerate(order):
            b = day_batch(s, n, True, rng)
            loss = cnn.day_loss(logit(b), b[4], b[5], LAM) / ACCUM
            loss.backward()
            tot += float(loss) * ACCUM
            k += 1
            if (i + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
        model.eval()
        vr = np.random.default_rng(123)
        with torch.no_grad():
            vl = float(np.mean([float(cnn.day_loss(logit(b), b[4], b[5], LAM)) for b in (day_batch(s, n, True, vr) for s, n in val_days)]))
        say(f"  epoch {ep + 1}: train loss {tot / k:.4f}, validation loss {vl:.4f} ({time.time() - t0:.0f} s)")
        if vl < best:
            best, best_state = vl, {k_: v.detach().clone() for k_, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    wd = wb2.cache_path("cnn_pool")
    wd.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, wd / f"w_{scope}_{tag}_s{seed}.pt")
    for s in STATES:
        x = S[s]
        preds, bgl = [], []
        with torch.no_grad():
            for n in x.test_days:
                b = day_batch(s, int(n), False, rng)
                preds.append(torch.sigmoid(logit(b)).cpu().numpy())
                bgl.append(x.rows(int(n))[0])
        np.savez(x.dir / f"p89_{scope}_{tag}_s{seed}.npz", p=np.concatenate(preds).astype(np.float32), bg=np.concatenate(bgl))
        say(f"  scored {s}: {sum(len(p) for p in preds):,} test rows, mean prediction {np.concatenate(preds).mean() * 100:.3f}%")
    say(f"best validation loss {best:.4f}")
    (wb2.cache_path("audit") / f"cnn_pooled_train_{scope}_{tag}_s{seed}.txt").write_text("\n".join(OUT) + "\n")


def run_eval(seeds: list[int], scope: str) -> None:
    e75 = _load("e75", "75_task_metrics.py")
    e85 = _load("e85", "85_state_group_baseline.py")
    metrics = (("bg_auc", "bg_w", "block-group AUC"), ("bg_cap_n_0.05", None, "BG top-5% capture"), ("bg_ndcg_p", None, "BG NDCG5 paid"),
               ("c_auc", "c_w", "county AUC"), ("c_cap_n_0.1", None, "county top-10% capture"))
    if scope != "S0":
        raise SystemExit("eval pairs the CNN with the S0 boosting rows of script 85; S2 needs its own row table")
    for st in STATES:
        x = State(st, scope)
        R = e85.features(st)
        yr = pd.DatetimeIndex(R["days"]).year.to_numpy()[R["day"]]
        rows = np.flatnonzero(yr >= 2017)
        T = e85.table(R, rows)
        scores = {"policies only": ("n_pol", "sum")}
        base = np.load(x.dir / "e45_pred.npz")
        for b in ("own", "tx", "pool", "loso"):
            if f"p_{b}" in base.files:
                T[f"b_{b}"] = base[f"p_{b}"]
                scores[f"boosting {b}"] = (f"b_{b}", "sum")
        fits = {"own": f"own_{st}", "pool": "pool", "loso": f"loso_{st}"}
        if st != "TX":
            fits["tx"] = "own_TX"
        for k, tag in fits.items():
            ps = []
            for sd in seeds:
                f = x.dir / f"p89_{scope}_{tag}_s{sd}.npz"
                if not f.exists():
                    continue
                z = np.load(f)
                assert len(z["p"]) == len(T) and (z["bg"] == R["bg"][rows]).all(), f"row alignment {f}"
                ps.append(z["p"])
            if ps:
                T[f"c_{k}"] = np.mean(ps, axis=0)
                scores[f"CNN {k} ({len(ps)} seed{'s' if len(ps) > 1 else ''})"] = (f"c_{k}", "sum")
        say(f"\n== test {st}: rows {len(T):,}, days {T.sysid.nunique()}, claimed block groups {int((T.n > 0).sum()):,}")
        say("    score order: " + " / ".join(scores))
        tabs = e75.event_table(T, scores)
        names = list(scores)
        cnn_names = [n for n in names if n.startswith("CNN")]
        by_key = {n.split(" ")[1]: n for n in cnn_names}
        for col, w, lab in metrics:
            pt = {k: e75.pooled(t, col, w) for k, t in tabs.items()}
            line = f"    {lab:24s} " + " / ".join(f"{pt[k]:.3f}" for k in names)
            say(line)
        say("    paired differences (point [95% over days]):")
        pairs = [("pool", "boosting pool"), ("own", "boosting own"), ("loso", "boosting loso"), ("pool", "own"), ("loso", "own"), ("tx", "own")]
        for a, b in pairs:
            na = by_key.get(a)
            nb = by_key.get(b) or (b if b in tabs else None)
            if na is None or nb is None or na == nb:
                continue
            parts = []
            for col, w, lab in metrics:
                _, dd = e75.boot({na: tabs[na], nb: tabs[nb]}, col, w, ref=nb)
                parts.append(f"{lab.split()[-1] if 'county' not in lab else 'cty ' + lab.split()[-1]} {dd[na][0]:+.3f} [{dd[na][1]:+.3f}, {dd[na][2]:+.3f}]")
            say(f"      {na} - {nb}: " + "; ".join(parts))
    (wb2.cache_path("audit") / f"cnn_pooled_{scope}.txt").write_text("\n".join(OUT) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval"])
    ap.add_argument("--train", default="TX+LA+FL", help="states to train on, joined by +")
    ap.add_argument("--tag", default="pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--frac", type=float, default=1.0, help="fraction of training months (learning curve)")
    ap.add_argument("--scope", choices=["S0", "S2"], default="S0")
    a = ap.parse_args()
    if a.mode == "train":
        run_train(a.train.split("+"), a.tag, a.seed, a.epochs, a.frac, a.scope)
    else:
        run_eval(a.seeds, a.scope)


if __name__ == "__main__":
    main()
