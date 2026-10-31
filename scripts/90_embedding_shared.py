"""Is the CNN's embedding one shared decision space across states? (RESULTS stage 50)

Frozen encoder of a script-89 fit (own-pixel 16-dim embedding at the block-group pixel; tabular branch and head NOT used), rows of the S0 rows of script 89.
Three probes:
  A  read-out transfer: a logistic read-out on the embedding alone is fitted in state X (training years: all claimed block-group days + 5% of the others, weight 20)
     and scored in state Y on test days (row AUC and mean within-day AUC over claimed days); reference = the same read-out on hand rain summaries of the pixel
     (log1p peak, log1p 2-day sum, log1p 5 x 5 maximum of the peak). Equal diagonal / off-diagonal values mean the code means the same thing in every state.
  B  state identifiability: a boosted classifier predicts the state from the embedding, from the hand summaries, and from the embedding inside bins of the pixel's peak
     (balanced accuracy). A shared space with only rain information should not reveal the state beyond what the rain statistics already do.
  C  geometry: first principal direction of the embedding per state (sign set by its correlation with log peak), cosine between states, and the median of that
     coordinate by peak bin per state.
Inputs : cnn_<st>/ (script 84), cnn_pool/w_S0_<tag>_s<seed>.pt (script 89)
Output : $WXRISK_DATA/audit/embedding_shared_<tag>_s<seed>.txt
Run    : python scripts/90_embedding_shared.py --tag pool --seed 0 [--control norain]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402

STATES = ["TX", "LA", "FL"]
OUT: list[str] = []
CONTROL = ""
BINS = [0, 1, 5, 12.5, 25, 50, 100, 1e9]


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def collect(S, embed, days, rng, sample_neg: float | None):
    """Embedding, hand summaries, label and day id of the rows of `days`; negatives subsampled when sample_neg is given (weight 1 / sample_neg)."""
    import torch

    E, Hs, Y, Wt, Dd, Pk = [], [], [], [], [], []
    for n in days:
        bgs, _, cnt = S.rows(int(n))
        y = cnt > 0
        keep = np.ones(len(bgs), bool) if sample_neg is None else (y | (rng.random(len(bgs)) < sample_neg))
        if not keep.any():
            continue
        img = S.image(int(n))
        with torch.no_grad():
            inp = img.copy()
            if CONTROL == "norain":  # control: the encoder sees only the validity mask (geometry), rain channels zeroed; summaries stay real
                inp[:2] = 0
            e = embed.embed(torch.from_numpy(inp[None]).to(DEV))[0].reshape(embed.emb, -1).T.cpu().numpy()
        pix = S.pix[bgs[keep]]
        mx = maximum_filter(img[0], size=5).ravel()
        E.append(e[pix])
        Hs.append(np.column_stack([img[0].ravel()[pix], img[1].ravel()[pix], mx[pix]]))
        Pk.append(np.expm1(img[0].ravel()[pix] * 3))
        Y.append(y[keep])
        Wt.append(np.where(y[keep], 1.0, 1.0 / (sample_neg or 1.0)))
        Dd.append(np.full(int(keep.sum()), int(n)))
    return {k: np.concatenate(v) for k, v in dict(E=E, H=Hs, y=Y, w=Wt, day=Dd, pk=Pk).items()}


def day_auc(p, y, day):
    from sklearn.metrics import roc_auc_score

    out = []
    for d in np.unique(day[y]):
        m = day == d
        if y[m].sum() and (~y[m]).sum():
            out.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(out))


def main() -> None:
    global DEV, CONTROL
    import torch
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    from wxrisk import cnn, compute

    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ndays", type=int, default=250)
    ap.add_argument("--control", choices=["", "norain"], default="", help="norain: zero the rain channels at the encoder input (geometry-only code)")
    a = ap.parse_args()
    CONTROL = a.control
    DEV = compute.device()
    e89 = _load("e89", "89_cnn_pooled.py")
    model = cnn.Hybrid(3, 9, dilations=(1, 2, 4, 8, 16)).to(DEV)
    model.load_state_dict(torch.load(wb2.cache_path("cnn_pool") / f"w_S0_{a.tag}_s{a.seed}.pt", map_location=DEV))
    model.eval()
    rng = np.random.default_rng(0)
    say(f"embedding probes of the fit '{a.tag}' seed {a.seed} (S0 rows; {a.ndays} train and {a.ndays} test claimed days per state)")
    tr, te = {}, {}
    for s in STATES:
        S = e89.State(s, "S0")
        d_tr = rng.choice(np.flatnonzero(S.is_train & S.has_pos), a.ndays, replace=False)
        d_te = rng.choice(S.test_days, a.ndays, replace=False)
        tr[s] = collect(S, model, d_tr, rng, 0.05)
        te[s] = collect(S, model, d_te, rng, None)
        say(f"  {s}: train rows {len(tr[s]['y']):,} (claimed {int(tr[s]['y'].sum()):,}), test rows {len(te[s]['y']):,} (claimed {int(te[s]['y'].sum()):,})")

    say("\n== A. read-out transfer (row AUC | mean within-day AUC); rows = fitted in, columns = scored in")
    for lab, key in (("embedding (16)", "E"), ("hand rain summaries (3)", "H")):
        say(f"  {lab}:   " + "   ".join(f"{s:>13s}" for s in STATES))
        fits = {}
        for s in STATES:
            sc = StandardScaler().fit(tr[s][key])
            fits[s] = (sc, LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(tr[s][key]), tr[s]["y"], sample_weight=tr[s]["w"]))
        for s in STATES:
            cells = []
            for t in STATES:
                p = fits[s][1].decision_function(fits[s][0].transform(te[t][key]))
                cells.append(f"{roc_auc_score(te[t]['y'], p):.3f} | {day_auc(p, te[t]['y'], te[t]['day']):.3f}")
            say(f"    fitted in {s}   " + "   ".join(cells))

    say("\n== B. state identifiability (balanced accuracy of a boosted classifier, 3 classes; chance 0.333)")
    n = min(len(tr[s]["y"]) for s in STATES)

    def stack(d, key, m):
        X = np.concatenate([d[s][key][:m] for s in STATES])
        return X, np.repeat(np.arange(3), [min(m, len(d[s]["y"])) for s in STATES])

    def subs(d, m):
        r = np.random.default_rng(1)
        return {s: {k: v[r.permutation(len(v))[:m]] for k, v in d[s].items()} for s in STATES}

    tr_s, te_s = subs(tr, min(n, 100000)), subs(te, 100000)
    res = {}
    for lab, key in (("embedding", "E"), ("hand rain summaries", "H")):
        Xtr, ytr = stack(tr_s, key, 10 ** 9)
        Xte, yte = stack(te_s, key, 10 ** 9)
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xtr, ytr)
        res[key] = (clf, Xte, yte)
        say(f"  {lab:22s} {balanced_accuracy_score(yte, clf.predict(Xte)):.3f}")
    pk_te = np.concatenate([te_s[s]["pk"] for s in STATES])
    clf, Xte, yte = res["E"]
    say("  embedding inside bins of the pixel's peak (mm); hand summaries in brackets")
    clfh, Xh, _ = res["H"]
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (pk_te >= lo) & (pk_te < hi)
        if m.sum() > 300 and len(np.unique(yte[m])) == 3:
            say(f"    {lo:g}-{hi:g}: {balanced_accuracy_score(yte[m], clf.predict(Xte[m])):.3f} [{balanced_accuracy_score(yte[m], clfh.predict(Xh[m])):.3f}] ({int(m.sum()):,} rows)")

    say("\n== C. geometry of the embedding")
    pcs, coord = {}, {}
    Z = {}
    for s in STATES:
        X = np.concatenate([tr[s]["E"], te[s]["E"]])
        pk = np.concatenate([tr[s]["pk"], te[s]["pk"]])
        Xc = X - X.mean(0)
        u, sv, vt = np.linalg.svd(Xc[np.random.default_rng(2).permutation(len(Xc))[:200000]], full_matrices=False)
        v = vt[0]
        c = Xc @ v
        if spearmanr(c[::20], np.log1p(pk[::20])).statistic < 0:
            v, c = -v, -c
        pcs[s], Z[s] = v, (c, pk)
        say(f"  {s}: PC1 share of variance {sv[0] ** 2 / (sv ** 2).sum():.2f}, Spearman(PC1, log peak) {spearmanr(c[::20], np.log1p(pk[::20])).statistic:.3f}")
    say("  cosine between the PC1 directions: " + ", ".join(f"{p}-{q} {abs(float(pcs[p] @ pcs[q])):.2f}" for i, p in enumerate(STATES) for q in STATES[i + 1:]))
    say("  median PC1 coordinate by peak bin (mm), rescaled by each state's 1st-99th percentile range:")
    say("    bin        " + "   ".join(f"{s:>6s}" for s in STATES))
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        cells = []
        for s in STATES:
            c, pk = Z[s]
            r = np.percentile(c, [1, 99])
            m = (pk >= lo) & (pk < hi)
            cells.append(f"{(np.median(c[m]) - r[0]) / (r[1] - r[0]):6.2f}" if m.sum() > 100 else "     -")
        say(f"    {lo:g}-{hi:g}".ljust(12) + "   ".join(cells))
    (wb2.cache_path("audit") / f"embedding_shared_{a.tag}_s{a.seed}{'_' + a.control if a.control else ''}.txt").write_text("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
