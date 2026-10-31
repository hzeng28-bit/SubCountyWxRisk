"""Figures and summary text of script 87 (RESULTS stage 47). Called as `python scripts/87_cnn_filters.py plot --seed S`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]  # colour-blind-safe group colours


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    e87 = _load("e87", "87_cnn_filters.py")
    D, FIG, HALF, RINGS = e87.D, e87.FIG, e87.HALF, e87.RINGS
    FIG.mkdir(parents=True, exist_ok=True)
    F = dict(np.load(D / f"filters_s{seed}.npz", allow_pickle=True))
    A = dict(np.load(D / f"chan_abl_s{seed}.npz", allow_pickle=True))
    M = dict(np.load(D / f"chan_abl_metrics_s{seed}.npz", allow_pickle=True))
    lab_e, lab_s = A["lab_e"], A["lab_s"]
    ke, ks = lab_e.max() + 1, lab_s.max() + 1
    txt: list[str] = []

    def say(s=""):
        print(s)
        txt.append(s)

    # canonical sign of every embedding channel: positive signed gradient at the pixel and its 1-pixel ring
    sg = F["fp_signed"].sum(1)
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1]
    rr = np.hypot(yy, xx)
    orient = np.where(sg[:, rr < 2].sum(1) >= 0, 1.0, -1.0)
    ordr = sorted(range(16), key=lambda k: (lab_e[k], k))
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # ---------------------------------------------------------------- 1 stem filters
    W = F["stem_W"]
    order = sorted(range(32), key=lambda i: (lab_s[i], -F["stem_peak_dc"][i]))
    vmax = np.abs(W[:, :2]).max()
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(6, 8, hspace=0.45, wspace=0.12)
    for t, i in enumerate(order):
        ax = fig.add_subplot(gs[t // 8, t % 8])
        im = np.concatenate([W[i, 0], np.full((3, 1), np.nan), W[i, 1]], 1)
        ax.imshow(im, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(COL[lab_s[i]])
            sp.set_linewidth(3)
        ax.set_title(f"f{i}", fontsize=8, color=COL[lab_s[i]])
    fig.text(0.5, 0.955, "Stem filters (3 x 3): left rain peak, right rain 2-day sum; red = positive weight, blue = negative; border = k-means group", ha="center")
    Z = e87.stem_features(F)
    ax = fig.add_subplot(gs[4:, 0:3])
    for g in range(ks):
        m = lab_s == g
        ax.scatter(A["pc_s"][m, 0], A["pc_s"][m, 1], color=COL[g], s=40, label=f"group {g} (n={int(m.sum())})")
    for i in range(32):
        ax.annotate(str(i), A["pc_s"][i], fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({A['ev_s'][0]:.0%})")
    ax.set_ylabel(f"PC2 ({A['ev_s'][1]:.0%})")
    ax.set_title("PCA of the filter descriptors", fontsize=9)
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[4:, 3:6])
    names = ["peak DC", "peak |grad|", "peak centre-surround", "sum DC", "sum |grad|", "sum centre-surround"]
    vals = lambda m: [F["stem_peak_dc"][m].mean(), np.hypot(F["stem_peak_gx"][m], F["stem_peak_gy"][m]).mean(), F["stem_peak_cs"][m].mean(),
                      F["stem_sum_dc"][m].mean(), np.hypot(F["stem_sum_gx"][m], F["stem_sum_gy"][m]).mean(), F["stem_sum_cs"][m].mean()]
    for g in range(ks):
        ax.bar(np.arange(6) + g * 0.8 / ks, vals(lab_s == g), 0.8 / ks, color=COL[g])
    ax.set_xticks(np.arange(6) + 0.4)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Mean kernel descriptors by group", fontsize=9)
    ax = fig.add_subplot(gs[4:, 6:8])
    nrm = F["stem_norm"]
    share = lambda m: np.array([nrm[m][:, [0, 1]].sum(), nrm[m][:, [3, 4, 5]].sum(), nrm[m][:, [6, 7]].sum(), nrm[m][:, 2].sum()]) / nrm[m].sum()
    for g in range(ks):
        ax.bar(np.arange(4) + g * 0.8 / ks, share(lab_s == g), 0.8 / ks, color=COL[g])
    ax.set_xticks(np.arange(4) + 0.4)
    ax.set_xticklabels(["rain", "exposure", "lat/lon", "valid"], fontsize=8)
    ax.set_title("Input-channel share of the kernel norm", fontsize=9)
    fig.savefig(FIG / f"stem_filters_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    say("== stem filters")
    for g in range(ks):
        m = lab_s == g
        say(f"  group {g}: n={int(m.sum())}, filters {list(np.flatnonzero(m))}; peak DC {F['stem_peak_dc'][m].mean():+.2f}, |grad| "
            f"{np.hypot(F['stem_peak_gx'][m], F['stem_peak_gy'][m]).mean():.2f}, centre-surround {F['stem_peak_cs'][m].mean():+.2f}; input share rain/exposure/coord/valid "
            f"{'/'.join(f'{v:.2f}' for v in share(m))}")
    rain_share = (nrm[:, [0, 1]].sum(1)) / nrm.sum(1)
    say(f"  rain share of the kernel norm per filter: min {rain_share.min():.2f}, median {np.median(rain_share):.2f}, max {rain_share.max():.2f}; "
        f"filters with a rain kernel whose |DC| < 30% of its L1 norm (sign-mixed): peak {int((np.abs(F['stem_peak_dc']) < 0.3 * np.abs(W[:, 0]).sum((1, 2))).sum())} / 32")

    # ---------------------------------------------------------------- 2 activations by layer
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    rng = np.random.default_rng(0)
    for li, l in enumerate(e87.LAYERS):
        c = F[f"act_{l}_corr"]
        s = F[f"act_{l}_std"]
        for ax, v in ((axs[0], np.abs(c[:, :4]).max(1)), (axs[1], np.abs(c[:, 4])), (axs[2], s / s.mean())):
            ax.scatter(li + rng.uniform(-0.2, 0.2, len(v)), v, s=14, color=COL[0], alpha=0.8)
            ax.plot([li - 0.3, li + 0.3], [np.median(v)] * 2, color="k", lw=1.5)
    for ax, t in zip(axs[:3], ["max |corr| with rain (1-33 px)", "|corr| with centre-minus-surround", "std over pixels / layer mean"]):
        ax.set_xticks(range(7))
        ax.set_xticklabels(e87.LAYERS, rotation=40, ha="right", fontsize=8)
        ax.set_title(t, fontsize=9)
    cc = F["act_emb_corr"] * orient[:, None]
    im = axs[3].imshow(cc[ordr], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axs[3].set_xticks(range(5))
    axs[3].set_xticklabels(["1 px", "3 px", "9 px", "33 px", "centre -\nsurround"], fontsize=8)
    axs[3].set_yticks(range(16))
    axs[3].set_yticklabels([f"e{k}" for k in ordr], fontsize=7)
    for t, k in zip(axs[3].get_yticklabels(), ordr):
        t.set_color(COL[lab_e[k]])
    axs[3].set_title("embedding channels x rain scale (sign-oriented)", fontsize=9)
    fig.colorbar(im, ax=axs[3], fraction=0.04)
    fig.savefig(FIG / f"activations_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    say("\n== activation statistics over 100 training days with claims (valid pixels)")
    for l in e87.LAYERS:
        c, s = F[f"act_{l}_corr"], F[f"act_{l}_std"]
        say(f"  {l:7s} channels {len(s):2d}: dead (std < 1e-3) {int((s < 1e-3).sum())}, median max|corr| with rain {np.median(np.abs(c[:, :4]).max(1)):.2f}, "
            f"median |corr| with centre-minus-surround {np.median(np.abs(c[:, 4])):.2f}, best scale r1/r3/r9/r33 {np.bincount(np.abs(c[:, :4]).argmax(1), minlength=4).tolist()}")

    # ---------------------------------------------------------------- 3 embedding footprints
    fp = (F["fp_signed"].sum(1) * orient[:, None, None])
    fig, axs = plt.subplots(2, 8, figsize=(17, 5.2))
    for t, k in enumerate(ordr):
        ax = axs[t // 8, t % 8]
        zm = 12
        cr = fp[k][HALF - zm:HALF + zm + 1, HALF - zm:HALF + zm + 1]
        v = np.abs(cr).max()
        ax.imshow(cr / v, cmap="RdBu_r", vmin=-1, vmax=1, extent=(-zm - 0.5, zm + 0.5, zm + 0.5, -zm - 0.5))
        for r in (2, 4, 8):
            ax.add_patch(plt.Circle((0, 0), r, fill=False, color="k", lw=0.4, ls=":"))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(COL[lab_e[k]])
            sp.set_linewidth(3)
        ax.set_title(f"e{k}", fontsize=8, color=COL[lab_e[k]])
    fig.suptitle("Embedding-channel footprints: mean input gradient (rain peak + sum) at the pixel of a claimed block group, zoom +-12 px (4 km), dotted circles 2 / 4 / 8 px; "
                 "each panel scaled to its own maximum, channel sign oriented", fontsize=9)
    fig.savefig(FIG / f"embedding_footprints_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # radial profiles
    bins = np.arange(0, HALF + 1)
    prof = np.array([[fp[k][(rr >= b) & (rr < b + 1)].mean() for b in bins] for k in range(16)])
    cum = np.array([[(fp[k][rr < b + 1]).sum() for b in bins] for k in range(16)])
    tot = np.abs(F["fp_abs"].sum(1)).reshape(16, -1).sum(1)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for k in range(16):
        axs[0].plot(bins, prof[k] / np.abs(prof[k]).max(), color=COL[lab_e[k]], alpha=0.7)
        axs[1].plot(bins, cum[k] / tot[k], color=COL[lab_e[k]], alpha=0.7)
    for g in range(ke):
        axs[2].bar(np.arange(len(RINGS) - 1) + g * 0.8 / ke, [np.mean([sg[k][(rr >= RINGS[j]) & (rr < RINGS[j + 1])].sum() * orient[k] / tot[k] for k in np.flatnonzero(lab_e == g)])
                                                             for j in range(len(RINGS) - 1)], 0.8 / ke, color=COL[g], label=f"group {g}")
    axs[2].set_xticks(np.arange(len(RINGS) - 1) + 0.4)
    axs[2].set_xticklabels(["<1", "1-2", "2-4", "4-8", "8-16", "16-32", ">32"], fontsize=8)
    axs[2].legend(fontsize=7)
    axs[0].set_title("gradient per pixel vs distance (px), each channel scaled to its max", fontsize=9)
    axs[1].set_title("cumulative signed gradient within radius / total |gradient|", fontsize=9)
    axs[2].set_title("signed gradient share by distance ring (group mean)", fontsize=9)
    for ax in axs[:2]:
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("distance (px, 1 px = 4 km)")
    fig.savefig(FIG / f"footprint_profiles_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    say("\n== embedding footprints (mean input gradient at claimed block-group pixels, oriented so the centre is positive)")
    for g in range(ke):
        ks_ = np.flatnonzero(lab_e == g)
        shares = [np.mean([sg[k][(rr >= RINGS[j]) & (rr < RINGS[j + 1])].sum() * orient[k] / tot[k] for k in ks_]) for j in range(len(RINGS) - 1)]
        pk = [int(bins[np.argmax(np.abs(prof[k]))]) for k in ks_]
        zc = [np.flatnonzero(np.sign(prof[k][1:]) != np.sign(prof[k][:-1])) for k in ks_]
        say(f"  group {g}: channels {list(ks_)}; signed share by ring <1/1-2/2-4/4-8/8-16/16-32/>32 px: " + " / ".join(f"{v:+.2f}" for v in shares)
            + f"; peak |gradient| at r={pk}; first sign change at r={[int(z[0]) + 1 if len(z) else None for z in zc]}")

    # ---------------------------------------------------------------- 4 response maps
    offs = F["map_offsets"]
    ext = (offs[0] - 1, offs[-1] + 1, offs[-1] + 1, offs[0] - 1)
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), constrained_layout=True)
    for j, (si, bgn) in enumerate([(0, "dry"), (0, "wet"), (1, "dry"), (1, "wet")]):
        mp = F[f"site{si}_{bgn}_map_logit"] - F[f"site{si}_{bgn}_logit"][0]
        v = np.abs(mp).max()
        im = axs[j].imshow(mp, cmap="RdBu_r", vmin=-v, vmax=v, extent=ext)
        axs[j].plot(0, 0, "k+", ms=10)
        for r in (4, 16, 32):
            axs[j].add_patch(plt.Circle((0, 0), r, fill=False, color="k", lw=0.4, ls=":"))
        axs[j].set_title(f"site {si}, {bgn} background, 50 mm blob (sigma 1.5 px)", fontsize=8)
        fig.colorbar(im, ax=axs[j], fraction=0.045, label="change of the logit")
        axs[j].set_xlabel("blob offset east (px)")
    axs[0].set_ylabel("blob offset south (px)")
    fig.savefig(FIG / f"logit_response_maps_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(2, 8, figsize=(17, 5.2))
    for t, k in enumerate(ordr):
        mp = F["site0_dry_map_emb"][:, :, k] - F["site0_dry_emb"][0, k]
        mp = mp * orient[k]
        v = np.abs(mp).max()
        ax = axs[t // 8, t % 8]
        ax.imshow(mp / v, cmap="RdBu_r", vmin=-1, vmax=1, extent=ext)
        ax.plot(0, 0, "k+", ms=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"e{k}  (max {v:.2f})", fontsize=8, color=COL[lab_e[k]])
    fig.suptitle("Response of each own-pixel embedding channel to a 50 mm blob moved around the block group (site 0, dry background), channel sign oriented", fontsize=9)
    fig.savefig(FIG / f"embedding_response_maps_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    say("\n== response of the logit to a 50 mm sigma-1.5 px blob (change from the empty background), by offset")
    for si in (0, 1):
        for bgn in ("dry", "wet"):
            mp = F[f"site{si}_{bgn}_map_logit"] - F[f"site{si}_{bgn}_logit"][0]
            ci = len(offs) // 2
            r = np.hypot(*np.meshgrid(offs, offs))
            ring = lambda lo, hi: mp[(r >= lo) & (r < hi)].mean()
            say(f"  site {si} {bgn}: centre {mp[ci, ci]:+.2f}; mean by ring <3 {ring(0, 3):+.2f}, 3-8 {ring(3, 8):+.2f}, 8-16 {ring(8, 16):+.2f}, 16-32 {ring(16, 32):+.2f}, 32-56 {ring(32, 57):+.2f} "
                f"(range {mp.min():+.2f} .. {mp.max():+.2f}; the logit of the empty background is {F[f'site{si}_{bgn}_logit'][0]:+.2f})")

    say("\n== anisotropy of the logit response (mean change by sector, blobs 4-20 px from the block group, offsets in px)")
    gy, gx = np.meshgrid(offs, offs, indexing="ij")
    for si in (0, 1):
        for bgn in ("dry", "wet"):
            mp = F[f"site{si}_{bgn}_map_logit"] - F[f"site{si}_{bgn}_logit"][0]
            rg = np.hypot(gx, gy)
            band = (rg >= 4) & (rg <= 20)
            sec = {"west": band & (gx < 0) & (np.abs(gy) <= np.abs(gx)), "east": band & (gx > 0) & (np.abs(gy) <= np.abs(gx)),
                   "north": band & (gy < 0) & (np.abs(gx) < np.abs(gy)), "south": band & (gy > 0) & (np.abs(gx) < np.abs(gy))}
            say(f"  site {si} {bgn}: " + ", ".join(f"{k} {mp[m].mean():+.3f}" for k, m in sec.items()))
    sgn_all = F["fp_signed"].sum(1) * orient[:, None, None]
    ew = [(sgn_all[k][:, :HALF].sum(), sgn_all[k][:, HALF + 1:].sum()) for k in range(16)]
    say("  mean input gradient summed west / east of the block-group column (oriented, mean over channels): "
        f"{np.mean([a for a, _ in ew]):+.4f} / {np.mean([b for _, b in ew]):+.4f}")

    # ---------------------------------------------------------------- 5 synthetic patterns
    names = [str(n) for n in F["site0_names"]]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    ypos = np.arange(len(names))
    for j, (si, bgn, c) in enumerate([(0, "dry", COL[0]), (0, "wet", "#56B4E9"), (1, "dry", COL[1]), (1, "wet", "#E69F00")]):
        lg = F[f"site{si}_{bgn}_logit"]
        axs[0].barh(ypos + (j - 1.5) * 0.2, lg[1:] - lg[0], 0.2, color=c, label=f"site {si}, {bgn} background")
    axs[0].set_yticks(ypos)
    axs[0].set_yticklabels(names, fontsize=8)
    axs[0].invert_yaxis()
    axs[0].axvline(0, color="k", lw=0.5)
    axs[0].set_xlabel("change of the logit against the same background without the pattern")
    axs[0].legend(fontsize=8)
    dE = (F["site0_dry_emb"][1:] - F["site0_dry_emb"][0]) * orient[None]
    v = np.abs(dE).max()
    im = axs[1].imshow(dE[:, ordr], cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    axs[1].set_xticks(range(16))
    axs[1].set_xticklabels([f"e{k}" for k in ordr], rotation=90, fontsize=7)
    for t, k in zip(axs[1].get_xticklabels(), ordr):
        t.set_color(COL[lab_e[k]])
    axs[1].set_yticks(ypos)
    axs[1].set_yticklabels([])
    axs[1].set_title("own-pixel embedding change, site 0 dry (oriented)", fontsize=9)
    fig.colorbar(im, ax=axs[1], fraction=0.04)
    fig.savefig(FIG / f"synthetic_patterns_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    say("\n== synthetic patterns: change of the logit (site 0 dry / site 0 wet / site 1 dry / site 1 wet)")
    for i, nme in enumerate(names):
        say(f"  {nme:24s} " + " / ".join(f"{F[f'site{si}_{b}_logit'][i + 1] - F[f'site{si}_{b}_logit'][0]:+.2f}" for si, b in ((0, 'dry'), (0, 'wet'), (1, 'dry'), (1, 'wet'))))

    # ---------------------------------------------------------------- 6 top activations
    crop = F["top_hi"].shape[-1] // 2
    ext2 = (-crop - 0.5, crop + 0.5, crop + 0.5, -crop - 0.5)
    ref = F["claimed_crop"][0]
    vmx = max(F["top_hi"][:, 0].max(), F["top_lo"][:, 0].max(), ref.max())
    fig, axs = plt.subplots(8, 4, figsize=(8.5, 15))
    for r in range(8):
        for blk in range(2):
            k = ordr[blk * 8 + r]
            for cc, key in enumerate(("top_hi", "top_lo")):
                ax = axs[r, blk * 2 + cc]
                ax.imshow(F[key][k, 0], cmap="magma", vmin=0, vmax=vmx, extent=ext2)
                ax.plot(0, 0, "c+", ms=5)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title("most active 200" if cc == 0 else "least active 200", fontsize=8)
                if cc == 0:
                    ax.set_ylabel(f"e{k}", color=COL[lab_e[k]], fontsize=9, rotation=0, labelpad=14)
    fig.suptitle("Mean rain-peak crop (+-20 px) around the 200 pixels where each embedding channel is most / least active (test days); scale 0-%.2f of log1p(mm)/3\n"
                 "reference: mean crop around claimed block groups has centre %.2f and 20-px edge %.2f" % (vmx, ref[crop, crop], ref[crop, 0]), fontsize=8)
    fig.savefig(FIG / f"top_activations_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    say("\n== mean rain peak (log1p(mm)/3) around the extreme pixels of each channel: centre / mean of the ring 8-16 px / mean of the ring 16-20 px")
    yy2, xx2 = np.mgrid[-crop:crop + 1, -crop:crop + 1]
    r2 = np.hypot(yy2, xx2)
    for k in ordr:
        out = []
        for key in ("top_hi", "top_lo"):
            m = F[key][k, 0]
            out.append(f"{m[crop, crop]:.2f}/{m[(r2 >= 8) & (r2 < 16)].mean():.2f}/{m[(r2 >= 16) & (r2 <= 20)].mean():.2f}")
        say(f"  e{k:<2d} group {lab_e[k]}: most active {out[0]}   least active {out[1]}")
    say(f"  reference (claimed block groups): {ref[crop, crop]:.2f}/{ref[(r2 >= 8) & (r2 < 16)].mean():.2f}/{ref[(r2 >= 16) & (r2 <= 20)].mean():.2f}")

    # ---------------------------------------------------------------- 7 ablation, redundancy, groups
    own = np.load(D / f"own_emb_s{seed}.npy", mmap_mode="r")
    sub = np.asarray(own[np.random.default_rng(0).choice(len(own), 400000, replace=False)], np.float32)
    sub = sub * orient[None]
    C = np.corrcoef(sub.T)
    ev = np.linalg.eigvalsh(np.cov(sub.T))[::-1]
    ev = ev / ev.sum()
    var = M["variants"].tolist()
    ml = M["metrics"].tolist()
    ij = {n: i for i, n in enumerate(var)}
    fig, axs = plt.subplots(1, 4, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    xs = np.arange(16)
    for mi, (mname, off) in enumerate((("BG top-5%", -0.2), ("BG AUC", 0.2))):
        j = ml.index(mname)
        d_ = np.array([M["delta"][ij[f"ch{k}"], j] for k in ordr])
        axs[0].bar(xs + off, d_[:, 0], 0.4, yerr=[d_[:, 0] - d_[:, 1], d_[:, 2] - d_[:, 0]], color=[COL[lab_e[k]] for k in ordr], alpha=1.0 if mi == 0 else 0.5, capsize=1.5,
                   label=mname)
    axs[0].set_xticks(xs)
    axs[0].set_xticklabels([f"e{k}" for k in ordr], rotation=90, fontsize=7)
    axs[0].set_title("replace one channel by its training mean\n(solid: top-5% capture, light: BG AUC; 95% bootstrap over days)", fontsize=8)
    axs[0].axhline(0, color="k", lw=0.5)
    axs[0].legend(fontsize=7)
    im = axs[1].imshow(C[np.ix_(ordr, ordr)], cmap="RdBu_r", vmin=-1, vmax=1)
    axs[1].set_xticks(range(16))
    axs[1].set_yticks(range(16))
    axs[1].set_xticklabels([f"e{k}" for k in ordr], rotation=90, fontsize=6)
    axs[1].set_yticklabels([f"e{k}" for k in ordr], fontsize=6)
    axs[1].set_title("correlation of the own-pixel embedding\nchannels over rows (oriented)", fontsize=8)
    for g in range(ke):
        m = lab_e == g
        axs[2].scatter(A["pc_e"][m, 0], A["pc_e"][m, 1], color=COL[g], s=40, label=f"group {g} (n={int(m.sum())})")
    for k in range(16):
        axs[2].annotate(f"e{k}", A["pc_e"][k], fontsize=7, xytext=(2, 2), textcoords="offset points")
    axs[2].set_xlabel(f"PC1 ({A['ev_e'][0]:.0%})")
    axs[2].set_ylabel(f"PC2 ({A['ev_e'][1]:.0%})")
    axs[2].set_title("PCA of the channel descriptors\n(footprint rings + rain correlations)", fontsize=8)
    axs[2].legend(fontsize=7)
    axs[3].plot(np.arange(1, 17), np.cumsum(ev), "o-", color=COL[0])
    axs[3].set_xlabel("principal components of the own-pixel embedding")
    axs[3].set_ylabel("cumulative variance share")
    axs[3].set_title("effective dimension of the embedding", fontsize=8)
    fig.colorbar(im, ax=axs[1], fraction=0.04)
    fig.savefig(FIG / f"embedding_channels_s{seed}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    say("\n== embedding channels: groups and ablation (change from the unablated model; pooled test days)")
    say(f"  k-means silhouette by k: {dict(zip(A['sil_e'][:, 0].astype(int).tolist(), np.round(A['sil_e'][:, 1], 2).tolist()))}")
    say(f"  own-pixel embedding PCA: first 1 / 2 / 3 / 4 / 6 components explain {' / '.join(f'{np.cumsum(ev)[i]:.2f}' for i in (0, 1, 2, 3, 5))} of the variance")
    base = {ml[j]: M["pooled"][ij["none"], j] for j in range(len(ml))}
    say("  unablated: " + ", ".join(f"{k} {v:.3f}" for k, v in base.items()))
    for nme in [f"ch{k}" for k in ordr] + [f"group{g}" for g in range(ke)]:
        i = ij[nme]
        say(f"  {nme:7s} " + "  ".join(f"{ml[j]} {M['delta'][i, j, 0]:+.3f} [{M['delta'][i, j, 1]:+.3f},{M['delta'][i, j, 2]:+.3f}]" for j in range(len(ml))))
    out_file = e87.wb2.cache_path("audit") / f"cnn_filters_s{seed}.txt"
    out_file.write_text("\n".join(txt) + "\n")
    print(f"figures in {FIG}; text in {out_file}")
