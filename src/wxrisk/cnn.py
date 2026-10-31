"""CNN hybrid for the flood ranking product (docs/CNN_HYBRID_DESIGN.md).

A small fully convolutional encoder turns the day's field stack into an embedding map at the resolution of its input; every block group reads the
embedding at its own pixel, the mean and the maximum of the embedding over its county, and its tabular features, and a small head turns them into a logit.
One forward pass of the encoder serves all block groups of a day. torch is an optional dependency; import this module only where it is needed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = ["Hybrid", "MultiScale", "pad_index", "day_loss"]


class Block(nn.Module):
    """Residual 3x3 convolution with a dilation, group normalisation and GELU (padding keeps the size)."""

    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, x):
        return x + nn.functional.gelu(self.norm(self.conv(x)))


class Hybrid(nn.Module):
    """Encoder + read-out + head. `forward(img, pix, county, tab, pool_idx, pool_ok)` returns one logit per row.

    img      [1, C, H, W]   the day's field stack
    pix      [R] long       flat pixel index (y * W + x) of every row (block group)
    county   [R] long       county position (row of pool_idx) of every row
    tab      [R, n_tab]     standardised tabular features
    pool_idx [K, L] long    flat pixel indices of the pixels of every county, padded; pool_ok [K, L] bool marks the real entries
    """

    def __init__(self, in_ch: int, n_tab: int, emb: int = 16, ch: int = 32, dilations=(1, 2, 4, 1), hidden: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU())
        self.blocks = nn.Sequential(*[Block(ch, d) for d in dilations])
        self.out = nn.Conv2d(ch, emb, 1)
        self.tab = nn.Sequential(nn.Linear(n_tab, 32), nn.GELU())
        self.head = nn.Sequential(nn.Linear(emb * 3 + 32, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.emb = emb

    def embed(self, img):
        return self.out(self.blocks(self.stem(img)))  # [1, emb, H, W]

    def forward(self, img, pix, county, tab, pool_idx, pool_ok):
        flat = self.embed(img)[0].reshape(self.emb, -1).T  # [H*W, emb]
        own = flat[pix]
        g = flat[pool_idx.clamp(min=0)]  # [K, L, emb]
        ok = pool_ok.unsqueeze(-1)
        mean = (g * ok).sum(1) / ok.sum(1).clamp(min=1)
        mx = g.masked_fill(~ok, float("-inf")).amax(1)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        z = torch.cat([own, mean[county], mx[county], self.tab(tab)], dim=1)
        return self.head(z).squeeze(-1)


class MultiScale(nn.Module):
    """Pyramid of small CNNs with a learned vote (stage 48): one encoder per scale s sees the day's stack average-pooled by s (s = 1 is the full-resolution
    image), its embedding is bilinearly brought back to the grid, and a per-scale head turns the own-pixel embedding + tabular branch into a logit. A gate reads the
    tabular branch and all own-pixel embeddings and returns softmax weights over the scales; the prediction is the weighted sum of the per-scale logits
    (`forward` returns that, the per-scale logits and the weights, so every scale can also be trained and scored on its own). No county pooling (stage 46: it carries nothing).

    img [1, C, H, W]; pix [R] flat pixel index of every row; tab [R, n_tab].
    """

    def __init__(self, in_ch: int, n_tab: int, scales=(1, 2, 4, 8, 16), emb: int = 16, ch: int = 16, dilations=(1, 2, 4), hidden: int = 32):
        super().__init__()
        self.scales = tuple(scales)
        self.emb = emb
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Sequential(nn.Conv2d(in_ch, ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU()), *[Block(ch, d) for d in dilations], nn.Conv2d(ch, emb, 1))
            for _ in self.scales])
        self.tab = nn.Sequential(nn.Linear(n_tab, 32), nn.GELU())
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(emb + 32, hidden), nn.GELU(), nn.Linear(hidden, 1)) for _ in self.scales])
        self.gate = nn.Sequential(nn.Linear(32 + emb * len(self.scales), hidden), nn.GELU(), nn.Linear(hidden, len(self.scales)))

    def own_embeddings(self, img, pix):
        out = []
        for s, enc in zip(self.scales, self.enc):
            x = nn.functional.avg_pool2d(img, s, s, ceil_mode=True) if s > 1 else img
            e = enc(x)
            if s > 1:
                e = nn.functional.interpolate(e, size=img.shape[-2:], mode="bilinear", align_corners=False)
            out.append(e[0].reshape(self.emb, -1).T[pix])
        return out

    def forward(self, img, pix, tab):
        own = self.own_embeddings(img, pix)
        t = self.tab(tab)
        logits = torch.cat([h(torch.cat([o, t], 1)) for h, o in zip(self.heads, own)], 1)  # [R, S]
        w = torch.softmax(self.gate(torch.cat([t] + own, 1)), dim=1)
        return (w * logits).sum(1), logits, w


def pad_index(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of index arrays to [K, L]; returns the indices (0 in the padding) and the validity mask."""
    L = max(len(g) for g in groups)
    idx = np.zeros((len(groups), L), np.int64)
    ok = np.zeros((len(groups), L), bool)
    for i, g in enumerate(groups):
        idx[i, : len(g)] = g
        ok[i, : len(g)] = True
    return idx, ok


def day_loss(logit: torch.Tensor, n: torch.Tensor, w: torch.Tensor, lam: float = 0.3) -> torch.Tensor:
    """Listwise softmax cross entropy against claim counts (negatives re-weighted by w inside the normaliser) + lam x weighted binary cross entropy per claimed row.

    logit, n, w are the rows of one day (all positives and a sample of negatives); w = 1 for positives, (negatives in the day) / (sampled) for negatives.
    """
    pos = n > 0
    bce = nn.functional.binary_cross_entropy_with_logits(logit, pos.to(logit.dtype), reduction="none")
    loss = lam * (w * bce).sum() / pos.sum().clamp(min=1)
    if pos.any():
        logp = torch.log_softmax(logit + torch.log(w), dim=0)
        t = n / n[pos].sum()
        loss = loss - (t[pos] * logp[pos]).sum()
    return loss
