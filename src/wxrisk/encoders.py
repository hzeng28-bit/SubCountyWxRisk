"""Multi-source neural encoder: every input type gets its own small encoder and all of them meet in one embedding per block group.

The aim is to replace hand-built feature assembly (pixel value, neighbourhood maxima, county summaries, merged tables) by learned encoders that are declared
per source and fused at the right level:

  fine grid sources     (4 km: observed rain, exposure maps, HAND / flood-zone fractions, ...)  1 x 1 projection + norm per source, concatenated, shared dilated trunk
  coarse grid sources   (0.25 degree: forecast members, ERA5 / forecast moisture and wind)       own trunk at the native resolution, brought to the fine grid by a cell
                                                                                                 index and used as FiLM (scale + shift) on the fine features, identity at init
  row sources           (one row per block group: policies, shares, claim history, geometry)      numeric columns + categorical embeddings (index 0 = unknown) -> MLP
  set sources           (several records per block group, e.g. policies)                         `SetPool`: per-record MLP and a segment mean (DeepSets)

`modality_dropout` zeroes a whole grid source for a day with that probability in training (the network must not rely on one source and a missing source
at test time stays in distribution). The head reads the block group's own-pixel embedding and the row branch; the county mean / max of the embedding is optional
(`county_pool`, stage 46: it carries no skill). Everything is fully convolutional, one day per call, any grid size.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .cnn import Block

__all__ = ["GridSpec", "SetPool", "MultiSourceNet"]


@dataclass(frozen=True)
class GridSpec:
    """A grid source: its name and number of channels."""

    name: str
    channels: int


def _proj(c: int, p: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c, p, 1), nn.GroupNorm(min(4, p), p), nn.GELU())


class SetPool(nn.Module):
    """DeepSets over the records of a group: per-record MLP, then the mean over the records of every group (segment mean)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, items: torch.Tensor, seg: torch.Tensor, n_seg: int) -> torch.Tensor:
        h = self.mlp(items)
        out = torch.zeros(n_seg, h.shape[1], dtype=h.dtype, device=h.device).index_add_(0, seg, h)
        cnt = torch.zeros(n_seg, dtype=h.dtype, device=h.device).index_add_(0, seg, torch.ones(len(seg), dtype=h.dtype, device=h.device))
        return out / cnt.clamp(min=1).unsqueeze(1)


class MultiSourceNet(nn.Module):
    """`forward(fine, coarse, up, pix, num, cat, county=None, pool_idx=None, pool_ok=None, extra=None)` returns one logit per row.

    fine    {name: [1, C, H, W]}   every fine source declared in `fine`
    coarse  {name: [1, C, h, w]}   every coarse source declared in `coarse` (may be empty)
    up      [H * W] long           flat coarse cell of every fine pixel (ignored without coarse sources)
    pix     [R] long               flat fine pixel of every row
    num     [R, n_numeric]         standardised numeric row features (n_numeric may be 0)
    cat     [R, len(cat_cards)]    categorical codes, 0 = unknown / missing, 1..card = values
    extra   [R, set_out]           optional output of a SetPool, appended to the row branch
    """

    def __init__(self, fine: list[GridSpec], coarse: list[GridSpec] | None = None, n_numeric: int = 0, cat_cards: tuple[int, ...] = (), *,
                 proj: int = 8, ch: int = 32, dilations=(1, 2, 4, 8, 16), emb: int = 16, coarse_ch: int = 16, coarse_dilations=(1, 2),
                 cat_dim: int = 4, tab_dim: int = 32, set_out: int = 0, hidden: int = 64, county_pool: bool = False, modality_dropout: float = 0.0):
        super().__init__()
        self.fine_specs, self.coarse_specs = list(fine), list(coarse or [])
        self.emb, self.county_pool, self.p_drop = emb, county_pool, modality_dropout
        self.fine_proj = nn.ModuleDict({s.name: _proj(s.channels, proj) for s in self.fine_specs})
        self.stem = nn.Sequential(nn.Conv2d(proj * len(self.fine_specs), ch, 3, padding=1), nn.GroupNorm(8, ch), nn.GELU())
        self.blocks = nn.Sequential(*[Block(ch, d) for d in dilations])
        self.out = nn.Conv2d(ch, emb, 1)
        if self.coarse_specs:
            self.coarse_proj = nn.ModuleDict({s.name: _proj(s.channels, proj) for s in self.coarse_specs})
            self.coarse_trunk = nn.Sequential(nn.Conv2d(proj * len(self.coarse_specs), coarse_ch, 3, padding=1), nn.GroupNorm(8, coarse_ch), nn.GELU(),
                                              *[Block(coarse_ch, d) for d in coarse_dilations])
            self.film = nn.Conv2d(coarse_ch, 2 * ch, 1)
            nn.init.zeros_(self.film.weight)  # the coarse branch starts as the identity: it must earn its influence
            nn.init.zeros_(self.film.bias)
        self.cats = nn.ModuleList([nn.Embedding(c + 1, cat_dim) for c in cat_cards])
        self.n_row = n_numeric + cat_dim * len(cat_cards) + set_out
        self.row = nn.Sequential(nn.Linear(self.n_row, tab_dim), nn.GELU()) if self.n_row else None
        self.head = nn.Sequential(nn.Linear(emb * (3 if county_pool else 1) + (tab_dim if self.row else 0), hidden), nn.GELU(), nn.Linear(hidden, 1))

    def _drop(self, h: torch.Tensor) -> torch.Tensor:
        if self.training and self.p_drop > 0 and torch.rand(()) < self.p_drop:
            return torch.zeros_like(h)
        return h

    def embed(self, fine: dict, coarse: dict | None = None, up: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([self._drop(self.fine_proj[s.name](fine[s.name])) for s in self.fine_specs], dim=1)
        x = self.stem(x)
        if self.coarse_specs:
            c = torch.cat([self._drop(self.coarse_proj[s.name](coarse[s.name])) for s in self.coarse_specs], dim=1)
            c = self.coarse_trunk(c)  # [1, coarse_ch, h, w]
            H, W = x.shape[-2:]
            gb = self.film(c).flatten(2)[0][:, up].reshape(2, -1, H, W)  # scale and shift on the fine grid
            x = x * (1 + gb[0:1]) + gb[1:2]
        return self.out(self.blocks(x))  # [1, emb, H, W]

    def forward(self, fine, coarse, up, pix, num, cat, county=None, pool_idx=None, pool_ok=None, extra=None):
        flat = self.embed(fine, coarse, up)[0].reshape(self.emb, -1).T
        z = [flat[pix]]
        if self.county_pool:
            g = flat[pool_idx.clamp(min=0)]
            ok = pool_ok.unsqueeze(-1)
            mean = (g * ok).sum(1) / ok.sum(1).clamp(min=1)
            mx = g.masked_fill(~ok, float("-inf")).amax(1)
            mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
            z += [mean[county], mx[county]]
        if self.row is not None:
            r = [num] if num.shape[1] else []
            r += [e(cat[:, i]) for i, e in enumerate(self.cats)]
            if extra is not None:
                r.append(extra)
            z.append(self.row(torch.cat(r, dim=1)))
        return self.head(torch.cat(z, dim=1)).squeeze(-1)
