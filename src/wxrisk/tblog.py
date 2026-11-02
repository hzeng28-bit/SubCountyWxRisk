"""TensorBoard logging for the training scripts: one `RunLog` per run, written under `$WXRISK_DATA/runs/<group>/<run>`.

A log must never break a training run: without the optional `tensorboard` package (extra `monitor`) or with `WXRISK_TB=0` every call is a no-op.
Open the dashboard with `scripts/tb.py` (or `tensorboard --logdir $WXRISK_DATA/runs`).
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import wb2

__all__ = ["RunLog", "runs_root"]


def runs_root() -> Path:
    return wb2.cache_path("runs")


class RunLog:
    """Scalars, text and weight histograms of one run. `name` is `group/run`; an existing directory of the same name is replaced (a restarted run must not mix curves)."""

    def __init__(self, name: str, hparams: dict | None = None, root: Path | None = None):
        self.w = None
        self.t0 = time.time()
        if os.environ.get("WXRISK_TB", "1") == "0":
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:  # tensorboard (or torch) not installed
            return
        path = (root or runs_root()) / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
        self.dir = path
        self.w = SummaryWriter(str(path), flush_secs=10)
        if hparams:
            self.text("hparams", "\n".join(f"- **{k}**: {v}" for k, v in hparams.items()))

    @property
    def active(self) -> bool:
        return self.w is not None

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.w is not None:
            self.w.add_scalar(tag, float(value), step)

    def scalars(self, tag: str, values: dict, step: int) -> None:
        if self.w is not None:  # same run, tags tag/<key>: add_scalars would open one sub-run per key and clutter the run list
            for k, v in values.items():
                self.w.add_scalar(f"{tag}/{k}", float(v), step)

    def text(self, tag: str, s: str, step: int = 0) -> None:
        if self.w is not None:
            self.w.add_text(tag, s, step)

    def weights(self, model, step: int) -> None:
        """Histograms of every parameter and of its gradient (the gradients of the last step)."""
        if self.w is None:
            return
        for k, p in model.named_parameters():
            self.w.add_histogram(f"weights/{k}", p.detach().cpu(), step)
            if p.grad is not None:
                self.w.add_histogram(f"grads/{k}", p.grad.detach().cpu(), step)

    def close(self) -> None:
        if self.w is not None:
            self.w.flush()
            self.w.close()
            self.w = None
