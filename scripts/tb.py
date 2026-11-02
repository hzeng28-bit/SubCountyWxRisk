"""Training dashboard: serve TensorBoard on the run logs, or backfill the epoch curves of earlier runs from the audit logs.

Training scripts (81, 89, 91, 93) write one run per fit under `$WXRISK_DATA/runs/<group>/<run>` through `wxrisk.tblog.RunLog`: per-step loss, gradient norm and
learning rate, per-epoch train / validation loss, best validation loss, days per second, weight and gradient histograms, hyper-parameters as text.
`backfill` turns the epoch lines of the older audit logs ("epoch N: train loss X, validation loss Y") into runs under `runs/past/` so the earlier fits show up in the same panel
(per-epoch curves only; no per-step data existed). Needs the extra `monitor` (`uv sync --extra torch --extra monitor`).
Run    : python scripts/tb.py serve [--port 6006] ; python scripts/tb.py backfill
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wxrisk import wb2  # noqa: E402
from wxrisk.tblog import RunLog, runs_root  # noqa: E402

LINE = re.compile(r"epoch (\d+): train loss ([\d.]+), validation loss ([\d.]+)(?: \((\d+) s\))?")
SKIP = ("mrms_finetune",)  # these runs are logged live


def backfill() -> None:
    n = 0
    for f in sorted(wb2.cache_path("audit").glob("*train*.txt")):
        if f.name.startswith(SKIP):
            continue
        rows = [(int(m[1]), float(m[2]), float(m[3]), float(m[4]) if m[4] else None) for m in map(LINE.search, f.read_text().splitlines()) if m]
        if not rows:
            continue
        log = RunLog(f"past/{f.stem}", {"source": f.name, "note": "backfilled from the audit log: epoch curves only"})
        for ep, tr, va, sec in rows:
            log.scalar("epoch/loss/train", tr, ep)
            log.scalar("epoch/loss/val", va, ep)
            if sec is not None:
                log.scalar("epoch/seconds", sec, ep)
        log.close()
        n += 1
    print(f"backfilled {n} runs under {runs_root() / 'past'}")


def serve(port: int) -> None:
    root = runs_root()
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "tensorboard.main", "--logdir", str(root), "--port", str(port), "--reload_interval", "10", "--samples_per_plugin", "histograms=20"], check=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve", "backfill"])
    ap.add_argument("--port", type=int, default=6006)
    a = ap.parse_args()
    backfill() if a.mode == "backfill" else serve(a.port)
