"""Forecast-mode inference pipeline as one callable, and its cost on a laptop (scenario report, RESULTS stage 56).

Rebuilds the `fc + expo` multi-source network of script 91 (seed 0 weights `cnn/ms_fc+expo_s0.pt`) with the same row standardisation, runs one day (GEFS fields + exposure
maps + 9 tabular columns per block group) and checks the predictions against the stored ones of the evaluation (`cnn/ms_fc+expo_s0.npy`). Then it times the phases of one
day on this machine: cold start (imports, day tensors, weights), per-day feature preparation, forward pass on CPU and on the accelerator `compute.device()` picks, and
the resident memory. Download of the GEFS fields and the 4 km exposure maps are NOT in this timing (they are prepared offline here; see the scenario report).
Inputs : cnn/ (script 80), cnn/ms_fc+expo_s0.pt / .npy (script 91)
Output : audit/scenario_latency.json
Run    : python scripts/97_scenario_pipeline.py
"""

from __future__ import annotations

import importlib.util
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wxrisk import wb2  # noqa: E402
from wxrisk import zstio  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class Pipeline:
    """fc + expo network (script 91) with its row standardisation, one-day inference."""

    def __init__(self, device: str | None = None, seed: int = 0):
        import torch

        from wxrisk import compute
        from wxrisk.encoders import GridSpec, MultiSourceNet

        self.torch = torch
        self.e81 = _load("e81", "81_cnn_e1.py")
        self.d = d = self.e81.Data()
        self.dev = torch.device(device) if device else compute.device()
        rng = np.random.default_rng(seed)
        train_idx = np.flatnonzero(d.is_train)
        sample_days = rng.choice(train_idx, 300, replace=False)
        sample = np.concatenate([np.asarray(d.rtab[d.ptr[n]:d.ptr[n + 1]], np.float32) for n in sample_days])
        tf = self.e81.tab_features(sample)
        self.mu, self.sd = tf.mean(0), tf.std(0) + 1e-6
        self.valid_tx = d.st["valid_tx"].astype(np.float32)[None]
        self.model = MultiSourceNet([GridSpec("expo", 3)], [GridSpec("fc", 8)], 9, proj=8, ch=32, dilations=(1, 2, 4, 8, 16), modality_dropout=0.0)
        self.model.load_state_dict(torch.load(self.e81.D / f"ms_fc+expo_s{seed}.pt", map_location="cpu"))
        self.model.to(self.dev).eval()
        self.up = torch.from_numpy(d.cell4).to(self.dev)

    def day(self, n: int, timing: dict | None = None):
        """Probabilities of all block groups with a policy on day index n; returns (bg ids, p, counts)."""
        torch, d = self.torch, self.d
        t0 = time.perf_counter()
        bgs, tab, cnt, _ = d.rows(n)
        num = (self.e81.tab_features(tab) - self.mu) / self.sd
        img = d.image("4km", n)
        g = np.asarray(d.G[n], np.float32)
        f = {"expo": torch.from_numpy(img[3:6][None]).to(self.dev)}
        c = {"fc": torch.from_numpy(np.concatenate([np.log1p(np.clip(np.nan_to_num(g[0:7]), 0, None)) / 3, self.valid_tx])[None]).to(self.dev)}
        pix = torch.from_numpy(d.bpix4[bgs]).to(self.dev)
        numt = torch.from_numpy(num.astype(np.float32)).to(self.dev)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.model(f, c, self.up, pix, numt, torch.zeros((len(bgs), 0), dtype=torch.long, device=self.dev))
            p = torch.sigmoid(out).cpu().numpy()
        t2 = time.perf_counter()
        if timing is not None:
            timing.setdefault("prep", []).append(t1 - t0)
            timing.setdefault("forward", []).append(t2 - t1)
        return bgs, p, cnt


def main() -> None:
    t0 = time.perf_counter()
    pipe = Pipeline(device="cpu")
    cold = time.perf_counter() - t0
    d = pipe.d
    test = np.flatnonzero(d.is_test)
    lo = d.ptr[test[0]]
    stored = zstio.load(pipe.e81.D / "ms_fc+expo_s0.npy")
    rng = np.random.default_rng(0)
    days = rng.choice(test, 30, replace=False)
    res = {"cold_start_s": cold}
    err = []
    for dev in ("cpu", "mps"):
        try:
            p_ = pipe if dev == "cpu" else Pipeline(device=dev)
        except Exception as e:  # noqa: BLE001
            res[dev] = {"error": str(e)[:100]}
            continue
        tm: dict = {}
        p_.day(int(days[0]))  # warm-up
        for n in days:
            bgs, p, _ = p_.day(int(n), tm)
            if dev == "cpu":
                ref = stored[d.ptr[n] - lo:d.ptr[n + 1] - lo]
                err.append(float(np.abs(p - ref).max()))
        res[dev] = {"prep_ms": 1000 * float(np.median(tm["prep"])), "forward_ms": 1000 * float(np.median(tm["forward"])),
                    "total_ms_p95": 1000 * float(np.percentile(np.array(tm["prep"]) + np.array(tm["forward"]), 95)), "rows_median": int(np.median([d.ptr[n + 1] - d.ptr[n] for n in days]))}
    res["max_abs_diff_vs_stored_predictions"] = float(max(err))
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    res["parameters"] = int(sum(p.numel() for p in pipe.model.parameters()))
    print(json.dumps(res, indent=1))
    (wb2.cache_path("audit") / "scenario_latency.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
