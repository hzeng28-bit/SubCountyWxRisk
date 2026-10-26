"""Device layer: one place that decides where tensors live, so scripts do not name a backend.

`device()` picks cuda > xpu > mps > cpu, or whatever `WXRISK_DEVICE` says (e.g. `cpu` to debug, `cuda` on Colab).
Kernels here take and return numpy arrays, run in float32 on the device (MPS has no float64) and have a numpy
reference in `wxrisk.motion`; tests/test_compute.py keeps the two within tolerance. Nothing here is imported
unless a script asks for it, so the default install still runs without torch.
"""

from __future__ import annotations

import functools
import os

import numpy as np

__all__ = ["device", "describe", "estimate_shift_batch"]


@functools.cache
def device() -> str:
    import torch

    want = os.environ.get("WXRISK_DEVICE", "auto").lower()
    if want != "auto":
        return want
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe() -> str:
    import torch

    d = device()
    name = {"cuda": lambda: torch.cuda.get_device_name(), "xpu": lambda: torch.xpu.get_device_name()}.get(d)
    return f"torch {torch.__version__} on {d}" + (f" ({name()})" if name else "")


def estimate_shift_batch(a: np.ndarray, b: np.ndarray, amp_frac: float = 1e-3, eps: float = 1e-12) -> np.ndarray:
    """Batched `motion.estimate_shift` (no window): a, b of shape (B, nx, ny) -> (B, 3) = sx, sy, coherence.

    Same algorithm and sign convention (b(x) ~ a(x - s)): integer peak of the amplitude-masked phase
    correlation, then the weighted phase-slope fit for the fractional part.
    """
    import torch

    dev = device()
    ta = torch.as_tensor(np.asarray(a), dtype=torch.float32, device=dev)
    tb = torch.as_tensor(np.asarray(b), dtype=torch.float32, device=dev)
    ta = ta - ta.mean(dim=(1, 2), keepdim=True)
    tb = tb - tb.mean(dim=(1, 2), keepdim=True)
    B, nx, ny = ta.shape
    fa, fb = torch.fft.fft2(ta), torch.fft.fft2(tb)
    cross = fb * fa.conj()
    mag = fa.abs() * fb.abs()
    keep = mag >= amp_frac * mag.amax(dim=(1, 2), keepdim=True)
    cross = torch.where(keep, cross / (cross.abs() + eps), torch.zeros_like(cross))
    corr = torch.fft.ifft2(cross).real
    flat = corr.reshape(B, -1)
    peak, idx = flat.max(dim=1)
    ix, iy = idx // ny, idx % ny
    sx_int = ((ix + nx / 2) % nx) - nx / 2
    sy_int = ((iy + ny / 2) % ny) - ny / 2
    wx = (2 * torch.pi * torch.fft.fftfreq(nx, device=dev))[None, :, None]
    wy = (2 * torch.pi * torch.fft.fftfreq(ny, device=dev))[None, None, :]
    ramp = wx * sx_int[:, None, None] + wy * sy_int[:, None, None]
    phase_res = torch.angle(cross * torch.polar(torch.ones_like(ramp), ramp))
    w = mag * keep
    gxx, gyy, gxy = (w * wx**2).sum((1, 2)), (w * wy**2).sum((1, 2)), (w * wx * wy).sum((1, 2))
    rx, ry = -(w * wx * phase_res).sum((1, 2)), -(w * wy * phase_res).sum((1, 2))
    det = gxx * gyy - gxy * gxy
    ok = det.abs() > 1e-20
    safe = torch.where(ok, det, torch.ones_like(det))
    fx = torch.where(ok, (gyy * rx - gxy * ry) / safe, torch.zeros_like(det)).clamp(-0.5, 0.5)
    fy = torch.where(ok, (gxx * ry - gxy * rx) / safe, torch.zeros_like(det)).clamp(-0.5, 0.5)
    coh = peak / (corr.abs().mean(dim=(1, 2)) + eps)
    return torch.stack([sx_int + fx, sy_int + fy, coh], dim=1).cpu().numpy().astype(np.float64)
