"""Probe which compute devices PyTorch can use here and whether the kernels wxrisk needs work on them.

For every available device (cpu, mps, cuda, xpu) each kernel is run on random data and compared with the CPU
result (max relative error), with wall time at a small (150x150, one MRMS crop) and a large (1024x1024) size.
Share the printed table when asking "does this machine's GPU work for us".

Run: python scripts/probe_devices.py
"""

from __future__ import annotations

import platform
import time

import torch
import torch.nn.functional as F


def devices() -> list[str]:
    out = ["cpu"]
    if torch.cuda.is_available():
        out.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        out.append("xpu")
    if torch.backends.mps.is_available():
        out.append("mps")
    return out


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()
    elif dev == "xpu":
        torch.xpu.synchronize()


def k_rfft2(x):
    return torch.fft.irfft2(torch.fft.rfft2(x) * 0.5, s=x.shape[-2:])


def k_c2c(x):
    z = torch.complex(x, x.flip(-1))
    return torch.fft.ifft2(torch.fft.fft2(z) * z.conj()).abs()


def k_phase(x):
    a, b = x, x.roll(3, dims=-1)
    c = torch.fft.fft2(a) * torch.fft.fft2(b).conj()
    return torch.fft.ifft2(c / (c.abs() + 1e-6)).real


def k_warp(x):
    n, h, w = x.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device), torch.linspace(-1, 1, w, device=x.device),
                            indexing="ij")
    g = torch.stack([xx + 0.01, yy - 0.01], -1).expand(n, h, w, 2)
    return F.grid_sample(x[:, None], g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]


def k_gauss(x):
    t = torch.arange(-12, 13, dtype=x.dtype, device=x.device)
    k = torch.exp(-t**2 / (2 * 4.0**2))
    k = k / k.sum()
    y = F.conv2d(x[:, None], k.view(1, 1, 1, -1), padding=(0, 12))
    return F.conv2d(y, k.view(1, 1, -1, 1), padding=(12, 0))[:, 0]


def k_idw(x):
    n, h, w = x.shape
    pts = torch.rand(256, 2, generator=torch.Generator().manual_seed(1)).to(x.device) * torch.tensor([h, w], device=x.device)
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    grid = torch.stack([gy, gx], -1).reshape(-1, 2).to(x.dtype)
    d, i = torch.topk(torch.cdist(grid, pts.to(x.dtype)), 8, largest=False)
    wgt = 1 / (d**2 + 1e-3)
    return (wgt / wgt.sum(1, keepdim=True)).sum(1).reshape(h, w)


KERNELS = {"rfft2": k_rfft2, "c2c_fft": k_c2c, "phase_corr": k_phase, "grid_sample": k_warp,
           "sep_conv": k_gauss, "idw_topk": k_idw}


def run(fn, x, dev):
    xd = x.to(dev)
    try:
        fn(xd)  # warm-up (shader compile)
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(3):
            y = fn(xd)
        sync(dev)
        return y.cpu(), (time.perf_counter() - t0) / 3
    except Exception as e:  # noqa: BLE001 - an unsupported op is a probe result, not a failure
        return None, f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"


def main() -> None:
    print(f"{platform.platform()} | python {platform.python_version()} | torch {torch.__version__}")
    devs = devices()
    print("devices:", devs)
    for size, n in ((150, 24), (1024, 4)):
        x = torch.rand(n, size, size, generator=torch.Generator().manual_seed(0))
        print(f"\n== batch {n} x {size}x{size} float32 ==")
        print(f"{'kernel':12s}" + "".join(f"{d:>26s}" for d in devs))
        for name, fn in KERNELS.items():
            ref, t_cpu = run(fn, x, "cpu")
            cells = []
            for d in devs:
                y, t = ("ref", t_cpu) if d == "cpu" else run(fn, x, d)
                if d == "cpu":
                    cells.append(f"{t * 1e3:8.1f} ms")
                elif y is None:
                    cells.append(t)
                else:
                    err = ((y - ref).abs().max() / (ref.abs().max() + 1e-12)).item()
                    cells.append(f"{t * 1e3:8.1f} ms  err {err:.1e}")
            print(f"{name:12s}" + "".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    main()
