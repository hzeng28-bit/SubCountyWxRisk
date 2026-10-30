import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wxrisk.cnn import Hybrid, day_loss, pad_index  # noqa: E402


def _batch(seed=0, H=9, W=11, R=30, K=4, n_tab=6, C=5):
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(1, C, H, W, generator=g)
    pix = torch.randint(0, H * W, (R,), generator=g)
    county = torch.randint(0, K, (R,), generator=g)
    tab = torch.randn(R, n_tab, generator=g)
    idx, ok = pad_index([np.arange(k, H * W, K + 1) for k in range(K)])
    return img, pix, county, tab, torch.from_numpy(idx), torch.from_numpy(ok)


def test_shapes_and_determinism():
    torch.manual_seed(0)
    m = Hybrid(5, 6, ch=16, dilations=(1, 2))
    b = _batch()
    out = m(*b)
    assert out.shape == (30,)
    assert torch.allclose(out, m(*b))


def test_gradients_reach_the_encoder_and_the_image():
    torch.manual_seed(0)
    m = Hybrid(5, 6, ch=16, dilations=(1, 2))
    img, *rest = _batch()
    img.requires_grad_(True)
    m(img, *rest).sum().backward()
    assert img.grad.abs().sum() > 0
    assert m.stem[0].weight.grad.abs().sum() > 0


def test_padding_does_not_change_county_pooling():
    torch.manual_seed(0)
    m = Hybrid(5, 6, ch=16, dilations=(1,))
    img, pix, county, tab, idx, ok = _batch()
    out = m(img, pix, county, tab, idx, ok)
    wider = torch.cat([idx, torch.zeros(idx.shape[0], 7, dtype=torch.long)], 1)
    wider_ok = torch.cat([ok, torch.zeros(ok.shape[0], 7, dtype=torch.bool)], 1)
    assert torch.allclose(out, m(img, pix, county, tab, wider, wider_ok), atol=1e-6)


def test_day_loss_prefers_the_claimed_rows_and_handles_days_without_claims():
    n = torch.tensor([3.0, 0.0, 0.0, 1.0])
    w = torch.ones(4)
    good = torch.tensor([3.0, -3.0, -3.0, 1.0])
    bad = torch.tensor([-3.0, 3.0, 3.0, -1.0])
    assert day_loss(good, n, w) < day_loss(bad, n, w)
    assert torch.isfinite(day_loss(good, torch.zeros(4), w))
