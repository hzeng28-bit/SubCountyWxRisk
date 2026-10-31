import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wxrisk.encoders import GridSpec, MultiSourceNet, SetPool  # noqa: E402

H, W, h, w, R = 12, 14, 3, 4, 25


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    fine = {"rain": torch.randn(1, 3, H, W, generator=g), "expo": torch.randn(1, 2, H, W, generator=g)}
    coarse = {"fc": torch.randn(1, 4, h, w, generator=g)}
    cy, cx = np.minimum(np.arange(H) * h // H, h - 1), np.minimum(np.arange(W) * w // W, w - 1)
    up = torch.from_numpy((cy[:, None] * w + cx[None, :]).ravel())
    pix = torch.randint(0, H * W, (R,), generator=g)
    num = torch.randn(R, 5, generator=g)
    cat = torch.randint(0, 4, (R, 2), generator=g)
    return fine, coarse, up, pix, num, cat


def _net(**kw):
    torch.manual_seed(0)
    return MultiSourceNet([GridSpec("rain", 3), GridSpec("expo", 2)], [GridSpec("fc", 4)], 5, (3, 3), ch=16, dilations=(1, 2), coarse_ch=8,
                          coarse_dilations=(1,), **kw)


def test_shapes_and_determinism():
    m, x = _net(), _inputs()
    out = m(*x)
    assert out.shape == (R,)
    assert torch.allclose(out, m(*x))


def test_coarse_branch_is_identity_at_init_and_trainable():
    m, (fine, coarse, up, pix, num, cat) = _net(), _inputs()
    base = m(fine, coarse, up, pix, num, cat)
    other = {"fc": coarse["fc"] * 5 + 3}
    assert torch.allclose(base, m(fine, other, up, pix, num, cat))  # zero-initialised FiLM
    base.sum().backward()
    assert m.film.weight.grad.abs().sum() > 0  # but it receives gradient, so it can learn to matter
    torch.nn.init.normal_(m.film.weight, std=0.5)
    assert not torch.allclose(base, m(fine, other, up, pix, num, cat))


def test_every_source_receives_gradient():
    m, (fine, coarse, up, pix, num, cat) = _net(), _inputs()
    torch.nn.init.normal_(m.film.weight, std=0.1)
    for t in list(fine.values()) + list(coarse.values()):
        t.requires_grad_(True)
    m(fine, coarse, up, pix, num, cat).sum().backward()
    assert all(t.grad.abs().sum() > 0 for t in list(fine.values()) + list(coarse.values()))
    assert all(e.weight.grad is not None for e in m.cats)


def test_modality_dropout_only_in_training():
    m, (fine, coarse, up, pix, num, cat) = _net(modality_dropout=1.0), _inputs()
    m.eval()
    a = m(fine, coarse, up, pix, num, cat)
    m.train()
    b = m(fine, coarse, up, pix, num, cat)  # every source zeroed
    assert not torch.allclose(a, b)
    m.eval()
    assert torch.allclose(a, m(fine, coarse, up, pix, num, cat))


def test_unknown_category_code_is_a_value_of_its_own():
    m, (fine, coarse, up, pix, num, cat) = _net(), _inputs()
    m.eval()
    a = m(fine, coarse, up, pix, num, cat)
    cat0 = torch.zeros_like(cat)
    assert not torch.allclose(a, m(fine, coarse, up, pix, num, cat0))


def test_without_coarse_and_rows():
    torch.manual_seed(0)
    m = MultiSourceNet([GridSpec("rain", 3)], ch=16, dilations=(1,))
    fine, _, _, pix, _, _ = _inputs()
    out = m({"rain": fine["rain"]}, {}, None, pix, torch.zeros(R, 0), torch.zeros(R, 0, dtype=torch.long))
    assert out.shape == (R,)


def test_county_pool_option_matches_padding_invariance():
    torch.manual_seed(0)
    m = MultiSourceNet([GridSpec("rain", 3)], n_numeric=2, ch=16, dilations=(1,), county_pool=True).eval()
    fine, _, _, pix, _, _ = _inputs()
    county = torch.randint(0, 2, (R,))
    idx = torch.tensor([[0, 5, 9, -1], [1, 2, -1, -1]])
    ok = idx >= 0
    num = torch.randn(R, 2)
    a = m({"rain": fine["rain"]}, {}, None, pix, num, torch.zeros(R, 0, dtype=torch.long), county, idx, ok)
    idx2 = torch.cat([idx, torch.full((2, 3), -1)], 1)
    b = m({"rain": fine["rain"]}, {}, None, pix, num, torch.zeros(R, 0, dtype=torch.long), county, idx2, idx2 >= 0)
    assert torch.allclose(a, b, atol=1e-6)


def test_set_pool_is_a_segment_mean_and_order_free():
    torch.manual_seed(0)
    sp = SetPool(3, 4)
    items = torch.randn(7, 3)
    seg = torch.tensor([0, 0, 1, 1, 1, 3, 3])
    out = sp(items, seg, 4)
    h = sp.mlp(items)
    assert torch.allclose(out[0], h[:2].mean(0), atol=1e-6)
    assert torch.allclose(out[1], h[2:5].mean(0), atol=1e-6)
    assert torch.all(out[2] == 0)  # empty group
    perm = torch.randperm(7)
    assert torch.allclose(out, sp(items[perm], seg[perm], 4), atol=1e-6)
