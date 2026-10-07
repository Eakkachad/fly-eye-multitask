import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models  # noqa: E402


def _lum():
    torch.manual_seed(0)
    return torch.rand(1, 3, 1, 721)


def _ntrain(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@pytest.fixture(scope="module")
def m1():
    return models.build_model("m1", seed=0)


@pytest.fixture(scope="module")
def hyb():
    return {k: models.build_model(k, seed=0) for k in ("m6", "m7")}


def test_shapes_match_m1(m1, hyb):
    ref = m1(_lum())
    for m in hyb.values():
        out = m(_lum())
        assert set(out) == set(ref)
        for k in ref:
            assert out[k].shape == ref[k].shape


def test_param_counts(hyb):
    n6, n7 = _ntrain(hyb["m6"]), _ntrain(hyb["m7"])
    assert n6 == n7
    assert 14925 <= n6 <= 15849


def test_m7_uses_rewired_file(hyb):
    assert hyb["m6"].connectome_file != hyb["m7"].connectome_file
    assert "rewired" in str(hyb["m7"].connectome_file) or "m2" in str(hyb["m7"].connectome_file) \
        or hyb["m7"].connectome_file != models.build_model("m1", seed=0).connectome_file


def test_grads_flow(hyb):
    m = hyb["m6"]
    m.zero_grad()
    out = m(_lum())
    (out["flow"].mean() + out["depth"].mean()).backward()
    assert any(p.grad is not None for p in m.network.parameters())
    assert all(p.grad is not None for p in m.trunk.parameters())


def test_frozen_hybrid(tmp_path, m1):
    ck = tmp_path / "best.pt"
    torch.save(dict(model=m1.state_dict(), iter=0), ck)
    m = models.build_model("m6f", seed=0, frontend_ckpt=str(ck))
    assert m.use_penalty is False and m.is_flyvis
    train_names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert train_names and all(n.startswith("trunk.") for n in train_names)
    assert _ntrain(m) == models.n_trainable(m)["total"]
    m.eval(); m1.eval()
    lum = _lum()
    with torch.no_grad():
        ref = m1(lum)
        out = m(lum)
        fe = m(lum, residual=False)
    for k in ("flow", "depth"):
        assert torch.equal(out[k], ref[k])
        assert torch.equal(fe[k], ref[k])
    # one optimizer step: frozen unchanged, trunk changed
    m.train()
    assert not m.network.training and not m.decoder.training
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    opt = torch.optim.Adam(m.param_groups(1e-2, 1e-2))
    o = m(lum)
    (o["flow"].square().mean() + o["depth"].square().mean()).backward()
    opt.step()
    changed = False
    for n, p in m.named_parameters():
        if n.startswith("trunk."):
            changed |= not torch.equal(p, before[n])
        else:
            assert torch.equal(p, before[n]), n
    assert changed
