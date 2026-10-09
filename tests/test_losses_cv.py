import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))
import losses as L  # noqa: E402
import splits as S  # noqa: E402


def test_si_invariant_to_constant_when_lambda_1():
    torch.manual_seed(0)
    p, g = torch.randn(2, 3, 1, 50), torch.randn(2, 3, 1, 50)
    a = L.si_loss(p, g, lam=1.0)
    b = L.si_loss(p + 3.7, g, lam=1.0)
    assert torch.allclose(a, b, atol=1e-5)
    # lambda=0 is plain mse, not invariant
    assert not torch.allclose(L.si_loss(p, g, 0.0), L.si_loss(p + 3.7, g, 0.0))
    assert torch.allclose(L.si_loss(p, g, 0.0), ((p - g) ** 2).mean(), atol=1e-6)


def test_si_hand_value_and_log_scale():
    d = torch.tensor([1.0, 3.0]).view(1, 1, 1, 2)
    z = torch.zeros_like(d)
    # mean(d^2)=5, mean(d)^2=4 -> 5 - 0.5*4 = 3
    assert L.si_loss(d, z, 0.5).item() == pytest.approx(3.0)
    assert L.si_loss(d, z, 0.5, log_scale=2.0).item() == pytest.approx(12.0)


def test_epe_hand_computation():
    gt = torch.zeros(1, 1, 2, 2)
    pred = torch.tensor([[[[3.0, 0.0], [4.0, 0.0]]]])  # hexal0 err 5, hexal1 err 0
    assert L.flow_epe_loss(pred, gt, eps=0.0).item() == pytest.approx(2.5)
    assert L.flow_epe_loss(pred, gt).item() == pytest.approx(2.5, abs=1e-3)


def test_robust_hand_computation():
    gt = torch.zeros(1, 1, 2, 1)
    pred = torch.tensor([[[[3.0], [4.0]]]])
    assert L.flow_robust_loss(pred, gt, eps=0.0).item() == pytest.approx(3.5)


def test_default_path_bit_identical_to_l2norm():
    from flyvis.task.objectives import l2norm

    torch.manual_seed(1)
    out = dict(flow=torch.randn(2, 5, 2, 721), depth=torch.randn(2, 5, 1, 721))
    b = dict(flow=torch.randn(2, 5, 2, 721), depth=torch.randn(2, 5, 1, 721))
    r = L.task_losses(out, b)
    for t in ("flow", "depth"):
        assert torch.equal(r[t], l2norm(out[t], b[t]) / 2)
    import train
    ref = train.l2norm_losses(out, b)
    assert all(torch.equal(r[t], ref[t]) for t in r)


def test_si_l2_combo_and_weights():
    torch.manual_seed(2)
    out = dict(flow=torch.randn(1, 2, 2, 9), depth=torch.randn(1, 2, 1, 9))
    b = dict(flow=torch.randn(1, 2, 2, 9), depth=torch.randn(1, 2, 1, 9))
    r = L.task_losses(out, b, depth_loss="si+l2", si_lambda=0.5, si_l2_weight=0.1)
    exp = (L.si_loss(out["depth"], b["depth"], 0.5)
           + 0.1 * ((out["depth"] - b["depth"]) ** 2).mean()) / 2
    assert torch.allclose(r["depth"], exp)
    r2 = L.task_losses(out, b, flow_loss="epe", flow_weight=10.0)
    assert torch.allclose(r2["flow"], L.flow_epe_loss(out["flow"], b["flow"]) * 5)


SPL = json.loads((ROOT / "splits.json").read_text())


@pytest.mark.parametrize("n", [2, 3, 4])
def test_cv_folds(n):
    pool = set(SPL["train"]) | set(SPL["val"])
    vals = []
    for k in range(n):
        sp = S.cv_split(SPL, k, n)
        tr, va = set(sp["train"]), set(sp["val"])
        assert tr | va == pool and not tr & va
        assert not (tr | va) & set(SPL["test"])
        assert not {S.family(s) for s in tr} & {S.family(s) for s in va}
        assert va
        vals.append(va)
        assert S.cv_split(SPL, k, n)["val"] == sp["val"]  # deterministic
    assert set().union(*vals) == pool
    assert sum(len(v) for v in vals) == len(pool)  # disjoint val folds


def test_m9_params_and_k():
    import hex_models as h
    import models
    m9 = models.build_model("m9")
    n = sum(p.numel() for p in m9.parameters())
    n5 = sum(p.numel() for p in models.build_model("m5").parameters())
    assert n == n5 and abs(n - 600_000) / 600_000 <= 0.03
    assert isinstance(m9, h.HexConvGRUNetK)
    m9s, m8 = models.build_model("m9s"), models.build_model("m8")
    assert sum(p.numel() for p in m9s.parameters()) == sum(p.numel() for p in m8.parameters()) == 15402
    x = torch.rand(1, 2, 1, 721)
    with torch.no_grad():
        assert not torch.allclose(m9(x, k=1)["flow"], m9(x, k=3)["flow"])
        m9.k_max = 2
        assert torch.equal(m9(x)["depth"], m9(x, k=2)["depth"])


def test_fast_gather_matches_default_and_m9m_params():
    import hex_models as h
    import models
    torch.manual_seed(0)
    a, b = h.make_large_k(hid_ch=8, n_layers=2, head_ch=8, fast_gather=False), \
        h.make_large_k(hid_ch=8, n_layers=2, head_ch=8, fast_gather=True)
    b.load_state_dict(a.state_dict())
    assert a.state_dict().keys() == b.state_dict().keys()
    x = torch.rand(1, 3, 1, 721)
    outs = []
    for m in (a, b):
        o = m(x, k=2)
        (o["flow"].square().mean() + o["depth"].square().mean()).backward()
        outs.append(o)
    assert torch.equal(outs[0]["flow"], outs[1]["flow"])
    for p, q in zip(a.parameters(), b.parameters()):
        assert torch.allclose(p.grad, q.grad, rtol=1e-4, atol=1e-7)
    assert sum(p.numel() for p in models.build_model("m9m").parameters()) == 274_323
