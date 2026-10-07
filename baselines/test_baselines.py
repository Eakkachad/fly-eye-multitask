"""Pytest suite for fly-eye multi-task hexagonal neural network baselines.

Tests:
1. Shape contract for make_small and make_large on CPU and CUDA.
2. Parameter count ranges for both models.
3. Gradient flow through all parameters and inputs.
4. Metric correctness on hand-computed toy examples and edge cases.
5. Neighbour table structure and border padding verification.
"""

import math
import numpy as np
import pytest
import torch

from flyvis.utils.hex_utils import get_hex_coords
from hex_models import (
    HexConv,
    HexConvGRUCell,
    HexConvGRUNet,
    count_parameters,
    get_neighbour_table,
    make_large,
    make_small,
    make_small_matched,
)
from metrics import angular_error_deg, binned, depth_absrel, depth_rmse, epe

AVAILABLE_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# -----------------------------------------------------------------------------
# 1. Parameter Counts
# -----------------------------------------------------------------------------
def test_param_counts():
    """Verify trainable parameter count targets."""
    small = make_small()
    large = make_large()

    p_small = count_parameters(small)
    p_large = count_parameters(large)

    assert 3000 <= p_small <= 6000, f"make_small() params {p_small} out of [3000, 6000]"
    assert (
        400000 <= p_large <= 1000000
    ), f"make_large() params {p_large} out of [400000, 1000000]"


def test_make_small_matched_param_counts():
    """Verify make_small_matched is within +-5% of M1 total params (15,387)."""
    matched = make_small_matched()
    p_matched = count_parameters(matched)
    target = 15387
    low = target * 0.95
    high = target * 1.05
    assert low <= p_matched <= high, (
        f"make_small_matched() params {p_matched} out of [{low:.1f}, {high:.1f}] (+-5% of {target})"
    )


# -----------------------------------------------------------------------------
# 2. Neighbour Table Correctness & HexConv Padding
# -----------------------------------------------------------------------------
def test_neighbour_table_correctness():
    """Verify that each interior hexal has 6 distinct neighbours and border padding works."""
    table = get_neighbour_table(extent=15)
    assert table.shape == (721, 7)

    # Column 0 is always self (index i)
    assert torch.equal(table[:, 0], torch.arange(721))

    u_arr, v_arr = get_hex_coords(15)
    # Axial distance from origin: max(|u|, |v|, |u + v|)
    dist = np.maximum(np.maximum(np.abs(u_arr), np.abs(v_arr)), np.abs(u_arr + v_arr))
    interior_indices = np.where(dist < 15)[0]
    border_indices = np.where(dist == 15)[0]

    assert len(interior_indices) == 631
    assert len(border_indices) == 90

    # Interior hexals: all 6 neighbours must be in 0..720, distinct, not self
    for i in interior_indices:
        nbrs = table[i, 1:].tolist()
        assert 721 not in nbrs, f"Interior hexal {i} has border padding"
        assert len(set(nbrs)) == 6, f"Interior hexal {i} does not have 6 distinct neighbours"
        assert i not in nbrs, f"Interior hexal {i} has self among neighbours"

    # Border hexals: must have at least one neighbour out of bounds (pointing to 721)
    for i in border_indices:
        nbrs = table[i, 1:].tolist()
        assert 721 in nbrs, f"Border hexal {i} lacks border padding"


def test_hexconv_border_zero_padding():
    """Verify that HexConv properly pads missing border neighbours with zeros."""
    conv = HexConv(in_ch=1, out_ch=1, bias=False, extent=15)
    conv.weight.data.fill_(1.0)

    # Input of all ones: interior hexals should sum 7 * 1 = 7.
    # Border hexals should sum to (7 - n_missing) * 1 < 7.
    x = torch.ones(1, 1, 721)
    out = conv(x).squeeze()

    u_arr, v_arr = get_hex_coords(15)
    dist = np.maximum(np.maximum(np.abs(u_arr), np.abs(v_arr)), np.abs(u_arr + v_arr))
    interior_indices = np.where(dist < 15)[0]
    border_indices = np.where(dist == 15)[0]

    assert torch.allclose(out[interior_indices], torch.tensor(7.0))
    assert (out[border_indices] < 7.0).all()


# -----------------------------------------------------------------------------
# 3. Model Output Shapes & Contract
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [make_small, make_small_matched, make_large])
@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_forward_shapes(factory, device):
    """Verify exact output shapes for flow (B, T, 2, N) and depth (B, T, 1, N)."""
    model = factory().to(device)
    model.eval()

    B, T, N = 2, 19, 721
    lum = torch.randn(B, T, 1, N, device=device)

    with torch.no_grad():
        out = model(lum)

    assert isinstance(out, dict), "Output must be a dictionary"
    assert "flow" in out and "depth" in out, "Dict must have 'flow' and 'depth' keys"
    assert out["flow"].shape == (B, T, 2, N)
    assert out["depth"].shape == (B, T, 1, N)


# -----------------------------------------------------------------------------
# 4. Gradient Flow
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("factory", [make_small, make_small_matched, make_large])
@pytest.mark.parametrize("device", AVAILABLE_DEVICES)
def test_gradient_flow(factory, device):
    """Verify that gradients propagate to inputs and all trainable parameters."""
    model = factory().to(device)
    model.train()

    B, T, N = 2, 5, 721  # Shorter T for faster grad test
    lum = torch.randn(B, T, 1, N, device=device, requires_grad=True)

    out = model(lum)
    loss = out["flow"].sum() + out["depth"].sum()
    loss.backward()

    # Input grad
    assert lum.grad is not None, "Input lum did not receive gradient"
    assert not torch.isnan(lum.grad).any(), "Input grad contains NaN"
    assert lum.grad.norm().item() > 0.0, "Input grad is all zero"

    # All trainable parameters receive gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} grad has NaN"


# -----------------------------------------------------------------------------
# 5. Metrics on Hand-Computed Toy Examples
# -----------------------------------------------------------------------------
def test_metric_epe():
    """Verify End-Point Error on toy examples."""
    # (0, 0) vs (3, 4) -> L2 norm = sqrt(9 + 16) = 5.0
    pred = torch.tensor([0.0, 0.0])
    gt = torch.tensor([3.0, 4.0])
    assert math.isclose(epe(pred, gt).item(), 5.0, rel_tol=1e-5)

    # Identical flow -> 0.0
    pred_4d = torch.randn(2, 5, 2, 721)
    assert math.isclose(epe(pred_4d, pred_4d).item(), 0.0, abs_tol=1e-6)

    # With mask: two hexals, one has error 5, one error 0; mask out error 0
    p = torch.tensor([[[[0.0, 0.0], [0.0, 0.0]]]])  # (1, 1, 2, 2)
    g = torch.tensor([[[[3.0, 0.0], [4.0, 0.0]]]])  # (1, 1, 2, 2)
    mask = torch.tensor([[[[True, False]]]])  # (1, 1, 1, 2)
    assert math.isclose(epe(p, g, mask=mask).item(), 5.0, rel_tol=1e-5)


def test_metric_angular_error_deg():
    """Verify angular error in degrees on toy examples."""
    # Identical vectors -> 0.0 deg
    p = torch.tensor([3.0, 4.0])
    assert math.isclose(angular_error_deg(p, p).item(), 0.0, abs_tol=1e-6)

    # Identical in 4D
    p_4d = torch.randn(2, 5, 2, 721)
    assert math.isclose(angular_error_deg(p_4d, p_4d).item(), 0.0, abs_tol=1e-6)

    # (0, 0, 1) vs (1, 0, 1): dot=1, norm1=1, norm2=sqrt(2) -> cos=1/sqrt(2) -> 45 deg
    p_zero = torch.tensor([0.0, 0.0])
    p_one = torch.tensor([1.0, 0.0])
    assert math.isclose(angular_error_deg(p_zero, p_one).item(), 45.0, rel_tol=1e-5)

    # With mask
    p_batch = torch.tensor([[[[0.0, 3.0], [0.0, 4.0]]]])  # (1, 1, 2, 2)
    g_batch = torch.tensor([[[[1.0, 3.0], [0.0, 4.0]]]])  # hexal 0 is 45 deg, hexal 1 is 0 deg
    mask = torch.tensor([[[[True, False]]]])
    assert math.isclose(angular_error_deg(p_batch, g_batch, mask=mask).item(), 45.0, rel_tol=1e-5)


def test_metric_depth_rmse():
    """Verify depth RMSE on toy examples."""
    # pred=3, gt=0 -> RMSE = 3.0
    p = torch.tensor([[[[3.0]]]])
    g = torch.tensor([[[[0.0]]]])
    assert math.isclose(depth_rmse(p, g).item(), 3.0, rel_tol=1e-5)

    # pred=[1, 2], gt=[1, 6] -> errors=[0, 4] -> RMSE = sqrt((0 + 16)/2) = sqrt(8)
    p2 = torch.tensor([[[[1.0, 2.0]]]])
    g2 = torch.tensor([[[[1.0, 6.0]]]])
    expected = math.sqrt(8.0)
    assert math.isclose(depth_rmse(p2, g2).item(), expected, rel_tol=1e-5)

    # With mask: select only the second pixel
    mask = torch.tensor([[[[False, True]]]])
    assert math.isclose(depth_rmse(p2, g2, mask=mask).item(), 4.0, rel_tol=1e-5)


def test_metric_depth_absrel():
    """Verify depth AbsRel on toy examples."""
    # pred=1.5, gt=1.0 -> |1.5 - 1.0| / (1.0 + 1e-6) ≈ 0.5
    p = torch.tensor([[[[1.5]]]])
    g = torch.tensor([[[[1.0]]]])
    assert math.isclose(depth_absrel(p, g).item(), 0.5, rel_tol=1e-4)

    # With mask
    p2 = torch.tensor([[[[1.5, 9.0]]]])
    g2 = torch.tensor([[[[1.0, 1.0]]]])
    mask = torch.tensor([[[[True, False]]]])
    assert math.isclose(depth_absrel(p2, g2, mask=mask).item(), 0.5, rel_tol=1e-4)


def test_metric_binned():
    """Verify binned metric analysis across bin ranges."""
    values = torch.tensor([10.0, 20.0, 30.0, 40.0])
    bin_values = torch.tensor([0.2, 0.8, 1.5, 2.5])
    edges = [0.0, 1.0, 2.0, 3.0]

    # Bins: [0, 1) has [10, 20] -> mean 15
    #       [1, 2) has [30] -> mean 30
    #       [2, 3] has [40] -> mean 40
    res = binned(values, bin_values, edges)
    expected = torch.tensor([15.0, 30.0, 40.0])
    assert torch.allclose(res, expected)

    # Empty bin produces NaN
    edges_empty = [0.0, 0.1, 1.0]
    res_empty = binned(values, bin_values, edges_empty)
    assert torch.isnan(res_empty[0])
    assert math.isclose(res_empty[1].item(), 15.0, rel_tol=1e-4)


# ----------------------------- M8 (HexConvGRU-K) -----------------------------
def test_m8_k1_equals_m4_and_params():
    from hex_models import make_small_matched_k
    torch.manual_seed(0)
    m4 = make_small_matched().eval()
    m8 = make_small_matched_k().eval()
    assert count_parameters(m8) == count_parameters(m4)
    m8.load_state_dict(m4.state_dict())
    x = torch.randn(2, 5, 1, m4.encoder[0].neighbour_table.shape[0])
    with torch.no_grad():
        a, b = m4(x), m8(x, k=1)
    for key in ("flow", "depth"):
        assert torch.equal(a[key], b[key])


def test_m8_k_shapes_grad_anytime():
    from hex_models import make_small_matched_k
    m = make_small_matched_k()
    n = m.encoder[0].neighbour_table.shape[0]
    x = torch.randn(2, 4, 1, n)
    outs = {}
    for k in (1, 2, 3, 4):
        o = m(x, k=k)
        assert o["flow"].shape == (2, 4, 2, n) and o["depth"].shape == (2, 4, 1, n)
        outs[k] = o["flow"].detach()
    assert not torch.allclose(outs[1], outs[4])
    m.zero_grad()
    (m(x, k=4)["flow"].sum() + m(x, k=4)["depth"].sum()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert m(x).get("flow").shape == (2, 4, 2, n)  # default k = k_max
