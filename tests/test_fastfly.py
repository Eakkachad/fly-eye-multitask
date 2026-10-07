"""CPU tests for fastfly (Triton interpreter). Run:
  TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest course/tests/test_fastfly.py -s
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from fastfly import (FusedFlyvisRollout, build_plan, patch_network, reference_rollout,
                     unpatch_network)


def _tiny_graph(n=7, e=30, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, n, (e,), generator=g)
    tgt = torch.randint(0, n - 1, (e,), generator=g)  # last node has no inputs
    w = (torch.rand(e, generator=g, dtype=dtype) - 0.4) * 0.5
    return src, tgt, w


def test_function_gradcheck_float64():
    n, B, T, dt = 7, 2, 3, 0.02
    src, tgt, w = _tiny_graph(n)
    plan = build_plan(src, tgt, n)
    g = torch.Generator().manual_seed(1)
    v0 = torch.randn(B, n, generator=g, dtype=torch.float64).requires_grad_()
    x = torch.randn(T, B, n, generator=g, dtype=torch.float64).requires_grad_()
    bias = torch.rand(n, generator=g, dtype=torch.float64).requires_grad_()
    tau = (0.03 + 0.05 * torch.rand(n, generator=g, dtype=torch.float64)).requires_grad_()
    wt = w[plan.perm_t].clone().requires_grad_()
    # dt=0.5 makes the Euler step strongly coupled so grads are non-trivial
    for step in (dt, 0.5):
        f = lambda *a: FusedFlyvisRollout.apply(*a, plan, step)
        assert torch.autograd.gradcheck(f, (v0, x, bias, tau, wt), eps=1e-6, atol=1e-6,
                                        rtol=1e-5)
        ref = reference_rollout(v0, x, bias, tau, wt[plan.perm_t.argsort()], src, tgt, step)
        torch.testing.assert_close(f(v0, x, bias, tau, wt), ref, atol=1e-12, rtol=1e-10)


def _small_net(extent=3, seed=0):
    from datamate import Namespace
    from flyvis import Network
    import flyvis
    node_config = Namespace(
        bias=Namespace(type="RestingPotential", groupby=["type"], initial_dist="Normal",
                       mode="sample", requires_grad=True, mean=0.5, std=0.05,
                       penalize=Namespace(activity=True), seed=seed),
        time_const=Namespace(type="TimeConstant", groupby=["type"], initial_dist="Value",
                             value=0.05, requires_grad=True))
    conn = Namespace(type="ConnectomeFromAvgFilters", file=str(flyvis.connectome_file),
                     extent=extent, n_syn_fill=1)
    return Network(connectome=conn, node_config=node_config)


def test_patched_network_matches_flyvis():
    torch.manual_seed(0)
    net = _small_net()
    # perturb the trainable params so grads are not degenerate
    with torch.no_grad():
        for p in net.parameters():
            if p.requires_grad:
                p.mul_(1 + 0.1 * torch.randn_like(p))
    params = [p for p in net.parameters() if p.requires_grad]
    B, T, dt = 1, 4, 0.02
    ss = net.steady_state(t_pre=0.1, dt=dt, batch_size=B, value=0.5)
    net.stimulus.zero(B, T)
    net.stimulus.add_input(torch.rand(B, T, 1, _n_in(net)))
    stim = net.stimulus().detach().clone()
    wgt = torch.randn(B, T, net.n_nodes)

    def run():
        out = net(stim, dt, state=ss)
        loss = (out * wgt).sum()
        return out, torch.autograd.grad(loss, params)

    o_ref, g_ref = run()
    patch_network(net)
    try:
        o_fast, g_fast = run()
        assert net.forward.__func__.__name__ == "_fast_forward"
    finally:
        unpatch_network(net)
    assert "forward" not in net.__dict__
    err_o = (o_fast - o_ref).abs().max().item()
    rel_o = err_o / o_ref.abs().max().item()
    print(f"\nout max abs err {err_o:.3e} (rel {rel_o:.3e}); n_params={len(params)}")
    assert rel_o < 1e-5
    worst = 0.0
    for p, a, b in zip(params, g_ref, g_fast):
        scale = a.abs().max().clamp_min(1e-12)
        r = ((a - b).abs().max() / scale).item()
        worst = max(worst, r)
        assert r < 1e-4, (tuple(p.shape), r)
    print(f"grad worst max-abs err relative to per-param max |grad|: {worst:.3e}")


def _n_in(net):
    return net.stimulus.n_input_elements if hasattr(net.stimulus, "n_input_elements") else 721
