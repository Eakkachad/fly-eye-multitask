"""Fused Triton rollout for flyvis ``Network`` (PPNeuronIGRSynapses + ReLU).

Per Euler step flyvis computes (all per batch element b, node n, dt = frame dt):
    S[n]   = sum_{e: tgt(e)=n} w[e] * relu(v[src(e)])        (gather, mul, scatter_add)
    v'[n]  = v[n] + dt * 1/max(tau[n], dt) * (-v[n] + bias[n] + S[n] + x[n])
with w = sign * syn_count * syn_strength (expanded to edges ONCE per forward).

Here the whole T-step rollout is one autograd.Function. Forward: one kernel per step
(CSR by target; gather+relu+mul+segment-sum+Euler fused, no atomics, no (B,E) tensor).
Backward (reverse over steps): kernel A (by target) computes dS, d bias, d tau, d x,
and accumulates the per-edge weight gradient in place over steps and batch; kernel B
(by source, CSR transpose) gathers dS[tgt] * w and applies the relu mask -> d v.
"""
from __future__ import annotations

import types

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- kernels


@triton.jit
def _fwd_kernel(v_ptr, x_ptr, bias_ptr, tau_ptr, w_ptr, src_ptr, rowptr_ptr,
                vout_ptr, inner_ptr, B, N, dt,
                BP: tl.constexpr, BLOCK_E: tl.constexpr, ACC: tl.constexpr):
    # one program per target node; edges loaded once and reused for all batch rows
    t = tl.program_id(0)
    start = tl.load(rowptr_ptr + t)
    end = tl.load(rowptr_ptr + t + 1)
    tau = tl.load(tau_ptr + t)
    bias = tl.load(bias_ptr + t)
    inv = 1.0 / tl.maximum(tau, dt)
    bs = tl.arange(0, BP)
    bm = bs < B
    acc = tl.zeros([BP, BLOCK_E], ACC)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + tl.arange(0, BLOCK_E)
        m = offs < end
        w = tl.load(w_ptr + offs, mask=m, other=0.0)
        s = tl.load(src_ptr + offs, mask=m, other=0)
        vs = tl.load(v_ptr + bs[:, None] * N + s[None, :],
                     mask=bm[:, None] & m[None, :], other=0.0)
        acc += w[None, :] * tl.maximum(vs, 0.0)
    S = tl.sum(acc, 1)
    v = tl.load(v_ptr + bs * N + t, mask=bm, other=0.0)
    x = tl.load(x_ptr + bs * N + t, mask=bm, other=0.0)
    inner = -v + bias + S + x
    tl.store(inner_ptr + bs * N + t, inner, mask=bm)
    tl.store(vout_ptr + bs * N + t, v + inv * inner * dt, mask=bm)


@triton.jit
def _bwd_target_kernel(G_ptr, gout_ptr, v_ptr, inner_ptr, tau_ptr, w_ptr, src_ptr,
                       rowptr_ptr, dS_ptr, gw_ptr, gbias_ptr, gtau_ptr, gx_ptr,
                       B, N, dt,
                       WANT_GX: tl.constexpr, WANT_GW: tl.constexpr,
                       BP: tl.constexpr, BLOCK_E: tl.constexpr, ACC: tl.constexpr):
    t = tl.program_id(0)
    start = tl.load(rowptr_ptr + t)
    end = tl.load(rowptr_ptr + t + 1)
    tau = tl.load(tau_ptr + t)
    inv = 1.0 / tl.maximum(tau, dt)
    # torch.max(tau, dt) routes the gradient to tau if tau > dt, half if equal
    c = tl.where(tau > dt, 1.0, tl.where(tau == dt, 0.5, 0.0))
    bs = tl.arange(0, BP)
    bm = bs < B
    g = tl.load(G_ptr + bs * N + t, mask=bm, other=0.0) \
        + tl.load(gout_ptr + bs * N + t, mask=bm, other=0.0)
    dS = g * dt * inv
    inner = tl.load(inner_ptr + bs * N + t, mask=bm, other=0.0)
    tl.store(dS_ptr + bs * N + t, dS, mask=bm)
    if WANT_GX:
        tl.store(gx_ptr + bs * N + t, dS, mask=bm)
    tl.store(gbias_ptr + t, tl.load(gbias_ptr + t) + tl.sum(dS, 0))
    tl.store(gtau_ptr + t, tl.load(gtau_ptr + t)
             + tl.sum(g * dt * inner * (-inv * inv) * c, 0))
    if WANT_GW:
        for e0 in range(start, end, BLOCK_E):
            offs = e0 + tl.arange(0, BLOCK_E)
            m = offs < end
            s = tl.load(src_ptr + offs, mask=m, other=0)
            vs = tl.load(v_ptr + bs[:, None] * N + s[None, :],
                         mask=bm[:, None] & m[None, :], other=0.0)
            acc = tl.sum(dS[:, None] * tl.maximum(vs, 0.0), 0)
            old = tl.load(gw_ptr + offs, mask=m, other=0.0)
            tl.store(gw_ptr + offs, old + acc, mask=m)


@triton.jit
def _bwd_source_kernel(G_ptr, Gnew_ptr, gout_ptr, v_ptr, dS_ptr, tau_ptr, w_ptr,
                       tgt_ptr, pos_ptr, rowptr_ptr, B, N, dt,
                       BP: tl.constexpr, BLOCK_E: tl.constexpr, ACC: tl.constexpr):
    s = tl.program_id(0)
    start = tl.load(rowptr_ptr + s)
    end = tl.load(rowptr_ptr + s + 1)
    tau = tl.load(tau_ptr + s)
    inv = 1.0 / tl.maximum(tau, dt)
    bs = tl.arange(0, BP)
    bm = bs < B
    acc = tl.zeros([BP, BLOCK_E], ACC)
    for e0 in range(start, end, BLOCK_E):
        offs = e0 + tl.arange(0, BLOCK_E)
        m = offs < end
        p = tl.load(pos_ptr + offs, mask=m, other=0)
        tg = tl.load(tgt_ptr + offs, mask=m, other=0)
        w = tl.load(w_ptr + p, mask=m, other=0.0)
        d = tl.load(dS_ptr + bs[:, None] * N + tg[None, :],
                    mask=bm[:, None] & m[None, :], other=0.0)
        acc += w[None, :] * d
    tot = tl.sum(acc, 1)
    v = tl.load(v_ptr + bs * N + s, mask=bm, other=0.0)
    g = tl.load(G_ptr + bs * N + s, mask=bm, other=0.0) \
        + tl.load(gout_ptr + bs * N + s, mask=bm, other=0.0)
    gv = g * (1.0 - dt * inv) + tl.where(v > 0.0, tot, 0.0)
    tl.store(Gnew_ptr + bs * N + s, gv, mask=bm)


SKIP_EDGE_GRAD = [False]  # set via skip_edge_grads(); backward then returns a zero w grad
CFG = dict(block_e=64, warps_fwd=1, warps_bt=1, warps_bs=1)  # tuning knobs


# --------------------------------------------------------------------------- plan


class Plan:
    """CSR (by target, and transpose by source) of the edge list. Built once."""

    def __init__(self, src, tgt, n_nodes):
        dev = src.device
        E = src.numel()
        perm_t = torch.argsort(tgt.long(), stable=True)          # target-sorted order
        self.perm_t = perm_t
        self.src_t = src[perm_t].to(torch.int32).contiguous()
        cnt_t = torch.bincount(tgt.long(), minlength=n_nodes)
        self.rowptr_t = torch.cat([cnt_t.new_zeros(1), cnt_t.cumsum(0)]).to(torch.int32)
        perm_s = torch.argsort(src.long(), stable=True)          # source-sorted order
        inv_t = torch.empty_like(perm_t)
        inv_t[perm_t] = torch.arange(E, device=dev)
        self.pos_s = inv_t[perm_s].to(torch.int32).contiguous()  # -> target-sorted pos
        self.tgt_s = tgt[perm_s].to(torch.int32).contiguous()
        cnt_s = torch.bincount(src.long(), minlength=n_nodes)
        self.rowptr_s = torch.cat([cnt_s.new_zeros(1), cnt_s.cumsum(0)]).to(torch.int32)
        self.n_nodes, self.n_edges = n_nodes, E


def build_plan(source_indices, target_indices, n_nodes) -> Plan:
    return Plan(source_indices, target_indices, n_nodes)


def _acc(dtype):
    return {torch.float32: tl.float32, torch.float64: tl.float64}[dtype]


# --------------------------------------------------------------------------- autograd


class FusedFlyvisRollout(torch.autograd.Function):
    """(v0 (B,N), x (T,B,N), bias (N), tau (N), w_t (E, target-sorted)) -> acts (T,B,N)."""

    @staticmethod
    def forward(ctx, v0, x, bias, tau, w_t, plan, dt, block_e=None):
        block_e = block_e or CFG['block_e']
        T, B, N = x.shape
        dev, dt_ = v0.device, v0.dtype
        v0, x, bias, tau, w_t = [a.detach().contiguous() for a in (v0, x, bias, tau, w_t)]
        traj = torch.empty((T + 1, B, N), device=dev, dtype=dt_)
        traj[0] = v0
        inner = torch.empty((T, B, N), device=dev, dtype=dt_)
        for i in range(T):
            _fwd_kernel[(N,)](traj[i], x[i], bias, tau, w_t, plan.src_t, plan.rowptr_t,
                              traj[i + 1], inner[i], B, N, float(dt),
                              BP=triton.next_power_of_2(B), BLOCK_E=block_e, ACC=_acc(dt_), num_warps=CFG['warps_fwd'])
        ctx.save_for_backward(traj, inner, tau, w_t)
        ctx.plan, ctx.dt, ctx.block_e = plan, float(dt), block_e
        return traj[1:]

    @staticmethod
    def backward(ctx, gout):
        traj, inner, tau, w_t = ctx.saved_tensors
        plan, dt, be = ctx.plan, ctx.dt, ctx.block_e
        T, B, N = inner.shape
        dtype, dev = traj.dtype, traj.device
        gout = gout.contiguous()
        G = torch.zeros((B, N), device=dev, dtype=dtype)
        Gn = torch.empty_like(G)
        dS = torch.empty_like(G)
        gw = torch.zeros_like(w_t)
        gbias = torch.zeros(N, device=dev, dtype=dtype)
        gtau = torch.zeros(N, device=dev, dtype=dtype)
        want_gx = ctx.needs_input_grad[1]
        want_gw = not SKIP_EDGE_GRAD[0]
        gx = torch.empty((T, B, N), device=dev, dtype=dtype) if want_gx else dS
        for i in reversed(range(T)):
            _bwd_target_kernel[(N,)](G, gout[i], traj[i], inner[i], tau, w_t, plan.src_t,
                                     plan.rowptr_t, dS, gw, gbias, gtau,
                                     gx[i] if want_gx else dS, B, N, dt,
                                     WANT_GX=want_gx, WANT_GW=want_gw,
                                     BP=triton.next_power_of_2(B), BLOCK_E=be, ACC=_acc(dtype),
                                     num_warps=CFG['warps_bt'])
            _bwd_source_kernel[(N,)](G, Gn, gout[i], traj[i], dS, tau, w_t, plan.tgt_s,
                                     plan.pos_s, plan.rowptr_s, B, N, dt,
                                     BP=triton.next_power_of_2(B), BLOCK_E=be, ACC=_acc(dtype),
                                     num_warps=CFG['warps_bs'])
            G, Gn = Gn, G
        return (G if ctx.needs_input_grad[0] else None, gx if want_gx else None,
                gbias, gtau, gw, None, None, None)


# --------------------------------------------------------------------------- reference


def reference_rollout(v0, x, bias, tau, w, src, tgt, dt):
    """Exact flyvis numerics in pure torch. v0 (B,N), x (T,B,N), w (E,), returns (T,B,N)."""
    N = v0.shape[-1]
    dt_t = torch.tensor(dt).float()
    outs, v = [], v0
    for i in range(x.shape[0]):
        vs = v.index_select(-1, src)
        cur = w * torch.relu(vs)
        S = torch.zeros_like(v).scatter_add_(-1, tgt.expand(*cur.shape), cur)
        vel = 1 / torch.max(tau, dt_t) * (-v + bias + S + x[i])
        v = v + vel * dt
        outs.append(v)
    return torch.stack(outs, 0)


import contextlib


@contextlib.contextmanager
def skip_edge_grads():
    """Use around autograd.grad(..., inputs=[bias]) (e.g. the activity-penalty pass):
    the per-edge weight gradient is then NOT computed (returned as zeros)."""
    SKIP_EDGE_GRAD[0] = True
    try:
        yield
    finally:
        SKIP_EDGE_GRAD[0] = False


# --------------------------------------------------------------------------- patching


class ExpandStrength(torch.autograd.Function):
    """w_t[e] = sc_t[e] * strength[idx_t[e]]   (edges in target-sorted order).

    Backward is a segment-sum over type-pair sorted edges (torch's index_select backward
    does a sort + index_put over 1.5M edges into 604 bins, ~2.5 ms each on a 4060)."""

    @staticmethod
    def forward(ctx, strength, idx_t, sc_t, perm2, lengths):
        ctx.save_for_backward(sc_t, perm2, lengths)
        ctx.k = strength.numel()
        return sc_t * strength.index_select(0, idx_t)

    @staticmethod
    def backward(ctx, gw):
        sc_t, perm2, lengths = ctx.saved_tensors
        g = (gw * sc_t).index_select(0, perm2)
        out = torch.segment_reduce(g, "sum", lengths=lengths)
        if out.numel() < ctx.k:  # types with no edges
            out = torch.cat([out, out.new_zeros(ctx.k - out.numel())])
        return out, None, None, None, None


def _plan_for(net):
    plan = getattr(net, "_fastfly_plan", None)
    dev = net._source_indices.device
    if plan is None or plan.src_t.device != dev:
        plan = Plan(net._source_indices, net._target_indices, net.n_nodes)
        net.__dict__["_fastfly_plan"] = plan
    return plan


def _fast_ok(net, x, state, as_states):
    from flyvis.network.dynamics import PPNeuronIGRSynapses
    return (not as_states and state is not None and not net._state_hooks
            and type(net.dynamics) is PPNeuronIGRSynapses
            and isinstance(net.dynamics.activation, torch.nn.ReLU)
            and x.dtype in (torch.float32, torch.float64))


def _weights_fast(net, plan):
    """w in target-sorted order, with an efficient backward onto the type-pair strengths.
    Returns None if the edge-parameter structure is not the standard sign*count*strength."""
    ep = net.edge_params
    if set(ep.keys()) != {"sign", "syn_count", "syn_strength"}:
        return None
    if ep["sign"].raw_values.requires_grad or ep["syn_count"].raw_values.requires_grad:
        return None
    if not hasattr(ep["syn_strength"], "indices") or ep["syn_strength"].indices.dim() != 1:
        return None
    strength = ep["syn_strength"].semantic_values
    cache = net.__dict__.get("_fastfly_wcache")
    dev = strength.device
    idx = ep["syn_strength"].indices
    if cache is None or cache["idx"] is not idx or cache["dev"] != dev:
        idx_t = idx.to(dev)[plan.perm_t].contiguous()
        perm2 = torch.argsort(idx_t, stable=True)
        lengths = torch.bincount(idx_t, minlength=strength.numel())
        cache = dict(idx=idx, dev=dev, idx_t=idx_t, perm2=perm2, lengths=lengths)
        net.__dict__["_fastfly_wcache"] = cache
    with torch.no_grad():  # sign and syn_count are fixed
        sc = ep["sign"].semantic_values[ep["sign"].indices] * \
            ep["syn_count"].semantic_values[ep["syn_count"].indices]
        sc_t = sc[plan.perm_t].contiguous()
    return ExpandStrength.apply(strength, cache["idx_t"], sc_t, cache["perm2"], cache["lengths"])


def _fast_forward(net, x, dt, state=None, as_states=False):
    orig = net.__dict__["_fastfly_orig_forward"]
    if not _fast_ok(net, x, state, as_states):
        return orig(x, dt, state=state, as_states=as_states)
    net.clamp()
    plan = _plan_for(net)
    w_t = _weights_fast(net, plan)
    if w_t is not None:  # skip flyvis' per-forward (E,) weight expansion (unused here)
        npar = net.node_params
        bias = npar["bias"].semantic_values.index_select(-1, npar["bias"].readers["nodes"])
        tau = npar["time_const"].semantic_values.index_select(
            -1, npar["time_const"].readers["nodes"])
    else:
        params = net._param_api()
        bias, tau = params.nodes.bias, params.nodes.time_const
        w_t = params.edges.weight.index_select(0, plan.perm_t.to(x.device))
    v0 = state.nodes.activity
    out = FusedFlyvisRollout.apply(v0, x.transpose(0, 1), bias, tau, w_t, plan, dt)
    return out.transpose(0, 1).contiguous()


def patch_network(net):
    """Swap net.forward for the fused rollout (falls back for as_states / hooks / no state)."""
    if is_patched(net):
        return net
    net.__dict__["_fastfly_orig_forward"] = net.forward
    net.forward = types.MethodType(_fast_forward, net)
    return net


def unpatch_network(net):
    if is_patched(net):
        del net.__dict__["_fastfly_orig_forward"]
        del net.__dict__["forward"]
        net.__dict__.pop("_fastfly_plan", None)
        net.__dict__.pop("_fastfly_wcache", None)
    return net


def is_patched(net):
    return "_fastfly_orig_forward" in net.__dict__
