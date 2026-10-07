"""GPU benchmark: unpatched vs fastfly-patched M1 train step (batch 4, 19 frames).
Run from course/:  ../.venv/bin/python fastfly/bench_fastfly.py [--batch 4] [--frames 19] [--iters 10]
Writes fastfly/bench_result.json. Full M1 forward+backward incl. decoders; the loss is a
simple sum of the flow+depth heads weighted by fixed random tensors (same for both runs).
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import torch

from fastfly import patch_network, unpatch_network


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--model", default="m1")
    ap.add_argument("--out", default=str(HERE / "bench_result.json"))
    a = ap.parse_args()
    a_ = a
    dev = "cuda"
    from models import build_model
    m = build_model(a.model, seed=0).to(dev).train()
    torch.manual_seed(0)
    lum = torch.rand(a.batch, a.frames, 1, 721, device=dev)
    tgt = {k: torch.randn(a.batch, a.frames, c, 721, device=dev)
           for k, c in (("flow", 2), ("depth", 1))}
    params = [p for p in m.parameters() if p.requires_grad]
    m.refresh_steady_state(a.batch)  # fixed grey-screen state for both runs
    refresh = m.refresh_steady_state
    m.refresh_steady_state = lambda b: None  # keep identical initial state (stored in m._ss)

    names = [n for n, p in m.named_parameters() if p.requires_grad]

    def step():
        torch.manual_seed(1)  # same decoder dropout mask each call
        out = m(lum, return_activity=True)
        loss = sum(((out[k] - tgt[k]) ** 2).mean() for k in tgt)
        grads = torch.autograd.grad(loss, params)
        return out, loss, grads

    def timed():
        for _ in range(a.warmup):
            step()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(a.iters):
            step()
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / a.iters

    res = dict(batch=a.batch, frames=a.frames, model=a.model, device=torch.cuda.get_device_name())
    o0, l0, g0 = step()
    _, _, g0b = step()  # unpatched twice: nondeterminism (atomics in index_add/scatter) floor
    res["s_per_iter_unpatched"] = timed()
    patch_network(m.network)
    o1, l1, g1 = step()
    res["s_per_iter_patched"] = timed()
    unpatch_network(m.network)
    res["speedup"] = res["s_per_iter_unpatched"] / res["s_per_iter_patched"]
    res["max_abs_err_activity"] = (o0["activity"] - o1["activity"]).abs().max().item()
    res["max_abs_err_out"] = max((o0[k] - o1[k]).abs().max().item() for k in tgt)
    res["max_rel_err_out"] = max(((o0[k] - o1[k]).abs().max() / o0[k].abs().max()).item() for k in tgt)
    res["loss_abs_err"] = abs(l0.item() - l1.item())
    gmax = max(x.abs().max().item() for x in g0)
    tab, worst = {}, 0.0
    for n, x, y, z in zip(names, g0, g1, g0b):
        mx = x.abs().max().item()
        e, noise = (x - y).abs().max().item(), (x - z).abs().max().item()
        # near-zero-gradient tensors (e.g. conv bias before BatchNorm) are judged against
        # 1e-3 * global max|g|, otherwise rel err is pure float noise
        rel = e / max(mx, 1e-3 * gmax)
        worst = max(worst, rel)
        tab[n] = dict(max_abs_grad=mx, abs_err=e, unpatched_vs_unpatched_noise=noise, rel_err=rel)
    res["grad_table"] = tab
    res["max_rel_err_grad_floored"] = worst
    res["max_abs_err_grad"] = max(v["abs_err"] for v in tab.values())
    res["gate_2x_and_match"] = bool(res["speedup"] >= 2 and worst < 1e-3)
    res["peak_mem_MB"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps({k: v for k, v in res.items() if k != "grad_table"}, indent=1))
    Path(a_.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
