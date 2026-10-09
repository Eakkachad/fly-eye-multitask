# R2 — training-speed benchmark (RTX 4060 8 GB, 2026-10-08)
Tool: `bench/profile_train.py` (same step as `train.py`), 50 timed iters after 10 warm-up, batch 4 unless noted.
Raw JSON: `bench/results/` (gitignored); table: `bench/results/summary_r2.md`. Torch 2.14.1+cu130; Triton via zig `cc` shim.

| config | s/iter | speed-up | fwd / bwd / "opt" s | VRAM GB | note |
|---|---|---|---|---|---|
| M1 fp32 (baseline, with activity penalty) | 0.202 | 1.00× | 0.057 / 0.075 / 0.062 | 1.53 | "opt" bucket includes the penalty's 2nd backward |
| M1 tf32 | 0.199 | 1.02× | | 1.53 | |
| M1 bf16 | 0.194 | 1.04× | | 1.47 | |
| M1 bf16, batch 8 / 16 | 0.370 / 0.695 | samples/s 21.6 / 23.0 (vs 20.7) | | 2.7 / 5.2 | ~linear in batch → bandwidth-bound, not launch-bound |
| M1 torch.compile (default / reduce-overhead) | fail | – | | | in-place param update before penalty backward / CUDA-graph output overwritten |
| **M1 no activity penalty** | **0.135** | **1.49×** | 0.053 / 0.076 / 0.000 | 1.32 | changes the pre-registered method → needs amendment |
| M1 no penalty + compile | 0.134 | 1.51× | 0.034 / 0.093 / 0.000 | 0.38 | compile gives ~nothing on M1 |
| M4 fp32 baseline | 0.138 | 1.00× | 0.027 / 0.104 / 0 | 0.60 | |
| M4 bf16 / tf32 | 0.134 / 0.136 | 1.03× / 1.01× | | | |
| **M4 torch.compile reduce-overhead** | **0.027** | **5.0×** | 0.008 / 0.011 / 0 | 0.04 | data loading becomes 24–29 % |
| M4 compile-ro batch 8 / 16 | 0.051 / 0.105 | samples/s ≈157 flat | | | **data-loader-bound** |

## Where M1 time goes (torch.profiler, no penalty, 10 iters; self CUDA time)
elementwise mul 29 % · gather 11 % · threshold_backward 10 % · index_put (indexing backward) 9 % · index_add 8 % · scatter_add 8 % ·
clamp_min 6 % · decoder conv 6 % + conv backward 3 %. GPU time per iter ≈ wall time → GPU-bound.
→ ~70 % is the flyvis sparse message passing (gather → mul → scatter/index_add over the edge list, 40 ODE steps × 19 frames)
done as many separate memory-bound kernels, each re-reading the edge arrays.

## Conclusions (numbers only)
1. Mixed precision / TF32 do not help either model (≤ 4 %).
2. M4/M5: `torch.compile(mode="reduce-overhead")` = 5×; after that the Python DataLoader is the bottleneck → keep the (tiny) dataset
   pre-rendered on the GPU and augment on GPU. No Rust needed.
3. M1–M3: compile does not help; the cost is memory-bound sparse gather/scatter. The activity penalty costs 1/3 of the step.
   The remaining lever is a **fused message-passing kernel with a hand-written backward** (one pass over edges per ODE step instead of ~6).
   This is the only place where custom GPU code (Triton, or Rust cudarc+NVRTC as in our earlier Rust/CUDA GPU engine) can plausibly win ≥ 2×. Not yet built or measured.

## Update — activity penalty kept, fused implementation (C022)
| M1 bs4 | s/iter | VRAM |
|---|---|---|
| penalty, original flyvis path | 0.200 | 1.53 |
| penalty, fused path (default now; equivalence-tested) | 0.177 (1.13×) | 1.39 |
| no penalty (reference only) | 0.135 | 1.32 |
The penalty is active only for the first 60 % of iterations, so the whole-run saving is smaller than 1.13×.

## Update — fused Triton rollout `fastfly/` (C023) — **passes the gate (≥ 2× + parity)**
| M1 bs4, 40 frames, penalty (fused impl) | s/iter | VRAM GB |
|---|---|---|
| PyTorch flyvis | 0.177 | 1.39 |
| `--fastfly` (Triton) | **0.0738 (2.40×)** | 0.48 |
Overall vs the original R2 baseline 0.202 s/iter: **2.7×**. Parity: outputs ~1e-7 rel (CPU interpreter), grads 4e-7 rel; fixed-batch losses after
training identical to ~1e-7. What it does: one autograd Function for the whole rollout; CSR by target, gather+relu+mul+segment-sum+Euler fused
per step, batch processed as a tile (each edge loaded once), reverse-time backward with by-target and by-source (CSR-transpose) kernels, and a
segment-sum backward for the 1.5 M-edge → 604 type-pair weight expansion. Graph: 45,669 nodes, 1,513,231 edges, in-degree mean 33 / max 208.
Needs `CC=$HOME/flyproj/tools/zigcc/cc` (+ zigcc on PATH) for Triton on this machine.
Rust/cudarc version: not built. Remaining M1 step after fastfly is ~25 % kernel time, so a Rust port of the same algorithm has little headroom;
a persistent whole-rollout kernel (state kept on-chip) is the only Rust-specific idea left (estimate, unmeasured).
