"""Benchmark script for fly-eye multi-task hexagonal neural network baselines.

Measures forward + backward runtime and peak VRAM on CUDA for:
- make_small() (param-matched baseline)
- make_large() (deep ConvGRU baseline)
Batch size: B=4, Time frames: T=19, Hexals: N=721.
"""

import sys
import time
import torch

from hex_models import count_parameters, make_large, make_small


def benchmark_model(
    name: str,
    model_fn,
    batch_size: int = 4,
    time_frames: int = 19,
    n_hexals: int = 721,
    warmup_iters: int = 3,
    benchmark_iters: int = 10,
    device: str = "cuda",
):
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA is not available on this system. Exiting.")
        sys.exit(1)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    model = model_fn().to(device)
    model.train()
    param_count = count_parameters(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Synthetic stimulus and targets matching multi-task contract
    x = torch.randn(batch_size, time_frames, 1, n_hexals, device=device)
    target_flow = torch.randn(batch_size, time_frames, 2, n_hexals, device=device)
    target_depth = torch.randn(batch_size, time_frames, 1, n_hexals, device=device)

    # Warmup runs
    for _ in range(warmup_iters):
        optimizer.zero_grad()
        out = model(x)
        loss = (
            (out["flow"] - target_flow).pow(2).mean()
            + (out["depth"] - target_depth).pow(2).mean()
        )
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Timed runs
    t_start = time.time()
    for _ in range(benchmark_iters):
        optimizer.zero_grad()
        out = model(x)
        loss = (
            (out["flow"] - target_flow).pow(2).mean()
            + (out["depth"] - target_depth).pow(2).mean()
        )
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    total_time = time.time() - t_start
    s_per_iter = total_time / benchmark_iters

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_vram_mb = peak_bytes / (1024**2)
    peak_vram_gb = peak_bytes / (1024**3)

    return {
        "name": name,
        "params": param_count,
        "s_per_iter": s_per_iter,
        "peak_vram_mb": peak_vram_mb,
        "peak_vram_gb": peak_vram_gb,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    print(f"Running benchmark on device: {device} ({device_name})")
    print("Settings: B=4, T=19, N=721 hexals, forward + backward (Adam)")
    print("-" * 75)
    print(
        f"{'Model':<10} | {'Params':>10} | {'s/iter':>10} | {'Peak VRAM (MB)':>16} | {'Peak VRAM (GB)':>16}"
    )
    print("-" * 75)

    for name, fn in [("small", make_small), ("large", make_large)]:
        res = benchmark_model(name, fn, device=device)
        print(
            f"{res['name']:<10} | {res['params']:>10,d} | {res['s_per_iter']:>9.4f}s | "
            f"{res['peak_vram_mb']:>14.1f} MB | {res['peak_vram_gb']:>14.3f} GB"
        )
    print("-" * 75)


if __name__ == "__main__":
    main()
