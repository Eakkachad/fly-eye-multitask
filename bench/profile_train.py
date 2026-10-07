"""Profile the SAME training step as train.py under configurable speed variants.

Reuses train.py (prepare_batch, l2norm_losses, lr_at, SeededSampler, seed_everything,
versions) and data/models/splits. The step body mirrors train.py main():
  zero_grad -> forward (+activity for flyvis) -> l2norm losses -> backward
  (retain_graph if activity penalty) -> Adam step -> activity penalty.

Variants: --amp, --compile, --tf32, --device. Report is written as JSON.
CUDA graphs: not exposed as a separate flag. Whole-step manual capture is not
practical here (flyvis Network re-allocates stimulus buffers and refreshes the
steady state, the activity Penalty has its own optimizer and Python control flow,
the DataLoader feeds fresh tensors). `--compile reduce-overhead` is the supported
CUDA-graph route (torch.compile mode that wraps compiled regions in CUDA graphs).

Example:
  CUDA_VISIBLE_DEVICES="" python bench/profile_train.py --model m4 --device cpu \
      --iters 3 --warmup 1 --batch-size 1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import statistics
import sys
import time
from pathlib import Path

COURSE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COURSE_DIR))

import train as T  # noqa: E402  (reuse prepare_batch, l2norm_losses, ...)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["m1", "m2", "m3", "m4", "m5"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none")
    p.add_argument("--compile", choices=["none", "default", "reduce-overhead",
                                         "max-autotune"], default="none")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="torch.profiler for --profile-iters iters after timing")
    p.add_argument("--profile-iters", type=int, default=10)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--data-fraction", type=float, default=1.0)
    p.add_argument("--no-activity-penalty", action="store_true")
    p.add_argument("--out", default=None, help="JSON output path")
    return p.parse_args(argv)


class Clock:
    """cuda events on cuda, perf_counter on cpu."""

    def __init__(self, cuda):
        import torch
        self.cuda, self.torch = cuda, torch
        self.marks = []

    def mark(self):
        if self.cuda:
            e = self.torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append(e)
        else:
            self.marks.append(time.perf_counter())

    def deltas(self):
        """Seconds between consecutive marks (call after synchronize)."""
        m = self.marks
        if self.cuda:
            d = [m[i].elapsed_time(m[i + 1]) / 1e3 for i in range(len(m) - 1)]
        else:
            d = [m[i + 1] - m[i] for i in range(len(m) - 1)]
        self.marks = []
        return d


def main(argv=None):
    args = parse_args(argv)
    logging.disable(logging.INFO)
    import torch
    from torch.utils.data import DataLoader

    import data as D
    import models
    import splits as S

    cuda = args.device == "cuda"
    dev = torch.device(args.device)
    if cuda and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    T.seed_everything(args.seed)
    splits = S.load_splits(str(COURSE_DIR / "splits.json"))
    ds = D.make_datasets(splits, S.train_scenes(splits, args.data_fraction),
                         which=("train",))["train"]
    depth_tf = D.DepthTransform.from_file()
    model = models.build_model(args.model, seed=args.seed).to(dev)
    is_flyvis = getattr(model, "is_flyvis", False)
    groups = (model.param_groups(args.lr, args.lr) if is_flyvis else
              [dict(params=list(model.parameters()), lr=args.lr, name="all")])
    opt = torch.optim.Adam(groups)
    penalty = None
    if is_flyvis and not args.no_activity_penalty:
        from datamate import Namespace
        from flyvis.solver import Penalty
        penalty = Penalty(Namespace(
            activity_penalty=Namespace(activity_baseline=5.0, activity_penalty=0.1,
                                       stop_iter=10**9,
                                       below_baseline_penalty_weight=1.0,
                                       above_baseline_penalty_weight=0.1),
            optim="SGD"), model.network)
    if is_flyvis:
        model.refresh_steady_state(args.batch_size)

    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    use_scaler = args.amp == "fp16" and cuda
    scaler = torch.amp.GradScaler("cuda") if use_scaler else None

    def autocast():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=args.device, dtype=amp_dtype)

    fwd_model = model
    result = dict(
        config=dict(vars(args), versions=T.versions(), n_params=models.n_trainable(model),
                    activity_penalty=penalty is not None,
                    n_train_sequences=len(ds)),
        status="ok")
    out_path = Path(args.out) if args.out else None

    def finish():
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=1, default=str))
        print(json.dumps({k: v for k, v in result.items() if k != "profile"},
                         default=str), flush=True)

    if args.compile != "none":
        try:
            kw = {} if args.compile == "default" else dict(mode=args.compile)
            fwd_model = torch.compile(model, **kw)
        except Exception as e:  # noqa: BLE001
            result["status"] = f"compile_failed: {type(e).__name__}: {str(e)[:300]}"
            finish()
            return result

    loader = DataLoader(ds, batch_size=args.batch_size, drop_last=True,
                        sampler=T.SeededSampler(len(ds), args.seed))

    def batches():
        while True:
            for b in loader:
                yield b

    def to_dev(b):
        return {k: v.to(dev) for k, v in b.items()}

    # ---- fixed batch for numerics (own sampler seed, independent of training order)
    fixed_loader = DataLoader(ds, batch_size=args.batch_size, drop_last=True,
                              sampler=T.SeededSampler(len(ds), 12345))
    torch.manual_seed(999)
    fixed = T.prepare_batch(to_dev(next(iter(fixed_loader))), depth_tf)

    def fixed_loss():
        """Eval-mode (no dropout) per-task unweighted loss on the fixed batch."""
        model.eval()
        with torch.no_grad(), autocast():
            out = fwd_model(fixed["lum"])
        ls = T.l2norm_losses({k: v.float() for k, v in out.items()}, fixed)
        model.train()
        return dict(flow=float(ls["flow"]) * 2, depth=float(ls["depth"]) * 2)

    def step(b, it, clock):
        """One train.py step. Marks: start, after fwd+loss, after bwd, after opt."""
        clock.mark()
        b = T.prepare_batch(b, depth_tf)
        opt.zero_grad(set_to_none=True)
        with autocast():
            out = fwd_model(b["lum"], return_activity=True) if penalty is not None \
                else fwd_model(b["lum"])
        losses = T.l2norm_losses({k: (v.float() if v.is_floating_point() else v)
                                  for k, v in out.items()}, b)
        loss = sum(losses.values())
        clock.mark()
        if scaler:
            scaler.scale(loss).backward(retain_graph=penalty is not None)
        else:
            loss.backward(retain_graph=penalty is not None)
        clock.mark()
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
        if penalty is not None:
            penalty(activity=out["activity"], iteration=it)
        clock.mark()
        return loss.detach(), losses

    try:
        model.train()
        result["fixed_batch_loss_initial"] = fixed_loss()
        it = 0
        src = batches()
        t_compile0 = time.perf_counter()
        for w in range(args.warmup):
            b = to_dev(next(src))
            step(b, it, Clock(cuda))
            it += 1
            sync()
            if w == 0:
                result["first_iter_s_incl_compile"] = time.perf_counter() - t_compile0
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        data_s, tot_s, fwd_s, bwd_s, opt_s = [], [], [], [], []
        last = None
        for _ in range(args.iters):
            sync()
            t0 = time.perf_counter()
            b = next(src)
            b = to_dev(b)
            sync()
            t1 = time.perf_counter()
            clock = Clock(cuda)
            _, losses = step(b, it, clock)
            sync()
            t2 = time.perf_counter()
            f, bw, o = clock.deltas()
            data_s.append(t1 - t0)
            tot_s.append(t2 - t1)
            fwd_s.append(f)
            bwd_s.append(bw)
            opt_s.append(o)
            last = {k: float(v) * 2 for k, v in losses.items()}
            it += 1
        iter_s = [d + c for d, c in zip(data_s, tot_s)]
        med = statistics.median
        result.update(
            s_per_iter_median=med(iter_s), s_per_iter_mean=statistics.fmean(iter_s),
            samples_per_s=args.batch_size / med(iter_s),
            data_s_median=med(data_s), compute_s_median=med(tot_s),
            data_fraction=med(data_s) / med(iter_s),
            fwd_s_median=med(fwd_s), bwd_s_median=med(bwd_s), opt_s_median=med(opt_s),
            peak_vram_allocated_gb=(torch.cuda.max_memory_allocated() / 2**30
                                    if cuda else None),
            peak_vram_reserved_gb=(torch.cuda.max_memory_reserved() / 2**30
                                   if cuda else None),
            final_train_loss=last,
            fixed_batch_loss_final=fixed_loss(),
        )
        if args.profile:
            result["profile"] = run_profile(args, step, src, to_dev, it, cuda, sync)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        result["status"] = (f"compile_failed: {msg}" if args.compile != "none"
                            else f"failed: {msg}")
    finish()
    return result


def run_profile(args, step, src, to_dev, it, cuda, sync):
    import torch
    from torch.profiler import ProfilerActivity, profile

    base = Path(args.out).with_suffix("") if args.out else Path(
        COURSE_DIR / "bench" / "results" / f"profile_{args.model}")
    base.parent.mkdir(parents=True, exist_ok=True)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    with profile(activities=acts, record_shapes=False, profile_memory=False) as prof:
        for _ in range(args.profile_iters):
            step(to_dev(next(src)), it, Clock(cuda))
            it += 1
            sync()
    trace = str(base) + "_trace.json"
    prof.export_chrome_trace(trace)
    txt = []
    ka = prof.key_averages()
    tables = [("self CPU time", "self_cpu_time_total")]
    if cuda:
        tables.insert(0, ("self CUDA time", "self_device_time_total"))
    for title, key in tables:
        txt.append(f"== top 25 ops by {title} ({args.profile_iters} iters) ==")
        try:
            txt.append(ka.table(sort_by=key, row_limit=25, max_name_column_width=70))
        except Exception as e:  # noqa: BLE001
            txt.append(f"(table failed: {e})")
    tpath = str(base) + "_top_ops.txt"
    Path(tpath).write_text("\n".join(txt))
    return dict(trace=trace, top_ops=tpath)


if __name__ == "__main__":
    main()
