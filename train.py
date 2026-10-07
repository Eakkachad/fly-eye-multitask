"""Multi-task (optic flow + depth) training harness for M1-M5.

Identical data, loss, optimizer and schedule for every model:
  data   : SplitSintel (flyvis MultiTaskSintel rendering + augmentation) restricted
           to the pre-registered train scenes (splits.json); val = val scenes,
           all frames, no augmentation.
  loss   : flyvis default per task (objectives.l2norm), equal task weights,
           normalised by the weight sum exactly like flyvis.Task.loss.
  optim  : Adam, flyvis default lr 5e-5 (network and decoders), flyvis
           'stepwise' schedule 5e-5 -> 5e-6 in 10 steps over n_iters.
  flyvis models additionally use flyvis' activity penalty on resting
  potentials (flyvis default; stopped at 60% of n_iters = flyvis' 150k/250k).

Outputs runs/<name>/: config.json, train_log.jsonl, val_log.jsonl,
best.pt (best total val loss), last.pt, summary.json.

Example:
  timeout --signal=KILL 4000 flock -s ~/flyproj/.orchestra/gpu.lock \
    ~/flyproj/.venv/bin/python train.py --model m1 --seed 0 --n-iters 30000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COURSE_DIR))



def penalty_grads(penalty, activity, iteration):
    """Fused-impl part 1: activity-penalty gradient w.r.t. ONLY the penalised params
    (flyvis Penalty.activity_penalty_step maths), computed BEFORE opt.step.

    Equivalent to flyvis: there the penalty backward runs after opt.step but through
    the graph saved at forward time (autograd saved tensors are the pre-step values),
    so the gradient is the same; restricting to `inputs` skips the weight/time-const/
    decoder grad work. Returns None when inactive (stop_iter reached)."""
    import torch
    from flyvis.solver import asymmetric_weighting

    if penalty.activity_optim is None:
        return None
    if (penalty.activity_penalty_stop_iter is not None
            and iteration >= penalty.activity_penalty_stop_iter):
        penalty.activity_optim = None  # same as Penalty.__call__: permanently off
        return None
    n_frames = activity.shape[1]
    mean = activity[:, n_frames // 4:, penalty.central_cells_index].mean(dim=1)
    pen = penalty.activity_penalty * (asymmetric_weighting(
        penalty.activity_baseline - mean, penalty.below_baseline_penalty_weight,
        penalty.above_baseline_penalty_weight) ** 2).mean()
    params = [p for g in penalty.activity_optim.param_groups for p in g["params"]]
    net = penalty.network
    skip = None
    if "_fastfly_orig_forward" in net.__dict__ and not any(
            p is net.edges_syn_strength for p in params):
        from fastfly import skip_edge_grads  # fused rollout: skip the per-edge weight grad
        skip = skip_edge_grads()
    if skip is None:
        return params, torch.autograd.grad(pen, params, retain_graph=True)
    with skip:
        return params, torch.autograd.grad(pen, params, retain_graph=True)


def penalty_apply(penalty, grads, lr):
    """Fused-impl part 2 (after opt.step): SGD step of the penalty optimizer with the
    precomputed grads at lr, then network.clamp(), like Penalty.activity_penalty_step."""
    params, gs = grads
    for g in penalty.activity_optim.param_groups:
        g["lr"] = lr
    for p, g in zip(params, gs):
        p.grad = g
    penalty.activity_optim.step()
    penalty.network.clamp()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["m1", "m2", "m3", "m4", "m5"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--null-seed", type=int, default=None,
                   help="seed of the M2/M3 null graph (default: = --seed)")
    p.add_argument("--n-iters", type=int, default=30000)
    p.add_argument("--data-fraction", type=float, default=1.0,
                   help="fraction of train scenes (pre-registered subsets in splits.json)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5, help="start lr (net and decoder)")
    p.add_argument("--lr-stop-factor", type=float, default=0.1)
    p.add_argument("--lr-steps", type=int, default=10)
    p.add_argument("--val-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--ss-every", type=int, default=0,
                   help="recompute flyvis steady state every k iters (0 = once per epoch)")
    p.add_argument("--no-activity-penalty", action="store_true")
    p.add_argument("--penalty-impl", choices=["flyvis", "fused"], default="fused",
                   help="flyvis: original flyvis Penalty call after opt.step; fused: "
                        "same maths, penalty grad taken before opt.step restricted to "
                        "the penalised params (see penalty_grads/penalty_apply)")
    p.add_argument("--fastfly", action="store_true",
                   help="patch the flyvis Network with the fused Triton rollout (fastfly/)")
    p.add_argument("--max-minutes", type=float, default=0, help="stop early (0 = off)")
    p.add_argument("--out-dir", default=str(COURSE_DIR / "runs"))
    p.add_argument("--name", default=None)
    p.add_argument("--splits", default=str(COURSE_DIR / "splits.json"))
    return p.parse_args(argv)


def lr_at(it, n_iters, start, stop, steps):
    """flyvis HyperParamScheduler.stepwise."""
    idx = min(int(it / max(n_iters / steps, 1e-9)), steps - 1)
    return float(np.linspace(start, stop, steps)[idx])


def git_rev():
    try:
        rev = subprocess.check_output(["git", "-C", str(COURSE_DIR), "rev-parse", "HEAD"],
                                      text=True).strip()
        dirty = subprocess.call(["git", "-C", str(COURSE_DIR), "diff", "--quiet"]) != 0
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    import flyvis
    import torch

    return dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, flyvis=flyvis.__version__,
                numpy=np.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2norm_losses(out, batch, targets=("flow", "depth")):
    """flyvis objectives.l2norm per task, weighted 1/len(tasks) (flyvis Task.loss)."""
    from flyvis.task.objectives import l2norm

    return {t: l2norm(out[t], batch[t]) / len(targets) for t in targets}


def prepare_batch(batch, depth_tf):
    batch = dict(batch)
    batch["depth"] = depth_tf(batch["depth"])
    return batch


def evaluate(model, dataset, depth_tf, per_sequence=False, return_preds=False):
    """Loss + metrics over every sequence of ``dataset`` (batch 1, all frames)."""
    import torch

    import evalmetrics as M

    model.eval()
    losses = {"flow": [], "depth": []}
    epe_all, ae_all, se_all, ar_all, seqs = [], [], [], [], []
    preds = []
    with torch.no_grad():
        for i in range(len(dataset)):
            raw = dataset[i]
            raw = {k: v[None] for k, v in raw.items()}
            b = prepare_batch(raw, depth_tf)
            out = model(b["lum"])
            for t, v in l2norm_losses(out, b).items():
                losses[t].append(float(v) * 2)  # unweighted per-task loss
            depth_pred = depth_tf.inverse(out["depth"])
            e = M.epe_map(out["flow"], raw["flow"])
            a = M.angular_error_map(out["flow"], raw["flow"])
            # RMSE in normalised target units, abs-rel in (clipped) metric depth
            s = M.sq_err_map(out["depth"], b["depth"])
            r = M.absrel_map(depth_pred, depth_tf.clip(raw["depth"]))
            epe_all.append(e.flatten())
            ae_all.append(a.flatten())
            se_all.append(s.flatten())
            ar_all.append(r.flatten())
            if per_sequence:
                seqs.append(dict(index=i, epe=float(e.mean()), angular=float(a.mean()),
                                 depth_rmse=float(s.mean().sqrt()),
                                 depth_absrel=float(r.mean()),
                                 loss_flow=losses["flow"][-1],
                                 loss_depth=losses["depth"][-1]))
            if return_preds:
                preds.append(dict(flow=out["flow"][0].cpu(), depth=depth_pred[0].cpu(),
                                  depth_t=out["depth"][0].cpu()))
    model.train()
    cat = lambda xs: torch.cat(xs)  # noqa: E731
    res = dict(
        loss_flow=float(np.mean(losses["flow"])),
        loss_depth=float(np.mean(losses["depth"])),
        epe=float(cat(epe_all).mean()),
        angular_deg=float(cat(ae_all).mean()),
        depth_rmse=float(cat(se_all).mean().sqrt()),
        depth_absrel=float(cat(ar_all).mean()),
    )
    res["loss"] = 0.5 * (res["loss_flow"] + res["loss_depth"])
    if per_sequence:
        res["sequences"] = seqs
    if return_preds:
        res["preds"] = preds
    return res


def main(argv=None):
    args = parse_args(argv)
    logging.disable(logging.INFO)
    import torch
    from torch.utils.data import DataLoader

    import data as D
    import models
    import splits as S

    seed_everything(args.seed)
    splits = S.load_splits(args.splits)
    D.check_disjoint(splits)
    train_sc = S.train_scenes(splits, args.data_fraction)
    name = args.name or (f"{args.model}_s{args.seed}_f{args.data_fraction:g}"
                         f"_it{args.n_iters}")
    run = Path(args.out_dir) / name
    run.mkdir(parents=True, exist_ok=True)

    t_setup = time.time()
    ds = D.make_datasets(splits, train_sc, which=("train", "val"))
    assert set(ds["train"].sequence_scenes()) <= set(splits["train"])
    assert set(ds["val"].sequence_scenes()) <= set(splits["val"])
    depth_tf = D.DepthTransform.from_file()
    model = models.build_model(args.model, seed=args.seed, null_seed=args.null_seed)
    dev = torch.device("cuda")
    model.to(dev)
    is_flyvis = getattr(model, "is_flyvis", False)
    if args.fastfly:
        if not is_flyvis:
            raise SystemExit("--fastfly needs a flyvis model (m1-m3)")
        from fastfly import patch_network
        patch_network(model.network)

    if is_flyvis:
        groups = model.param_groups(args.lr, args.lr)
    else:
        groups = [dict(params=list(model.parameters()), lr=args.lr, name="all")]
    opt = torch.optim.Adam(groups)
    penalty = None
    if is_flyvis and not args.no_activity_penalty:
        from datamate import Namespace
        from flyvis.solver import Penalty

        penalty = Penalty(Namespace(
            activity_penalty=Namespace(activity_baseline=5.0, activity_penalty=0.1,
                                       stop_iter=int(0.6 * args.n_iters),
                                       below_baseline_penalty_weight=1.0,
                                       above_baseline_penalty_weight=0.1),
            optim="SGD"), model.network)

    loader = DataLoader(ds["train"], batch_size=args.batch_size, drop_last=True,
                        sampler=SeededSampler(len(ds["train"]), args.seed))
    iters_per_epoch = len(loader)
    ss_every = args.ss_every or iters_per_epoch

    config = dict(
        args=vars(args), name=name, run_dir=str(run), git=git_rev(), versions=versions(),
        host=platform.node(), argv=sys.argv,
        train_scenes=train_sc, val_scenes=splits["val"], splits_sha256=sha256(args.splits),
        n_train_sequences=len(ds["train"]), n_val_sequences=len(ds["val"]),
        iters_per_epoch=iters_per_epoch, ss_every=ss_every,
        n_params=models.n_trainable(model), depth_transform=depth_tf.describe(),
        connectome_file=getattr(model, "connectome_file", None),
        connectome_sha256=(sha256(model.connectome_file)
                           if getattr(model, "connectome_file", None) else None),
        loss="flyvis.task.objectives.l2norm per task, weights 0.5/0.5",
        optimizer=f"Adam lr {args.lr} stepwise -> {args.lr * args.lr_stop_factor} "
                  f"in {args.lr_steps} steps",
        activity_penalty=penalty is not None,
        dataset=D.FLYVIS_TASK_DEFAULTS,
    )
    (run / "config.json").write_text(json.dumps(config, indent=1, default=str))
    setup_s = time.time() - t_setup

    torch.cuda.reset_peak_memory_stats()
    flog = open(run / "train_log.jsonl", "w")
    vlog = open(run / "val_log.jsonl", "w")
    best = float("inf")
    it, t0 = 0, time.time()
    data_t = comp_t = 0.0
    acc = {"loss": 0.0, "flow": 0.0, "depth": 0.0, "n": 0}
    status = "completed"
    model.train()

    def do_val(it):
        nonlocal best
        tv = time.time()
        res = evaluate(model, ds["val"], depth_tf)
        res.update(iter=it, wall_s=time.time() - t0, val_s=time.time() - tv)
        improved = res["loss"] < best
        if improved:
            best = res["loss"]
            torch.save(dict(model=model.state_dict(), iter=it, val=res, config=config),
                       run / "best.pt")
        res["best"] = improved
        vlog.write(json.dumps(res) + "\n")
        vlog.flush()
        return res

    while it < args.n_iters and status == "completed":
        for batch in _timed(loader):
            td, batch = batch
            data_t += td
            tc = time.time()
            lr = lr_at(it, args.n_iters, args.lr, args.lr * args.lr_stop_factor,
                       args.lr_steps)
            for g in opt.param_groups:
                g["lr"] = lr
            if is_flyvis and (it % ss_every == 0):
                model.refresh_steady_state(args.batch_size)
            batch = prepare_batch(batch, depth_tf)
            opt.zero_grad(set_to_none=True)
            out = model(batch["lum"], return_activity=penalty is not None) \
                if is_flyvis else model(batch["lum"])
            losses = l2norm_losses(out, batch)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                status = "diverged"
                break
            fused = penalty is not None and args.penalty_impl == "fused"
            pgrads = None
            if fused:
                pgrads = penalty_grads(penalty, out["activity"], it)
            loss.backward(retain_graph=penalty is not None and not fused)
            opt.step()
            if fused:
                if pgrads is not None:
                    penalty_apply(penalty, pgrads, lr)
            elif penalty is not None:
                if penalty.activity_optim is not None:
                    for g in penalty.activity_optim.param_groups:
                        g["lr"] = lr
                penalty(activity=out["activity"], iteration=it)
            if fused:
                # logs stay on the GPU; host sync only at the log interval
                acc["loss"] += loss.detach()
                acc["flow"] += losses["flow"].detach() * 2
                acc["depth"] += losses["depth"].detach() * 2
                if (it + 1) % args.log_every == 0:
                    torch.cuda.synchronize()
            else:
                torch.cuda.synchronize()
                acc["loss"] += float(loss)
                acc["flow"] += float(losses["flow"]) * 2
                acc["depth"] += float(losses["depth"]) * 2
            comp_t += time.time() - tc
            acc["n"] += 1
            it += 1
            if it % args.log_every == 0:
                n = acc.pop("n")
                rec = {k: float(v) / n for k, v in acc.items()}
                rec.update(iter=it, lr=lr, wall_s=time.time() - t0,
                           s_per_iter=(time.time() - t0) / it,
                           data_s_per_iter=data_t / it, compute_s_per_iter=comp_t / it)
                flog.write(json.dumps(rec) + "\n")
                flog.flush()
                acc = {"loss": 0.0, "flow": 0.0, "depth": 0.0, "n": 0}
            if it % args.val_every == 0 or it == args.n_iters:
                r = do_val(it)
                print(f"[{name}] it {it} val loss {r['loss']:.4f} epe {r['epe']:.4f} "
                      f"depth_rmse {r['depth_rmse']:.4f} ({(time.time()-t0)/it:.3f}"
                      f" s/it)", flush=True)
            if args.max_minutes and (time.time() - t0) > 60 * args.max_minutes:
                status = "time_limit"
                break
            if it >= args.n_iters:
                break
    wall = time.time() - t0
    if status != "diverged" and (it % args.val_every != 0 and it != args.n_iters):
        do_val(it)
    torch.save(dict(model=model.state_dict(), iter=it, config=config), run / "last.pt")
    summary = dict(
        status=status, iters=it, wall_s=wall, setup_s=setup_s,
        s_per_iter=wall / max(it, 1), data_s_per_iter=data_t / max(it, 1),
        compute_s_per_iter=comp_t / max(it, 1),
        peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30,
        peak_vram_reserved_gb=torch.cuda.max_memory_reserved() / 2**30,
        best_val_loss=best,
        best_iter=(torch.load(run / "best.pt", map_location="cpu", weights_only=False)
                   ["iter"] if (run / "best.pt").exists() else None),
        pid=os.getpid(),
    )
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)
    return summary


class SeededSampler:
    """Random permutation per epoch from a numpy RNG (torch's default device is
    CUDA under flyvis, which breaks torch.randperm with a CPU generator)."""

    def __init__(self, n, seed):
        self.n, self.rng = n, np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        return iter(self.rng.permutation(self.n).tolist())


def _timed(loader):
    """Yields (seconds spent fetching, batch)."""
    import torch

    itr = iter(loader)
    while True:
        t = time.time()
        try:
            b = next(itr)
        except StopIteration:
            return
        torch.cuda.synchronize()
        yield time.time() - t, b


if __name__ == "__main__":
    main()
