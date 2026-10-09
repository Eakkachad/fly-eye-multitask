"""Optional training objectives (phase 2).  Defaults in train.py keep the flyvis l2norm.

Tensors: flow (B, T, 2, N) [du, dv per hexal], depth (B, T, 1, N) standardised log-depth.
All functions return a scalar.  Task weighting (1/len(tasks)) is applied by the caller.

FLOW
  l2norm : flyvis objectives.l2norm = mean_b sqrt(sum_{t,c,n} (pred-gt)^2)   (sequence-level norm)
  epe    : mean_{b,t,n} sqrt(du^2 + dv^2 + eps),  eps = 1e-6   (per-hexal endpoint error;
           the evaluation metric itself, smoothed so the gradient exists at 0)
  robust : mean_{b,t,c,n} sqrt(d^2 + eps^2), eps = 1e-3        (Charbonnier / smooth-L1 per
           component; ~L1 for |d| >> eps, quadratic near 0; less dominated by outliers)

DEPTH
  l2norm : as above.
  si     : scale-invariant log-depth loss (Eigen, Puhrsch & Fergus, NeurIPS 2014), per frame
           d_n = log(pred_n) - log(gt_n),  L = mean_n d^2 - lam * (mean_n d)^2,  then mean over (B,T).
           Space: our target is t = (log clip(depth) - mean) / std (data.DepthTransform), so
           d = std * (t_pred - t_gt); the dataset mean cancels in the difference, the std must
           be applied (``log_scale=std``) to obtain true log-ratio units.  With lam = 1 the loss is
           the per-frame variance of d: invariant to adding any constant to pred (= multiplying
           metric depth by a constant); lam = 0 is plain MSE in log space.
  si+l2  : si + w * mean((t_pred - t_gt)^2) (standardised units, w = --si-l2-weight, default 0.1),
           so the absolute depth level is still (weakly) learned.

  Why it targets the bias failure: docs/ERROR_ANALYSIS.md found depth predictions are almost
  constant within a frame, with the per-frame offset (bias) making up ~66 % of the MSE.  The
  l2 loss spends its gradient on this global offset (a few degrees of freedom that the
  network can only guess from textureless input).  SI with lam close to 1 removes the
  per-frame offset from the objective, so gradient capacity goes to relative depth structure
  (what varies across hexals); lam = 0.5 keeps half of the offset penalty, and si+l2 a small
  residual absolute term.  Magnitude: SI is O(std^2) while l2norm is a sequence-level norm
  (O(100)); use --depth-loss-weight / --flow-loss-weight to rebalance (default 1).
"""

from __future__ import annotations

import torch

FLOW_LOSSES = ("l2norm", "epe", "robust")
DEPTH_LOSSES = ("l2norm", "si", "si+l2")
EPE_EPS = 1e-6
ROBUST_EPS = 1e-3


def flow_epe_loss(pred, gt, eps: float = EPE_EPS):
    return torch.sqrt(((pred - gt) ** 2).sum(dim=2) + eps).mean()


def flow_robust_loss(pred, gt, eps: float = ROBUST_EPS):
    return torch.sqrt((pred - gt) ** 2 + eps ** 2).mean()


def si_loss(pred, gt, lam: float = 0.5, log_scale: float = 1.0):
    """Scale-invariant loss, per frame over hexals (and channels), mean over (B, T)."""
    d = (pred - gt) * log_scale
    d = d.flatten(start_dim=2)  # (B, T, C*N)
    return ((d ** 2).mean(dim=-1) - lam * d.mean(dim=-1) ** 2).mean()


def task_losses(out, batch, tasks=("flow", "depth"), flow_loss="l2norm",
                depth_loss="l2norm", si_lambda=0.5, si_l2_weight=0.1, log_scale=1.0,
                flow_weight=1.0, depth_weight=1.0):
    """Per-task losses weighted 1/len(tasks) (times optional extra weights, default 1).
    With defaults this is exactly train.l2norm_losses."""
    from flyvis.task.objectives import l2norm

    res = {}
    for t in tasks:
        p, g = out[t], batch[t]
        kind = flow_loss if t == "flow" else depth_loss
        if kind == "l2norm":
            v = l2norm(p, g)
        elif kind == "epe":
            v = flow_epe_loss(p, g)
        elif kind == "robust":
            v = flow_robust_loss(p, g)
        elif kind == "si":
            v = si_loss(p, g, si_lambda, log_scale)
        elif kind == "si+l2":
            v = si_loss(p, g, si_lambda, log_scale) + si_l2_weight * ((p - g) ** 2).mean()
        else:
            raise ValueError(kind)
        w = flow_weight if t == "flow" else depth_weight
        res[t] = v / len(tasks) if w == 1.0 else v * w / len(tasks)
    return res
