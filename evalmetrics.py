"""Metrics on hex-lattice predictions. All tensors (B, T, C, N); masks (B, T, 1, N).

Per-pixel functions return (B, T, N); the mean versions average over valid
pixels. API mirrors baselines/metrics.py (epe, angular_error_deg, depth_rmse,
depth_absrel, binned) so either implementation can be used; eval.py cross-checks
them when baselines/metrics.py is importable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def _mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    m = torch.ones_like(x, dtype=torch.bool) if mask is None else mask[:, :, 0].bool()
    return m & torch.isfinite(x)


def epe_map(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - gt, dim=2)


def angular_error_map(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Angle (deg) between (u, v, 1) and (u_gt, v_gt, 1) (Barron et al. 1994)."""
    one = torch.ones_like(pred[:, :, :1])
    a = torch.cat([pred, one], 2)
    b = torch.cat([gt, one], 2)
    cos = (a * b).sum(2) / (a.norm(dim=2) * b.norm(dim=2))
    return torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0)))


def sq_err_map(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return ((pred - gt) ** 2)[:, :, 0]


def absrel_map(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return ((pred - gt).abs() / (gt.abs() + eps))[:, :, 0]


def _mean(v: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    m = _mask(v, mask)
    return v[m].mean()


def epe(pred_flow, gt_flow, mask=None):
    return _mean(epe_map(pred_flow, gt_flow), mask)


def angular_error_deg(pred_flow, gt_flow, mask=None):
    return _mean(angular_error_map(pred_flow, gt_flow), mask)


def depth_rmse(pred, gt, mask=None):
    return _mean(sq_err_map(pred, gt), mask).sqrt()


def depth_absrel(pred, gt, mask=None, eps=1e-6):
    return _mean(absrel_map(pred, gt, eps), mask)


def binned(values: np.ndarray, bin_values: np.ndarray, edges: np.ndarray):
    """Mean of ``values`` in bins of ``bin_values`` (np.digitize with ``edges``).
    Returns (means, counts) with len(edges) - 1 bins; empty bins -> nan."""
    values = np.asarray(values).ravel()
    bin_values = np.asarray(bin_values).ravel()
    ok = np.isfinite(values) & np.isfinite(bin_values)
    idx = np.digitize(bin_values[ok], edges) - 1
    n = len(edges) - 1
    means = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    for i in range(n):
        sel = idx == i
        counts[i] = sel.sum()
        if counts[i]:
            means[i] = values[ok][sel].mean()
    return means, counts
