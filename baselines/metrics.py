"""Evaluation metrics for optic flow and depth on fly-eye video sequences.

All metric functions take prediction and target tensors of shape (B, T, C, N)
and an optional boolean mask of shape (B, T, 1, N).
Also includes binned error analysis across feature distributions.
"""

from typing import Optional, Union

import torch


def epe(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean end-point error (L2 norm over the 2 flow channels).

    Args:
        pred_flow: Predicted flow tensor of shape (B, T, 2, N) or (2, N) or (2,).
        gt_flow: Ground truth flow tensor of shape matching pred_flow.
        mask: Optional boolean mask of valid pixels (B, T, 1, N) or matching shape.

    Returns:
        Scalar torch.Tensor with the mean end-point error.
    """
    diff = pred_flow - gt_flow
    if diff.ndim == 1:
        error = torch.linalg.norm(diff, dim=0, keepdim=True)
    else:
        error = torch.linalg.norm(diff, dim=-2, keepdim=True)

    if mask is not None:
        mask = mask.bool()
        if mask.shape != error.shape:
            mask = mask.expand_as(error)
        valid = error[mask]
        if valid.numel() == 0:
            return torch.tensor(0.0, device=pred_flow.device, dtype=pred_flow.dtype)
        return valid.mean()

    return error.mean()


def angular_error_deg(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Mean angular error in degrees between space-time vectors (u, v, 1).

    Args:
        pred_flow: Predicted flow tensor of shape (B, T, 2, N) or (2, N) or (2,).
        gt_flow: Ground truth flow tensor of shape matching pred_flow.
        mask: Optional boolean mask of valid pixels (B, T, 1, N) or matching shape.

    Returns:
        Scalar torch.Tensor with the mean angular error in degrees.
    """
    orig_dtype = pred_flow.dtype
    p = pred_flow.to(torch.float64)
    g = gt_flow.to(torch.float64)

    if p.ndim == 1:
        u_p, v_p = p[0:1], p[1:2]
        u_g, v_g = g[0:1], g[1:2]
    else:
        u_p = p.narrow(dim=-2, start=0, length=1)
        v_p = p.narrow(dim=-2, start=1, length=1)
        u_g = g.narrow(dim=-2, start=0, length=1)
        v_g = g.narrow(dim=-2, start=1, length=1)

    identical = (u_p == u_g) & (v_p == v_g)
    dot = u_p * u_g + v_p * v_g + 1.0
    norm_p = torch.sqrt(u_p * u_p + v_p * v_p + 1.0)
    norm_g = torch.sqrt(u_g * u_g + v_g * v_g + 1.0)
    denom = torch.clamp(norm_p * norm_g, min=1e-12)
    cos = torch.clamp(dot / denom, -1.0, 1.0)
    angle_deg = torch.rad2deg(torch.acos(cos)).to(orig_dtype)
    angle_deg = torch.where(identical, torch.zeros_like(angle_deg), angle_deg)

    if mask is not None:
        mask = mask.bool()
        if mask.shape != angle_deg.shape:
            mask = mask.expand_as(angle_deg)
        valid = angle_deg[mask]
        if valid.numel() == 0:
            return torch.tensor(0.0, device=pred_flow.device, dtype=orig_dtype)
        return valid.mean()

    return angle_deg.mean()


def depth_rmse(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Root Mean Squared Error (RMSE) for depth predictions.

    Args:
        pred: Predicted depth tensor of shape (B, T, 1, N) or matching gt.
        gt: Ground truth depth tensor.
        mask: Optional boolean mask of valid pixels (B, T, 1, N) or matching shape.

    Returns:
        Scalar torch.Tensor with the RMSE.
    """
    sq_err = (pred - gt) ** 2
    if mask is not None:
        mask = mask.bool()
        if mask.shape != sq_err.shape:
            mask = mask.expand_as(sq_err)
        valid = sq_err[mask]
        if valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return torch.sqrt(valid.mean())

    return torch.sqrt(sq_err.mean())


def depth_absrel(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mean absolute relative error (|pred - gt| / (|gt| + eps)).

    Args:
        pred: Predicted depth tensor of shape (B, T, 1, N) or matching gt.
        gt: Ground truth depth tensor.
        mask: Optional boolean mask of valid pixels (B, T, 1, N) or matching shape.
        eps: Small epsilon added to denominator to avoid division by zero.

    Returns:
        Scalar torch.Tensor with the AbsRel metric.
    """
    abs_rel = torch.abs(pred - gt) / (torch.abs(gt) + eps)
    if mask is not None:
        mask = mask.bool()
        if mask.shape != abs_rel.shape:
            mask = mask.expand_as(abs_rel)
        valid = abs_rel[mask]
        if valid.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return valid.mean()

    return abs_rel.mean()


def binned(
    metric_values_per_pixel: torch.Tensor,
    bin_values: torch.Tensor,
    edges: Union[torch.Tensor, list, tuple],
) -> torch.Tensor:
    """Compute mean metric value per bin defined by edges.

    Args:
        metric_values_per_pixel: Tensor of per-pixel metric values.
        bin_values: Tensor of conditioning values (e.g. speed, texture) of same shape.
        edges: 1D sequence or tensor of K+1 bin boundaries defining K bins.
            Bins 0..K-2 are half-open [edges[i], edges[i+1]), and the last bin K-1
            is closed [edges[K-1], edges[K]].

    Returns:
        1D torch.Tensor of length K containing the mean metric for each bin.
        Bins with zero pixels contain NaN.
    """
    if not isinstance(edges, torch.Tensor):
        edges = torch.as_tensor(edges, device=bin_values.device, dtype=bin_values.dtype)

    m_flat = metric_values_per_pixel.flatten()
    b_flat = bin_values.flatten()

    if m_flat.shape != b_flat.shape:
        raise ValueError(
            f"Shape mismatch: metric_values_per_pixel {metric_values_per_pixel.shape} "
            f"vs bin_values {bin_values.shape}"
        )

    n_bins = len(edges) - 1
    if n_bins < 1:
        raise ValueError("edges must contain at least two boundary values.")

    means = []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (b_flat >= low) & (b_flat <= high)
        else:
            mask = (b_flat >= low) & (b_flat < high)

        vals = m_flat[mask]
        if vals.numel() > 0:
            means.append(vals.float().mean())
        else:
            means.append(
                torch.tensor(float("nan"), device=m_flat.device, dtype=torch.float32)
            )

    return torch.stack(means)
