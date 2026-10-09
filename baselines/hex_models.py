"""Hexagonal lattice neural network baselines for fly-eye multi-task video.

Provides:
- HexConv: 2D convolution over a regular hexagonal lattice (7-neighbourhood).
- HexConvGRUCell: Recurrent ConvGRU cell on the hexagonal lattice.
- HexConvGRUNet: Video model with HexConv encoder, temporal ConvGRU, and multi-task heads.
- make_small / make_small_matched / make_large: Model factories matched to target parameter budgets.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flyvis.utils.hex_utils import get_hex_coords

# 7-neighbourhood offsets in axial coordinates (u, v):
# self (0, 0) + 6 axial neighbours:
# (+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, +1)
HEX_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, -1),
    (-1, 1),
)

_TABLE_CACHE: Dict[int, torch.Tensor] = {}


def get_neighbour_table(extent: int = 15) -> torch.Tensor:
    """Build or retrieve a cached neighbour index table of shape (N, 7).

    For each hexal i in 0..N-1, entries 0..6 correspond to the 7 axial offsets:
    (0, 0), (+1, 0), (-1, 0), (0, +1), (0, -1), (+1, -1), (-1, +1).
    Missing neighbours at the boundary point to index N (zero-padding row).

    Args:
        extent: Integer radius of the hexagonal lattice (15 gives N=721).

    Returns:
        torch.LongTensor of shape (N, 7).
    """
    if extent not in _TABLE_CACHE:
        u_arr, v_arr = get_hex_coords(extent)
        n_hexals = len(u_arr)
        coord_to_idx = {
            (int(u_arr[i]), int(v_arr[i])): i for i in range(n_hexals)
        }

        table = torch.empty((n_hexals, 7), dtype=torch.long)
        for i in range(n_hexals):
            ui, vi = int(u_arr[i]), int(v_arr[i])
            for k, (du, dv) in enumerate(HEX_OFFSETS):
                nbr_coord = (ui + du, vi + dv)
                table[i, k] = coord_to_idx.get(nbr_coord, n_hexals)

        _TABLE_CACHE[extent] = table

    return _TABLE_CACHE[extent].clone()


class HexConv(nn.Module):
    """Convolution on the hexagonal lattice using the 7-neighbourhood.

    Weights shape: (out_ch, in_ch, 7) + bias (out_ch).
    Missing neighbours at lattice boundaries are padded with zeros.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bias: bool = True,
        extent: int = 15,
        neighbour_table: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.extent = extent

        if neighbour_table is None:
            neighbour_table = get_neighbour_table(extent)
        self.register_buffer("neighbour_table", neighbour_table.clone())

        self.weight = nn.Parameter(torch.empty(out_ch, in_ch, 7))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_ch))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and bias with uniform distribution based on fan-in."""
        fan_in = self.in_ch * 7
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply hexagonal convolution over spatial hexals.

        Args:
            x: Tensor of shape (..., in_ch, N).

        Returns:
            Tensor of shape (..., out_ch, N).
        """
        c = x.shape[-2]
        n = x.shape[-1]
        if c != self.in_ch:
            raise ValueError(f"Expected {self.in_ch} input channels, got {c}")
        if n != self.neighbour_table.shape[0]:
            raise ValueError(
                f"Expected {self.neighbour_table.shape[0]} hexals, got {n}"
            )

        # Pad last dimension with zero at index N for boundary hexals
        x_pad = F.pad(x, (0, 1), value=0.0)  # (..., in_ch, N + 1)
        # Gather 7 neighbours for each hexal -> (..., in_ch, N, 7)
        nbrs = x_pad[..., self.neighbour_table]

        # Rearrange to put channel and kernel dimensions at the end:
        # (..., in_ch, N, 7) -> (..., N, in_ch, 7) -> (..., N, in_ch * 7)
        nbrs_flat = nbrs.movedim(-3, -2).flatten(start_dim=-2, end_dim=-1)
        w_flat = self.weight.flatten(start_dim=1)  # (out_ch, in_ch * 7)

        # Linear projection and move channel dimension back before hexals
        out = F.linear(nbrs_flat, w_flat, self.bias)  # (..., N, out_ch)
        return out.movedim(-1, -2)  # (..., out_ch, N)


class HexConvGRUCell(nn.Module):
    """ConvGRU cell on the hexagonal lattice using HexConv for all gates."""

    def __init__(
        self,
        in_ch: int,
        hid_ch: int,
        extent: int = 15,
        neighbour_table: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.hid_ch = hid_ch
        self.extent = extent

        if neighbour_table is None:
            neighbour_table = get_neighbour_table(extent)

        # Joint convolution for reset and update gates
        self.conv_gates = HexConv(
            in_ch + hid_ch,
            2 * hid_ch,
            bias=True,
            extent=extent,
            neighbour_table=neighbour_table,
        )
        # Candidate hidden state convolution
        self.conv_cand = HexConv(
            in_ch + hid_ch,
            hid_ch,
            bias=True,
            extent=extent,
            neighbour_table=neighbour_table,
        )

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Return zero initial hidden state of shape (B, hid_ch, N)."""
        n_hexals = self.conv_gates.neighbour_table.shape[0]
        return torch.zeros(batch_size, self.hid_ch, n_hexals, device=device, dtype=dtype)

    def forward(
        self, x: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Step GRU forward by one time frame.

        Args:
            x: Input tensor of shape (B, in_ch, N).
            h: Previous hidden state of shape (B, hid_ch, N), or None for zero init.

        Returns:
            Updated hidden state of shape (B, hid_ch, N).
        """
        if h is None:
            h = self.init_hidden(x.shape[0], device=x.device, dtype=x.dtype)

        # Concatenate input and previous hidden state along channel dimension
        cat_input = torch.cat([x, h], dim=-2)
        gates = self.conv_gates(cat_input)
        r, z = torch.split(gates, self.hid_ch, dim=-2)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)

        # Candidate state with reset gate applied to h
        cand_input = torch.cat([x, r * h], dim=-2)
        cand = torch.tanh(self.conv_cand(cand_input))

        # Hidden state interpolation
        h_next = (1.0 - z) * h + z * cand
        return h_next


class HexConvGRUNet(nn.Module):
    """Hexagonal ConvGRU video network for multi-task optic flow and depth.

    Architecture:
    1. HexConv encoder stack (n_layers layers).
    2. Temporal ConvGRU cell maintaining hidden state across frames.
    3. Multi-task decoder heads producing flow (2 ch) and depth (1 ch) at every step.
    """

    def __init__(
        self,
        hid_ch: int = 16,
        n_layers: int = 2,
        head_ch: int = 16,
        in_ch: int = 1,
        extent: int = 15,
    ):
        super().__init__()
        self.hid_ch = hid_ch
        self.n_layers = n_layers
        self.head_ch = head_ch
        self.in_ch = in_ch
        self.extent = extent

        table = get_neighbour_table(extent)

        # Encoder: n_layers HexConv layers with ReLU activations
        encoder_layers = []
        curr_in = in_ch
        for _ in range(n_layers):
            encoder_layers.append(
                HexConv(curr_in, hid_ch, bias=True, extent=extent, neighbour_table=table)
            )
            encoder_layers.append(nn.ReLU(inplace=True))
            curr_in = hid_ch
        self.encoder = nn.Sequential(*encoder_layers)

        # Temporal HexConvGRU
        self.gru = HexConvGRUCell(
            hid_ch, hid_ch, extent=extent, neighbour_table=table
        )

        # Multi-task decoder heads
        if head_ch > 0:
            self.flow_head = nn.Sequential(
                HexConv(hid_ch, head_ch, bias=True, extent=extent, neighbour_table=table),
                nn.ReLU(inplace=True),
                HexConv(head_ch, 2, bias=True, extent=extent, neighbour_table=table),
            )
            self.depth_head = nn.Sequential(
                HexConv(hid_ch, head_ch, bias=True, extent=extent, neighbour_table=table),
                nn.ReLU(inplace=True),
                HexConv(head_ch, 1, bias=True, extent=extent, neighbour_table=table),
            )
        else:
            self.flow_head = HexConv(
                hid_ch, 2, bias=True, extent=extent, neighbour_table=table
            )
            self.depth_head = HexConv(
                hid_ch, 1, bias=True, extent=extent, neighbour_table=table
            )

    def forward(
        self,
        lum: Optional[torch.Tensor] = None,
        h_0: Optional[torch.Tensor] = None,
        *,
        x: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process video sequence and return optic flow and depth predictions.

        Args:
            lum: Input luminance sequence of shape (B, T, 1, N) or (B, T, N).
            h_0: Optional initial hidden state for ConvGRU.
            x: Alias for lum if passed as keyword.

        Returns:
            Dict containing:
                "flow": (B, T, 2, N)
                "depth": (B, T, 1, N)
        """
        if lum is None:
            if x is None:
                raise ValueError("Must provide input tensor 'lum' or 'x'.")
            lum = x

        if lum.dim() == 3:
            lum = lum.unsqueeze(2)

        B, T, C, N = lum.shape

        # Run spatial encoder across all frames simultaneously: (B * T, C, N)
        enc_input = lum.reshape(B * T, C, N)
        enc_feats = self.encoder(enc_input)  # (B * T, hid_ch, N)
        enc_feats = enc_feats.view(B, T, self.hid_ch, N)

        # Recurrent temporal processing frame by frame
        h = h_0
        h_seq = []
        for t in range(T):
            h = self.gru(enc_feats[:, t], h)
            h_seq.append(h)
        h_all = torch.stack(h_seq, dim=1)  # (B, T, hid_ch, N)

        # Decoder heads applied across all frames
        h_flat = h_all.reshape(B * T, self.hid_ch, N)
        flow_flat = self.flow_head(h_flat)  # (B * T, 2, N)
        depth_flat = self.depth_head(h_flat)  # (B * T, 1, N)

        flow = flow_flat.view(B, T, 2, N)
        depth = depth_flat.view(B, T, 1, N)

        return {"flow": flow, "depth": depth}


class HexConvGRUNetK(HexConvGRUNet):
    """M8: HexConvGRU-K. Same weights/params as M4; the GRU cell is applied K times per frame
    (tied weights), re-injecting that frame's encoder features at every inner step. K=1 == M4."""

    def __init__(self, *args, k_max: int = 4, z_bias_init: float = -1.0, **kw):
        super().__init__(*args, **kw)
        self.k_max = k_max
        # update-gate bias toward "copy" (small z keeps previous h); value only, no new params
        with torch.no_grad():
            self.gru.conv_gates.bias[self.hid_ch:].fill_(z_bias_init)

    def forward(self, lum=None, h_0=None, *, x=None, k: Optional[int] = None):
        k = self.k_max if k is None else int(k)
        if lum is None:
            if x is None:
                raise ValueError("Must provide input tensor 'lum' or 'x'.")
            lum = x
        if lum.dim() == 3:
            lum = lum.unsqueeze(2)
        B, T, C, N = lum.shape
        feats = self.encoder(lum.reshape(B * T, C, N)).view(B, T, self.hid_ch, N)
        h = h_0
        h_seq = []
        for t in range(T):
            f = feats[:, t]
            for _ in range(k):
                h = self.gru(f, h)
            h_seq.append(h)
        h_flat = torch.stack(h_seq, dim=1).reshape(B * T, self.hid_ch, N)
        return {"flow": self.flow_head(h_flat).view(B, T, 2, N),
                "depth": self.depth_head(h_flat).view(B, T, 1, N)}


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_small() -> HexConvGRUNet:
    """Factory for small HexConvGRUNet (3,000 - 6,000 params, param-matched to connectome)."""
    return HexConvGRUNet(hid_ch=8, n_layers=2, head_ch=8, in_ch=1, extent=15)


def make_small_matched() -> HexConvGRUNet:
    """Factory for param-matched HexConvGRUNet (~15,387 params, within +-5% of M1 connectome model)."""
    return HexConvGRUNet(hid_ch=15, n_layers=2, head_ch=18, in_ch=1, extent=15)


def make_small_matched_k(k_max: int = 4) -> HexConvGRUNetK:
    """M8 factory: M4 architecture (same param count), GRU iterated K<=k_max times per frame."""
    return HexConvGRUNetK(hid_ch=15, n_layers=2, head_ch=18, in_ch=1, extent=15, k_max=k_max)


def make_large() -> HexConvGRUNet:
    """Factory for large HexConvGRUNet (400,000 - 1,000,000 params)."""
    return HexConvGRUNet(hid_ch=96, n_layers=3, head_ch=64, in_ch=1, extent=15)


def make_large_k(k_max: int = 4) -> HexConvGRUNetK:
    """M9 factory: M5 architecture/size (same param count), GRU iterated K<=k_max times per frame."""
    return HexConvGRUNetK(hid_ch=96, n_layers=3, head_ch=64, in_ch=1, extent=15, k_max=k_max)


if __name__ == "__main__":
    small_model = make_small()
    matched_model = make_small_matched()
    large_model = make_large()

    small_p = count_parameters(small_model)
    matched_p = count_parameters(matched_model)
    large_p = count_parameters(large_model)

    print("=== HexConvGRUNet Parameter Counts ===")
    print(f"make_small(): {small_p:,} params (target: 3,000 - 6,000)")
    print(f"make_small_matched(): {matched_p:,} params (target: 15,387 +- 5% -> [14,618, 16,156])")
    print(f"make_large(): {large_p:,} params (target: 400,000 - 1,000,000)")

    assert 3000 <= small_p <= 6000, f"Small model params out of range: {small_p}"
    assert int(15387 * 0.95) <= matched_p <= int(15387 * 1.05) + 1, (
        f"Matched model params out of range: {matched_p}"
    )
    assert 400000 <= large_p <= 1000000, f"Large model params out of range: {large_p}"
    print("All parameter count checks passed!")
