"""Model zoo with one contract: lum (B, T, 1, 721) -> {"flow": (B,T,2,721), "depth": (B,T,1,721)}.

M1-M3: flyvis ``Network`` (connectome-constrained dynamical model, flyvis default
config) with real / rewired / random type graph + two flyvis ``DecoderGAVP``
heads (flyvis default decoder config, shape [8, 2] for flow, [8, 1] for depth).
M4/M5: baselines.hex_models.make_small / make_large (written by another worker),
imported if present.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch import nn

COURSE_DIR = Path(__file__).resolve().parent

FLYVIS_MODELS = ("m1", "m2", "m3")
GENERIC_MODELS = ("m4", "m5", "m8")
HYBRID_MODELS = ("m6", "m7")
FROZEN_HYBRID_MODELS = ("m6f", "m7f")
_HYBRID_CONNECTOME = {"m6": "m1", "m7": "m2", "m6f": "m1", "m7f": "m2"}
# Trunk size: total trainable (network + trunk) must be 15,387 +- 3 % (see HybridMultiTask).
HYBRID_HID_CH, HYBRID_HEAD_CH, HYBRID_LAYERS = 14, 14, 1  # total 15,423 (net 734 + trunk 14,689)

DECODER_DEFAULTS = dict(
    type="DecoderGAVP", kernel_size=5, const_weight=0.001, n_out_features=None,
    p_dropout=0.5,
)


def _build_network(model: str, seed: int, null_seed: Optional[int]):
    """flyvis Network for m1/m2/m3 (shared by FlyvisMultiTask and the hybrids)."""
    from datamate import Namespace
    from flyvis import Network

    import nulls

    null_seed = seed if null_seed is None else null_seed
    conn = nulls.connectome_config(model, null_seed)
    node_config = Namespace(
        bias=Namespace(type="RestingPotential", groupby=["type"],
                       initial_dist="Normal", mode="sample", requires_grad=True,
                       mean=0.5, std=0.05, penalize=Namespace(activity=True),
                       seed=seed),
        time_const=Namespace(type="TimeConstant", groupby=["type"],
                             initial_dist="Value", value=0.05, requires_grad=True),
    )
    return Network(connectome=Namespace(**conn), node_config=node_config), conn["file"]


class FlyvisMultiTask(nn.Module):
    """flyvis Network + per-task DecoderGAVP, behind the common contract."""

    is_flyvis = True

    def __init__(self, model: str = "m1", seed: int = 0, null_seed: Optional[int] = None,
                 dt: float = 0.02, t_pre: float = 0.5):
        super().__init__()
        from flyvis.task.decoder import DecoderGAVP

        self.network, conn_file = _build_network(model, seed, null_seed)
        cfg = {k: v for k, v in DECODER_DEFAULTS.items() if k != "type"}
        self.decoder = nn.ModuleDict({
            "flow": DecoderGAVP(self.network.connectome, shape=[8, 2], **cfg),
            "depth": DecoderGAVP(self.network.connectome, shape=[8, 1], **cfg),
        })
        self.dt, self.t_pre = dt, t_pre
        self.connectome_file = conn_file
        self._ss = None
        self._ss_batch = None

    def refresh_steady_state(self, batch_size: int) -> None:
        """Grey-screen steady state (flyvis recomputes it once per epoch)."""
        with torch.no_grad():
            self._ss = self.network.steady_state(
                t_pre=self.t_pre, dt=self.dt, batch_size=batch_size, value=0.5)
        self._ss_batch = batch_size

    def forward(self, lum: torch.Tensor, return_activity: bool = False):
        B, T = lum.shape[:2]
        if self._ss is None or self._ss_batch != B or not self.training:
            self.refresh_steady_state(B)
        net = self.network
        net.stimulus.zero(B, T)
        net.stimulus.add_input(lum)
        act = net(net.stimulus(), self.dt, state=self._ss)
        out = {k: d(act) for k, d in self.decoder.items()}
        if return_activity:
            out["activity"] = act
        return out

    def param_groups(self, lr_net: float, lr_dec: float):
        return [dict(params=list(self.network.parameters()), lr=lr_net, name="net"),
                dict(params=list(self.decoder.parameters()), lr=lr_dec, name="dec")]


class HybridMultiTask(nn.Module):
    """Connectome front-end + HexConvGRU trunk (PLAN amendments A5 / A6.2).

    m6/m7  : flyvis network (real / M2-rewired connectome) trained jointly with the
             trunk; relu(output-cell-type activity) (B,T,C,721) -> HexConvGRUNet.
    m6f/m7f: front-end = trained M1/M2 checkpoint (network AND decoders frozen,
             eval mode, no activity penalty); output = frozen decoder output +
             trunk output, final layers of both trunk heads zero-initialised so that
             at step 0 output == frozen M1/M2 output.  ``forward(.., residual=False)``
             returns the front-end-only prediction.
    Trainable params: m6/m7 network + trunk; m6f/m7f trunk only.
    """

    is_flyvis = True

    def __init__(self, model: str = "m6", seed: int = 0, null_seed: Optional[int] = None,
                 dt: float = 0.02, t_pre: float = 0.5, hid_ch: int = HYBRID_HID_CH,
                 head_ch: int = HYBRID_HEAD_CH, n_layers: int = HYBRID_LAYERS,
                 frontend_ckpt: Optional[str] = None):
        super().__init__()
        from flyvis.utils.activity_utils import LayerActivity

        model = model.lower()
        self.frozen = model in FROZEN_HYBRID_MODELS
        self.use_penalty = not self.frozen  # nothing in the network trains when frozen
        base = _HYBRID_CONNECTOME[model]
        self.network, self.connectome_file = _build_network(base, seed, null_seed)
        self.dt, self.t_pre = dt, t_pre
        self.dvs_channels = LayerActivity(None, self.network.connectome, use_central=False)
        C = len(self.network.connectome.output_cell_types)
        self.trunk = hex_models().HexConvGRUNet(hid_ch=hid_ch, n_layers=n_layers,
                                               head_ch=head_ch, in_ch=C)
        self._ss = None
        self._ss_batch = None
        if self.frozen:
            from flyvis.task.decoder import DecoderGAVP

            cfg = {k: v for k, v in DECODER_DEFAULTS.items() if k != "type"}
            self.decoder = nn.ModuleDict({
                "flow": DecoderGAVP(self.network.connectome, shape=[8, 2], **cfg),
                "depth": DecoderGAVP(self.network.connectome, shape=[8, 1], **cfg),
            })
            self.frontend_ckpt = frontend_ckpt
            if frontend_ckpt is not None:
                ck = torch.load(frontend_ckpt, map_location="cpu", weights_only=False)
                sd = {k: v for k, v in ck["model"].items()
                      if k.startswith(("network.", "decoder."))}
                res = self.load_state_dict(sd, strict=False)
                bad = [k for k in res.unexpected_keys] + [
                    k for k in res.missing_keys if k.startswith(("network.", "decoder."))]
                if bad:
                    raise RuntimeError(f"front-end checkpoint mismatch: {bad[:5]}")
            for m in (self.network, self.decoder):
                for p in m.parameters():
                    p.requires_grad_(False)
            # zero-init the final layer of both heads: output == front-end at step 0
            for head in (self.trunk.flow_head, self.trunk.depth_head):
                last = head[-1] if isinstance(head, nn.Sequential) else head
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            self.network.eval()
            self.decoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:  # frozen front-end stays in eval (BatchNorm/Dropout)
            self.network.eval()
            self.decoder.eval()
        return self

    def refresh_steady_state(self, batch_size: int) -> None:
        with torch.no_grad():
            self._ss = self.network.steady_state(
                t_pre=self.t_pre, dt=self.dt, batch_size=batch_size, value=0.5)
        self._ss_batch = batch_size

    def forward(self, lum: torch.Tensor, return_activity: bool = False,
                residual: bool = True):
        B, T = lum.shape[:2]
        if self._ss is None or self._ss_batch != B or not self.training:
            self.refresh_steady_state(B)
        net = self.network
        net.stimulus.zero(B, T)
        net.stimulus.add_input(lum)
        act = net(net.stimulus(), self.dt, state=self._ss)
        if self.frozen:
            out = {k: d(act) for k, d in self.decoder.items()}
            if residual:
                self.dvs_channels.update(act)
                res = self.trunk(torch.relu(self.dvs_channels.output))
                out = {k: out[k] + res[k] for k in out}
        else:
            self.dvs_channels.update(act)
            out = self.trunk(torch.relu(self.dvs_channels.output))
        if return_activity:
            out["activity"] = act
        return out

    def param_groups(self, lr_net: float, lr_dec: float):
        groups = []
        if not self.frozen:
            groups.append(dict(params=list(self.network.parameters()), lr=lr_net,
                               name="net"))
        groups.append(dict(params=list(self.trunk.parameters()), lr=lr_dec, name="dec"))
        return groups


def hex_models():
    path = COURSE_DIR / "baselines" / "hex_models.py"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (baselines not written yet)")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    import hex_models as hm  # type: ignore

    return hm


def build_model(model: str, seed: int = 0, null_seed: Optional[int] = None,
                frontend_ckpt: Optional[str] = None) -> nn.Module:
    model = model.lower()
    if model in HYBRID_MODELS or model in FROZEN_HYBRID_MODELS:
        return HybridMultiTask(model, seed=seed, null_seed=null_seed,
                               frontend_ckpt=frontend_ckpt)
    if model in FLYVIS_MODELS:
        return FlyvisMultiTask(model, seed=seed, null_seed=null_seed)
    if model in GENERIC_MODELS:
        path = COURSE_DIR / "baselines" / "hex_models.py"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (baselines not written yet)")
        sys.path.insert(0, str(path.parent))
        import hex_models  # type: ignore

        torch.manual_seed(seed)
        # M4 is parameter-matched to M1 *including* decoders (15,387): make_small_matched (15,402).
        return {"m4": hex_models.make_small_matched, "m5": hex_models.make_large,
                "m8": hex_models.make_small_matched_k}[model]()
    raise ValueError(f"unknown model {model}")


def n_trainable(module: nn.Module) -> Dict[str, int]:
    out = {"total": sum(p.numel() for p in module.parameters() if p.requires_grad)}
    if isinstance(module, HybridMultiTask):
        out["network"] = sum(p.numel() for p in module.network.parameters()
                              if p.requires_grad)
        out["trunk"] = sum(p.numel() for p in module.trunk.parameters()
                           if p.requires_grad)
    if isinstance(module, FlyvisMultiTask):
        out["network"] = sum(p.numel() for p in module.network.parameters()
                             if p.requires_grad)
        out["decoders"] = sum(p.numel() for p in module.decoder.parameters()
                              if p.requires_grad)
    return out
