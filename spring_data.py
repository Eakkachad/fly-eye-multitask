"""Spring (Mehl et al., CVPR 2023, CC BY 4.0, DOI 10.18419/darus-3376) on the fly-eye
hex lattice, rendered EXACTLY like our Sintel data (flyvis ``RenderedSintel`` +
``MultiTaskSintel`` evaluation path).  Held-out TEST data only: this module never
trains or evaluates anything.

The generic renderer ``render_arrays`` is the flyvis recipe (split -> BoxEye
mean / sum / median) applied to cartesian arrays.  Two loaders feed it:

* ``load_sintel_sequence``  -- uses flyvis' own sample_lum/flow/depth, no resizing
  (used to validate that this module reproduces data.py's Sintel tensors).
* ``load_spring_sequence``  -- Spring files, resized to the Sintel frame height.

Conventions copied from flyvis (RenderedSintel):
  * lum / depth use frames 1..N-1, flow uses files 0..N-2 (flow k = frame k -> k+1,
    paired with lum[k] = frame k+1)  => N-1 samples per sequence, native frame rate.
  * lum in [0,1] (L channel / 255); flow in units of image-heights per frame with
    the y axis flipped (up = positive); depth raw (metric), no clipping here.
  * center crop 0.7 of the width, 3 overlapping vertical splits, BoxEye(extent=15,
    kernel_size=13): lum 'mean', flow 'sum' (sic, as flyvis), depth 'median'.
  * temporal handling (get_item): lum piecewise-constant resampled, flow/depth
    linearly interpolated from the native rate to 1/dt = 50 Hz (Interpolate with
    original_framerate = 24 for Sintel).  GT flow stays "per original frame interval"
    in image-heights/frame in both pipelines, i.e. it is NOT rescaled by the
    resampling (flyvis does not rescale it); we copy that behaviour.

Spring assumptions (official docs are NOT inside the zips; verified empirically):
  * frames 1920x1080 RGB PNG; flow (.flo5, key 'flow') and disparity (.dsp5, key
    'disparity') are float16 HDF5 arrays at 2x resolution (2160x3840).
  * FLOW VALUES ARE IN 1920x1080 PIXEL UNITS even though stored on the 2x grid
    (verified by photometric warping: error minimal for scale 1.0 vs 0.5/2.0).
    -> flow / 1080 = image-heights per frame; no value rescaling.
  * DISPARITY VALUES ARE IN 3840x2160 (2x-grid) PIXEL UNITS: static-scene check
    |flow|/disp 99th pct = 0.617 ~= |dt_cam|/(2*B) with |dt_cam|=0.0798 (extrinsics),
    B=0.065.  Hence disparity_1080 = disp/2 and
        depth = f * B / disparity_1080 = 2 f B / disp,
    f = fx from cam_data/intrinsics.txt (per frame, 1920x1080 pixels, cx=960,cy=540),
    B = 0.065 m (owner-specified).  Depth is in metres under that assumption.
  * invalid = non-finite or <= 0 disparity/flow (stored as NaN in the files).
  * frame rate 24 fps (Blender Studio 'Spring' movie; assumption, not in the zips).
  * spatial: frames/GT are resized to the Sintel frame height (436) so one BoxEye
    pixel subtends the same fraction of image height as for Sintel (width 775 vs
    1024; aspect 16:9 vs 2.35:1).  lum/flow: area averaging; depth/validity:
    nearest-exact decimation (no mixing across depth edges).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

COURSE_DIR = Path(__file__).resolve().parent
DATA_SINTEL = Path.home() / "flyproj/data/sintel"
DATA_SPRING_RAW = Path.home() / "flyproj/data/spring_raw"
DATA_SPRING_HEX = Path.home() / "flyproj/data/spring_hex"

SINTEL_HEIGHT = 436
SPRING_FPS = 24
SPRING_BASELINE_M = 0.065
DT = 0.02
BOXFILTER = dict(extent=15, kernel_size=13)
VERTICAL_SPLITS = 3
CENTER_CROP = 0.7
VALID_THRESHOLD = 0.5  # hexal is valid if >= half of its 13x13 kernel is valid
DEPTH_FILL = 2522.0753222656253  # depth_norm.json 'hi'; fills invalid px before the median
CHUNK = 6


# -- generic renderer (flyvis recipe) ------------------------------------------------

def _hex(x: torch.Tensor, boxeye, ftype: str, chunk: int = CHUNK) -> torch.Tensor:
    """x: (T,H,W) cartesian -> (splits, T, 1, hexals), as RenderedSintel."""
    from flyvis.datasets.rendering.utils import split

    xs = split(x, boxeye.min_frame_size[1] + 2 * boxeye.kernel_size, VERTICAL_SPLITS, CENTER_CROP)
    outs = [boxeye(xs[:, a:a + chunk].clone(), ftype=ftype).cpu() for a in range(0, xs.shape[1], chunk)]
    return torch.cat(outs, dim=1)


def render_arrays(lum, flow, depth, depth_valid=None, flow_valid=None) -> Dict[str, torch.Tensor]:
    """lum (T,H,W), flow (T,2,H,W) [image-heights/frame, y flipped], depth (T,H,W),
    valid masks (T,H,W) in {0,1}.  Returns tensors (splits, T, C, hexals)."""
    from flyvis.datasets.rendering import BoxEye

    boxeye = BoxEye(**BOXFILTER)
    out = {"lum": _hex(lum, boxeye, "mean")}
    out["flow"] = torch.cat([_hex(flow[:, 0], boxeye, "sum"), _hex(flow[:, 1], boxeye, "sum")], dim=2)
    if depth_valid is None:
        out["depth"] = _hex(depth, boxeye, "median")
        out["depth_valid"] = torch.ones_like(out["depth"], dtype=torch.bool)
    else:
        filled = torch.where(depth_valid > 0, depth, torch.full_like(depth, DEPTH_FILL))
        out["depth"] = _hex(filled, boxeye, "median")
        out["depth_valid"] = _hex(depth_valid, boxeye, "mean") >= VALID_THRESHOLD
    if flow_valid is None:
        out["flow_valid"] = torch.ones_like(out["depth"], dtype=torch.bool)
    else:
        out["flow_valid"] = _hex(flow_valid, boxeye, "mean") >= VALID_THRESHOLD
    return out


# -- loaders -----------------------------------------------------------------------

def load_sintel_sequence(name: str, root: Path = DATA_SINTEL):
    """flyvis' own file readers: lum/depth from frame 1, flow from file 0."""
    from flyvis.datasets.sintel_utils import load_sequence, sample_depth, sample_flow, sample_lum

    root = Path(root)
    lum = load_sequence(root / "training/final" / name, sample_lum, start=1)
    flow = load_sequence(root / "training/flow" / name, sample_flow)
    depth = load_sequence(root / "training/depth" / name, sample_depth, start=1)
    return dict(lum=lum, flow=flow, depth=depth)


def _area(x: torch.Tensor, size) -> torch.Tensor:
    return x if tuple(x.shape[-2:]) == tuple(size) else torch.nn.functional.interpolate(x, size=size, mode="area")


def _nearest(x: torch.Tensor, size) -> torch.Tensor:
    return x if tuple(x.shape[-2:]) == tuple(size) else torch.nn.functional.interpolate(x, size=size, mode="nearest-exact")


def spring_file_lists(seq: str, raw: Path = DATA_SPRING_RAW):
    base = Path(raw) / "spring/train" / seq
    fr = sorted((base / "frame_left").glob("frame_left_*.png"))
    fl = sorted((base / "flow_FW_left").glob("flow_FW_left_*.flo5"))
    dp = sorted((base / "disp1_left").glob("disp1_left_*.dsp5"))
    assert len(fr) == len(dp) == len(fl) + 1, (len(fr), len(fl), len(dp))
    return base, fr, fl, dp


def load_spring_sequence(seq: str, raw: Path = DATA_SPRING_RAW, target_height: int = SINTEL_HEIGHT,
                         baseline: float = SPRING_BASELINE_M):
    import h5py
    from PIL import Image

    base, fr, fl, dp = spring_file_lists(seq, raw)
    intr = np.loadtxt(base / "cam_data/intrinsics.txt")  # (N, 4): fx fy cx cy, 1920x1080 px
    assert intr.shape[0] == len(fr)
    W0, H0 = Image.open(fr[0]).size
    assert (W0, H0) == (1920, 1080)
    size = (target_height, int(round(W0 * target_height / H0)))
    lum, flow, fvalid, depth, dvalid = [], [], [], [], []
    for k in range(len(fl)):  # sample k <-> lum/depth frame k+1, flow file k (as Sintel)
        im = np.float32(Image.open(fr[k + 1]).convert("L")) / 255
        lum.append(_area(torch.from_numpy(im)[None, None], size)[0, 0])
        with h5py.File(fl[k], "r") as h:
            f = torch.from_numpy(h["flow"][()].astype(np.float32))  # (2160,3840,2), 1080p-pixel units
        ok = torch.isfinite(f).all(-1)
        f = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).permute(2, 0, 1) / H0  # image-heights
        f = f * torch.tensor([1.0, -1.0])[:, None, None]  # y up, as sample_flow
        flow.append(_area(f[None], size)[0])
        fvalid.append(_area(ok[None, None].float(), size)[0, 0])
        with h5py.File(dp[k + 1], "r") as h:
            d = torch.from_numpy(h["disparity"][()].astype(np.float32))  # (2160,3840), 2x-grid px
        ok = torch.isfinite(d) & (d > 0)
        z = torch.where(ok, 2.0 * float(intr[k + 1, 0]) * baseline / d.clamp_min(1e-6),
                        torch.zeros_like(d))  # = f*B/(d/2)
        depth.append(_nearest(z[None, None], size)[0, 0])
        dvalid.append(_nearest(ok[None, None].float(), size)[0, 0])
    st = torch.stack
    return dict(lum=st(lum), flow=st(flow), depth=st(depth), depth_valid=st(dvalid), flow_valid=st(fvalid))


def render_sintel(name: str, root: Path = DATA_SINTEL):
    return render_arrays(**load_sintel_sequence(name, root))


def render_spring(seq: str, raw: Path = DATA_SPRING_RAW):
    return render_arrays(**load_spring_sequence(seq, raw))


# -- hex cache + dataset ---------------------------------------------------------------

def build_cache(seqs: List[str], raw: Path = DATA_SPRING_RAW, out: Path = DATA_SPRING_HEX) -> List[Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in seqs:
        r = render_spring(s, raw)
        arrs = {k: (v.numpy().astype(np.uint8) if v.dtype == torch.bool else v.numpy().astype(np.float32))
                for k, v in r.items()}
        p = out / f"spring_{s}.npz"
        np.savez_compressed(p, **arrs)
        paths.append(p)
    return paths


class SpringHex:
    """Pre-rendered Spring test samples; one sample per (sequence, vertical split).

    ``get_item`` mirrors MultiTaskSintel.get_item with augment=False, all_frames=True:
    lum piecewise-constant, flow/depth linear, to 1/dt Hz; masks are conservative
    (valid only if valid at both neighbouring native frames).  ``depth`` is raw metric
    depth; use ``depth_transform`` (data.DepthTransform from train stats) for targets.
    """

    def __init__(self, cache: Path = DATA_SPRING_HEX, dt: float = DT, fps: float = SPRING_FPS,
                 seqs: Optional[List[str]] = None):
        from flyvis.datasets.augmentation.temporal import Interpolate

        cache = Path(cache)
        files = sorted(cache.glob("spring_*.npz"))
        if seqs is not None:
            files = [f for f in files if f.stem.split("_")[1] in seqs]
        self.names, self.samples = [], []
        for f in files:
            z = np.load(f)
            for j in range(z["lum"].shape[0]):
                self.names.append(f"{f.stem}_split_{j:02d}")
                self.samples.append({k: torch.from_numpy(z[k][j]) for k in z.files})
        self.piecewise = Interpolate(fps, 1 / dt, mode="nearest-exact")
        self.linear = Interpolate(fps, 1 / dt, mode="linear")
        from data import DepthTransform
        self.depth_transform = DepthTransform.from_file()

    def __len__(self):
        return len(self.samples)

    def get_item(self, i: int) -> Dict[str, torch.Tensor]:
        s = self.samples[i]
        out = {"lum": self.piecewise(s["lum"]), "flow": self.linear(s["flow"]),
               "depth": self.linear(s["depth"])}
        for k in ("depth_valid", "flow_valid"):
            out[k] = self.linear(s[k].float()) >= 0.999
        return out

    __getitem__ = get_item

    def sequence_ids(self) -> List[str]:
        return [n.split("_")[1] for n in self.names]
