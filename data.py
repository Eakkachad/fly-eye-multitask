"""Sintel on the fly-eye hex lattice, restricted to our pre-registered scene splits.

Uses flyvis' own rendering (``RenderedSintel``) and augmentation code
(``MultiTaskSintel``) but replaces flyvis' fold / original-split logic: each
``SplitSintel`` instance only ever contains the sequences of the scenes passed
to it, so a training loader physically cannot see val/test scenes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

COURSE_DIR = Path(__file__).resolve().parent
DATA_SINTEL = Path.home() / "flyproj/data/sintel"

# flyvis.datasets.MultiTaskSintel settings from flyvis config/task/task.yaml
FLYVIS_TASK_DEFAULTS = dict(
    boxfilter=dict(extent=15, kernel_size=13),
    vertical_splits=3,
    n_frames=19,
    center_crop_fraction=0.7,
    dt=0.02,
    random_temporal_crop=True,
    all_frames=False,
    resampling=True,
    interpolate=True,
    p_flip=0.5,
    p_rot=0.5,
    contrast_std=0.2,
    brightness_std=0.1,
    gaussian_white_noise=0.08,
    gamma_std=None,
    flip_axes=[0, 1, 2, 3],
)

_NAME = re.compile(r"^sequence_\d+_(?P<scene>.+)_split_\d+$")


def scene_of(name: str) -> str:
    m = _NAME.match(name)
    if not m:
        raise ValueError(name)
    return m.group("scene")


def ensure_sintel_link() -> Path:
    """Make flyvis.sintel_dir a symlink to our extracted copy (no copying)."""
    import flyvis

    target = Path(flyvis.sintel_dir)
    for sub in ("training/final", "training/flow", "training/depth", "test"):
        if not (DATA_SINTEL / sub).exists():
            raise FileNotFoundError(f"{DATA_SINTEL / sub} missing (download done?)")
    if target.is_symlink():
        if target.resolve() != DATA_SINTEL.resolve():
            raise RuntimeError(f"{target} points elsewhere: {target.resolve()}")
    elif target.exists():
        if any(target.iterdir()):
            raise RuntimeError(f"{target} exists and is not our symlink")
        target.rmdir()
        os.symlink(DATA_SINTEL, target)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(DATA_SINTEL, target)
    return target


def make_split_sintel_class():
    from flyvis.datasets.sintel import MultiTaskSintel

    class SplitSintel(MultiTaskSintel):
        """MultiTaskSintel holding only the sequences of ``scenes``."""

        def __init__(self, scenes: Sequence[str], **kwargs):
            kwargs = {**FLYVIS_TASK_DEFAULTS, **kwargs}
            kwargs.setdefault("tasks", ["flow", "depth"])
            super().__init__(_init_cache=False, sintel_path=ensure_sintel_link(),
                             **kwargs)
            scenes = list(scenes)
            all_scenes = {scene_of(n) for n in self.arg_df.name}
            missing = set(scenes) - all_scenes
            if missing:
                raise ValueError(f"unknown scenes {missing}")
            keep = self.arg_df.name.map(scene_of).isin(scenes).values
            object.__setattr__(self, "arg_df", self.arg_df[keep].reset_index(drop=True))
            object.__setattr__(self, "scenes", sorted(scenes))
            self.init_cache()

        def init_cache(self) -> None:
            # arg_df["index"] indexes into the sorted rendered sequences
            object.__setattr__(self, "cached_sequences", [
                {
                    k: torch.tensor(v, dtype=torch.float32)
                    for k, v in self.rendered(int(i)).items()
                    if k in self.data_keys
                }
                for i in self.arg_df["index"].values
            ])
            for cached, name in zip(self.cached_sequences, self.arg_df.name):
                assert scene_of(name) in self.scenes

        def sequence_scenes(self) -> List[str]:
            return [scene_of(n) for n in self.arg_df.name]

    return SplitSintel


def make_datasets(splits: dict, train_scenes: Sequence[str], which=("train", "val")):
    """Train set: flyvis augmentation + random 19-frame crops.
    Val / test sets: no augmentation, all frames of every sequence."""
    SplitSintel = make_split_sintel_class()
    out = {}
    for w in which:
        if w == "train":
            out[w] = SplitSintel(train_scenes, augment=True)
        else:
            out[w] = SplitSintel(splits[w], augment=False, all_frames=True,
                                 random_temporal_crop=False)
    return out


def check_disjoint(splits: dict) -> None:
    sets = {k: set(splits[k]) for k in ("train", "val", "test")}
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise AssertionError(f"{a} and {b} share {sets[a] & sets[b]}")
    for sub in splits["train_fraction_subsets"].values():
        assert set(sub) <= sets["train"]


# -- depth target transform --------------------------------------------------------

DEPTH_NORM_FILE = COURSE_DIR / "depth_norm.json"


class DepthTransform:
    """Maps rendered Sintel depth to the regression target and back.

    kind 'log_std': target = (log(clip(depth, lo, hi)) - mean) / std, with
    lo/hi/mean/std computed on TRAIN scenes only (see compute_depth_norm).
    kind 'identity': raw depth.
    Metrics: depth RMSE is reported in target (normalised) units and abs-rel in
    clipped metric depth (see eval.py / train.evaluate).
    """

    def __init__(self, kind="identity", lo=None, hi=None, mean=0.0, std=1.0):
        self.kind, self.lo, self.hi, self.mean, self.std = kind, lo, hi, mean, std

    def __call__(self, d):
        if self.kind == "identity":
            return d
        return (torch.log(d.clamp(self.lo, self.hi)) - self.mean) / self.std

    def inverse(self, t):
        if self.kind == "identity":
            return t
        return torch.exp(t * self.std + self.mean)

    def clip(self, d):
        return d if self.kind == "identity" else d.clamp(self.lo, self.hi)

    def describe(self):
        return dict(kind=self.kind, lo=self.lo, hi=self.hi, mean=self.mean, std=self.std)

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

    @classmethod
    def from_file(cls, path=DEPTH_NORM_FILE):
        return cls(**json.loads(Path(path).read_text()))


def compute_depth_norm(splits: dict, q_lo=0.001, q_hi=0.999) -> dict:
    """Clip range and log-depth mean/std from the TRAIN scenes (all frames)."""
    SplitSintel = make_split_sintel_class()
    ds = SplitSintel(splits["train"], augment=False, all_frames=True,
                     random_temporal_crop=False)
    d = torch.cat([s["depth"].flatten() for s in ds.cached_sequences]).double().cpu()
    d = d[torch.isfinite(d)]
    sub = d[torch.randperm(len(d), device="cpu", generator=torch.Generator().manual_seed(0))[:2_000_000]]
    lo, hi = (float(x) for x in torch.quantile(sub, torch.tensor([q_lo, q_hi], device="cpu",
                                                                  dtype=sub.dtype)))
    ld = torch.log(d.clamp(lo, hi))
    out = dict(kind="log_std", lo=lo, hi=hi, mean=float(ld.mean()), std=float(ld.std()))
    return out
