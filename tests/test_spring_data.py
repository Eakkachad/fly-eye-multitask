"""Spring hex-cache tests (held-out test data; no model is evaluated here)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spring_data as sd  # noqa: E402

CACHE = sd.DATA_SPRING_HEX
SEL = json.load(open(Path(sd.COURSE_DIR) / "spring_sequences.json"))["test_selection"]
have_cache = len(list(CACHE.glob("spring_*.npz"))) == 8
need_cache = pytest.mark.skipif(not have_cache, reason="spring_hex cache not built")


def test_selection_rule_reproducible():
    meta = json.load(open(Path(sd.COURSE_DIR) / "spring_sequences.json"))
    seqs = meta["sequences"]
    elig = sorted(k for k, v in seqs.items() if v["n_frames"] >= 40)
    assert elig == SEL["eligible"]
    pick = sorted(np.random.default_rng(20261010).choice(elig, size=8, replace=False).tolist())
    assert pick == SEL["selected"] and len(pick) == 8


def test_render_arrays_shapes_synthetic():
    T, H, W = 3, 436, 775
    g = torch.Generator().manual_seed(0)
    out = sd.render_arrays(torch.rand(T, H, W, generator=g), torch.randn(T, 2, H, W, generator=g) * 1e-3,
                           torch.rand(T, H, W, generator=g) + 1, torch.ones(T, H, W), torch.ones(T, H, W))
    assert out["lum"].shape == (3, T, 1, 721) and out["flow"].shape == (3, T, 2, 721)
    assert out["depth"].shape == (3, T, 1, 721) and out["depth_valid"].all()


@need_cache
def test_cache_shapes_units_and_masks():
    ds = sd.SpringHex(CACHE)
    assert len(ds) == 24
    assert sorted(set(ds.sequence_ids())) == SEL["selected"]
    meta = json.load(open(Path(sd.COURSE_DIR) / "spring_sequences.json"))["sequences"]
    for i in range(len(ds)):
        raw = ds.samples[i]
        n = meta[ds.sequence_ids()[i]]["n_frames"] - 1
        assert raw["lum"].shape == (n, 1, 721) and raw["flow"].shape == (n, 2, 721)
        assert raw["depth"].shape == (n, 1, 721)
        assert raw["lum"].min() >= 0 and raw["lum"].max() <= 1.0 + 1e-6
        # flow in image heights per frame; Sintel hex-sum flow is O(1-100), here 169x px/h
        assert torch.isfinite(raw["flow"]).all() and torch.isfinite(raw["depth"]).all()
        assert raw["flow"].abs().max() < 169 * 0.5  # < half an image height per frame
        item = ds.get_item(i)
        T2 = int(np.ceil(50 / 24 * n))
        assert item["lum"].shape == (T2, 1, 721) and item["flow"].shape == (T2, 2, 721)
        assert item["depth_valid"].shape == (T2, 1, 721) and item["depth_valid"].dtype == torch.bool


@need_cache
def test_mask_fraction_and_depth_transform():
    ds = sd.SpringHex(CACHE)
    v = torch.cat([s["depth_valid"].flatten() for s in ds.samples]).float()
    assert 0.5 < v.mean() <= 1.0, v.mean()  # most hexals have valid depth
    d = torch.cat([s["depth"][s["depth_valid"].bool()] for s in ds.samples])
    assert (d > 0).all()
    t = ds.depth_transform(d)
    assert torch.isfinite(t).all()


@need_cache
def test_determinism_cache_reload():
    a, b = sd.SpringHex(CACHE), sd.SpringHex(CACHE)
    for i in (0, 7, 23):
        for k, v in a.get_item(i).items():
            assert torch.equal(v, b.get_item(i)[k])
