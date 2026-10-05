import json
import sys
from pathlib import Path

import pytest

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
import data as D  # noqa: E402
import splits as S  # noqa: E402


@pytest.fixture(scope="module")
def sp():
    return S.load_splits()


def test_disjoint_and_complete(sp):
    tr, va, te = map(set, (sp["train"], sp["val"], sp["test"]))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == 23 == len(sp["frames"])
    assert (len(tr), len(va), len(te)) == (15, 4, 4)
    D.check_disjoint(sp)


def test_no_family_crosses_splits(sp):
    where = {}
    for k in ("train", "val", "test"):
        for s in sp[k]:
            where.setdefault(S.family(s), set()).add(k)
    assert all(len(v) == 1 for v in where.values()), where


def test_subsets(sp):
    assert set(sp["train_fraction_subsets"]["0.25"]) < set(sp["train"])
    assert S.train_scenes(sp, 1.0) == sp["train"]
    assert S.train_scenes(sp, 0.25) == sp["train_fraction_subsets"]["0.25"]


def test_reproducible(sp):
    again = S.make_splits(sp["frames"], seed=sp["seed"])
    for k in ("train", "val", "test", "train_fraction_subsets"):
        assert again[k] == sp[k]


@pytest.mark.skipif(not (D.DATA_SINTEL / "training/depth").exists(),
                    reason="Sintel not extracted")
def test_datasets_only_contain_their_scenes(sp):
    """The flyvis-based datasets hold exactly the sequences of their split."""
    ds = D.make_datasets(sp, sp["train"], which=("train", "val", "test"))
    for k in ("train", "val", "test"):
        seen = set(ds[k].sequence_scenes())
        assert seen == set(sp[k]), (k, seen)
        assert len(ds[k]) == len(ds[k].cached_sequences) == 3 * len(sp[k])
        # rendered names agree with the cached data length
        assert len(ds[k].arg_df) == len(ds[k])
    assert not ds["val"].augment and not ds["test"].augment and ds["train"].augment
    # samples have the contract shapes
    x = ds["train"][0]
    assert x["lum"].shape[1:] == (1, 721) and x["flow"].shape[1:] == (2, 721)
    assert x["depth"].shape[1:] == (1, 721)
    assert x["lum"].shape[0] == x["flow"].shape[0] == x["depth"].shape[0]
