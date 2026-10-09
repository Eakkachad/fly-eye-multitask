"""Scene-level train/val/test split of the 23 MPI Sintel training scenes.

Scenes are grouped by *family* (the name before the trailing number, e.g.
``ambush_2`` and ``ambush_4`` both belong to ``ambush``): scenes of one family
share characters, environment and textures, so we never put two scenes of one
family into different splits (stricter than a plain per-scene split).

Procedure (deterministic, uses only scene names and frame counts, no labels or
model results):
  1. enumerate every assignment of families to test / val such that test and
     val each contain exactly N_EVAL (=4) scenes;
  2. score = |frames(test) - target| + |frames(val) - target| with
     target = N_EVAL / 23 * total frames;
  3. among the assignments with the minimal score, pick one with
     ``numpy.random.default_rng(SEED)``.
The 25 % training subset (data-efficiency arm) is chosen the same way from
the training scenes (round(0.25 * 15) = 4 scenes, frame-balanced to 25 % of
train frames, ties broken with the same seed; families may be split here since
this only subsamples the training set).

Usage: python splits.py  (writes splits.json next to this file)
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
SPLITS_FILE = COURSE_DIR / "splits.json"
SINTEL_DIR = Path.home() / "flyproj/data/sintel"
SEED = 20261005
N_EVAL = 4


def family(scene: str) -> str:
    return re.sub(r"_\d+$", "", scene)


def scene_frames(sintel_dir: Path = SINTEL_DIR) -> Dict[str, int]:
    final = sintel_dir / "training" / "final"
    return {p.name: len(list(p.glob("*.png"))) for p in sorted(final.iterdir())}


def make_splits(frames: Dict[str, int], seed: int = SEED) -> dict:
    scenes = sorted(frames)
    fams: Dict[str, List[str]] = {}
    for s in scenes:
        fams.setdefault(family(s), []).append(s)
    names = sorted(fams)
    total = sum(frames.values())
    target = N_EVAL / len(scenes) * total

    def nsc(group):
        return sum(len(fams[f]) for f in group)

    def nfr(group):
        return sum(frames[s] for f in group for s in fams[f])

    candidates = []
    subsets = [
        c for r in range(1, len(names) + 1) for c in itertools.combinations(names, r)
        if nsc(c) == N_EVAL
    ]
    for test in subsets:
        for val in subsets:
            if set(test) & set(val):
                continue
            score = abs(nfr(test) - target) + abs(nfr(val) - target)
            candidates.append((score, test, val))
    best = min(c[0] for c in candidates)
    tied = sorted([c for c in candidates if c[0] == best], key=lambda c: (c[1], c[2]))
    rng = np.random.default_rng(seed)
    _, test_f, val_f = tied[rng.integers(len(tied))]
    test = sorted(s for f in test_f for s in fams[f])
    val = sorted(s for f in val_f for s in fams[f])
    train = sorted(s for s in scenes if s not in test and s not in val)

    # 25% training subset (by scene, frame-balanced)
    k = int(round(0.25 * len(train)))
    tgt = 0.25 * sum(frames[s] for s in train)
    combos = list(itertools.combinations(train, k))
    sc = [abs(sum(frames[s] for s in c) - tgt) for c in combos]
    tied25 = [c for c, v in zip(combos, sc) if v == min(sc)]
    train_25 = sorted(tied25[rng.integers(len(tied25))])

    return dict(
        description=(
            "MPI Sintel training scenes split by scene family; test scenes are "
            "never used for any choice. See splits.py for the procedure."
        ),
        seed=seed,
        n_candidates_tied=len(tied),
        train=train,
        val=val,
        test=test,
        train_fraction_subsets={"0.25": train_25, "1.0": train},
        frames={s: frames[s] for s in scenes},
        frames_per_split={
            k_: sum(frames[s] for s in v)
            for k_, v in dict(train=train, val=val, test=test, train_25=train_25).items()
        },
        families={f: fams[f] for f in names},
    )


def load_splits(path: Path = SPLITS_FILE) -> dict:
    return json.loads(Path(path).read_text())


def train_scenes(splits: dict, fraction: float) -> List[str]:
    key = f"{float(fraction):g}" if fraction != 1 else "1.0"
    subsets = splits["train_fraction_subsets"]
    if key in subsets:
        return subsets[key]
    if f"{float(fraction)}" in subsets:
        return subsets[f"{float(fraction)}"]
    raise KeyError(f"no pre-registered train subset for fraction {fraction}")


def cv_split(splits: dict, fold: int, n_folds: int) -> dict:
    """Scene-family k-fold CV over the union of the registered train+val scenes.

    Test scenes are never touched (the pool is train+val only, asserted).  Families are
    atomic: all scenes of a family go to the same fold.  Assignment is deterministic and
    uses only names/frame counts: families are ordered by (-frames, name) and each is
    given to the currently lightest fold (frames, then fold index) -- a greedy balance,
    no randomness.  Returns a copy of ``splits`` with train/val replaced (test kept
    only for the disjointness check), 25 % subsets dropped, plus a ``cv`` record.
    """
    if not (n_folds >= 2 and 0 <= fold < n_folds):
        raise ValueError(f"bad fold {fold}/{n_folds}")
    pool = sorted(set(splits["train"]) | set(splits["val"]))
    assert not set(pool) & set(splits["test"]), "test scene in CV pool"
    frames = splits["frames"]
    fams: Dict[str, List[str]] = {}
    for s in pool:
        fams.setdefault(family(s), []).append(s)
    if len(fams) < n_folds:
        raise ValueError(f"{len(fams)} families < {n_folds} folds")
    order = sorted(fams, key=lambda f: (-sum(frames[s] for s in fams[f]), f))
    load = [0] * n_folds
    fold_of: Dict[str, int] = {}
    for f in order:
        j = min(range(n_folds), key=lambda i: (load[i], i))
        fold_of[f] = j
        load[j] += sum(frames[s] for s in fams[f])
    val = sorted(s for f in fams if fold_of[f] == fold for s in fams[f])
    train = sorted(s for s in pool if s not in val)
    out = dict(splits)
    out.update(train=train, val=val, train_fraction_subsets={"1.0": train})
    out["cv"] = dict(fold=fold, n_folds=n_folds,
                     family_fold={f: fold_of[f] for f in sorted(fold_of)},
                     frames_per_fold=load)
    return out


if __name__ == "__main__":
    out = make_splits(scene_frames())
    SPLITS_FILE.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("train", "val", "test", "frames_per_split")},
                     indent=1))
