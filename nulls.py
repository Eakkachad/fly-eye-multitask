"""Null connectomes for the flyvis network (M2, M3).

The flyvis connectome (fib25-fib19_v2.2.json) is a *type graph*: 65 cell types,
one entry per (source type -> target type) edge with a spatial kernel
(list of (du, dv) offsets with synapse counts), a sign ``alpha`` and a
``lambda_mult`` (synapse count certainty). flyvis compiles it into a cell-level
graph with ``ConnectomeFromAvgFilters``. One JSON edge (Lawf1 -> Lawf1, a single
offset that never lands on a strided Lawf1 cell) compiles to zero cell edges, so
the compiled real connectome has 604 type-pair edges; we drop that edge from the
base spec so all models have exactly 604 type-pair edges.

M2 (degree-preserving rewiring, Maslov-Sneppen): repeatedly pick two edges
a->b, c->d and swap their targets to a->d, c->b. Each edge keeps its source,
kernel, sign and lambda_mult, so every type's out-degree, in-degree and the
multiset of signs per source type are preserved exactly (Dale's law as in the
data: sign stays a property of the source). Swaps are rejected if they would
create a duplicate type pair, a new self-loop, or a kernel that compiles to zero
cell edges (strided Lawf types).

M3 (random / Erdos-Renyi type graph): same number of edges and the same
number of excitatory / inhibitory edges. Each type gets the majority sign of
its outgoing edges in M1 (types without outgoing edges in M1, Mi11 and Tm30,
are not used as sources since they have no sign). Excitatory edges are drawn
uniformly from (excitatory source type, non-input target type) pairs and
likewise for inhibitory edges; kernels / lambda_mult are a random permutation
of the M1 kernels of the same sign. No duplicate pairs, no self-loops, no
edges into the photoreceptors R1-R8 (they stay pure input cells).

In both nulls the node set, the input cell types (R1-R8) and the output cell
types read by the decoders are unchanged.

Generated specs are written as JSON (same schema as the flyvis file) and loaded
through flyvis' own ``ConnectomeFromAvgFilters`` (absolute file path), so the
resulting network is a normal flyvis Network.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
CONNECTOME_DIR = COURSE_DIR / "connectomes"
EXTENT = 15


def flyvis_connectome_file() -> Path:
    import flyvis

    return Path(flyvis.connectome_file)


def load_spec(path: Optional[Path] = None) -> dict:
    return json.loads(Path(path or flyvis_connectome_file()).read_text())


# -- geometry: does a kernel compile to at least one cell edge? -----------------


def _type_positions(spec: dict, extent: int = EXTENT) -> Dict[str, set]:
    pos = {}
    for n in spec["nodes"]:
        pattern, args = n["pattern"]
        if pattern == "single":
            pos[n["name"]] = {(0, 0)}
            continue
        su, sv = (args, args) if pattern == "tile" else args
        s = set()
        for u in range(-extent, extent + 1):
            for v in range(max(-extent, -extent - u), min(extent, extent - u) + 1):
                if u % su == 0 and v % sv == 0:
                    s.add((u, v))
        pos[n["name"]] = s
    return pos


class KernelChecker:
    """Checks that a (src, tar, offsets) edge compiles to >= 1 cell-level edge."""

    def __init__(self, spec: dict, extent: int = EXTENT):
        self.pos = _type_positions(spec, extent)
        self.full = {k for k, v in self.pos.items() if len(v) == len(self.pos["R1"])}
        self._cache: Dict[Tuple, bool] = {}

    def __call__(self, src: str, tar: str, offsets: list) -> bool:
        # stride-1 source and target always produce edges for |offset| <= extent
        if src in self.full and tar in self.full:
            return True
        key = (src, tar, tuple(tuple(o[0]) for o in offsets))
        if key not in self._cache:
            tpos = self.pos[tar]
            self._cache[key] = any(
                (u + du, v + dv) in tpos
                for (du, dv) in key[2]
                for (u, v) in self.pos[src]
            )
        return self._cache[key]


def base_spec(spec: Optional[dict] = None) -> dict:
    """The real spec with edges that compile to zero cell edges removed."""
    spec = copy.deepcopy(spec or load_spec())
    check = KernelChecker(spec)
    spec["edges"] = [e for e in spec["edges"] if check(e["src"], e["tar"], e["offsets"])]
    return spec


# -- edge table helpers -----------------------------------------------------------


def type_pairs(spec: dict) -> List[Tuple[str, str]]:
    return [(e["src"], e["tar"]) for e in spec["edges"]]


def degrees(spec: dict) -> Tuple[Counter, Counter]:
    pairs = type_pairs(spec)
    return Counter(s for s, _ in pairs), Counter(t for _, t in pairs)


def signs_by_source(spec: dict) -> Dict[str, Counter]:
    out: Dict[str, Counter] = {}
    for e in spec["edges"]:
        out.setdefault(e["src"], Counter())[e["alpha"]] += 1
    return out


def majority_sign(spec: dict) -> Dict[str, int]:
    """Majority sign of each type's outgoing edges (ties -> -1)."""
    return {
        t: (1 if c[1] > c[-1] else -1) for t, c in signs_by_source(spec).items()
    }


# -- M2: degree-preserving rewiring ---------------------------------------------


def rewire_degree_preserving(
    spec: Optional[dict] = None,
    seed: int = 0,
    swaps_per_edge: int = 10,
    max_tries_per_edge: int = 1000,
) -> Tuple[dict, dict]:
    """Maslov-Sneppen target swaps. Returns (new_spec, info)."""
    spec = base_spec(spec)
    check = KernelChecker(spec)
    rng = np.random.default_rng(seed)
    edges = spec["edges"]
    n = len(edges)
    src = [e["src"] for e in edges]
    tar = [e["tar"] for e in edges]
    pairs = set(zip(src, tar))
    target_swaps = swaps_per_edge * n
    accepted, tries = 0, 0
    while accepted < target_swaps and tries < max_tries_per_edge * n:
        tries += 1
        i, j = rng.integers(0, n, size=2)
        if i == j or tar[i] == tar[j] or src[i] == src[j]:
            continue
        a, b, c, d = src[i], tar[i], src[j], tar[j]
        if a == d or c == b:  # would create a self-loop
            continue
        if (a, d) in pairs or (c, b) in pairs:  # duplicate type pair
            continue
        if not (check(a, d, edges[i]["offsets"]) and check(c, b, edges[j]["offsets"])):
            continue
        pairs -= {(a, b), (c, d)}
        pairs |= {(a, d), (c, b)}
        tar[i], tar[j] = d, b
        accepted += 1
    new = copy.deepcopy(spec)
    for e, t in zip(new["edges"], tar):
        e["tar"] = t
    info = dict(
        model="M2",
        seed=seed,
        accepted_swaps=accepted,
        tries=tries,
        n_edges=n,
        frac_edges_changed=float(np.mean([t != e["tar"] for t, e in zip(tar, edges)])),
        self_loops_before=sum(e["src"] == e["tar"] for e in edges),
        self_loops_after=sum(s == t for s, t in zip(src, tar)),
    )
    return new, info


# -- M3: random type graph -------------------------------------------------------


def random_type_graph(spec: Optional[dict] = None, seed: int = 0) -> Tuple[dict, dict]:
    spec = base_spec(spec)
    check = KernelChecker(spec)
    rng = np.random.default_rng(seed)
    types = [n["name"] for n in spec["nodes"]]
    inputs = set(spec["input_units"])
    msign = majority_sign(spec)
    targets = [t for t in types if t not in inputs]
    new_edges: List[dict] = []
    pairs: set = set()
    for sgn in (1, -1):
        sources = [t for t in types if msign.get(t) == sgn]
        kernels = [e for e in spec["edges"] if e["alpha"] == sgn]
        order = rng.permutation(len(kernels))
        for k in order:
            proto = kernels[k]
            for _ in range(100_000):
                s = sources[rng.integers(len(sources))]
                t = targets[rng.integers(len(targets))]
                if s == t or (s, t) in pairs or not check(s, t, proto["offsets"]):
                    continue
                break
            else:  # pragma: no cover
                raise RuntimeError("could not place random edge")
            e = copy.deepcopy(proto)
            e["src"], e["tar"], e["alpha"] = s, t, sgn
            e["alpha_references"] = []
            pairs.add((s, t))
            new_edges.append(e)
    new = copy.deepcopy(spec)
    new["edges"] = new_edges
    info = dict(
        model="M3",
        seed=seed,
        n_edges=len(new_edges),
        n_exc=sum(e["alpha"] == 1 for e in new_edges),
        n_inh=sum(e["alpha"] == -1 for e in new_edges),
        self_loops_after=0,
    )
    return new, info


# -- writing / loading as flyvis connectome -------------------------------------


def null_spec(model: str, seed: int) -> Tuple[dict, dict]:
    model = model.lower()
    if model == "m1":
        return base_spec(), dict(model="M1", seed=None)
    if model == "m2":
        return rewire_degree_preserving(seed=seed)
    if model == "m3":
        return random_type_graph(seed=seed)
    raise ValueError(model)


def connectome_file(model: str, seed: int = 0, out_dir: Path = CONNECTOME_DIR) -> Path:
    """Path of a JSON connectome spec for model m1/m2/m3 (generated if needed).

    M1 returns the original flyvis file (identical compiled graph to base_spec).
    """
    model = model.lower()
    if model == "m1":
        return flyvis_connectome_file()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{model}_seed{seed}.json"
    spec, info = null_spec(model, seed)
    text = json.dumps(spec, separators=(",", ":"))
    if not path.exists() or path.read_text() != text:
        tmp = path.with_suffix(f".tmp{np.random.randint(1 << 30)}")
        tmp.write_text(text)
        tmp.replace(path)
    (out_dir / f"{model}_seed{seed}.info.json").write_text(json.dumps(info, indent=1))
    return path


def connectome_config(model: str, seed: int = 0) -> dict:
    """flyvis Network(connectome=...) config for M1/M2/M3."""
    return dict(
        type="ConnectomeFromAvgFilters",
        file=str(connectome_file(model, seed)),
        extent=EXTENT,
        n_syn_fill=1,
    )


def compiled_type_pairs(connectome) -> List[Tuple[str, str]]:
    """Unique (source_type, target_type) pairs of a compiled flyvis connectome."""
    st = connectome.edges.source_type[:]
    tt = connectome.edges.target_type[:]
    return sorted({(s.decode(), t.decode()) for s, t in zip(st, tt)})


def reachable_outputs(spec: dict) -> float:
    """Fraction of output types reachable from the input types (type graph)."""
    adj: Dict[str, set] = {}
    for s, t in type_pairs(spec):
        adj.setdefault(s, set()).add(t)
    seen = set(spec["input_units"])
    stack = list(seen)
    while stack:
        for t in adj.get(stack.pop(), ()):
            if t not in seen:
                seen.add(t)
                stack.append(t)
    outs = spec["output_units"]
    return sum(o in seen for o in outs) / len(outs)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    for m in ("m2", "m3"):
        for s in a.seeds:
            path = connectome_file(m, s)
            spec = load_spec(path)
            info = json.loads((CONNECTOME_DIR / f"{m}_seed{s}.info.json").read_text())
            print(path.name, info, "reach_out=%.2f" % reachable_outputs(spec))
