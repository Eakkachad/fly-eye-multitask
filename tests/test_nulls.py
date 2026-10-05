import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nulls  # noqa: E402


@pytest.fixture(scope="module")
def base():
    return nulls.base_spec()


@pytest.fixture(scope="module", params=[0, 1, 2])
def m2(request):
    return nulls.rewire_degree_preserving(seed=request.param)


@pytest.fixture(scope="module", params=[0, 1, 2])
def m3(request):
    return nulls.random_type_graph(seed=request.param)


def test_base_has_604_edges(base):
    assert len(base["edges"]) == 604
    assert len(set(nulls.type_pairs(base))) == 604


def test_m2_preserves_degrees_signs(base, m2):
    spec, info = m2
    assert len(spec["edges"]) == len(base["edges"])
    out0, in0 = nulls.degrees(base)
    out1, in1 = nulls.degrees(spec)
    assert out0 == out1 and in0 == in1
    # sign multiset per source type identical (sign travels with the source)
    assert nulls.signs_by_source(base) == nulls.signs_by_source(spec)
    # kernels travel with their edge: multiset of (src, sign, kernel) unchanged
    key = lambda e: (e["src"], e["alpha"], str(e["offsets"]), e["lambda_mult"])  # noqa
    assert Counter(map(key, base["edges"])) == Counter(map(key, spec["edges"]))
    pairs = nulls.type_pairs(spec)
    assert len(set(pairs)) == len(pairs)  # no duplicate type pairs
    assert info["self_loops_after"] <= info["self_loops_before"]
    assert info["frac_edges_changed"] > 0.8  # well mixed
    # node set / roles unchanged
    assert spec["nodes"] == base["nodes"]
    assert spec["input_units"] == base["input_units"]
    assert spec["output_units"] == base["output_units"]


def test_m3_counts_and_dale(base, m3):
    spec, info = m3
    assert len(spec["edges"]) == len(base["edges"])
    c0 = Counter(e["alpha"] for e in base["edges"])
    c1 = Counter(e["alpha"] for e in spec["edges"])
    assert c0 == c1
    msign = nulls.majority_sign(base)
    for e in spec["edges"]:
        assert e["alpha"] == msign[e["src"]]  # Dale: sign = source-type sign
        assert e["src"] != e["tar"]
        assert e["tar"] not in base["input_units"]
    pairs = nulls.type_pairs(spec)
    assert len(set(pairs)) == len(pairs)
    assert spec["output_units"] == base["output_units"]


def test_deterministic():
    a, _ = nulls.rewire_degree_preserving(seed=7)
    b, _ = nulls.rewire_degree_preserving(seed=7)
    c, _ = nulls.rewire_degree_preserving(seed=8)
    assert nulls.type_pairs(a) == nulls.type_pairs(b) != nulls.type_pairs(c)
    a, _ = nulls.random_type_graph(seed=7)
    b, _ = nulls.random_type_graph(seed=7)
    assert nulls.type_pairs(a) == nulls.type_pairs(b)


@pytest.mark.parametrize("model", ["m1", "m2", "m3"])
def test_compiles_as_flyvis_connectome(model):
    """The spec loads through flyvis' ConnectomeFromAvgFilters and every type
    pair compiles to cell-level edges (604 type pairs, degrees intact)."""
    from flyvis.connectome import ConnectomeFromAvgFilters

    path = nulls.connectome_file(model, seed=0)
    spec = nulls.base_spec(nulls.load_spec(path))
    conn = ConnectomeFromAvgFilters(file=str(path), extent=15, n_syn_fill=1)
    pairs = nulls.compiled_type_pairs(conn)
    assert len(pairs) == 604
    assert sorted(pairs) == sorted(nulls.type_pairs(spec))
    assert [t.decode() for t in conn.output_cell_types[:]] == spec["output_units"]
    assert [t.decode() for t in conn.input_cell_types[:]] == spec["input_units"]
