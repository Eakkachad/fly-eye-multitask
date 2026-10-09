"""Decision logic of analyze.py on synthetic metrics (true and false cases for H1-H7)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze as A  # noqa: E402


def mk(spec, split="test"):
    """spec: {(model, frac, variant): (epe_list, rmse_list)} -> R dict as load_all returns."""
    R = {}
    for (m, f, v), (e, d) in spec.items():
        for s in range(3):
            R[(m, f, s, v)] = dict(epe=e[s], angular_deg=40.0, depth_rmse=d[s], depth_absrel=0.9)
    return R


def dec(R, split="test"):
    return {d["id"]: d for d in A.decide_all(R, split)}


def full(over=None):
    base = {("m1", 1.0, ""): ([4.0] * 3, [1.0] * 3), ("m2", 1.0, ""): ([5.0] * 3, [2.0] * 3),
            ("m4", 1.0, ""): ([4.5] * 3, [1.5] * 3), ("m5", 1.0, ""): ([4.6] * 3, [1.6] * 3),
            ("m1", 0.25, ""): ([4.2] * 3, [1.2] * 3), ("m2", 0.25, ""): ([5.5] * 3, [2.5] * 3),
            ("m5", 0.25, ""): ([5.2] * 3, [2.2] * 3),
            ("m6", 1.0, ""): ([4.0] * 3, [1.0] * 3), ("m7", 1.0, ""): ([4.5] * 3, [1.2] * 3),
            ("m8", 1.0, "k4"): ([4.0] * 3, [1.0] * 3),
            ("m6f", 1.0, ""): ([3.9] * 3, [1.0] * 3), ("m7f", 1.0, ""): ([4.4] * 3, [1.2] * 3)}
    base.update(over or {})
    return mk(base)


def test_all_true():
    d = dec(full())
    for h in ("H1", "H2", "H4", "H5", "H6", "H7"):
        assert d[h]["status"] == "SUPPORTED", h
        assert d[h]["level"] == "L1"
    assert d["H3"]["status"].startswith("SUPPORTED")


def test_val_split_is_never_L1():
    assert all(not x["level"].startswith("L1") for x in A.decide_all(full(), "val"))


def test_h1_needs_3_of_3_on_both_tasks():
    d = dec(full({("m1", 1.0, ""): ([4.0, 4.0, 5.0], [1.0] * 3)}))  # seed 2 not < (5.0 == 5.0)
    assert d["H1"]["status"] == "NOT SUPPORTED"
    d = dec(full({("m1", 1.0, ""): ([4.0] * 3, [1.0, 1.0, 2.5])}))  # depth fails once
    assert d["H1"]["status"] == "NOT SUPPORTED"


def test_h2_mean_condition():
    # one bad seed but better mean -> still supported (mean over seeds)
    d = dec(full({("m1", 1.0, ""): ([3.0, 3.0, 5.5], [1.0] * 3)}))
    assert d["H2"]["status"] == "SUPPORTED"
    d = dec(full({("m1", 1.0, ""): ([4.0] * 3, [1.0, 1.0, 5.0])}))  # depth mean 2.33 > 1.5
    assert d["H2"]["status"] == "NOT SUPPORTED"


def test_h3_descriptive_cases():
    # gap at 0.25 smaller than at 1.0 for everything -> not supported
    d = dec(full({("m2", 0.25, ""): ([4.3] * 3, [1.3] * 3), ("m5", 0.25, ""): ([4.3] * 3, [1.3] * 3)}))
    assert d["H3"]["status"] == "NOT SUPPORTED"
    # only flow vs M5 gap larger -> partial
    d = dec(full({("m2", 0.25, ""): ([4.3] * 3, [1.3] * 3), ("m5", 0.25, ""): ([6.0] * 3, [1.3] * 3)}))
    assert d["H3"]["status"].startswith("PARTIAL (1/4")


def test_h4_requires_mean_and_two_of_three_on_both_metrics():
    d = dec(full({("m6", 1.0, ""): ([4.0, 4.0, 5.0], [1.0] * 3)}))  # EPE 2/3 wins, mean 4.33<4.5
    assert d["H4"]["status"] == "SUPPORTED"
    d = dec(full({("m6", 1.0, ""): ([4.0, 5.0, 5.0], [1.0] * 3)}))  # 1/3 wins
    assert d["H4"]["status"] == "NOT SUPPORTED"
    d = dec(full({("m6", 1.0, ""): ([4.0] * 3, [1.0, 1.0, 9.0])}))  # depth: 2/3 wins but mean worse
    assert d["H4"]["status"] == "NOT SUPPORTED"


def test_h5_three_of_three():
    d = dec(full({("m6", 1.0, ""): ([4.0, 4.0, 4.6], [1.0] * 3)}))
    assert d["H5"]["status"] == "NOT SUPPORTED"
    assert "architecture" not in d["H5"]["detail"] or d["H4"]["status"] == "SUPPORTED"
    d = dec(full({("m6", 1.0, ""): ([4.0, 4.0, 4.6], [1.0] * 3)}))
    assert "not the wiring" in d["H5"]["detail"]  # H4 holds, H5 fails


def test_h6_two_of_three_uses_k4():
    d = dec(full({("m8", 1.0, "k4"): ([4.0, 4.0, 5.0], [1.0] * 3)}))
    assert d["H6"]["status"] == "SUPPORTED"
    d = dec(full({("m8", 1.0, "k4"): ([4.0, 5.0, 5.0], [1.0] * 3)}))
    assert d["H6"]["status"] == "NOT SUPPORTED"
    # a decoy plain-'' entry must not be used when k4 exists
    R = full({("m8", 1.0, "k4"): ([9.0] * 3, [1.0] * 3), ("m8", 1.0, ""): ([1.0] * 3, [1.0] * 3)})
    assert dec(R)["H6"]["status"] == "NOT SUPPORTED"


def test_h7_three_of_three_final_not_frontend():
    d = dec(full({("m6f", 1.0, ""): ([4.0, 4.0, 4.4], [1.0] * 3)}))  # tie at seed 2 -> not strict
    assert d["H7"]["status"] == "NOT SUPPORTED"
    R = full({("m6f", 1.0, "frontend"): ([99.0] * 3, [1.0] * 3)})  # front-end-only must not matter
    assert dec(R)["H7"]["status"] == "SUPPORTED"


def test_incomplete_when_seed_missing():
    R = full()
    del R[("m1", 1.0, 2, "")]
    d = dec(R)
    assert d["H1"]["status"] == "INCOMPLETE" and d["H1"]["level"].startswith("L2")
    assert d["H5"]["status"] == "SUPPORTED"


def test_floor_margin_and_end_to_end(tmp_path):
    runs, floors = tmp_path / "runs", tmp_path / "floors"
    grid = tmp_path / "g.tsv"
    names = [f"{m}_s{s}_f1.0" for m in ("m1", "m2") for s in range(3)]
    grid.write_text("\n".join(f"{n}\tx\t0\t1.0\t\t3" for n in names))
    for i, n in enumerate(names):
        p = runs / n / "val"
        p.mkdir(parents=True)
        (p / "metrics.json").write_text(json.dumps(dict(
            epe=4.0 + i * 0.01, angular_deg=40, depth_rmse=1.5, depth_absrel=0.9)))
    (floors / "val").mkdir(parents=True)
    (floors / "val" / "metrics.json").write_text(json.dumps(dict(floors={
        "flow_zero": dict(epe=5.0, angular_deg=50.0), "flow_lucas_kanade_hex": dict(epe=4.7, angular_deg=42.0),
        "oracle_constant_velocity": dict(epe=1.0, angular_deg=5.0, ORACLE="x"),
        "depth_train_mean": dict(depth_rmse=1.9, depth_absrel=0.98),
        "depth_train_mean_per_hexal": dict(depth_rmse=1.8, depth_absrel=0.94)})))
    A.main(["--split", "val", "--runs", str(runs), "--grid", str(grid), "--floors", str(floors),
            "--out-dir", str(tmp_path)])
    txt = (tmp_path / "docs" / "RESULTS_val.md").read_text()
    assert "flow_lucas_kanade_hex" in txt and "oracle" in txt.lower()
    csv_ = (tmp_path / "results" / "summary_val.csv").read_text()
    assert "m1" in csv_
    # margin of m1 EPE (mean 4.01) vs LK floor 4.7 -> +0.69
    R = A.load_all("val", runs, grid)
    rows, fv = A.arm_rows(R, A.load_floors("val", floors))
    r = next(r for r in rows if r["arm"] == "m1")
    assert abs(r["epe_margin_vs_floor"] - (4.7 - 4.01)) < 1e-9 and fv["depth_rmse"][1] == "depth_train_mean_per_hexal"
