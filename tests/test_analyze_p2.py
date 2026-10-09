"""Decision logic of analyze_p2.py on synthetic metrics (true/false/incomplete cases for P1-P4) + a file round trip."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import analyze_p2 as A  # noqa: E402

LK = 4.0
F = {"floors": {"flow_zero": dict(epe=5.0, angular_deg=50.0),
                "flow_lucas_kanade_hex": dict(epe=LK, angular_deg=45.0),
                "depth_constant": dict(depth_rmse_aligned=1.0, depth_rmse=1.8, depth_absrel=0.98)}}


def mk(spec):
    R = {}
    for arm, (e, d) in spec.items():
        for s in range(3):
            if e[s] is None:
                continue
            R[(arm, s)] = dict(epe=e[s], angular_deg=40.0, depth_rmse_aligned=d[s], depth_rmse=1.5, depth_absrel=0.9)
    return R


def base():
    spec = {a: ([3.5] * 3, [0.9] * 3) for a in A.P1_ARMS}
    spec["m4"] = ([3.0] * 3, [0.9] * 3)
    spec["oursL"] = ([2.0] * 3, [0.7] * 3)
    spec["oursS"] = ([2.5] * 3, [0.8] * 3)
    spec["noSI"] = ([2.5] * 3, [0.9] * 3)
    spec["noEPE"] = ([3.0] * 3, [0.8] * 3)
    spec["K1"] = ([2.8] * 3, [0.8] * 3)
    spec["m1"] = ([3.2] * 3, [0.95] * 3)
    return spec


def dec(spec, F=F, k4=None):
    return {d["id"]: d for d in A.decide_all(mk(spec), F, mk(k4) if k4 else None)}


def test_all_true():
    d = dec(base())
    for p in ("P1", "P2", "P3"):
        assert d[p]["status"] == "SUPPORTED" and d[p]["level"] == "L1", p
    assert d["P4"]["status"] == "DESCRIPTIVE"


def test_p1_false_cases():
    s = base(); s["oursL"] = ([3.5] * 3, [0.7] * 3)          # tie with phase-1 arms: strict < fails
    assert dec(s)["P1"]["status"] == "NOT SUPPORTED"
    s = base(); s["m6f"] = ([1.9] * 3, [0.9] * 3)            # one phase-1 arm better
    assert dec(s)["P1"]["status"] == "NOT SUPPORTED"
    s = base(); s["oursL"] = ([2.0] * 3, [0.7] * 3)
    f = json.loads(json.dumps(F)); f["floors"]["flow_lucas_kanade_hex"]["epe"] = 1.5   # LK floor better
    assert dec(s, f)["P1"]["status"] == "NOT SUPPORTED"


def test_p2_seed_rule():
    s = base(); s["oursS"] = ([2.5, 2.5, 3.5], [0.8, 0.8, 0.95])   # 2/3 on both -> true
    assert dec(s)["P2"]["status"] == "SUPPORTED"
    s["oursS"] = ([2.5, 3.5, 3.5], [0.8, 0.8, 0.8])                # EPE 1/3 -> false
    assert dec(s)["P2"]["status"] == "NOT SUPPORTED"
    s["oursS"] = ([2.5, 2.5, 2.5], [0.8, 1.2, 1.2])                # depth 1/3 -> false
    assert dec(s)["P2"]["status"] == "NOT SUPPORTED"


def test_p3_each_ablation():
    assert dec(base())["P3"]["status"] == "SUPPORTED"
    for abl, met in (("noSI", 1), ("noEPE", 0), ("K1", 0)):
        s = base()
        e, dp = s[abl]
        better = ([2.0] * 3, dp) if met == 0 else (e, [0.5] * 3)    # ablation better than Ours-S
        s[abl] = better
        d = dec(s)["P3"]
        assert d["status"] == "NOT SUPPORTED", abl
        assert "NOT SUPPORTED" in d["detail"]
    s = base(); s["K1"] = ([2.5, 2.4, 2.6], [0.8] * 3)             # ties/better in 1 seed, worse in 1 -> 1/3 worse
    assert dec(s)["P3"]["status"] == "NOT SUPPORTED"
    s = base(); s["K1"] = ([2.6, 2.6, 2.4], [0.8] * 3)             # worse in 2/3 -> ok
    assert dec(s)["P3"]["status"] == "SUPPORTED"


def test_incomplete_and_level():
    s = base(); s["oursS"] = ([2.5, 2.5, None], [0.8] * 3)
    d = dec(s)
    for p in ("P1", "P2", "P3", "P4"):
        if p == "P1":
            assert d[p]["status"] == "SUPPORTED"   # P1 does not involve Ours-S
        else:
            assert d[p]["status"] == "INCOMPLETE" and d[p]["level"].startswith("L2"), p
    assert dec(base(), F=None)["P1"]["status"] == "INCOMPLETE"


def test_roundtrip_files(tmp_path):
    grid1, grid2 = tmp_path / "g1.tsv", tmp_path / "g2.tsv"
    grid1.write_text("# h\n" + "".join(f"m4_s{s}_f1.0\tm4\t{s}\t1.0\t\t3\n" for s in range(3)))
    grid2.write_text("# h\n" + "".join(f"p2_oursS_s{s}\tm8\t{s}\t1.0\t\t3\n" for s in range(3)))
    for s in range(3):
        for name, e, dd in ((f"m4_s{s}_f1.0", 3.0, 0.8), (f"p2_oursS_s{s}", 2.0, 0.7)):
            d = tmp_path / "runs" / name / "spring_last"
            d.mkdir(parents=True)
            (d / "metrics.json").write_text(json.dumps(dict(epe=e, angular_deg=1.0, depth_rmse_aligned=dd,
                                                            depth_rmse=1.0, depth_absrel=0.9)))
    fl = tmp_path / "fl" / "spring"
    fl.mkdir(parents=True)
    (fl / "metrics.json").write_text(json.dumps(F))
    out = tmp_path / "out"
    out.mkdir()
    D = A.main(["--runs", str(tmp_path / "runs"), "--grids", str(grid1), str(grid2), "--floors", str(tmp_path / "fl"),
                "--out-dir", str(out)])
    assert {d["id"]: d["status"] for d in D}["P2"] == "SUPPORTED"
    assert (out / "docs" / "RESULTS_P2.md").exists() and (out / "results" / "summary_p2.csv").exists()
