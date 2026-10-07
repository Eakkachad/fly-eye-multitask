"""Summarise bench/results/*.json (profile_train.py output) as a markdown table.

  summarize_r2.py [--dir DIR] [files...]     markdown table
  summarize_r2.py --best-flags m1            print CLI flags of the fastest ok
                                             non-batch-sweep variant of model m1
Speedup is vs '<model>_baseline' (same model, same batch size) by median s/iter;
loss delta = max relative change of fixed-batch flow/depth loss (initial) vs baseline.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(paths):
    rows = []
    for p in sorted(paths):
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        if "config" not in d:
            continue
        d["_name"] = Path(p).stem
        rows.append(d)
    return rows


def flags(c):
    f = []
    if c.get("tf32"):
        f.append("--tf32")
    if c.get("amp", "none") != "none":
        f += ["--amp", c["amp"]]
    if c.get("compile", "none") != "none":
        f += ["--compile", c["compile"]]
    return f


def is_baseline(d):
    c = d["config"]
    return d["_name"].endswith("baseline") or (
        not c.get("tf32") and c.get("amp") == "none" and c.get("compile") == "none")


def rel_delta(a, b):
    return max(abs(a[k] - b[k]) / max(abs(b[k]), 1e-12) for k in ("flow", "depth"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dir", default=str(HERE / "results"))
    ap.add_argument("--best-flags")
    a = ap.parse_args()
    paths = a.files or [p for p in Path(a.dir).glob("*.json")]
    rows = load(paths)

    if a.best_flags:
        cand = [d for d in rows if d["config"]["model"] == a.best_flags
                and d["status"] == "ok" and "_best_bs" not in d["_name"]
                and d["config"]["batch_size"] == 4]
        if cand:
            best = min(cand, key=lambda d: d["s_per_iter_median"])
            print(" ".join(flags(best["config"])))
        return

    base = {}
    for d in rows:
        if is_baseline(d):
            base[(d["config"]["model"], d["config"]["batch_size"])] = d
    print("| variant | model | bs | s/iter (median) | samples/s | speedup vs baseline | "
          "fwd/bwd/opt s | data % | VRAM alloc/reserved GB | loss delta (rel, fixed batch) | status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for d in rows:
        c = d["config"]
        st = " ".join(d["status"].split())
        b = base.get((c["model"], c["batch_size"]))
        if "s_per_iter_median" not in d:
            print(f"| {d['_name']} | {c['model']} | {c['batch_size']} | - | - | - | - | - | - | - | {st} |")
            continue
        sp = f"{b['s_per_iter_median'] / d['s_per_iter_median']:.2f}x" \
            if b and "s_per_iter_median" in b else "n/a"
        ld = "n/a"
        if b and "fixed_batch_loss_initial" in b and b is not d:
            ld = f"{rel_delta(d['fixed_batch_loss_initial'], b['fixed_batch_loss_initial']):.2e}"
        elif b is d:
            ld = "0 (ref)"
        va, vr = d.get("peak_vram_allocated_gb"), d.get("peak_vram_reserved_gb")
        vram = f"{va:.2f} / {vr:.2f}" if va is not None else "n/a (cpu)"
        print(f"| {d['_name']} | {c['model']} | {c['batch_size']} | "
              f"{d['s_per_iter_median']:.4f} | {d['samples_per_s']:.2f} | {sp} | "
              f"{d['fwd_s_median']:.3f}/{d['bwd_s_median']:.3f}/{d['opt_s_median']:.3f} | "
              f"{100 * d['data_fraction']:.1f} | {vram} | {ld} | {st[:80]} |")


if __name__ == "__main__":
    main()
