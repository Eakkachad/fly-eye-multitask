"""Floors on the Spring test set (PLAN.md A9): zero-flow, Lucas-Kanade, constant depth.

Nothing is refitted on Spring.  LK window radius / lambda and the 2x2 unit calibration (fitted on
Sintel TRAIN only) are read from the frozen Sintel val report floors_out/val/metrics.json
(`lk_choice`); the constant-depth value is the Sintel train mean of the standardised target
(`train_means.depth_global_target` of the same report).  All floors are scored through
eval_spring.compute_metrics, i.e. exactly the metric code the models go through, so the constant-depth
floor under the per-clip median alignment reduces to the within-clip spread of the GT log-depth.

    python floors_spring.py --split spring --once          # the single Spring floor evaluation
    python floors_spring.py --split val --out-dir <dir>    # debugging on Sintel val (scratch dir)
Writes floors_out/spring/metrics.json.  Refuses a second run unless --force --reason.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COURSE_DIR))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="spring", choices=["spring", "val"])
    p.add_argument("--once", action="store_true", help="required for --split spring")
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default="")
    p.add_argument("--val-report", default=str(COURSE_DIR / "floors_out" / "val" / "metrics.json"))
    p.add_argument("--out-dir", default=None, help="default floors_out/spring; required for val")
    args = p.parse_args(argv)
    logging.disable(logging.INFO)

    if args.split == "spring" and not args.once:
        sys.exit("spring split requires --once (the Spring test is evaluated once)")
    if args.split == "val" and not args.out_dir:
        sys.exit("--split val needs --out-dir (scratch)")
    out_dir = Path(args.out_dir) if args.out_dir else COURSE_DIR / "floors_out" / "spring"
    if (out_dir / "metrics.json").exists():
        if not args.force:
            sys.exit(f"{out_dir}/metrics.json exists: refusing to overwrite (use --force --reason)")
    if args.force and not args.reason:
        sys.exit("--force needs --reason (it is logged)")
    vf = Path(args.val_report)
    if not vf.exists():
        sys.exit("floors_out/val/metrics.json missing: LK config must be frozen on Sintel val")
    rep = json.loads(vf.read_text())
    ch = rep["lk_choice"]
    dconst = float(rep["train_means"]["depth_global_target"])

    import data as D
    import eval_spring as ES
    import floors as FL

    depth_tf = D.DepthTransform.from_file()
    ds = ES.load_clips(args.split)
    clips = [ds[i] for i in range(len(ds))]
    names = list(ds.names)
    u, v = FL.hex_axial(15)
    lk = FL.HexLK(u, v, int(ch["radius"]), float(ch["lam"]))
    A = np.array(ch["calibration"])

    def run(flow_fn, depth_fn):
        preds = []
        import torch

        for c in clips:
            pr = dict(flow=None, depth_t=None)
            if flow_fn:
                pr["flow"] = torch.as_tensor(np.asarray(flow_fn(c)), dtype=torch.float32, device="cpu")
            if depth_fn:
                pr["depth_t"] = torch.as_tensor(np.asarray(depth_fn(c)), dtype=torch.float32, device="cpu")
            preds.append(pr)
        m, _ = ES.compute_metrics(clips, preds, depth_tf, names, pixels=False)
        return m

    res = dict(split=args.split, lk_choice=ch, n_sequences=len(clips),
               train_means=dict(depth_global_target=dconst), created=time.strftime("%Y-%m-%dT%H:%M:%S"),
               note="Sintel-fitted/frozen configuration; nothing refitted on this split; scored with eval_spring.compute_metrics")
    if args.force:
        res["force_reason"] = args.reason
    res["floors"] = {
        "flow_zero": run(lambda c: np.zeros(tuple(c["flow"].shape)), None),
        "flow_lucas_kanade_hex": run(lambda c: FL.apply_calibration(lk.flow(FL.lum_np(c)), A), None),
        "depth_constant": run(None, lambda c: np.full(tuple(c["depth"].shape), dconst)),
    }
    for f in res["floors"].values():
        f.pop("clips", None)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    if args.force:
        with open(out_dir / "force_log.txt", "a") as fh:
            fh.write(f"{res['created']} {args.reason}\n")
    for k, m in res["floors"].items():
        print(f"{k:24s}", {a: round(b, 4) for a, b in m.items() if isinstance(b, float)})


if __name__ == "__main__":
    main()
