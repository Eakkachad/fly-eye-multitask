"""Phase-2 evaluation of a trained run on the Spring test set (PLAN.md amendment A9).

    python eval_spring.py runs/<name> --split spring [--ckpt last|best] [--k K]
    python eval_spring.py runs/<name> --split val --out-dir <scratch>   # debugging through the SAME code path

Writes <run>/spring_<ckpt>[_k<K>]/{metrics.json, per_pixel.npz}.  Refuses to overwrite an existing
output unless --force --reason "<why>" (the reason is stored in metrics.json and appended to
<run>/spring_force_log.txt).  `--split val` (Sintel val, all hexals valid) is for debugging only and
requires an explicit --out-dir (it never writes into the run directory).

METRIC DEFINITIONS (all pooled over every valid hexal and frame of every clip; clip = one rendered
sample = one Spring sequence x one of its 3 vertical splits, all frames; every frame is scored):
  epe, angular_deg      flow, over hexals with flow_valid (evalmetrics.epe_map / angular_error_map)
  epe_by_speed          mean EPE in deciles of GT speed |flow| (edges from valid GT hexals only,
                        identical for every model)
  depth_rmse_aligned    PRIMARY depth metric, natural-log units.  Per clip:
                          g = log(clip(depth, lo, hi))            (GT, natural log of clipped metric depth)
                          p = pred_t * std + mean                 (prediction, standardised -> natural log;
                                                                   DepthTransform log_std inverse of the log)
                          s = median over the clip's depth-valid hexals (all frames) of (g - p)
                          e = (p + s) - g
                        value = sqrt(mean e^2) over all valid hexals of all clips.  (Equals std * RMSE of the
                        aligned standardised error because the standardisation mean cancels.)
  depth_rmse            SECONDARY: raw standardised RMSE, over depth-valid hexals (no alignment)
  depth_absrel          SECONDARY: mean |D_pred - D| / D, D = clipped metric depth, no alignment
  depth_rmse_aligned_seq  supplementary: same as aligned but one median shift per Spring sequence
                        (all 3 vertical splits together)
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

K_MODELS = ("m8", "m9", "m9m", "m9s")
SPEED_BINS = 10


# ---------------------------------------------------------------- data
class _ValClips:
    """Sintel val through the same interface as SpringHex (all hexals valid)."""

    def __init__(self, splits):
        import data as D

        self.ds = D.make_datasets(splits, [], which=("val",))["val"]
        assert set(self.ds.sequence_scenes()) == set(splits["val"])
        self.names = [str(n) for n in self.ds.arg_df.name]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        import torch

        r = dict(self.ds[i])
        r["flow_valid"] = torch.ones_like(r["depth"], dtype=torch.bool)
        r["depth_valid"] = torch.ones_like(r["depth"], dtype=torch.bool)
        return r

    def sequence_ids(self):
        return [n.rsplit("_split_", 1)[0] for n in self.names]


def load_clips(split, splits_path=None):
    if split == "val":
        import splits as S

        sp = S.load_splits(splits_path or str(COURSE_DIR / "splits.json"))
        return _ValClips(sp)
    import spring_data as SD

    ds = SD.SpringHex()
    sel = json.loads((COURSE_DIR / "spring_sequences.json").read_text())["test_selection"]["selected"]
    assert sorted(set(ds.sequence_ids())) == sorted(sel), (sorted(set(ds.sequence_ids())), sel)
    return ds


# ---------------------------------------------------------------- metrics
def compute_metrics(clips, preds, depth_tf, names=None, pixels=True):
    """clips: list of dicts of tensors (T,C,N) incl. flow_valid/depth_valid (bool).
    preds: list of dicts(flow=(T,2,N) or None, depth_t=(T,1,N) or None, standardised depth).
    Returns (metrics dict, per-pixel dict or None).  Used by eval_spring.py AND floors_spring.py."""
    import torch

    import evalmetrics as M

    assert depth_tf.kind == "log_std", "aligned depth metric is defined for the log_std transform"
    std, mean = float(depth_tf.std), float(depth_tf.mean)
    names = names or [f"clip{i}" for i in range(len(clips))]
    has_flow = preds[0].get("flow") is not None
    has_depth = preds[0].get("depth_t") is not None
    acc = {k: [] for k in ("epe", "ang", "speed", "se", "ar", "al")}
    pix = {k: [] for k in ("epe", "angular", "depth_sqerr", "depth_absrel", "depth_aligned_err",
                           "gt_speed", "flow_valid", "depth_valid")}
    per_clip, lengths = [], []
    for c, pr, nm in zip(clips, preds, names):
        T = c["flow"].shape[0]
        lengths.append(T)
        fv = c["flow_valid"][:, 0].bool().numpy()
        dv = c["depth_valid"][:, 0].bool().numpy()
        row = dict(name=nm, n_flow_valid=int(fv.sum()), n_depth_valid=int(dv.sum()))
        gt_speed = torch.linalg.vector_norm(c["flow"], dim=1).numpy().astype(np.float64)
        zero = np.zeros((T, c["flow"].shape[2]))
        e = a = zero
        if has_flow:
            pf = pr["flow"].float()
            assert torch.isfinite(pf).all(), f"non-finite flow prediction in {nm}"
            e = M.epe_map(pf[None], c["flow"][None].float())[0].numpy().astype(np.float64)
            a = M.angular_error_map(pf[None], c["flow"][None].float())[0].numpy().astype(np.float64)
            ok = fv & np.isfinite(e) & np.isfinite(a)
            acc["epe"].append(e[ok]); acc["ang"].append(a[ok]); acc["speed"].append(gt_speed[ok])
            row.update(epe=float(e[ok].mean()), angular=float(a[ok].mean()))
        se = ar = al = zero
        if has_depth:
            pdt = pr["depth_t"].float()
            assert torch.isfinite(pdt).all(), f"non-finite depth prediction in {nm}"
            d = c["depth"].float()
            gt_t = depth_tf(d)[:, 0].numpy().astype(np.float64)
            pd_t = pdt[:, 0].numpy().astype(np.float64)
            ok = dv & np.isfinite(gt_t)
            g = torch.log(depth_tf.clip(d))[:, 0].numpy().astype(np.float64)   # natural log, clipped metric depth
            p = pd_t * std + mean                                               # predicted natural log depth
            shift = float(np.median((g - p)[ok]))
            al = (p + shift) - g
            se = (pd_t - gt_t) ** 2
            ar = M.absrel_map(depth_tf.inverse(pdt[None]), depth_tf.clip(d)[None])[0].numpy().astype(np.float64)
            acc["se"].append(se[ok]); acc["ar"].append(ar[ok]); acc["al"].append(al[ok])
            row.update(depth_rmse_aligned=float(np.sqrt((al[ok] ** 2).mean())),
                       depth_rmse=float(np.sqrt(se[ok].mean())), depth_absrel=float(ar[ok].mean()),
                       align_shift_nat_log=shift, _g=g[ok], _p=p[ok], _ok=ok)
        per_clip.append(row)
        if pixels:
            for k, v in (("epe", e), ("angular", a), ("depth_sqerr", se), ("depth_absrel", ar),
                         ("depth_aligned_err", al), ("gt_speed", gt_speed)):
                pix[k].append(v.astype(np.float16))
            pix["flow_valid"].append(fv); pix["depth_valid"].append(dv)
    cat = lambda k: np.concatenate(acc[k])  # noqa: E731
    res = {}
    if has_flow:
        res["epe"] = float(cat("epe").mean())
        res["angular_deg"] = float(cat("ang").mean())
        sp = cat("speed")
        edges = np.quantile(sp, np.linspace(0, 1, SPEED_BINS + 1))
        edges[-1] += 1e-9
        m, n = M.binned(cat("epe"), sp, edges)
        res["epe_by_speed"] = dict(edges=edges.tolist(), epe=m.tolist(), counts=n.tolist(),
                                   note="deciles of GT speed over valid hexals of this test set")
    if has_depth:
        res["depth_rmse_aligned"] = float(np.sqrt((cat("al") ** 2).mean()))
        res["depth_rmse"] = float(np.sqrt(cat("se").mean()))
        res["depth_absrel"] = float(cat("ar").mean())
        # supplementary: one shift per sequence (all vertical splits of a sequence together)
        seq_of = [n.rsplit("_split_", 1)[0] for n in names]
        tot, cnt = 0.0, 0
        for s in sorted(set(seq_of)):
            rows = [r for r, q in zip(per_clip, seq_of) if q == s]
            g = np.concatenate([r["_g"] for r in rows]); p = np.concatenate([r["_p"] for r in rows])
            err = p + np.median(g - p) - g
            tot += float((err ** 2).sum()); cnt += err.size
        res["depth_rmse_aligned_seq"] = float(np.sqrt(tot / cnt))
    for r in per_clip:
        for k in ("_g", "_p", "_ok"):
            r.pop(k, None)
    res["clips"] = per_clip
    res["n_clips"] = len(clips)
    res["n_flow_valid"] = int(sum(r["n_flow_valid"] for r in per_clip))
    res["n_depth_valid"] = int(sum(r["n_depth_valid"] for r in per_clip))
    res["n_steps"] = int(sum(lengths))
    pp = None
    if pixels:
        pp = {k: np.concatenate(v, 0) for k, v in pix.items()}
        pp["offsets"] = np.concatenate([[0], np.cumsum(lengths)])
        pp["names"] = np.array(names)
    return res, pp


# ---------------------------------------------------------------- model
def predict(model, clips, device):
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for c in clips:
            out = model(c["lum"][None].to(device))
            preds.append(dict(flow=out["flow"][0].float().cpu(), depth_t=out["depth"][0].float().cpu()))
    return preds


def guard_output(out_dir, force, reason, run_dir=None):
    if (out_dir / "metrics.json").exists():
        if not force:
            sys.exit(f"{out_dir}/metrics.json exists: refusing to overwrite (use --force --reason)")
        if not reason:
            sys.exit("--force needs --reason (it is logged)")
        log = Path(run_dir or out_dir.parent) / "spring_force_log.txt"
        with open(log, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} overwrite {out_dir}: {reason}\n")
    elif force and not reason:
        sys.exit("--force needs --reason (it is logged)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", help="runs/<name> directory")
    p.add_argument("--split", default="spring", choices=["spring", "val"])
    p.add_argument("--ckpt", default="last", choices=["last", "best"], help="primary = last")
    p.add_argument("--k", type=int, default=None,
                   help="m8/m9/m9m only: GRU inner steps at eval (default = the run's training k_max); "
                        "output goes to spring_<ckpt>_k<K>/")
    p.add_argument("--out-dir", default=None, help="required for --split val (scratch dir, not runs/)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default="")
    p.add_argument("--no-per-pixel", action="store_true")
    p.add_argument("--fastfly", action="store_true", help="flyvis models: fused rollout (cuda; same numerics)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args(argv)
    logging.disable(logging.INFO)

    run = Path(args.run)
    suffix = f"_k{args.k}" if args.k is not None else ""
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.split == "val":
        sys.exit("--split val needs --out-dir (scratch); it never writes into the run dir")
    else:
        out_dir = run / f"spring_{args.ckpt}{suffix}"
    if args.split == "val" and run.resolve() in out_dir.resolve().parents:
        sys.exit("--split val must not write inside the run directory")
    guard_output(out_dir, args.force, args.reason, run)

    import torch

    import data as D
    import models

    ck = torch.load(run / f"{args.ckpt}.pt", map_location=args.device, weights_only=False)
    cfg = ck["config"]
    a = cfg["args"]
    model = models.build_model(a["model"], seed=a["seed"], null_seed=a["null_seed"])
    model.load_state_dict(ck["model"])
    k_eval = None
    if a["model"] in K_MODELS:
        k_eval = args.k if args.k is not None else int(a["k_max"])
        model.k_max = k_eval
    elif args.k is not None:
        sys.exit(f"--k is only for {K_MODELS}")
    model.to(args.device)
    if args.fastfly:
        if not getattr(model, "is_flyvis", False):
            sys.exit("--fastfly needs a flyvis model")
        from fastfly import patch_network

        patch_network(model.network)
    depth_tf = D.DepthTransform.from_config(cfg["depth_transform"])

    ds = load_clips(args.split)
    clips = [ds[i] for i in range(len(ds))]
    names = list(ds.names)
    t0 = time.time()
    preds = predict(model, clips, args.device)
    res, pp = compute_metrics(clips, preds, depth_tf, names, pixels=not args.no_per_pixel)
    out_dir.mkdir(parents=True, exist_ok=True)
    if pp is not None:
        np.savez_compressed(out_dir / "per_pixel.npz", **pp)
    res.update(run=str(run), model=a["model"], seed=a["seed"], ckpt=f"{args.ckpt}.pt",
               ckpt_iter=ck.get("iter"), split=args.split, k_eval=k_eval, depth_transform=cfg["depth_transform"],
               eval_s=time.time() - t0, fastfly=bool(args.fastfly),
               note="pooled over valid hexals and all frames; depth_rmse_aligned in natural-log units "
                    "(per-clip median alignment), depth_rmse in standardised units")
    if args.force:
        res["force_reason"] = args.reason
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("epe", "angular_deg", "depth_rmse_aligned", "depth_rmse",
                                          "depth_absrel")}))


if __name__ == "__main__":
    main()
