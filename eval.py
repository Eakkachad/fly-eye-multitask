"""Evaluate a trained run ONCE on the test scenes (all sequences, all frames,
no augmentation).

Writes runs/<name>/test/:
  metrics.json    flow EPE, angular error (deg), depth RMSE, depth abs-rel
                  (pooled over all valid test pixels and frames), per-sequence
                  metrics, binned EPE by GT flow speed and by local luminance
                  variance (texture), best/worst sequences, metric cross-check
                  against baselines/metrics.py when importable.
  per_pixel.npz   per-pixel (seq, frame, hexal) error arrays + bin variables
                  (float16), concatenated over sequences (ragged lengths ->
                  'offsets').
  examples.npz    lum / GT / prediction of 4 example sequences (2 best, 2 worst
                  by EPE of this model) + the fixed reference sequences
                  (indices 0 of each test scene) so all models share figures.
Refuses to run twice unless --force (test is evaluated once at the end).

Bin edges are deciles of GT-only quantities on the test set (identical for every
model, no model output involved).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COURSE_DIR))


def hex_neighbours(radius: int = 15) -> np.ndarray:
    """(N, 7) indices of self + 6 axial neighbours; missing -> N (padding)."""
    from flyvis.utils.hex_utils import get_hex_coords

    u, v = get_hex_coords(radius)
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(u, v))}
    offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    n = len(u)
    out = np.full((n, 7), n, dtype=np.int64)
    for i, (a, b) in enumerate(zip(u, v)):
        for j, (du, dv) in enumerate(offs):
            out[i, j] = idx.get((int(a) + du, int(b) + dv), n)
    return out


def local_variance(lum: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """lum (T, N) -> variance over the 7-hexal neighbourhood (T, N)."""
    pad = np.concatenate([lum, np.full((lum.shape[0], 1), np.nan)], 1)
    patch = pad[:, nb]  # (T, N, 7)
    return np.nanvar(patch, axis=2)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run", help="runs/<name> directory")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--force", action="store_true")
    p.add_argument("--splits", default=str(COURSE_DIR / "splits.json"))
    p.add_argument("--split", default="test", choices=["test", "val"],
                   help="'val' only for debugging this script without touching test")
    p.add_argument("--k", type=int, default=None,
                   help="m8/m9/m9s only: GRU inner steps at eval (default k_max); output goes to <split>_k<K>/")
    p.add_argument("--frontend-only", action="store_true",
                   help="m6f/m7f only (PLAN A6.2): prediction of the frozen front-end without the "
                        "residual trunk; output goes to <split>_frontend/")
    p.add_argument("--out-name", default=None,
                   help="output sub-directory name under the run (default <split>[_k<K>|_frontend]); "
                        "the per-directory refuse-to-overwrite guard applies to it (A9 second look: test_last)")
    p.add_argument("--device", default="cuda", help="torch device (default cuda; 'cpu' for debugging)")
    args = p.parse_args(argv)
    logging.disable(logging.INFO)

    import torch

    import data as D
    import evalmetrics as M
    import models
    import splits as S
    from train import evaluate

    run = Path(args.run)
    if args.frontend_only and args.k is not None:
        sys.exit("--frontend-only and --k are mutually exclusive")
    suffix = f"_k{args.k}" if args.k is not None else "_frontend" if args.frontend_only else ""
    out_dir = run / (args.out_name or f"{args.split}{suffix}")
    if (out_dir / "metrics.json").exists() and not args.force:
        sys.exit(f"{out_dir}/metrics.json exists: test already evaluated (use --force)")
    out_dir.mkdir(exist_ok=True)

    ck = torch.load(run / args.ckpt, map_location=args.device, weights_only=False)
    cfg = ck["config"]
    a = cfg["args"]
    splits = S.load_splits(args.splits)
    D.check_disjoint(splits)
    model = models.build_model(a["model"], seed=a["seed"], null_seed=a["null_seed"])
    model.load_state_dict(ck["model"])
    if args.k is not None:
        model.k_max = args.k  # m8: default forward K
    model.to(args.device)
    if args.frontend_only:
        if not getattr(model, "frozen", False):
            sys.exit("--frontend-only needs a frozen-front-end hybrid (m6f/m7f)")
        _fwd = model.forward
        model.forward = lambda lum, *a, **kw: _fwd(lum, *a, residual=False, **kw)
    depth_tf = D.DepthTransform.from_config(cfg["depth_transform"])
    ds = D.make_datasets(splits, [], which=(args.split,))[args.split]
    assert set(ds.sequence_scenes()) == set(splits[args.split])

    res = evaluate(model, ds, depth_tf, per_sequence=True, return_preds=True)
    preds = res.pop("preds")
    names = list(ds.arg_df.name)
    for s, n in zip(res["sequences"], names):
        s["name"] = n

    # per-pixel arrays
    nb = hex_neighbours(15)
    rows = {k: [] for k in ("epe", "angular", "depth_sqerr", "depth_absrel",
                            "gt_speed", "gt_depth", "texture")}
    lengths = []
    for i, pr in enumerate(preds):
        raw = ds[i]
        f, d = raw["flow"][None], raw["depth"][None]
        pf, pd = pr["flow"][None].to(f.device), pr["depth"][None].to(f.device)
        rows["epe"].append(M.epe_map(pf, f)[0].cpu().numpy())
        rows["angular"].append(M.angular_error_map(pf, f)[0].cpu().numpy())
        # RMSE in normalised target units, abs-rel in (clipped) metric depth
        pdt = pr["depth_t"][None].to(f.device)
        rows["depth_sqerr"].append(M.sq_err_map(pdt, depth_tf(d))[0].cpu().numpy())
        rows["depth_absrel"].append(M.absrel_map(pd, depth_tf.clip(d))[0].cpu().numpy())
        rows["gt_speed"].append(torch.linalg.vector_norm(f, dim=2)[0].cpu().numpy())
        rows["gt_depth"].append(d[0, :, 0].cpu().numpy())
        rows["texture"].append(local_variance(raw["lum"][:, 0].cpu().numpy(), nb))
        lengths.append(f.shape[1])
    cat = {k: np.concatenate(v, 0) for k, v in rows.items()}
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez_compressed(out_dir / "per_pixel.npz", offsets=offsets,
                        names=np.array(names),
                        **{k: v.astype(np.float16) for k, v in cat.items()})

    def deciles(x):
        e = np.nanquantile(x.ravel().astype(np.float64), np.linspace(0, 1, 11))
        e[-1] += 1e-9
        return e

    binned = {}
    for var in ("gt_speed", "texture"):
        edges = deciles(cat[var])
        binned[var] = dict(edges=edges.tolist())
        for met in ("epe", "angular", "depth_absrel"):
            m, c = M.binned(cat[met], cat[var], edges)
            binned[var][met] = m.tolist()
            binned[var]["counts"] = c.tolist()
    res["binned"] = binned

    # examples
    order = sorted(range(len(res["sequences"])), key=lambda i: res["sequences"][i]["epe"])
    best2, worst2 = order[:2], order[-2:]
    ref = sorted({names.index(n) for n in names
                  if n.endswith("_split_00")})
    keep = sorted(set(best2 + worst2 + ref))
    ex = {}
    for i in keep:
        raw = ds[i]
        ex[f"{i}_lum"] = raw["lum"].cpu().numpy().astype(np.float16)
        ex[f"{i}_flow_gt"] = raw["flow"].cpu().numpy().astype(np.float16)
        ex[f"{i}_depth_gt"] = raw["depth"].cpu().numpy().astype(np.float16)
        ex[f"{i}_flow_pred"] = preds[i]["flow"].numpy().astype(np.float16)
        ex[f"{i}_depth_pred"] = preds[i]["depth"].numpy().astype(np.float16)
    np.savez_compressed(out_dir / "examples.npz", **ex)
    res["examples"] = dict(best=[names[i] for i in best2], worst=[names[i] for i in worst2],
                           reference=[names[i] for i in ref], indices=keep)

    # cross-check with the baselines' metrics implementation, if present
    try:
        sys.path.insert(0, str(COURSE_DIR / "baselines"))
        import metrics as BM  # type: ignore

        cc = {}
        for i, pr in enumerate(preds[:3]):
            raw = ds[i]
            f, d = raw["flow"][None].cpu(), raw["depth"][None].cpu()
            pf, pd = pr["flow"][None], pr["depth"][None]
            cc[names[i]] = dict(
                epe=[float(M.epe(pf, f)), float(BM.epe(pf, f))],
                angular=[float(M.angular_error_deg(pf, f)),
                         float(BM.angular_error_deg(pf, f))],
                rmse=[float(M.depth_rmse(pr["depth_t"][None], depth_tf(d))),
                      float(BM.depth_rmse(pr["depth_t"][None], depth_tf(d)))],
                absrel=[float(M.depth_absrel(pd, depth_tf.clip(d))),
                        float(BM.depth_absrel(pd, depth_tf.clip(d)))])
        res["crosscheck_baselines_metrics"] = cc
    except Exception as e:  # noqa: BLE001
        res["crosscheck_baselines_metrics"] = f"not available: {e!r}"

    res.update(run=str(run), ckpt=args.ckpt, ckpt_iter=ck.get("iter"),
               split=args.split, scenes=splits[args.split], n_sequences=len(ds),
               note="pooled over all valid pixels and frames of all test sequences")
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("epe", "angular_deg", "depth_rmse",
                                          "depth_absrel", "loss")}))


if __name__ == "__main__":
    main()
