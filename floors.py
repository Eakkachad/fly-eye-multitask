"""Pre-registered floors (PLAN.md amendment A6 item 1): baselines without deep learning.

Flow:   zero-flow, train-mean flow (global and per-hexal), Lucas-Kanade computed on the
        hex lattice from the input luminance, and the ORACLE constant-velocity floor
        (prediction at frame t = GT flow at t-1; uses ground truth, not a competitor).
Depth:  train-mean and per-hexal train-mean in the normalised log-depth space of the models.

Scoring matches train.evaluate / eval.py exactly: every frame (incl. frame 0) of every
sequence, all 721 hexals, no mask, metrics pooled over all pixels and frames
(evalmetrics.epe_map / angular_error_map / sq_err_map / absrel_map). RMSE is in
normalised target units, AbsRel on clipped metric depth (depth_tf.inverse of the prediction).

Frame alignment (flyvis rendering): lum[t] is Sintel frame t+1 and flow[t] is Sintel flow
t -> t+1, so flow[t] is the motion between lum[t-1] and lum[t]. LK at frame t therefore
uses (lum[t-1], lum[t]) -- causal, same information a model has at t. Frame 0 has no
previous frame: LK predicts zero flow there (oracle: zero flow as well).

LK units: velocities are first estimated in hexal-spacings per frame in the lattice
cartesian frame (flyvis hex_to_pixel 'default', divided by sqrt(3)). The dataset's flow is
in other units/axis conventions (pixel/height, y flipped, sum-boxfilter), so a single 2x2
linear map (4 numbers, no intercept) from LK output to the dataset flow convention is fitted
by least squares on the TRAIN split only (pure unit/axis calibration). Window radius
(1 or 2 rings) and Tikhonov lambda are chosen on VAL only; the choice is logged.

Test split: needs --once; refuses a second run unless --force (with --reason), and uses the
LK configuration frozen in the val report. Usage:
    python floors.py --split val
    python floors.py --split test --once
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np

COURSE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(COURSE_DIR))

ORACLE_LABEL = "oracle_constant_velocity"
OUT_DIR = COURSE_DIR / "floors_out"
RADIUS_GRID = (1, 2)
LAMBDA_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SQRT3 = np.sqrt(3.0)


# -- hex geometry -------------------------------------------------------------------

def hex_axial(radius: int = 15):
    from flyvis.utils.hex_utils import get_hex_coords

    u, v = get_hex_coords(radius)
    return np.asarray(u, int), np.asarray(v, int)


def hex_xy(u, v):
    """Lattice cartesian positions in units of the hexal spacing."""
    x = 1.5 * v
    y = -SQRT3 * (u + v / 2.0)
    return np.stack([x, y], 1) / SQRT3


def ring_offsets(R: int):
    """Axial offsets with hex distance 1..R (excluding self)."""
    out = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if (a or b) and max(abs(a), abs(b), abs(a + b)) <= R:
                out.append((a, b))
    return out


def neighbour_index(u, v, offsets):
    """(N, len(offsets)) indices, missing -> N."""
    idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(u, v))}
    n = len(u)
    out = np.full((n, len(offsets)), n, dtype=np.int64)
    for i, (a, b) in enumerate(zip(u, v)):
        for j, (da, db) in enumerate(offsets):
            out[i, j] = idx.get((int(a) + da, int(b) + db), n)
    return out


class HexLK:
    """Lucas-Kanade on the hex lattice (numpy)."""

    def __init__(self, u, v, radius: int, lam: float):
        self.n = len(u)
        self.xy = hex_xy(u, v)
        self.nb1 = neighbour_index(u, v, ring_offsets(1))   # 6 neighbours -> gradients
        offs = [(0, 0)] + ring_offsets(radius)
        self.nbw = neighbour_index(u, v, offs)              # LK window incl. self
        self.lam = lam
        self.radius = radius
        pxy = np.concatenate([self.xy, np.zeros((1, 2))], 0)
        valid = self.nb1 < self.n
        d = (pxy[self.nb1] - self.xy[:, None]) * valid[..., None]      # (N,6,2)
        G = np.einsum("nki,nkj->nij", d, d) + 1e-9 * np.eye(2)
        self.d, self.valid = d, valid
        self.Ginv = np.linalg.inv(G)

    def gradients(self, lum):
        """lum (T,N) -> (Ix, Iy) (T,N): LS gradient from the 6 hex neighbours."""
        pad = np.concatenate([lum, np.zeros((lum.shape[0], 1))], 1)
        diff = (pad[:, self.nb1] - lum[:, :, None]) * self.valid          # (T,N,6)
        b = np.einsum("tnk,nki->tni", diff, self.d)
        g = np.einsum("nij,tnj->tni", self.Ginv, b)
        return g[..., 0], g[..., 1]

    def _wsum(self, x):
        pad = np.concatenate([x, np.zeros((x.shape[0], 1))], 1)
        return pad[:, self.nbw].sum(2)

    def flow(self, lum):
        """lum (T,N) -> flow (T,2,N) in spacings/frame (lattice cartesian); frame 0 = 0."""
        lum = np.asarray(lum, np.float64)
        T = lum.shape[0]
        out = np.zeros((T, 2, self.n))
        if T < 2:
            return out
        a, b = lum[:-1], lum[1:]
        ix, iy = self.gradients(0.5 * (a + b))
        it = b - a
        sxx, sxy, syy = self._wsum(ix * ix), self._wsum(ix * iy), self._wsum(iy * iy)
        bx, by = -self._wsum(ix * it), -self._wsum(iy * it)
        sxx, syy = sxx + self.lam, syy + self.lam
        det = sxx * syy - sxy ** 2
        out[1:, 0] = (syy * bx - sxy * by) / det
        out[1:, 1] = (sxx * by - sxy * bx) / det
        return out


def fit_calibration(lk_list, gt_list):
    """2x2 least-squares map A with gt ~= lk @ A (no intercept). Lists of (T,2,N)."""
    X = np.concatenate([x.transpose(0, 2, 1).reshape(-1, 2) for x in lk_list])
    Y = np.concatenate([y.transpose(0, 2, 1).reshape(-1, 2) for y in gt_list])
    A, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return A


def apply_calibration(flow, A):
    return np.einsum("tcn,cd->tdn", flow, A)


# -- floors that need no fitting ------------------------------------------------------

def oracle_prediction(gt_flow):
    """Constant velocity: pred[t] = gt[t-1]; frame 0 -> zeros. (T,2,N)."""
    out = np.zeros_like(gt_flow)
    out[1:] = gt_flow[:-1]
    return out


def score_sequences(seqs, flow_pred_fn, depth_pred_fn, depth_tf):
    """Pooled metrics exactly as train.evaluate (via evalmetrics). ``seqs`` = iterable of
    dataset items (dict of torch (T,C,N)); *_pred_fn(raw)->(T,2,N) / (T,1,N) normalised
    depth target (either may be None)."""
    import torch

    import evalmetrics as M

    epe_all, ae_all, se_all, ar_all = [], [], [], []
    for raw in seqs:
        f = raw["flow"][None]
        d = raw["depth"][None]
        if flow_pred_fn is not None:
            pf = torch.as_tensor(np.asarray(flow_pred_fn(raw)), dtype=torch.float32)[None]
            epe_all.append(M.epe_map(pf, f).flatten())
            ae_all.append(M.angular_error_map(pf, f).flatten())
        if depth_pred_fn is not None:
            pdt = torch.as_tensor(np.asarray(depth_pred_fn(raw)), dtype=torch.float32)[None]
            se_all.append(M.sq_err_map(pdt, depth_tf(d)).flatten())
            ar_all.append(M.absrel_map(depth_tf.inverse(pdt), depth_tf.clip(d)).flatten())
    res = {}
    if epe_all:
        res["epe"] = float(torch.cat(epe_all).mean())
        res["angular_deg"] = float(torch.cat(ae_all).mean())
    if se_all:
        res["depth_rmse"] = float(torch.cat(se_all).mean().sqrt())
        res["depth_absrel"] = float(torch.cat(ar_all).mean())
    return res


def train_means(train_seqs, depth_tf):
    """Global and per-hexal train means of flow (u,v) and normalised depth target."""
    import torch

    fl = torch.cat([r["flow"] for r in train_seqs]).double()             # (T,2,N)
    dp = torch.cat([depth_tf(r["depth"]) for r in train_seqs]).double()  # (T,1,N)
    return dict(
        flow_global=fl.mean((0, 2)).numpy(),           # (2,)
        flow_hexal=fl.mean(0).numpy(),                 # (2,N)
        depth_global=float(dp.mean()),
        depth_hexal=dp.mean(0).numpy(),                # (1,N)
    )


def lum_np(raw):
    return raw["lum"][:, 0].double().numpy()


def evaluate_all(seqs, means, lk_pred, depth_tf):
    """All floors on ``seqs`` (list of items). lk_pred(raw)->(T,2,N) calibrated LK flow."""
    zeros = lambda r: np.zeros(tuple(r["flow"].shape))  # noqa: E731
    out = {}
    out["flow_zero"] = score_sequences(seqs, zeros, None, depth_tf)
    fg = means["flow_global"]
    out["flow_train_mean"] = score_sequences(
        seqs, lambda r: np.broadcast_to(fg[None, :, None], r["flow"].shape), None, depth_tf)
    out["flow_train_mean_per_hexal"] = score_sequences(
        seqs, lambda r: np.broadcast_to(means["flow_hexal"][None], r["flow"].shape), None, depth_tf)
    out["flow_lucas_kanade_hex"] = score_sequences(seqs, lk_pred, None, depth_tf)
    out[ORACLE_LABEL] = score_sequences(seqs, lambda r: oracle_prediction(r["flow"].numpy()),
                                        None, depth_tf)
    out[ORACLE_LABEL]["ORACLE"] = ("uses ground-truth flow of the previous frame; "
                                   "not a competitor, an upper bound on constant-velocity")
    dg = means["depth_global"]
    out["depth_train_mean"] = score_sequences(
        seqs, None, lambda r: np.full(tuple(r["depth"].shape), dg), depth_tf)
    out["depth_train_mean_per_hexal"] = score_sequences(
        seqs, None, lambda r: np.broadcast_to(means["depth_hexal"][None], r["depth"].shape),
        depth_tf)
    return out


# -- CLI --------------------------------------------------------------------------------

def load_seqs(D, S, splits, which):
    cls = D.make_split_sintel_class()
    ds = cls(splits[which], augment=False, all_frames=True, random_temporal_crop=False)
    assert set(ds.sequence_scenes()) == set(splits[which])
    return [ds[i] for i in range(len(ds))]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--once", action="store_true", help="required for --split test")
    p.add_argument("--force", action="store_true", help="re-run test (needs --reason)")
    p.add_argument("--reason", default="")
    p.add_argument("--splits", default=str(COURSE_DIR / "splits.json"))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args(argv)
    logging.disable(logging.INFO)

    out_dir = Path(args.out_dir) / args.split
    val_file = Path(args.out_dir) / "val" / "metrics.json"
    if args.split == "test":
        if not args.once:
            sys.exit("test split requires --once (test is evaluated once at the end)")
        if (out_dir / "metrics.json").exists() and not args.force:
            sys.exit(f"{out_dir}/metrics.json exists: test already evaluated (use --force)")
        if args.force and not args.reason:
            sys.exit("--force needs --reason (it is logged)")
        if not val_file.exists():
            sys.exit("run `floors.py --split val` first: LK config must be frozen on val")

    import data as D
    import splits as S

    splits = S.load_splits(args.splits)
    D.check_disjoint(splits)
    depth_tf = D.DepthTransform.from_file()
    train = load_seqs(D, S, splits, "train")
    seqs = load_seqs(D, S, splits, args.split)
    means = train_means(train, depth_tf)
    u, v = hex_axial(15)

    def make_lk(radius, lam, A):
        lk = HexLK(u, v, radius, lam)
        return lambda r: apply_calibration(lk.flow(lum_np(r)), A)

    def calibrate(radius, lam):
        lk = HexLK(u, v, radius, lam)
        return fit_calibration([lk.flow(lum_np(r)) for r in train],
                               [r["flow"].double().numpy() for r in train])

    res = dict(split=args.split, scenes=splits[args.split], n_sequences=len(seqs),
               n_frames=int(sum(r["flow"].shape[0] for r in seqs)),
               note="pooled over all pixels and frames, every frame scored (as eval.py)")
    if args.split == "val":
        grid = []
        for R, lam in itertools.product(RADIUS_GRID, LAMBDA_GRID):
            A = calibrate(R, lam)
            sc = score_sequences(seqs, make_lk(R, lam, A), None, depth_tf)
            grid.append(dict(radius=R, lam=lam, calibration=A.tolist(), val_epe=sc["epe"],
                             val_angular_deg=sc["angular_deg"]))
            print(f"LK window radius={R} lam={lam:g}: val EPE {sc['epe']:.4f}", flush=True)
        best = min(grid, key=lambda g: g["val_epe"])
        res["lk_tuning_grid_val"] = grid
        res["lk_choice"] = dict(radius=best["radius"], lam=best["lam"],
                                calibration=best["calibration"],
                                selected_on="val EPE (train used only for the 2x2 unit calibration)")
    else:
        ch = json.loads(val_file.read_text())["lk_choice"]
        res["lk_choice"] = ch
        if args.force:
            res["force_reason"] = args.reason
    ch = res["lk_choice"]
    lk_pred = make_lk(ch["radius"], ch["lam"], np.array(ch["calibration"]))
    res["floors"] = evaluate_all(seqs, means, lk_pred, depth_tf)
    res["train_means"] = dict(flow_global=means["flow_global"].tolist(),
                              depth_global_target=means["depth_global"])
    best_flow = min((k for k in res["floors"] if k.startswith("flow_")),
                    key=lambda k: res["floors"][k]["epe"])
    res["best_non_oracle_flow_floor"] = best_flow
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(res, indent=1))
    for k, m in res["floors"].items():
        print(f"{k:32s}", {a: round(b, 4) for a, b in m.items() if isinstance(b, float)})
    print("LK choice:", {k: ch[k] for k in ("radius", "lam")}, "| best non-oracle flow floor:", best_flow)


if __name__ == "__main__":
    main()
