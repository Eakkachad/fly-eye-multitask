import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import floors as F  # noqa: E402


def test_lk_recovers_uniform_translation():
    u, v = F.hex_axial(15)
    xy = F.hex_xy(u, v)
    vel = np.array([0.12, -0.07])  # spacings / frame
    k = 2 * np.pi / 14.0
    def pat(t):
        p = xy - vel * t
        return (np.sin(k * p[:, 0] + 0.3) + np.cos(k * 0.8 * p[:, 1] - 0.5)
                + 0.5 * np.sin(k * (p[:, 0] + p[:, 1])))
    lum = np.stack([pat(t) for t in range(6)])
    lk = F.HexLK(u, v, radius=2, lam=1e-6)
    fl = lk.flow(lum)
    assert np.all(fl[0] == 0)
    interior = (np.abs(u) < 11) & (np.abs(v) < 11) & (np.abs(u + v) < 11)
    est = fl[1:, :, interior]
    assert np.abs(np.median(est[:, 0]) - vel[0]) < 0.02
    assert np.abs(np.median(est[:, 1]) - vel[1]) < 0.02
    assert np.abs(est.mean(axis=(0, 2)) - vel).max() < 0.03


def test_calibration_recovers_linear_map():
    rng = np.random.default_rng(0)
    A = np.array([[0.5, 0.1], [-0.2, -0.7]])
    x = [rng.normal(size=(4, 2, 30))]
    y = [F.apply_calibration(x[0], A)]
    assert np.allclose(F.fit_calibration(x, y), A)


def _seq(T=5, N=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(lum=torch.rand(T, 1, N, generator=g), flow=torch.randn(T, 2, N, generator=g),
                depth=torch.rand(T, 1, N, generator=g) + 1.0)


def test_zero_mean_floors_match_hand_computation():
    from data import DepthTransform

    tf = DepthTransform("log_std", lo=1.0, hi=2.0, mean=0.3, std=0.5)
    seqs = [_seq(seed=1), _seq(T=4, seed=2)]
    means = F.train_means(seqs, tf)
    f = torch.cat([s["flow"] for s in seqs])
    assert np.allclose(means["flow_global"], f.mean((0, 2)).numpy())
    assert np.allclose(means["flow_hexal"], f.mean(0).numpy())
    lk = lambda r: np.zeros(tuple(r["flow"].shape))  # noqa: E731
    out = F.evaluate_all(seqs, means, lk, tf)
    # zero flow: EPE = mean |gt|; angular by hand
    allf = f.double()
    assert out["flow_zero"]["epe"] == pytest.approx(float(allf.norm(dim=1).mean()), rel=1e-5)
    cos = 1.0 / (torch.sqrt(1 + (allf ** 2).sum(1)))
    assert out["flow_zero"]["angular_deg"] == pytest.approx(
        float(torch.rad2deg(torch.arccos(cos)).mean()), rel=1e-4)
    # global mean flow
    m = torch.as_tensor(means["flow_global"])[None, :, None]
    assert out["flow_train_mean"]["epe"] == pytest.approx(
        float((allf - m).norm(dim=1).mean()), rel=1e-5)
    # depth: constant in target space
    d = torch.cat([tf(s["depth"]) for s in seqs]).double()
    c = d.mean()
    assert out["depth_train_mean"]["depth_rmse"] == pytest.approx(float(((d - c) ** 2).mean().sqrt()), rel=1e-5)
    pdepth = torch.exp(c * 0.5 + 0.3)
    gt = torch.cat([s["depth"].clamp(1.0, 2.0) for s in seqs]).double()
    assert out["depth_train_mean"]["depth_absrel"] == pytest.approx(
        float(((pdepth - gt).abs() / (gt.abs() + 1e-6)).mean()), rel=1e-4)
    ph = d.mean(0, keepdim=True)
    assert out["depth_train_mean_per_hexal"]["depth_rmse"] == pytest.approx(
        float(((d - ph) ** 2).mean().sqrt()), rel=1e-5)
    # per-hexal mean can never be worse than the global mean in RMSE (in-sample)
    assert out["depth_train_mean_per_hexal"]["depth_rmse"] <= out["depth_train_mean"]["depth_rmse"] + 1e-9


def test_oracle_floor_labelled_and_lagged():
    s = _seq(T=4)
    pred = F.oracle_prediction(s["flow"].numpy())
    assert np.all(pred[0] == 0) and np.allclose(pred[1:], s["flow"].numpy()[:-1])
    assert "oracle" in F.ORACLE_LABEL
    from data import DepthTransform

    tf = DepthTransform("identity")
    means = F.train_means([s], tf)
    out = F.evaluate_all([s], means, lambda r: np.zeros(tuple(r["flow"].shape)), tf)
    assert "oracle" in " ".join(out).lower() and "ORACLE" in out[F.ORACLE_LABEL]
    f = s["flow"]
    ref = torch.cat([f[0].norm(dim=0), (f[1:] - f[:-1]).norm(dim=1).flatten()]).mean()
    assert out[F.ORACLE_LABEL]["epe"] == pytest.approx(float(ref), rel=1e-5)


def test_test_split_refuses_without_once(tmp_path):
    with pytest.raises(SystemExit):
        F.main(["--split", "test", "--out-dir", str(tmp_path)])
    with pytest.raises(SystemExit):  # --once but no frozen val choice
        F.main(["--split", "test", "--once", "--out-dir", str(tmp_path)])
