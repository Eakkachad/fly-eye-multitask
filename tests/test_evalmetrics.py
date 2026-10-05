import sys
from pathlib import Path

import numpy as np
import torch

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
import evalmetrics as M  # noqa: E402


def test_toy_values():
    z = torch.zeros(1, 1, 2, 3, device="cpu")
    g = torch.tensor([3.0, 4.0], device="cpu").view(1, 1, 2, 1).expand(1, 1, 2, 3)
    assert torch.isclose(M.epe(z, g), torch.tensor(5.0, device="cpu"))
    assert float(M.angular_error_deg(g, g)) < 1e-2
    d = torch.full((1, 1, 1, 3), 2.0, device="cpu")
    assert torch.isclose(M.depth_rmse(d + 1, d), torch.tensor(1.0, device="cpu"))
    assert torch.isclose(M.depth_absrel(d + 1, d), torch.tensor(0.5, device="cpu"))


def test_matches_baselines_metrics():
    sys.path.insert(0, str(COURSE / "baselines"))
    import metrics as BM

    g = torch.Generator(device="cpu").manual_seed(0)
    p, t = (torch.randn(2, 5, 2, 721, generator=g, device="cpu") for _ in range(2))
    dp, dt = (torch.rand(2, 5, 1, 721, generator=g, device="cpu") + 0.1 for _ in range(2))
    for a, b in [(M.epe(p, t), BM.epe(p, t)),
                 (M.angular_error_deg(p, t), BM.angular_error_deg(p, t)),
                 (M.depth_rmse(dp, dt), BM.depth_rmse(dp, dt)),
                 (M.depth_absrel(dp, dt), BM.depth_absrel(dp, dt))]:
        assert abs(float(a) - float(b)) < 1e-4 * max(1, abs(float(b)))


def test_binned_and_neighbours():
    m, c = M.binned(np.array([1, 2, 3, 4.0]), np.array([0, 0.5, 1.5, 2.5]),
                    np.array([0, 1, 2, 3]))
    assert np.allclose(m, [1.5, 3, 4]) and list(c) == [2, 1, 1]
    from eval import hex_neighbours, local_variance

    nb = hex_neighbours(15)
    assert nb.shape == (721, 7)
    interior = (nb < 721).all(1)
    assert interior.sum() == 721 - 6 * 15  # border ring of radius 15 has 90 hexals
    assert all(len(set(r)) == 7 for r in nb[interior])
    v = local_variance(np.ones((2, 721)), nb)
    assert np.allclose(v, 0)
