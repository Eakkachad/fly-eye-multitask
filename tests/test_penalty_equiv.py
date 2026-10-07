"""flyvis Penalty (after opt.step) vs fused penalty (grad before opt.step) on CPU."""
import sys
from pathlib import Path

import torch

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
import models  # noqa: E402
import train as T  # noqa: E402


def _run(impl, n_iters=4, stop_iter=3, B=1, frames=8):
    from datamate import Namespace
    from flyvis.solver import Penalty

    torch.manual_seed(0)
    model = models.build_model("m1", seed=0)
    model.train()
    opt = torch.optim.Adam(model.param_groups(5e-5, 5e-5))
    pen = Penalty(Namespace(
        activity_penalty=Namespace(activity_baseline=5.0, activity_penalty=0.1,
                                   stop_iter=stop_iter, below_baseline_penalty_weight=1.0,
                                   above_baseline_penalty_weight=0.1),
        optim="SGD"), model.network)
    g = torch.Generator().manual_seed(1)
    losses_log = []
    for it in range(n_iters):
        lr = 5e-5 * (1 - 0.1 * it)
        for gr in opt.param_groups:
            gr["lr"] = lr
        batch = dict(lum=torch.rand(B, frames, 1, 721, generator=g),
                     flow=torch.randn(B, frames, 2, 721, generator=g),
                     depth=torch.rand(B, frames, 1, 721, generator=g))
        model.refresh_steady_state(B)
        opt.zero_grad(set_to_none=True)
        torch.manual_seed(100 + it)  # decoder dropout
        out = model(batch["lum"], return_activity=True)
        ls = T.l2norm_losses(out, batch)
        loss = sum(ls.values())
        fused = impl == "fused"
        pg = T.penalty_grads(pen, out["activity"], it) if fused else None
        loss.backward(retain_graph=not fused)
        opt.step()
        if fused:
            if pg is not None:
                T.penalty_apply(pen, pg, lr)
        else:
            if pen.activity_optim is not None:
                for gr in pen.activity_optim.param_groups:
                    gr["lr"] = lr
            pen(activity=out["activity"], iteration=it)
        losses_log.append(float(loss))
    return model, losses_log


def test_fused_matches_flyvis_penalty():
    ma, la = _run("flyvis")
    mb, lb = _run("fused")
    assert la == lb or all(abs(x - y) < 1e-6 * max(1, abs(x)) for x, y in zip(la, lb))
    sa, sb = ma.state_dict(), mb.state_dict()
    assert sa.keys() == sb.keys()
    n_diff_from_init = 0
    init = models.build_model("m1", seed=0).state_dict()
    for k in sa:
        torch.testing.assert_close(sb[k], sa[k], rtol=1e-5, atol=1e-6, msg=k)
        if sa[k].is_floating_point() and not torch.equal(sa[k], init[k]):
            n_diff_from_init += 1
    assert n_diff_from_init > 0  # the test actually trained something
    # penalty must have an effect (otherwise the test proves nothing)
    mc, _ = _run("flyvis", stop_iter=0)
    bias = [k for k in sa if "bias" in k][0]
    assert not torch.allclose(mc.state_dict()[bias], sa[bias], rtol=0, atol=1e-9)
