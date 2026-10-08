# BRIEF: hybrid models M6/M7 (connectome front-end + HexConvGRU)
Working directory: /home/user/flyproj/course. You may ONLY edit `models.py` and create `tests/test_hybrid.py`.
Do NOT edit train.py, eval.py, nulls.py, baselines/, PLAN.md or anything else. No git. No GPU (CUDA_VISIBLE_DEVICES is empty).
No network, no pip install. Do not write personal names or emails anywhere.
Python: /home/user/flyproj/.venv/bin/python

## Spec (from PLAN.md amendment A5)
Add to models.py:
1. `HYBRID_MODELS = ("m6", "m7")`.
2. `class HybridMultiTask(nn.Module)` with class attribute `is_flyvis = True` (the training loop then uses refresh_steady_state and the activity penalty exactly like M1).
   - `__init__(self, model="m6", seed=0, null_seed=None, dt=0.02, t_pre=0.5, hid_ch=?, head_ch=?)`:
     - connectome: m6 -> `nulls.connectome_config("m1", null_seed)`, m7 -> `nulls.connectome_config("m2", null_seed)`; null_seed defaults to seed (same rule as FlyvisMultiTask).
     - build `self.network` exactly as in FlyvisMultiTask (same node_config, copy that code or factor it into a shared helper function used by both classes; FlyvisMultiTask behaviour must stay identical).
     - `from flyvis.task.decoder import ActivityDecoder` is the base class of DecoderGAVP; its `__init__(connectome)` creates `self.dvs_channels = LayerActivity(None, connectome, use_central=False)`. Create your own `self.dvs_channels = LayerActivity(None, self.network.connectome, use_central=False)` (import LayerActivity from the same module flyvis.task.decoder imports it from — read flyvis/task/decoder.py in /home/user/flyproj/.venv/lib/python3.12/site-packages/flyvis/).
     - C = len(self.network.connectome.output_cell_types).
     - `self.trunk = hex_models.HexConvGRUNet(hid_ch=hid_ch, n_layers=1, head_ch=head_ch, in_ch=C)` (import baselines/hex_models.py the same way build_model does).
     - Choose default hid_ch/head_ch so that the TOTAL trainable parameter count of HybridMultiTask (network + trunk) is within 15,387 ± 3 % (i.e. 14,925–15,849). Use n_layers=1 or 2, whatever fits; document the chosen numbers in a comment with the resulting count.
   - `refresh_steady_state(batch_size)` identical to FlyvisMultiTask.
   - `forward(lum, return_activity=False)`: same steady-state/stimulus logic as FlyvisMultiTask.forward to get `act`; then
     `self.dvs_channels.update(act)`; `x = torch.relu(self.dvs_channels.output)` -> shape (B, T, C, 721);
     `out = self.trunk(x)` -> dict with "flow" (B,T,2,721) and "depth" (B,T,1,721). Make sure the output dict has exactly the same keys/shapes
     as FlyvisMultiTask returns (check what DecoderGAVP returns in FlyvisMultiTask and match it; read train.py l2norm_losses to see the expected shapes).
     If return_activity: out["activity"] = act.
   - `param_groups(lr_net, lr_dec)`: "net" = network params, "dec" = trunk params.
3. `build_model`: route "m6"/"m7" to HybridMultiTask(model, seed=seed, null_seed=null_seed).

## Acceptance tests (write tests/test_hybrid.py, pytest, CPU only, fast)
- build m6 and m7 (seed 0): forward on a random lum of shape (1, 3, 721) (or whatever shape FlyvisMultiTask accepts — check data.py/train.py) runs on CPU;
  outputs have the same keys and shapes as `build_model("m1")` on the same input.
- total trainable params of m6 and m7 within 14,925–15,849, and equal to each other.
- m6 and m7 networks have different connectome files (m7 uses the rewired m2 file).
- a backward pass of (flow.mean()+depth.mean()) gives non-None grads on both network and trunk params.
Run: `cd /home/user/flyproj/course && CUDA_VISIBLE_DEVICES="" /home/user/flyproj/.venv/bin/python -m pytest -q tests/test_hybrid.py`
and also the existing tests `... -m pytest -q tests baselines/test_baselines.py` must still pass. Report the exact pytest output and the param counts.
