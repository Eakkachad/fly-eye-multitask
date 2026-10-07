# Task for you (agy): implement standard deep-learning BASELINES for a fly-eye multi-task video project

Work ONLY inside this directory: /home/user/flyproj/course/baselines/ (create files here; do not modify anything outside it; do not run git commands; do not download anything except pip packages already installed). Python env: /home/user/flyproj/.venv/bin/python (torch 2.14 + CUDA, flyvis 1.2.0 installed). Do not put any personal names or emails in files.

## Input / output contract (must match exactly)
- Input `lum`: float tensor (B, T, 1, N) — B batch, T frames (typically 19), N = 721 hexals (hexagonal lattice of radius 15, as in flyvis: `flyvis.utils.hex_utils.get_hex_coords(15)` gives (u, v) axial coordinates of the N hexals in flyvis order).
- Output: dict with `"flow"`: (B, T, 2, N) and `"depth"`: (B, T, 1, N).

## What to build (file `hex_models.py`)
1. `HexConv(in_ch, out_ch)`: convolution on the hex lattice using the 7-neighbourhood (self + 6 axial neighbours: (+1,0),(-1,0),(0,+1),(0,-1),(+1,-1),(-1,+1)); build a neighbour index table (N, 7) from the hex coords once (missing neighbours at the border → index N pointing to a zero-padding row). Weight shape (out_ch, in_ch, 7) + bias.
2. `HexConvGRUCell(in_ch, hid_ch)`: ConvGRU cell where all gates use HexConv.
3. `HexConvGRUNet(hid_ch, n_layers, head_ch)`: encoder HexConv stack → ConvGRU over time (process frames sequentially, keep hidden state) → two heads (HexConv layers) producing flow (2 ch) and depth (1 ch) at every time step. Constructor options so that we can build:
   - `small`: total trainable params between 3,000 and 6,000 (param-matched to the connectome model) — provide a factory `make_small()`.
   - `large`: total trainable params between 400,000 and 1,000,000 — factory `make_large()`.
   Print param counts in a `__main__` block.
4. `metrics.py`: pure torch functions, all taking prediction and target of shape (B, T, C, N) plus an optional boolean mask (B, T, 1, N) for valid pixels:
   - `epe(pred_flow, gt_flow, mask=None)` → mean end-point error (L2 norm over the 2 channels).
   - `angular_error_deg(pred_flow, gt_flow, mask=None)` → mean angle between (u,v,1) vectors in degrees.
   - `depth_rmse(pred, gt, mask=None)`, `depth_absrel(pred, gt, mask=None, eps=1e-6)`.
   - `binned(metric_values_per_pixel, bin_values, edges)` → mean metric per bin (for error analysis by speed / texture).
5. `test_baselines.py`: pytest tests: shapes for both factories on random input (B=2, T=19, N=721) on CPU and CUDA if available; param-count ranges; gradients flow (loss.backward works); metrics on hand-computed toy examples (e.g. epe of zero vs (3,4) = 5, angular error of identical = 0); neighbour table correctness (each interior hexal has 6 distinct neighbours, border padding works).
6. `bench.py`: time forward+backward for B=4, T=19 for small and large on CUDA; print s/iter and peak VRAM.

Run the tests (`/home/user/flyproj/.venv/bin/python -m pytest -q test_baselines.py`) and bench, and make them pass. At the end write `REPORT_AGY.md` in this directory with: files created, test output, param counts, bench numbers, and any limitations. Keep code clean and commented briefly.
