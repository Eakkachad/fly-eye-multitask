# Fly-Eye Multi-Task Video: does a connectome prior replace parameters and data?
_Course project "Deep Learning for Image Analysis" (difficulty class: Multi-task Learning + Video Analysis). Deadline 2026-10-09. Team of 8._
**PRE-REGISTERED 2026-10-05, before any training run on Sintel.**

## Problem
Given a short grayscale video sequence as seen by the fly eye (hexagonal photoreceptor lattice, 721 hexals, 19 frames at 50 Hz),
jointly predict, for every hexal at the final frames:
- Task A: **optic flow** (2-D vector per pixel; pixel regression)
- Task B: **depth** (scalar per pixel; pixel regression)
The model must integrate information ACROSS frames (motion is only defined between frames).

## Data
MPI Sintel (Butler et al., ECCV 2012), "final" rendering pass + flow ground truth + depth ground truth (training set, 23 scenes).
Rendered onto the fly-eye hex lattice with flyvis' `MultiTaskSintel` (Lappalainen et al., Nature 2024).
**Split by scene** (no frames of one scene in two splits): train / val / test scenes fixed in `splits.json` before training.
The Sintel official test set has no public GT → not used. Augmentation (flips, rotations, contrast, noise) only on train.

## Models (same input, same multi-task decoder heads, same loss, same iterations, same optimizer schedule)
| id | model | trainable params (approx.) | role |
|---|---|---|---|
| M1 | flyvis network with the **real connectome** wiring (trained from scratch) | ~0.7 k + decoders | hypothesis |
| M2 | flyvis network, **degree-preserving rewired** type graph (sign/Dale and spatial kernels kept, which type connects to which shuffled) | same | structural null |
| M3 | flyvis network, **random (ER) type graph** with the same number of type-pair edges and sign ratio | same | weak null |
| M4 | ConvGRU on the hex lattice, **parameter-matched** to M1 (incl. decoders) | same | standard DL, small |
| M5 | ConvGRU / U-Net-GRU, **large** (~0.5–1 M params) | large | standard DL, big |
| M0 | pretrained flyvis ensemble (flow-only, reference; not trained by us) | — | reference |
Seeds: 3 per model (M1–M5). Data-efficiency arm: M1, M2, M5 also trained on 25% of train scenes (3 seeds).

## Metrics (test scenes only; reported per task)
- Flow: mean end-point error (EPE, in hexal/frame-normalized units as in flyvis) + angular error.
- Depth: RMSE and abs-rel error (depth normalized per flyvis rendering).
- Combined: none (tasks reported separately as required).
- Error analysis: EPE by flow-speed bins and by texture (luminance variance) bins; best/worst example sequences with predicted vs GT maps.

## Pre-registered hypotheses
- H1 (connectome vs structural null): M1 test EPE < M2 test EPE in 3/3 paired seeds, and same for depth RMSE.
- H2 (connectome vs param-matched DL): M1 beats M4 on both tasks (mean over seeds).
- H3 (data efficiency): the M1−M2 and M1−M5 gaps (relative) at 25% data are larger than at 100% data (descriptive; reported either way).
No model is tuned on test scenes; hyper-parameters chosen on val only; test evaluated once at the end.

## Engineering
- Training: PyTorch (autodiff) via flyvis; several runs concurrently on the RTX 4060 (0.05 s/iter, 0.7 GB/run measured).
- Rust: used where profiling shows a real bottleneck (data rendering/augmentation, evaluation over all test frames), via PyO3; the null connectome generator reuses the kagpt-fly `nulls` methodology. Not forced where PyTorch is the right tool.
- Every run logs config, seed, package versions, peak VRAM, wall time (experiments/<run>/).

## Amendments (logged before any full training run)
- A1 (2026-10-05): M4 is parameter-matched to M1 **including decoders** (M1 total 15,387 = 734 network + 14,653 decoders) → `make_small_matched()` 15,402 params. The original brief targeted 3–6k (network-only count), which would have been unfair to M4.
- A2 (2026-10-05): Smoke test showed no model beats zero-flow on val at 2k iterations. A learnability probe (M1, M4 @ 10k iters) on the A100 decides whether the flow task stays on Sintel or moves to flyvis synthetic motion stimuli (depth stays on Sintel). Any change will be recorded here **before** the full grid.
- A3 (2026-10-05): The pretrained flyvis (M0) has seen bandage_2, cave_2, market_5 and market_6 in training. It is reported only on unseen scenes.
- Note: the deadline in the header (2026-10-09) refers to the original 4-day plan. The actual course deadlines are tracked in docs/TEAM.md.
- A4 (2026-10-07, pre-registration of the learnability probe, C014 — written before running): runs on the local RTX 4060 (A100 not yet available).
  Runs: M1 and M4, seed 0, `--n-iters 10000 --lr 5e-4 --val-every 1000`, all other flags default, full train split; output `runs_probe/`.
  (lr 5e-4: middle of the 2k-iter LR probe grid 5e-5/5e-4/2e-3, which showed no clear winner.)
  Metric: best val flow EPE (val only; test untouched). Reference: zero-flow val EPE 5.074.
  **PASS** if at least one model reaches val EPE ≤ 4.97 (≥ 2 % below zero-flow) → keep Sintel flow for the grid.
  **FAIL** otherwise → present ROADMAP R1 options (b) synthetic-motion curriculum, (c) central/speed-masked flow evaluation,
  (d) lower-resolution flow to the owner; the chosen change is logged as amendment A5 before any grid run.
  Depth val RMSE is reported alongside (not part of the decision).
- A4 outcome (2026-10-08): **PASS** — best val EPE M1 4.915 (−3.1 %), M4 4.501 (−11.3 %) vs zero-flow 5.074. Flow stays on Sintel. Details: `.orchestra/tasks/C014-probe-REPORT.md` (single seed, noisy val; not a test of H1–H3).
- A5 (2026-10-08, owner-approved, written before implementation/any run of these models): add a **hybrid** connectome-prior model.
  Motivation (disclosed): the C014 probe (val only) showed M4 > M1; this amendment is therefore *post-probe* but pre-grid.
  - **M6** = flyvis network with the real connectome (as M1, same activity penalty) → rectified activity of the connectome's output cell types
    on all 721 hexals (B,T,C,721) → shared HexConvGRU trunk (`baselines/hex_models.HexConvGRUNet`, in_ch=C) → flow and depth heads.
    Sized so total trainable params ≈ M1/M4 (15.4k ± 3 %).
  - **M7** = M6 with the M2 degree-preserving rewired connectome (same null_seed rule as M2) — the control that separates "connectome wiring"
    from "extra recurrent front-end".
  - Same data, loss, optimizer, iterations, 3 seeds as the grid; test evaluated once.
  - **H4:** M6 beats M4 on both val-selected test metrics (flow EPE, depth RMSE), mean over 3 seeds and ≥ 2/3 paired seeds.
  - **H5:** M6 beats M7 on flow EPE in 3/3 paired seeds (connectome-specific benefit). If H4 holds but H5 fails, we report that the gain comes
    from the architecture, not the wiring.
