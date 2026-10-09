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
- Rust: used where profiling shows a real bottleneck (data rendering/augmentation, evaluation over all test frames), via PyO3; the null connectome generator follows the degree-preserving rewiring methodology of our earlier Rust/CUDA GPU engine (Maslov & Sneppen 2002). Not forced where PyTorch is the right tool.
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
- A6 (2026-10-08, owner-approved, before any grid run; methodology ideas drawn from the group's methodology notes and the published literature cited below):
  1. **Claim ladder** (pre-registration: Nosek et al., "The preregistration revolution", PNAS 2018): every claim in the report is tagged **L1** = pre-registered + test set, **L2** = pre-registered but
     val-only or partial, **L3** = exploratory. **Floors** registered now, reported on test alongside models:
     flow — zero-flow, train-mean flow, Lucas–Kanade (Lucas & Kanade, IJCAI 1981) on the hex lattice (classical, no learning); *oracle* constant-velocity (previous-frame GT flow;
     labelled oracle, not a competitor); depth — train-mean, per-hexal train-mean. Models are reported as margin over the best non-oracle floor.
  2. **Frozen-front-end residual hybrids M6f/M7f** (zero-initialised residual on a frozen base: Zhang, Rao & Agrawala, "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet, zero convolutions), ICCV 2023; Johannink et al., "Residual Reinforcement Learning for Robot Control", ICRA 2019): front-end = trained M1 (resp. M2) checkpoint of the same seed, frozen
     (network + its decoders, penalty off); output = frozen decoder output + residual from a HexConvGRU trunk (as M6) whose last layer is
     zero-initialised, so training starts exactly at the M1/M2 solution. Report front-end-only and final. Trained after M1/M2 of the grid.
     **H7:** M6f beats M7f on test flow EPE in 3/3 paired seeds.
  3. **M8 = HexConvGRU-K** (recurrent depth / weight-tied iteration: Dehghani et al., "Universal Transformers", ICLR 2019; Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach", arXiv:2502.05171, 2025): M4 architecture/params, the GRU cell iterated K times per frame with tied weights and
     re-injected encoder input; train with K ~ U{1..4}; evaluate at K = 1..4 (any-time curve). Primary: K = 4.
     **H6:** M8 (K=4) beats M4 on test flow EPE in ≥ 2/3 paired seeds.
  4. **Diagnostics (L3, exploratory, no retraining)** on final checkpoints: front-end necessity (zeroing/ablating input cell-type groups),
     CKA between models with a seed-vs-seed band, recurrent-state stability (representation similarity: Kornblith et al., "Similarity of Neural Network Representations Revisited" (CKA), ICML 2019).
  5. **Ablation-superposition probe + divergence/curl (L3)**: silence cell-type / channel groups, test additivity with cosine
     AND magnitude ratio; EPE vs fraction silenced; divergence/curl of predicted vs GT flow on the hex lattice; looming vs predicted depth.
  Grid becomes 24 + M6/M7 (6) + M6f/M7f (6) + M8 (3) = 39 runs, 3 seeds each arm.
- A7 (2026-10-08, owner-approved, before the grid): grid runs locally on the RTX 4060 (`scripts/run_grid_local.sh`, `scripts/grid_v2.tsv`, 39 runs).
  **lr 5e-4** for all arms (the probe-validated value; the original 5e-5 was never tested beyond 2k iters), 30k iters, val every 1k, batch 4,
  best checkpoint by total val loss. Speed flags (numerically equivalent, R2): `--fastfly` for flyvis-based arms (M1–M3, M6/M7, M6f/M7f),
  fused penalty implementation (default), `--compile default` for M4/M5/M8. M6f/M7f use the grid's own M1/M2 `best.pt` of the same seed.
  No per-arm lr retuning (time budget) — listed as a limitation.
- A8 (2026-10-08, owner-approved, mid-grid, speed only): hybrid arms (M6/M7/M6f/M7f) get `--compile default` on the HexConvGRU trunk
  (measured 0.188 → 0.069 s/iter; loss after 400 iters 717.74 vs 717.51 — compile is not bit-identical). For consistency within the hybrid arms,
  the two hybrid runs already finished without compile (m6_s0, m7_s0) are **re-run** with it; the eager runs are kept in `runs_eager_ref/`
  as a reproducibility reference only (not reported as results). No other arm is affected.
- Note (2026-10-09): provenance wording in A6 and Engineering anonymised (internal project names removed); no change to design, hypotheses or analysis.
