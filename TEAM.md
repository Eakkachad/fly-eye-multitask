# Team plan (8 people). Each person owns one part and must be able to explain it in the Q&A.

| # | Role | Owns (code / artifacts) | Explains in Q&A |
|---|---|---|---|
| 1 | Problem & data lead | splits.json, data section, Sintel → hex-lattice rendering | why split by scene (leakage), what the fly-eye input is, augmentation only on train |
| 2 | Connectome model (M1) | flyvis network, multi-task decoders | what a connectome-constrained network is; why it has ~700 trainable params |
| 3 | Null models (M2, M3) | rewired/ER type graphs, tests | what a "null model" is, what degree-preserving rewiring keeps and what it breaks |
| 4 | Baselines (M4, M5) | baselines/hex_models.py (HexConv, ConvGRU) | hex convolution; ConvGRU temporal memory; param matching |
| 5 | Training & engineering | train.py, configs, GPU scheduling, Rust tooling | fair comparison (same data/iters/loss), seeds, VRAM/time budget |
| 6 | Evaluation | eval.py, metrics (EPE, angular error, RMSE, abs-rel) | metric definitions; test-once protocol |
| 7 | Error analysis & figures | binned errors, best/worst examples, data-efficiency curve | where and why models fail; limitations |
| 8 | Slides & presentation lead | deck, storyline, timing, rehearsal | overall story; connects to the research motivation (genome/connectome project) |

## Schedule (deadline 2026-10-09)
| day | work |
|---|---|
| D1 (10-05) | data + harness + nulls + baselines; smoke tests |
| D2 (10-06) | full training grid: M1–M5 × 3 seeds, plus the data-efficiency arm |
| D3 (10-07) | test-set evaluation (once), error analysis, figures |
| D4 (10-08) | slides, README with usage, rehearsal; buffer |
