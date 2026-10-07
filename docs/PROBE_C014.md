# C014 — learnability probe (pre-registered: course/PLAN.md A4, commit 0da4196) — RESULT: PASS
Local RTX 4060, 2026-10-07/08, seed 0, 10k iters, lr 5e-4, val every 1k. Outputs: course/runs_probe/ (gitignored).
Zero-flow val EPE = 5.074.

| model | best val EPE (iter) | vs zero-flow | EPE @10k | best-val-loss ckpt (iter): EPE / depth RMSE | best depth RMSE | s/iter | peak VRAM |
|---|---|---|---|---|---|---|---|
| M1 flyvis connectome | 4.915 (9k) | −3.1 % | 4.982 | 8k: 4.930 / 1.831 | 1.815 (6k) | 0.191 | 2.45 GB |
| M4 HexConvGRU (15.4k) | 4.501 (6k) | −11.3 % | 4.666 | 6k: 4.501 / 1.507 | 1.507 (6k) | 0.150 | 0.61 GB |

Decision rule (A4): PASS if any model ≤ 4.97 → both pass → **keep Sintel flow for the grid**.
Caveats (honest): one seed; val curves are noisy (M1 EPE ±0.05, M4 depth 1.51–1.84 between checks), so "best" is optimistic
(selected over 10 val checks). M1 only crosses the threshold at 5k/8k/9k; M4 wins clearly on both tasks at this budget.
Not a test of H1/H2 (single seed, 10k not 30k, val not test) — but it is an early warning that H2 (M1 > M4) may fail.
