# Engineering Report: Param-Matched M4 Baseline & Remote Portability (AGY 2)

**Repository**: `/home/user/flyproj/course`  
**Execution Environment**: Linux (x86_64), Python 3.12 (`/home/user/flyproj/.venv/bin/python`), PyTorch 2.14.1  
**Constraint Compliance**: All python commands executed strictly on CPU with `CUDA_VISIBLE_DEVICES=""` to avoid interfering with ongoing local GPU workloads. Zero git commands or pushes performed. No personal names or emails included.

---

## 1. Summary of Changes

| Path | Action | Description |
|---|---|---|
| [baselines/hex_models.py](file:///home/user/flyproj/course/baselines/hex_models.py) | **Modified** | Added `make_small_matched()` factory (15,402 parameters, within +0.10% of M1 total). Kept `make_small()` intact. Updated module docstring and `__main__` verification. |
| [baselines/test_baselines.py](file:///home/user/flyproj/course/baselines/test_baselines.py) | **Modified** | Added `test_make_small_matched_param_counts()` asserting ±5% parameter tolerance; added `make_small_matched` to parametrized forward shape and gradient flow tests. |
| [requirements.txt](file:///home/user/flyproj/course/requirements.txt) | **Created** | Pinned dependencies to local environment versions with explicit PyTorch CUDA wheel instructions for remote GPU environments. |
| [scripts/setup_remote.sh](file:///home/user/flyproj/course/scripts/setup_remote.sh) | **Created** | Idempotent environment and data setup: creates venv (`uv` or `venv`), installs requirements, downloads MPI Sintel complete + depth training zips (curl resume + custom User-Agent), extracts with python `zipfile`, symlinks `flyvis.sintel_dir`, generates nulls (`python nulls.py --seeds 0 1 2`), and runs `pytest tests`. |
| [scripts/grid.tsv](file:///home/user/flyproj/course/scripts/grid.tsv) | **Created** | Tab-separated experiment grid of 24 configurations: M1–M5 × seeds 0–2 at 1.0 (15 runs) and M1, M2, M5 × seeds 0–2 at 0.25 (9 runs), with `$N_ITERS` and `$LR` placeholders. |
| [scripts/run_grid.sh](file:///home/user/flyproj/course/scripts/run_grid.sh) | **Created** | Grid execution engine supporting configurable concurrent jobs per GPU (`JOBS_PER_GPU`), multi-GPU distribution, run resumption (`summary.json` with status `"completed"`), per-run stdout logging (`runs/<name>/stdout.log`), optional per-run timeout, and dry-run mode (`DRY_RUN=1`). |
| [scripts/slurm_array.sbatch](file:///home/user/flyproj/course/scripts/slurm_array.sbatch) | **Created** | SLURM batch array script (array 1–24) mapping each task index directly to the corresponding row in `scripts/grid.tsv`. |
| [README.md](file:///home/user/flyproj/course/README.md) | **Created** | Two-paragraph project description from PLAN.md, M4 parameter matching and factory switch guide, local & remote quick starts, grid execution, test evaluation protocol (`eval.py`), reproduction guide, and data citations. |

---

## 2. Part 1: Param-Matched Model M4

### Parameter Target Analysis
- **Connectome Model M1 Trainable Parameters**:
  - Network (resting potentials + time constants): **734**
  - Decoder Readout Heads (optic flow + depth): **14,653**
  - **Total**: **15,387**
- **Allowable ±5% Margin**:
  - Minimum (-5%): $\lfloor 15,387 \times 0.95 \rfloor = \mathbf{14,618}$
  - Maximum (+5%): $\lceil 15,387 \times 1.05 \rceil = \mathbf{16,156}$

### Architecture & Parameter Breakdown
The new factory `make_small_matched()` in `baselines/hex_models.py` configures `HexConvGRUNet` with:
- `hid_ch = 15`
- `n_layers = 2`
- `head_ch = 18`
- `in_ch = 1`
- `extent = 15` (721 hexals)

Trainable parameter breakdown:
- **Encoder (2 layers)**:
  - Layer 1 (`HexConv(1, 15)`): $15 \times (7 \times 1 + 1) = 120$
  - Layer 2 (`HexConv(15, 15)`): $15 \times (7 \times 15 + 1) = 1,590$
  - *Encoder subtotal*: $1,710$
- **ConvGRU Cell**:
  - Reset & Update Gates (`HexConv(30, 30)`): $30 \times (7 \times 30 + 1) = 6,330$
  - Candidate Hidden State (`HexConv(30, 15)`): $15 \times (7 \times 30 + 1) = 3,165$
  - *ConvGRU subtotal*: $9,495$
- **Multi-Task Readout Heads**:
  - Flow Head (`HexConv(15, 18)` + `HexConv(18, 2)`): $18 \times 106 + 2 \times 127 = 1,908 + 254 = 2,162$
  - Depth Head (`HexConv(15, 18)` + `HexConv(18, 1)`): $18 \times 106 + 1 \times 127 = 1,908 + 127 = 2,035$
  - *Heads subtotal*: $4,197$
- **Total Trainable Parameters**:
  $$1,710 + 9,495 + 4,197 = \mathbf{15,402}$$
  - Absolute difference: **+15 parameters**
  - Relative difference: **+0.10%** (well within the ±5.0% tolerance).

### One-Line Change in `models.py`
To train `--model m4` with the matched model rather than the legacy 4,315-parameter model, the single line change in `models.py` (line 103) is:

```diff
- return {"m4": hex_models.make_small, "m5": hex_models.make_large}[model]()
+ return {"m4": hex_models.make_small_matched, "m5": hex_models.make_large}[model]()
```

---

## 3. Part 2: Portability & Execution Scripts

### `requirements.txt`
Pinned packages match the local environment:
- `python >= 3.12`
- `torch==2.14.1` (with CUDA index documentation for A100 VM / clusters)
- `flyvis==1.2.0`
- `datamate==1.0.0`
- `h5py==3.16.0`
- `numpy==2.5.2`
- `scipy==1.18.1`
- `pandas==3.0.6`
- `matplotlib==3.11.2`
- `pytest==9.1.1`

### `scripts/setup_remote.sh`
- **Idempotency**: Detects existing venvs and existing extracted directories (`training/final`, `training/flow`, `training/depth`, `test`) to avoid redundant downloads or overwrites.
- **Venv Creation**: Uses `uv venv` if `uv` is available, falling back to `python3 -m venv`.
- **Download & Extraction**: Uses `curl -fSL -C - -A "fly-eye-multitask-research"` to resume interrupted downloads. Extracts archives via Python's built-in `zipfile.ZipFile` module and cleans up zip archives.
- **Link & Graph Initialization**: Reads `flyvis.sintel_dir` dynamically and symlinks to `DATA_DIR` (supports `DATA_DIR` override); precomputes M2/M3 null connectomes for seeds 0, 1, 2; runs `pytest -q tests`.

### `scripts/grid.tsv`
Defined 24 experiment grid rows:
- 15 runs at data fraction 1.0 (M1, M2, M3, M4, M5 × seeds 0, 1, 2)
- 9 runs at data fraction 0.25 (M1, M2, M5 × seeds 0, 1, 2)
- Columns: `run_name`, `model`, `seed`, `data_fraction`, `n_iters`, `lr`

### `scripts/run_grid.sh`
- **Concurrency**: Manages slots using `JOBS_PER_GPU` (default 1) across detected or specified GPUs (`GPUS` or `CUDA_VISIBLE_DEVICES`).
- **Resumability**: Checks if `runs/<name>/summary.json` contains `"status": "completed"` and skips previously completed runs.
- **Logging**: Per-run output redirected to `runs/<name>/stdout.log`.
- **Timeouts**: Optional per-run execution timeout via `TIMEOUT` env var.
- **Dry-Run**: `DRY_RUN=1` prints launch commands without executing.

### `scripts/slurm_array.sbatch`
- Standard SLURM template for 24 array tasks (`#SBATCH --array=1-24`).
- Maps `SLURM_ARRAY_TASK_ID` to row in `scripts/grid.tsv`.

---

## 4. Verification and Test Results

### 1. Parameter Count Verification (`baselines/hex_models.py`)
Command:
```bash
CUDA_VISIBLE_DEVICES="" /home/user/flyproj/.venv/bin/python baselines/hex_models.py
```
Output:
```
=== HexConvGRUNet Parameter Counts ===
make_small(): 4,315 params (target: 3,000 - 6,000)
make_small_matched(): 15,402 params (target: 15,387 +- 5% -> [14,618, 16,156])
make_large(): 604,835 params (target: 400,000 - 1,000,000)
All parameter count checks passed!
```

### 2. Full Test Suite Execution (`tests/` and `baselines/test_baselines.py`)
Command:
```bash
CUDA_VISIBLE_DEVICES="" /home/user/flyproj/.venv/bin/python -m pytest -q tests baselines/test_baselines.py
```
Output:
```
..................................                                       [100%]
=============================== warnings summary ===============================
tests/test_splits.py::test_datasets_only_contain_their_scenes
  /home/user/flyproj/.venv/lib/python3.12/site-packages/torch/utils/_device.py:122: UserWarning: Using a non-tuple sequence for multidimensional indexing is deprecated and will be changed in pytorch 2.9; use x[tuple(seq)] instead of x[seq]. In pytorch 2.9 this will be interpreted as tensor index, x[torch.tensor(seq)], which will result either in an error or a different result (Triggered internally at /__w/pytorch/pytorch/torch/csrc/autograd/python_variable_indexing.cpp:356.)
    return func(*args, **kwargs)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
34 passed, 1 warning in 5.90s
```

### 3. Shell Syntax Validation (`bash -n`)
Command:
```bash
bash -n scripts/setup_remote.sh && bash -n scripts/run_grid.sh && bash -n scripts/slurm_array.sbatch
```
Result: All scripts passed with exit code 0.

### 4. Runner Dry-Run Validation (`DRY_RUN=1 scripts/run_grid.sh`)
Command:
```bash
DRY_RUN=1 scripts/run_grid.sh
```
Output:
```
/home/user/flyproj/course/../.venv/bin/python train.py --model m1 --seed 0 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m1_s0_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m1_s0_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m1 --seed 1 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m1_s1_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m1_s1_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m1 --seed 2 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m1_s2_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m1_s2_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m2 --seed 0 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m2_s0_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m2_s0_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m2 --seed 1 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m2_s1_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m2_s1_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m2 --seed 2 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m2_s2_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m2_s2_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m3 --seed 0 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m3_s0_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m3_s0_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m3 --seed 1 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m3_s1_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m3_s1_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m3 --seed 2 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m3_s2_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m3_s2_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m4 --seed 0 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m4_s0_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m4_s0_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m4 --seed 1 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m4_s1_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m4_s1_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m4 --seed 2 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m4_s2_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m4_s2_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m5 --seed 0 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m5_s0_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m5_s0_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m5 --seed 1 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m5_s1_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m5_s1_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m5 --seed 2 --data-fraction 1.0 --n-iters 30000 --lr 5e-5 --name m5_s2_f1.0 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m5_s2_f1.0/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m1 --seed 0 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m1_s0_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m1_s0_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m1 --seed 1 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m1_s1_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m1_s1_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m1 --seed 2 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m1_s2_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m1_s2_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m2 --seed 0 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m2_s0_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m2_s0_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m2 --seed 1 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m2_s1_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m2_s1_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m2 --seed 2 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m2_s2_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m2_s2_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m5 --seed 0 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m5_s0_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m5_s0_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m5 --seed 1 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m5_s1_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m5_s1_f0.25/stdout.log 2>&1
/home/user/flyproj/course/../.venv/bin/python train.py --model m5 --seed 2 --data-fraction 0.25 --n-iters 30000 --lr 5e-5 --name m5_s2_f0.25 --out-dir /home/user/flyproj/course/runs > /home/user/flyproj/course/runs/m5_s2_f0.25/stdout.log 2>&1
```
