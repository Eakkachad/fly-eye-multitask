# Running the project (technical guide)

> Thai overview: see [../README.md](../README.md). This file: setup, grid, evaluation, reproduction.


## Project Overview

Given a short grayscale video sequence as seen by the fruit fly eye—modeled as a regular hexagonal photoreceptor lattice of 721 ommatidia (hexals) spanning 19 frames at 50 Hz—this project investigates whether biological connectome wiring acts as an effective inductive bias for multi-task video perception. The network must integrate temporal information across frames to jointly predict pixel-wise **optic flow** (2-D vector per hexal) and **depth** (scalar per hexal) at the final frames. We evaluate five architectures under an identical experimental protocol: the real connectome dynamical network (**M1**), a degree-preserving rewired structural null (**M2**), an Erdős–Rényi random type-graph null (**M3**), a parameter-matched hexagonal ConvGRU network (**M4**), and an overparameterized deep ConvGRU baseline (**M5**).

The core scientific question is whether the connectome prior can replace both raw parameters and training data. We test three pre-registered hypotheses: (H1) whether the real connectome (M1) outperforms structural nulls (M2/M3) across paired random seeds; (H2) whether M1 outperforms a standard deep learning model matched for parameter budget (M4); and (H3) whether connectome-constrained networks achieve superior sample efficiency when training data is restricted to 25% of training scenes. All models are trained from scratch using identical multi-task decoder heads, L2-norm loss objectives, Adam optimizer schedules, and pre-registered scene-level data splits on the MPI Sintel benchmark.

---

## Quick Start

### 1. Local Environment (CPU Testing)
To run the test suite on CPU only:

```bash
# Set CUDA_VISIBLE_DEVICES="" for all local commands
export CUDA_VISIBLE_DEVICES=""

# Run test suites on CPU
/home/user/flyproj/.venv/bin/python -m pytest -q tests baselines/test_baselines.py
```

### 2. Remote Setup (A100 VM / Generic Linux GPU Machine)
To set up a fresh remote Linux instance with CUDA (e.g. an A100 GPU box, Colab, or cloud VM):

```bash
# Clone repository and run automated setup
cd /path/to/course
bash scripts/setup_remote.sh
```

`scripts/setup_remote.sh` is completely idempotent and executes the following steps:
1. Creates a Python 3.12 virtual environment (using `uv` if installed, otherwise `python3 -m venv`).
2. Installs pinned dependencies from `requirements.txt`.
3. Downloads MPI Sintel complete and depth-training zip archives (with resume support and custom `User-Agent: kagpt-fly-research`).
4. Extracts archives using Python's built-in `zipfile` module (eliminating dependencies on system `unzip`) and deletes zips.
5. Dynamically queries `python -c "import flyvis; print(flyvis.sintel_dir)"` and symlinks `flyvis.sintel_dir` to the dataset directory.
6. Pre-generates null connectome graphs for M2 and M3 (`python nulls.py --seeds 0 1 2`).
7. Runs the test suite to verify installation (`python -m pytest -q tests`).

*Note: The dataset location defaults to `$HOME/flyproj/data/sintel` and can be overridden via `DATA_DIR`:*
```bash
DATA_DIR=/mnt/fast_storage/sintel bash scripts/setup_remote.sh
```

---

## Experiment Grid

The full pre-registered experiment grid is defined in `scripts/grid.tsv` across 24 configurations:
- **Full Data (1.0 fraction)**: Models `m1`, `m2`, `m3`, `m4`, `m5` across seeds `0`, `1`, `2` (15 runs).
- **Data-Efficiency Arm (0.25 fraction)**: Models `m1`, `m2`, `m5` across seeds `0`, `1`, `2` (9 runs).

### Running on a Single / Multi-GPU Server (`scripts/run_grid.sh`)
The runner script `scripts/run_grid.sh` manages execution, concurrency, and logging:

```bash
# Validate commands without running (Dry-run mode)
DRY_RUN=1 scripts/run_grid.sh

# Run grid on a single GPU (e.g. A100 with 2 concurrent jobs)
JOBS_PER_GPU=2 scripts/run_grid.sh

# Run grid distributed across 4 GPUs (GPUs 0, 1, 2, 3)
GPUS="0,1,2,3" JOBS_PER_GPU=2 scripts/run_grid.sh

# Custom iterations, learning rate, and per-run timeout
N_ITERS=30000 LR=5e-5 TIMEOUT=14400 scripts/run_grid.sh
```

**Key Features:**
- **Resumable**: Automatically checks `runs/<name>/summary.json` for `"status": "completed"` and skips already finished runs.
- **Isolated Logging**: Captures `stdout` and `stderr` into `runs/<name>/stdout.log`.
- **Configurable Concurrency**: Set `JOBS_PER_GPU` (default 1; multiple runs fit on an 80GB A100).
- **Graceful Timeouts**: Optional per-run wall-clock timeout enforced via `TIMEOUT`.

### Running on a SLURM Cluster (`scripts/slurm_array.sbatch`)
To dispatch the 24 grid runs as an array job on a SLURM cluster:

```bash
sbatch scripts/slurm_array.sbatch
```
Each task `$SLURM_ARRAY_TASK_ID` (1 to 24) maps to its respective row in `scripts/grid.tsv`, respects run resumption, and writes to `runs/<name>/stdout.log`.

---

## Evaluation on Test Scenes (`eval.py`)

### Strict Evaluation Protocol
To prevent data snooping and information leakage:
1. No model hyperparameters are tuned on the test split.
2. The test split is evaluated **exactly once** after training is finalized.
3. `eval.py` enforces this by refusing to re-run on a test directory unless `--force` is specified.

### Running Test Evaluation
After a model has finished training:

```bash
# Evaluate best checkpoint (best.pt) on the held-out test scenes
python eval.py runs/m1_s0_f1.0

# Optional: debug/verify evaluation pipeline on validation split without touching test
python eval.py runs/m1_s0_f1.0 --split val

# Force re-evaluation if necessary
python eval.py runs/m1_s0_f1.0 --force
```

### Generated Artifacts
Evaluation results are saved in `runs/<name>/test/`:
- `metrics.json`: End-Point Error (EPE), angular error (degrees), depth RMSE, depth AbsRel, per-sequence metrics, and binned errors by ground-truth flow speed and local texture (luminance variance).
- `per_pixel.npz`: Compressed per-pixel error arrays and bin indicators across all valid test frames.
- `examples.npz`: Visualizations comparing predictions against ground truth for best/worst test sequences.

---

## Reproduction Guide

To reproduce the study from scratch on a remote CUDA box:

1. **Environment & Data Setup**:
   ```bash
   bash scripts/setup_remote.sh
   ```
2. **M4** already uses `hex_models.make_small_matched` (15,402 params, matched to M1's 15,387).
3. **Execute Training Grid**:
   ```bash
   JOBS_PER_GPU=2 scripts/run_grid.sh
   ```
4. **Evaluate Test Splits**:
   ```bash
   for run in runs/*_f*; do
       if [ -d "$run" ]; then
           python eval.py "$run"
       fi
   done
   ```

---

## Data Sources & Licensing

- **MPI Sintel Dataset**:
  - Reference: Butler et al., *A naturalistic open source movie for optical flow evaluation*, European Conference on Computer Vision (ECCV), 2012.
  - License / Usage: Research use only. Extracted under permission for academic benchmarking.
  - Sintel dataset homepage: `https://files.is.tue.mpg.de/sintel/`
- **Flyvis Framework**:
  - Reference: Lappalainen et al., *Connectome-constrained networks predict neural activity in the fly visual system*, Nature, 2024.
  - License: MIT License.
