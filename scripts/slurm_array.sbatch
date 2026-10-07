#!/usr/bin/env bash
#SBATCH --job-name=fly_grid
#SBATCH --output=runs/slurm_%A_%a.out
#SBATCH --error=runs/slurm_%A_%a.err
#SBATCH --array=1-24
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# ==============================================================================
# scripts/slurm_array.sbatch
#
# SLURM array template to execute the 24 pre-registered experiment grid tasks.
# Each array task index (1-24) maps to one row in scripts/grid.tsv.
#
# Usage:
#   sbatch scripts/slurm_array.sbatch
# ==============================================================================

set -euo pipefail

COURSE_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${COURSE_DIR}"

GRID_FILE="${COURSE_DIR}/scripts/grid.tsv"
OUT_DIR="${COURSE_DIR}/runs"
mkdir -p "${OUT_DIR}"

# 1. Activate Virtual Environment
if [ -f "${COURSE_DIR}/.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${COURSE_DIR}/.venv/bin/activate"
elif [ -f "${COURSE_DIR}/../.venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "${COURSE_DIR}/../.venv/bin/activate"
fi

# 2. Extract Row for this Task (Task ID 1 maps to row 2 due to header)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
ROW_LINE=$(( TASK_ID + 1 ))
ROW=$(sed -n "${ROW_LINE}p" "${GRID_FILE}")

if [ -z "${ROW}" ]; then
    echo "Error: No entry found in ${GRID_FILE} at line ${ROW_LINE} (task ID ${TASK_ID})" >&2
    exit 1
fi

IFS=$'\t' read -r RUN_NAME MODEL SEED DATA_FRACTION N_ITERS LR <<< "${ROW}"

# Resolve placeholders
if [[ "${N_ITERS}" == *N_ITERS* ]] || [ -z "${N_ITERS}" ]; then
    N_ITERS="${N_ITERS:-30000}"
fi
if [[ "${LR}" == *LR* ]] || [ -z "${LR}" ]; then
    LR="${LR:-5e-5}"
fi

RUN_DIR="${OUT_DIR}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
SUMMARY_FILE="${RUN_DIR}/summary.json"
LOG_FILE="${RUN_DIR}/stdout.log"

# 3. Check Resumability
if [ -f "${SUMMARY_FILE}" ] && grep -q '"status": *"completed"' "${SUMMARY_FILE}" 2>/dev/null; then
    echo "[SKIP] ${RUN_NAME} has already completed."
    exit 0
fi

echo "=== Task ${TASK_ID}: Launching ${RUN_NAME} ==="
echo "Node: $(hostname)"
echo "GPU:  ${CUDA_VISIBLE_DEVICES:-none}"
echo "Model: ${MODEL}, Seed: ${SEED}, Fraction: ${DATA_FRACTION}, Iterations: ${N_ITERS}, LR: ${LR}"

# 4. Execute Training
python "${COURSE_DIR}/train.py" \
    --model "${MODEL}" \
    --seed "${SEED}" \
    --data-fraction "${DATA_FRACTION}" \
    --n-iters "${N_ITERS}" \
    --lr "${LR}" \
    --name "${RUN_NAME}" \
    --out-dir "${OUT_DIR}" \
    > "${LOG_FILE}" 2>&1

echo "=== Task ${TASK_ID}: Finished ${RUN_NAME} ==="
