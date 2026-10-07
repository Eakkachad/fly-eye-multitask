#!/usr/bin/env bash
# ==============================================================================
# scripts/run_grid.sh
#
# Launches experiment runs from scripts/grid.tsv.
#
# Features:
# - Resumable: automatically skips runs with runs/<name>/summary.json ("status": "completed").
# - Concurrent execution: configurable number of jobs per GPU (JOBS_PER_GPU, default 1).
# - Multi-GPU aware: automatically distributes jobs across GPUs in GPUS or CUDA_VISIBLE_DEVICES.
# - Logging: redirects per-run stdout and stderr to runs/<name>/stdout.log.
# - Optional timeout: per-run timeout enforced via TIMEOUT env var (e.g. TIMEOUT=4000).
# - Dry-run mode: DRY_RUN=1 prints commands only without launching.
#
# Environment variables:
#   GRID_FILE      Path to grid TSV (default: scripts/grid.tsv)
#   OUT_DIR        Output runs directory (default: runs/)
#   JOBS_PER_GPU   Concurrent jobs to run per GPU (default: 1)
#   GPUS           GPU IDs to use, comma- or space-separated (default: detected or 0)
#   TIMEOUT        Optional timeout per run (e.g. 3600s, 4000)
#   DRY_RUN        Set to 1 to only print commands without running
#   N_ITERS        Default iterations if placeholder used (default: 30000)
#   LR             Default learning rate if placeholder used (default: 5e-5)
#   PYTHON         Python executable to use (default: venv python if present, else python)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COURSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${COURSE_DIR}"

# 1. Configuration & Defaults
GRID_FILE="${GRID_FILE:-${COURSE_DIR}/scripts/grid.tsv}"
OUT_DIR="${OUT_DIR:-${COURSE_DIR}/runs}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
TIMEOUT="${TIMEOUT:-}"
DRY_RUN="${DRY_RUN:-0}"
N_ITERS_DEFAULT="${N_ITERS:-30000}"
LR_DEFAULT="${LR:-5e-5}"

# Locate Python
if [ -n "${PYTHON:-}" ]; then
    PY_BIN="${PYTHON}"
elif [ -f "${COURSE_DIR}/.venv/bin/python" ]; then
    PY_BIN="${COURSE_DIR}/.venv/bin/python"
elif [ -f "${COURSE_DIR}/../.venv/bin/python" ]; then
    PY_BIN="${COURSE_DIR}/../.venv/bin/python"
else
    PY_BIN="python"
fi

if [ ! -f "${GRID_FILE}" ]; then
    echo "Error: Grid file not found: ${GRID_FILE}" >&2
    exit 1
fi

# 2. Determine GPUs & Slots
# Check if CUDA_VISIBLE_DEVICES was explicitly set to empty (CPU mode)
if [ "${CUDA_VISIBLE_DEVICES+set}" = "set" ] && [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    GPU_LIST=("")
elif [ -n "${GPUS:-}" ]; then
    IFS=', ' read -r -a GPU_LIST <<< "${GPUS}"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=', ' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
else
    GPU_LIST=("0")
fi

NUM_GPUS=${#GPU_LIST[@]}
TOTAL_CONCURRENCY=$(( NUM_GPUS * JOBS_PER_GPU ))
if [ "${TOTAL_CONCURRENCY}" -lt 1 ]; then
    TOTAL_CONCURRENCY=1
fi

if [ "${DRY_RUN}" != "1" ]; then
    mkdir -p "${OUT_DIR}"
fi

# 3. Helper: check if a run has already completed
is_completed() {
    local run_name="$1"
    local summary_file="${OUT_DIR}/${run_name}/summary.json"
    if [ -f "${summary_file}" ]; then
        if grep -q '"status": *"completed"' "${summary_file}" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# 4. Job Slot Management
available_slots=()
for ((i=0; i<TOTAL_CONCURRENCY; i++)); do
    available_slots+=("$i")
done

active_pids=()
declare -A pid_slot_map=()

# 5. Read grid.tsv and execute / print
line_no=0
while IFS=$'\t' read -r run_name model seed data_fraction n_iters lr || [ -n "${run_name}" ]; do
    line_no=$((line_no + 1))
    # Skip header
    if [ "${line_no}" -eq 1 ] && [ "${run_name}" = "run_name" ]; then
        continue
    fi
    # Skip empty lines or comments
    [[ -z "${run_name}" || "${run_name}" =~ ^# ]] && continue

    # Resolve placeholders
    if [[ "${n_iters}" == *N_ITERS* ]] || [ -z "${n_iters}" ]; then
        n_iters="${N_ITERS_DEFAULT}"
    fi
    if [[ "${lr}" == *LR* ]] || [ -z "${lr}" ]; then
        lr="${LR_DEFAULT}"
    fi

    # Check for completion
    if is_completed "${run_name}"; then
        echo "[SKIP] ${run_name} already completed."
        continue
    fi

    # Dry-run handling
    if [ "${DRY_RUN}" = "1" ]; then
        cmd_str=""
        if [ -n "${TIMEOUT}" ]; then
            cmd_str+="timeout --signal=KILL ${TIMEOUT} "
        fi
        cmd_str+="${PY_BIN} train.py --model ${model} --seed ${seed} --data-fraction ${data_fraction} --n-iters ${n_iters} --lr ${lr} --name ${run_name} --out-dir ${OUT_DIR} > ${OUT_DIR}/${run_name}/stdout.log 2>&1"
        echo "${cmd_str}"
        continue
    fi

    # Wait for an available execution slot
    while [ "${#available_slots[@]}" -eq 0 ]; do
        wait -n 2>/dev/null || true
        new_active_pids=()
        for p in "${active_pids[@]}"; do
            if kill -0 "${p}" 2>/dev/null; then
                new_active_pids+=("${p}")
            else
                slot="${pid_slot_map[${p}]}"
                available_slots+=("${slot}")
                wait "${p}" 2>/dev/null || true
                unset "pid_slot_map[${p}]"
            fi
        done
        active_pids=("${new_active_pids[@]}")
    done

    # Assign slot and GPU
    slot="${available_slots[0]}"
    available_slots=("${available_slots[@]:1}")
    gpu_idx=$(( slot % NUM_GPUS ))
    target_gpu="${GPU_LIST[${gpu_idx}]}"

    run_dir="${OUT_DIR}/${run_name}"
    mkdir -p "${run_dir}"
    log_file="${run_dir}/stdout.log"

    echo "[LAUNCH] ${run_name} (model=${model}, seed=${seed}, fraction=${data_fraction}) on GPU '${target_gpu}', slot ${slot}"

    # Build launch command
    timeout_prefix=()
    if [ -n "${TIMEOUT}" ]; then
        timeout_prefix=("timeout" "--signal=KILL" "${TIMEOUT}")
    fi

    train_cmd=(
        "${PY_BIN}" "train.py"
        "--model" "${model}"
        "--seed" "${seed}"
        "--data-fraction" "${data_fraction}"
        "--n-iters" "${n_iters}"
        "--lr" "${lr}"
        "--name" "${run_name}"
        "--out-dir" "${OUT_DIR}"
    )

    # Launch background job
    (
        if [ -n "${target_gpu}" ]; then
            export CUDA_VISIBLE_DEVICES="${target_gpu}"
        fi
        "${timeout_prefix[@]}" "${train_cmd[@]}" > "${log_file}" 2>&1
    ) &
    pid=$!

    pid_slot_map["${pid}"]="${slot}"
    active_pids+=("${pid}")

done < "${GRID_FILE}"

# 6. Wait for remaining jobs (in non-dry-run mode)
if [ "${DRY_RUN}" != "1" ] && [ "${#active_pids[@]}" -gt 0 ]; then
    echo "Waiting for remaining ${#active_pids[@]} job(s) to complete..."
    wait
    echo "All grid jobs finished."
fi
