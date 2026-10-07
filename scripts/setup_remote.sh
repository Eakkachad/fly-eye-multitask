#!/usr/bin/env bash
# ==============================================================================
# scripts/setup_remote.sh
#
# Idempotent setup script for remote Linux + CUDA environments (A100 VM, Colab,
# SLURM cluster).
#
# Tasks performed:
# 1. Creates a Python 3.12 virtual environment (using uv if available, else python -m venv).
# 2. Installs dependencies from requirements.txt.
# 3. Downloads MPI Sintel complete and depth-training zip archives (with resume and
#    custom User-Agent) if not already present.
# 4. Extracts datasets using Python's zipfile module (no system unzip needed) and removes zips.
# 5. Symlinks flyvis.sintel_dir to the dataset directory.
# 6. Pre-generates null connectome graphs via `python nulls.py --seeds 0 1 2`.
# 7. Executes test suite via `python -m pytest -q tests`.
#
# Overridable environment variables:
#   DATA_DIR   Target dataset directory (default: $HOME/flyproj/data/sintel)
#   VENV_DIR   Virtual environment directory (default: .venv in repo root)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COURSE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${COURSE_DIR}"

# 1. Configuration & Data Directory
DATA_DIR="${DATA_DIR:-${HOME}/flyproj/data/sintel}"
VENV_DIR="${VENV_DIR:-${COURSE_DIR}/.venv}"
DEFAULT_COURSE_DATA="${HOME}/flyproj/data/sintel"

echo "=== Setup Remote Environment ==="
echo "Course directory: ${COURSE_DIR}"
echo "Data directory:   ${DATA_DIR}"
echo "Venv directory:   ${VENV_DIR}"

mkdir -p "${DATA_DIR}"

# Ensure standard path $HOME/flyproj/data/sintel points to DATA_DIR if overridden
if [ "${DATA_DIR}" != "${DEFAULT_COURSE_DATA}" ]; then
    mkdir -p "$(dirname "${DEFAULT_COURSE_DATA}")"
    if [ ! -e "${DEFAULT_COURSE_DATA}" ]; then
        ln -sfn "${DATA_DIR}" "${DEFAULT_COURSE_DATA}"
        echo "Linked ${DEFAULT_COURSE_DATA} -> ${DATA_DIR}"
    fi
fi

# 2. Virtual Environment Creation & Activation
if [ ! -f "${VENV_DIR}/bin/python" ]; then
    if command -v uv >/dev/null 2>&1; then
        echo "Creating virtual environment using uv..."
        uv venv "${VENV_DIR}" --python 3.12 2>/dev/null || uv venv "${VENV_DIR}"
    else
        echo "Creating virtual environment using python3 -m venv..."
        python3 -m venv "${VENV_DIR}"
    fi
else
    echo "Virtual environment already exists at ${VENV_DIR}."
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

# 3. Install Dependencies
echo "Installing requirements from requirements.txt..."
if command -v uv >/dev/null 2>&1; then
    uv pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

# 4. Download and Extract MPI Sintel Data (Idempotent)
COMPLETE_URL="https://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip"
DEPTH_URL="https://files.is.tue.mpg.de/jwulff/sintel/MPI-Sintel-depth-training-20150305.zip"

COMPLETE_ZIP="${DATA_DIR}/MPI-Sintel-complete.zip"
DEPTH_ZIP="${DATA_DIR}/MPI-Sintel-depth-training-20150305.zip"

# Check if complete dataset already extracted
if [ -d "${DATA_DIR}/training/final" ] && [ -d "${DATA_DIR}/training/flow" ] && [ -d "${DATA_DIR}/test" ]; then
    echo "MPI Sintel complete already extracted in ${DATA_DIR}."
else
    echo "Downloading MPI Sintel complete..."
    curl -fSL -C - -A "kagpt-fly-research" "${COMPLETE_URL}" -o "${COMPLETE_ZIP}"
    echo "Extracting MPI Sintel complete with python zipfile..."
    python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "${COMPLETE_ZIP}" "${DATA_DIR}"
    rm -f "${COMPLETE_ZIP}"
fi

# Check if depth training dataset already extracted
if [ -d "${DATA_DIR}/training/depth" ]; then
    echo "MPI Sintel depth training already extracted in ${DATA_DIR}."
else
    echo "Downloading MPI Sintel depth training..."
    curl -fSL -C - -A "kagpt-fly-research" "${DEPTH_URL}" -o "${DEPTH_ZIP}"
    echo "Extracting MPI Sintel depth training with python zipfile..."
    python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "${DEPTH_ZIP}" "${DATA_DIR}"
    rm -f "${DEPTH_ZIP}"
fi

# 5. Symlink flyvis.sintel_dir to DATA_DIR
FLYVIS_SINTEL_DIR="$(python -c "import flyvis; print(flyvis.sintel_dir)")"
echo "Target flyvis.sintel_dir: ${FLYVIS_SINTEL_DIR}"

if [ -L "${FLYVIS_SINTEL_DIR}" ]; then
    rm "${FLYVIS_SINTEL_DIR}"
elif [ -d "${FLYVIS_SINTEL_DIR}" ]; then
    # If empty directory, remove it so symlink can be created
    if [ -z "$(ls -A "${FLYVIS_SINTEL_DIR}")" ]; then
        rmdir "${FLYVIS_SINTEL_DIR}"
    fi
fi

mkdir -p "$(dirname "${FLYVIS_SINTEL_DIR}")"
ln -sfn "${DATA_DIR}" "${FLYVIS_SINTEL_DIR}"
echo "Symlinked ${FLYVIS_SINTEL_DIR} -> ${DATA_DIR}"

# 6. Pre-generate Null Connectomes (M2, M3 for seeds 0, 1, 2)
echo "Generating null connectomes (seeds 0, 1, 2)..."
python nulls.py --seeds 0 1 2

# 7. Run Pytest Suite
echo "Running test suite..."
python -m pytest -q tests

echo "=== Setup complete! ==="
