#!/bin/bash
# Environment setup for Illinois Campus Cluster batch jobs.
# Sources the uv-managed virtual environment and loads required cluster modules.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-${SLURM_SUBMIT_DIR:-/projects/illinois/eng/aero/huytran1/myuasa2/hrl-tl}}"
cd "${WORKSPACE_DIR}"

# Load CUDA module if module command is available
if command -v module >/dev/null 2>&1; then
  module load cuda/12.8 2>/dev/null || true
fi

# Activate uv-managed Python virtual environment
VENV_ACTIVATE="${WORKSPACE_DIR}/.venv/bin/activate"
if [[ ! -f "${VENV_ACTIVATE}" ]]; then
  echo "Error: Virtual environment not found at ${VENV_ACTIVATE}" >&2
  exit 1
fi
source "${VENV_ACTIVATE}"

# Set Python and runtime environment variables
export PYTHONPATH="${WORKSPACE_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Ensure log and output directories exist
mkdir -p "${WORKSPACE_DIR}/cluster/logs/train/hrl-train"
mkdir -p "${WORKSPACE_DIR}/cluster/logs/tune/hrl-tune"
mkdir -p "${WORKSPACE_DIR}/cluster/manifests"
mkdir -p "${WORKSPACE_DIR}/db"
mkdir -p "${WORKSPACE_DIR}/configs/tuning/best"

# Required for mujoco to run on cluster
export MUJOCO_GL="${MUJOCO_GL:-egl}"