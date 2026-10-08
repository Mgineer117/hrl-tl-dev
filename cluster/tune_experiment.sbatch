#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --job-name="hrl-tune"
#SBATCH --output=cluster/logs/tune/%x/%x_o%j.log

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-${SLURM_SUBMIT_DIR:-/projects/illinois/eng/aero/huytran1/myuasa2/hrl-tl}}"
cd "${WORKSPACE_DIR}"
source "${WORKSPACE_DIR}/cluster/env_setup.sh"

# ---------------------------------------------------------------------------
# Resolve target tuning script and parameters
# ---------------------------------------------------------------------------

TARGET_SCRIPT="${1:-${TUNE_SCRIPT:-}}"
if [[ "$#" -gt 0 && "${1}" == *.py ]]; then
  shift
fi

if [[ -z "${TARGET_SCRIPT}" ]]; then
  echo "Error: Target tuning script must be provided via argument or TUNE_SCRIPT." >&2
  echo "Usage: sbatch $0 <SCRIPT_PATH> [OPTIONS]" >&2
  exit 1
fi

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Error: Tuning script not found at ${TARGET_SCRIPT}" >&2
  exit 1
fi

ENV_NAME="${ENV_NAME:-fr_cont}"
DEVICE="${DEVICE:-cuda:0}"
EXTRA_ARGS="${EXTRA_ARGS:-$*}"

CMD_ARGS=(
  "--device=${DEVICE}"
)

if [[ -n "${ENV_NAME}" ]]; then
  CMD_ARGS+=("--env=${ENV_NAME}")
fi

if [[ -n "${TRIALS:-}" ]]; then
  CMD_ARGS+=("--trials=${TRIALS}")
fi

if [[ -n "${TIMESTEPS:-}" ]]; then
  CMD_ARGS+=("--timesteps=${TIMESTEPS}")
fi

if [[ "${TRAIN_REPLICATES:-false}" == "true" || "${TRAIN_REPLICATES:-0}" == "1" ]]; then
  CMD_ARGS+=("--train_replicates")
fi

echo "======================================================================"
echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Node:              $(hostname)"
echo "CUDA Devices:      ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Tuning Script:     ${TARGET_SCRIPT}"
echo "Target Env:        ${ENV_NAME}"
echo "Assigned Device:   ${DEVICE}"
echo "Optuna Args:       ${CMD_ARGS[*]} ${EXTRA_ARGS}"
echo "Start Time:        $(date)"
echo "======================================================================"

python "${TARGET_SCRIPT}" "${CMD_ARGS[@]}" ${EXTRA_ARGS}

echo "======================================================================"
echo "Tuning finished at $(date)"
echo "======================================================================"
