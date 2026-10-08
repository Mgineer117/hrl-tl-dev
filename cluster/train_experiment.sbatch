#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-task=1
#SBATCH --mem=150G
#SBATCH --job-name="hrl-train"
#SBATCH --output=cluster/logs/train/%x/%x_o%A_%a.log

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-${SLURM_SUBMIT_DIR:-/projects/illinois/eng/aero/huytran1/myuasa2/hrl-tl}}"
cd "${WORKSPACE_DIR}"
source "${WORKSPACE_DIR}/cluster/env_setup.sh"

# ---------------------------------------------------------------------------
# Resolve target Python script and configuration
# ---------------------------------------------------------------------------
# Precedence:
# 1. Positional arguments: $1 = script, $2 = config
# 2. Environment variables: EXP_SCRIPT, EXP_CONFIG
# 3. Experiment manifest: EXP_MANIFEST (reads line for SLURM_ARRAY_TASK_ID)
# ---------------------------------------------------------------------------

TARGET_SCRIPT="${1:-${EXP_SCRIPT:-}}"
CONFIG_PATH="${2:-${EXP_CONFIG:-}}"
DEVICE="${DEVICE:-cuda:0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
REPLICATE_ARRAY="${REPLICATE_ARRAY:-0}"

# Check if an experiment manifest file is specified
if [[ -n "${EXP_MANIFEST:-}" && -f "${EXP_MANIFEST}" ]]; then
  ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
  LINE_NUM="$((ARRAY_ID + 1))"
  MANIFEST_LINE="$(sed -n "${LINE_NUM}p" "${EXP_MANIFEST}" | xargs)"
  if [[ -z "${MANIFEST_LINE}" ]]; then
    echo "Error: Manifest line ${LINE_NUM} in ${EXP_MANIFEST} is empty." >&2
    exit 1
  fi

  # Line may be "script config" or just "config"
  read -r item1 item2 _ <<< "${MANIFEST_LINE}"
  if [[ "${item1}" == *.yaml || "${item1}" == *.yml ]]; then
    CONFIG_PATH="${item1}"
  else
    TARGET_SCRIPT="${item1}"
    CONFIG_PATH="${item2:-${CONFIG_PATH}}"
  fi
fi

if [[ -z "${TARGET_SCRIPT}" ]]; then
  echo "Error: Target Python script must be provided via argument, EXP_SCRIPT, or EXP_MANIFEST." >&2
  echo "Usage: sbatch $0 <SCRIPT_PATH> <CONFIG_PATH>" >&2
  exit 1
fi

if [[ -z "${CONFIG_PATH}" ]]; then
  echo "Error: Configuration path must be provided via argument, EXP_CONFIG, or EXP_MANIFEST." >&2
  echo "Usage: sbatch $0 <SCRIPT_PATH> <CONFIG_PATH>" >&2
  exit 1
fi

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Error: Python script not found at ${TARGET_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Error: Config YAML not found at ${CONFIG_PATH}" >&2
  exit 1
fi

# Build replicate arguments when running a replicate array
REPLICATE_ARGS=()
if [[ "${REPLICATE_ARRAY}" == "1" && -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${EXP_MANIFEST:-}" ]]; then
  REPLICATE_ARGS+=("--replicate_start_id=${SLURM_ARRAY_TASK_ID}" "--num_replicates=1")
fi

echo "======================================================================"
echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Array Task ID:     ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Replicate Array:   ${REPLICATE_ARRAY}"
echo "Node:              $(hostname)"
echo "CUDA Devices:      ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "Python Script:     ${TARGET_SCRIPT}"
echo "Config YAML:       ${CONFIG_PATH}"
echo "Assigned Device:   ${DEVICE}"
echo "Extra Args:        ${EXTRA_ARGS}"
echo "Start Time:        $(date)"
echo "======================================================================"

python "${TARGET_SCRIPT}" \
  --exp_config_path="${CONFIG_PATH}" \
  --device="${DEVICE}" \
  ${REPLICATE_ARGS[@]+"${REPLICATE_ARGS[@]}"} \
  ${EXTRA_ARGS}

echo "======================================================================"
echo "Training finished at $(date)"
echo "======================================================================"
