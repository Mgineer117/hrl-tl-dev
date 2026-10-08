#!/bin/bash
# ==============================================================================
# Illinois Campus Cluster (ICC) Job Submission Wrapper for hrl-tl
# ==============================================================================
# Usage:
#   ./cluster/queue_experiment.sh train <SCRIPT> <CONFIG> [OPTIONS]
#   ./cluster/queue_experiment.sh tune <SCRIPT> [OPTIONS]
#   ./cluster/queue_experiment.sh campaign <MANIFEST_FILE> [OPTIONS]
#
# Examples:
#   # Train a single CPC experiment on default partition (eng-research-gpu)
#   ./cluster/queue_experiment.sh train scripts/cpc/main_tl_hrl_meta_option_fr_cont_rep_sdsac.py \
#       configs/fr_cont/train/3.j/3.j.e_hrl_train_rep3.yaml
#
#   # Train 5 replicates in parallel across GPUs on eng-research-gpu
#   ./cluster/queue_experiment.sh train scripts/cpc/train_primitive_rep.py \
#       configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml --replicates=5
#
#   # Run hyperparameter tuning on eng-research-gpu
#   ./cluster/queue_experiment.sh tune scripts/cpc/tune_tl_hrl_meta_option.py \
#       --env=fr_cont --trials=25 --timesteps=2000000
#
#   # Run batch campaign from manifest across partitions (eng first, ic for overflow)
#   ./cluster/queue_experiment.sh campaign cluster/manifests/example_campaign.txt --partition=all
#
#   # Dry-run test (checks Slurm validity without submitting)
#   ./cluster/queue_experiment.sh train scripts/cpc/train_primitive_rep.py \
#       configs/fr_cont/cpc/pr_soft/3.l.b_train_primitive_rep.yaml --replicates=5 --dry-run
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${WORKSPACE_DIR}"

TRAIN_SBATCH="${SCRIPT_DIR}/train_experiment.sbatch"
TUNE_SBATCH="${SCRIPT_DIR}/tune_experiment.sbatch"

# Default partition settings
IC_ACCOUNT="huytran1-ic"
IC_PARTITION="IllinoisComputes-GPU"
IC_CONCURRENCY=4
IC_CONCURRENCY_HARD_CAP=4

ENG_ACCOUNT="huytran1-ae-eng"
ENG_PARTITION="eng-research-gpu"
ENG_CONCURRENCY=21
ENG_CONCURRENCY_HARD_CAP=21

SEC_ACCOUNT="huytran1-ae-eng"
SEC_PARTITION="secondary"
SEC_CONCURRENCY=10
SEC_MAX_TIME="04:00:00"

# User-configurable defaults
PARTITION_ALIAS="eng"
ACCOUNT_OVERRIDE=""
PARTITION_OVERRIDE=""
CONCURRENCY_OVERRIDE=""
WALLTIME=""
CPUS=""
MEM=""
DEVICE="cuda:0"
EXCLUDE_NODES=""
DRY_RUN=0
ARRAY_SPEC=""
JOB_NAME_OVERRIDE=""
NUM_REPLICATES=""
REPLICATE_START_ID=0
REPLICATE_ARRAY=0
EXTRA_SBATCH_FLAGS=()
EXTRA_PYTHON_ARGS=()

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

print_usage() {
  cat <<EOF
Usage: $0 <COMMAND> [ARGS...] [OPTIONS]

Commands:
  train <SCRIPT> <CONFIG>   Submit a training job with specified script and config
  tune <SCRIPT>             Submit an Optuna hyperparameter tuning job with specified script
  campaign <MANIFEST>       Submit a job array over an experiment manifest file

Options:
  -p, --partition=NAME      Partition to use: 'eng' (eng-research-gpu),
                            'ic' (IllinoisComputes-GPU), 'secondary', or 'all' (eng first, ic overflow)
                            [default: eng]
  -r, --replicates=N        Submit N replicate jobs in parallel as a Slurm job array
      --replicate-start-id=S Starting replicate ID offset (default: 0)
  -a, --array=SPEC          Slurm array specification (e.g. '0-4%4')
  -c, --concurrency=N       Array concurrency limit
  -t, --time=HH:MM:SS       Maximum job walltime (e.g. '24:00:00')
  -m, --mem=SIZE            Job memory (e.g. '128G')
      --cpus=N              Number of CPUs per task (default: 20)
      --device=ID           GPU device ordinal (default: 0)
  -J, --job-name=NAME       Override Slurm job name (defaults to script or manifest name)
      --account=NAME        Override Slurm account name
  -x, --exclude=NODES       Comma-separated list of nodes to exclude (e.g. 'ccc0314,ccc0315,ccc0316')
      --dry-run             Validate job submission with --test-only without submitting
  -h, --help                Show this help message

Tuning Options (for 'tune' command):
      --env=NAME            Target environment ('fr_cont', 'zone', 'nr', or 'fetch')
      --trials=N            Number of Optuna trials
      --timesteps=N         Number of training timesteps per trial
      --train-replicates    Train 5 replicates using best params after tuning

EOF
}

resolve_partition() {
  local p="${1:-${PARTITION_ALIAS}}"
  case "${p}" in
    eng|eng-gpu|eng-research-gpu)
      ACCOUNT="${ACCOUNT_OVERRIDE:-${ENG_ACCOUNT}}"
      PARTITION="${ENG_PARTITION}"
      CONCURRENCY="${CONCURRENCY_OVERRIDE:-${ENG_CONCURRENCY}}"
      if (( CONCURRENCY > ENG_CONCURRENCY_HARD_CAP )); then
        echo "Note: Clamping ENG concurrency to ${ENG_CONCURRENCY_HARD_CAP}."
        CONCURRENCY="${ENG_CONCURRENCY_HARD_CAP}"
      fi
      ;;
    ic|ic-gpu|IllinoisComputes-GPU)
      ACCOUNT="${ACCOUNT_OVERRIDE:-${IC_ACCOUNT}}"
      PARTITION="${IC_PARTITION}"
      CONCURRENCY="${CONCURRENCY_OVERRIDE:-${IC_CONCURRENCY}}"
      if (( CONCURRENCY > IC_CONCURRENCY_HARD_CAP )); then
        echo "Note: Clamping IC concurrency to ${IC_CONCURRENCY_HARD_CAP} (gpu4 QoS cap)."
        CONCURRENCY="${IC_CONCURRENCY_HARD_CAP}"
      fi
      ;;
    secondary|sec)
      ACCOUNT="${ACCOUNT_OVERRIDE:-${SEC_ACCOUNT}}"
      PARTITION="${SEC_PARTITION}"
      CONCURRENCY="${CONCURRENCY_OVERRIDE:-${SEC_CONCURRENCY}}"
      WALLTIME="${WALLTIME:-${SEC_MAX_TIME}}"
      ;;
    all|split)
      PARTITION="all"
      ;;
    *)
      if [[ -z "${ACCOUNT_OVERRIDE}" ]]; then
        echo "Error: Custom partition '${p}' requires --account to be specified." >&2
        exit 1
      fi
      ACCOUNT="${ACCOUNT_OVERRIDE}"
      PARTITION="${p}"
      CONCURRENCY="${CONCURRENCY_OVERRIDE:-1}"
      ;;
  esac
}

submit_job() {
  local sbatch_file="$1"
  local account="$2"
  local partition="$3"
  local array_arg="$4"
  local job_name="${5:-}"
  shift 5
  local sbatch_cmd=(sbatch)

  if (( DRY_RUN == 1 )); then
    sbatch_cmd+=(--test-only)
  else
    sbatch_cmd+=(--parsable)
  fi

  sbatch_cmd+=(--account="${account}" --partition="${partition}" --export="ALL,WORKSPACE_DIR=${WORKSPACE_DIR},REPLICATE_ARRAY=${REPLICATE_ARRAY:-0}")

  if [[ -n "${job_name}" ]]; then
    sbatch_cmd+=(--job-name="${job_name}")
  fi

  if [[ -n "${array_arg}" ]]; then
    sbatch_cmd+=(--array="${array_arg}")
  fi

  if [[ -n "${WALLTIME}" ]]; then
    sbatch_cmd+=(--time="${WALLTIME}")
  fi

  if [[ -n "${MEM}" ]]; then
    sbatch_cmd+=(--mem="${MEM}")
  fi

  if [[ -n "${CPUS}" ]]; then
    sbatch_cmd+=(--cpus-per-task="${CPUS}")
  fi

  if [[ -n "${EXCLUDE_NODES}" ]]; then
    sbatch_cmd+=(--exclude="${EXCLUDE_NODES}")
  fi

  if [[ "${#EXTRA_SBATCH_FLAGS[@]}" -gt 0 ]]; then
    sbatch_cmd+=("${EXTRA_SBATCH_FLAGS[@]}")
  fi

  sbatch_cmd+=("${sbatch_file}" "$@")

  echo "Submitting: ${sbatch_cmd[*]}"
  "${sbatch_cmd[@]}"
}

# ------------------------------------------------------------------------------
# Parse Subcommand
# ------------------------------------------------------------------------------

if [[ "$#" -eq 0 || "$1" == "-h" || "$1" == "--help" || "$1" == "help" ]]; then
  print_usage
  exit 0
fi

COMMAND="$1"
shift

# Collect positional args and flag options
POSITIONAL_ARGS=()
TUNE_ENV="fr_cont"
TUNE_TRIALS=""
TUNE_TIMESTEPS=""
TUNE_TRAIN_REPLICATES=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -p=*|--partition=*)
      PARTITION_ALIAS="${1#*=}"
      shift
      ;;
    -p|--partition)
      PARTITION_ALIAS="$2"
      shift 2
      ;;
    -r=*|--replicates=*)
      NUM_REPLICATES="${1#*=}"
      shift
      ;;
    -r|--replicates)
      NUM_REPLICATES="$2"
      shift 2
      ;;
    --replicate-start-id=*)
      REPLICATE_START_ID="${1#*=}"
      shift
      ;;
    --replicate-start-id)
      REPLICATE_START_ID="$2"
      shift 2
      ;;
    -a=*|--array=*)
      ARRAY_SPEC="${1#*=}"
      shift
      ;;
    -a|--array)
      ARRAY_SPEC="$2"
      shift 2
      ;;
    -c=*|--concurrency=*)
      CONCURRENCY_OVERRIDE="${1#*=}"
      shift
      ;;
    -c|--concurrency)
      CONCURRENCY_OVERRIDE="$2"
      shift 2
      ;;
    -t=*|--time=*)
      WALLTIME="${1#*=}"
      shift
      ;;
    -t|--time)
      WALLTIME="$2"
      shift 2
      ;;
    -m=*|--mem=*)
      MEM="${1#*=}"
      shift
      ;;
    -m|--mem)
      MEM="$2"
      shift 2
      ;;
    --cpus=*)
      CPUS="${1#*=}"
      shift
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --device=*)
      DEVICE="${1#*=}"
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    -J=*|--job-name=*)
      JOB_NAME_OVERRIDE="${1#*=}"
      shift
      ;;
    -J|--job-name)
      JOB_NAME_OVERRIDE="$2"
      shift 2
      ;;
    --account=*)
      ACCOUNT_OVERRIDE="${1#*=}"
      shift
      ;;
    --account)
      ACCOUNT_OVERRIDE="$2"
      shift 2
      ;;
    -x=*|--exclude=*|--exclude-nodes=*)
      EXCLUDE_NODES="${1#*=}"
      shift
      ;;
    -x|--exclude|--exclude-nodes)
      EXCLUDE_NODES="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --env=*)
      TUNE_ENV="${1#*=}"
      shift
      ;;
    --env)
      TUNE_ENV="$2"
      shift 2
      ;;
    --trials=*)
      TUNE_TRIALS="${1#*=}"
      shift
      ;;
    --trials)
      TUNE_TRIALS="$2"
      shift 2
      ;;
    --timesteps=*)
      TUNE_TIMESTEPS="${1#*=}"
      shift
      ;;
    --timesteps)
      TUNE_TIMESTEPS="$2"
      shift 2
      ;;
    --train-replicates)
      TUNE_TRAIN_REPLICATES=1
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      EXTRA_PYTHON_ARGS+=("$@")
      break
      ;;
    -*)
      EXTRA_PYTHON_ARGS+=("$1")
      shift
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

resolve_partition "${PARTITION_ALIAS}"

# ------------------------------------------------------------------------------
# Dispatch Subcommand
# ------------------------------------------------------------------------------

case "${COMMAND}" in
  train)
    if [[ "${#POSITIONAL_ARGS[@]}" -lt 2 ]]; then
      echo "Error: Both SCRIPT and CONFIG arguments are required for 'train'." >&2
      echo "Usage: $0 train <SCRIPT> <CONFIG> [OPTIONS]" >&2
      exit 1
    fi

    TARGET_SCRIPT="${POSITIONAL_ARGS[0]}"
    CONFIG_PATH="${POSITIONAL_ARGS[1]}"

    if [[ ! -f "${TARGET_SCRIPT}" ]]; then
      echo "Error: Target script '${TARGET_SCRIPT}' not found." >&2
      exit 1
    fi

    if [[ ! -f "${CONFIG_PATH}" ]]; then
      echo "Error: Configuration file '${CONFIG_PATH}' not found." >&2
      exit 1
    fi

    export EXP_SCRIPT="${TARGET_SCRIPT}"
    export EXP_CONFIG="${CONFIG_PATH}"
    export DEVICE="${DEVICE}"
    if [[ "${#EXTRA_PYTHON_ARGS[@]}" -gt 0 ]]; then
      export EXTRA_ARGS="${EXTRA_PYTHON_ARGS[*]}"
    fi

    SCRIPT_NAME="$(basename "${TARGET_SCRIPT}" .py)"
    mkdir -p "${WORKSPACE_DIR}/cluster/logs/train/${SCRIPT_NAME}"
    JOB_NAME="${JOB_NAME_OVERRIDE:-${SCRIPT_NAME}}"

    if [[ -n "${NUM_REPLICATES}" ]]; then
      if ! [[ "${NUM_REPLICATES}" =~ ^[0-9]+$ ]] || (( NUM_REPLICATES < 1 )); then
        echo "Error: --replicates must be a positive integer." >&2
        exit 1
      fi
      if ! [[ "${REPLICATE_START_ID}" =~ ^[0-9]+$ ]] || (( REPLICATE_START_ID < 0 )); then
        echo "Error: --replicate-start-id must be a non-negative integer." >&2
        exit 1
      fi

      REPLICATE_ARRAY=1
      export REPLICATE_ARRAY

      if [[ "${PARTITION}" == "all" ]]; then
        # Priority: use eng first up to capacity, then ic for overflow
        eng_cap="${CONCURRENCY_OVERRIDE:-${ENG_CONCURRENCY}}"
        if (( eng_cap > ENG_CONCURRENCY_HARD_CAP )); then eng_cap="${ENG_CONCURRENCY_HARD_CAP}"; fi
        ic_cap="${IC_CONCURRENCY_HARD_CAP}"

        if (( NUM_REPLICATES <= eng_cap )); then
          eng_count="${NUM_REPLICATES}"
          ic_count=0
        else
          eng_count="${eng_cap}"
          ic_count="$((NUM_REPLICATES - eng_cap))"
          if (( ic_count > ic_cap )); then
            echo "Note: Replicate count (${NUM_REPLICATES}) exceeds combined capacity (${eng_cap} on ENG + ${ic_cap} on IC). Clamping IC concurrency to ${ic_cap}."
          fi
        fi

        ENG_IDS=()
        for ((i=0; i<eng_count; i++)); do ENG_IDS+=("$((REPLICATE_START_ID + i))"); done
        IC_IDS=()
        for ((i=0; i<ic_count; i++)); do IC_IDS+=("$((REPLICATE_START_ID + eng_count + i))"); done

        if (( ${#ENG_IDS[@]} > 0 )); then
          eng_spec="$(IFS=,; echo "${ENG_IDS[*]}")%${eng_cap}"
          eng_res=$(submit_job "${TRAIN_SBATCH}" "${ENG_ACCOUNT}" "${ENG_PARTITION}" "${eng_spec}" "${JOB_NAME}" "${TARGET_SCRIPT}" "${CONFIG_PATH}")
          echo "Submitted to ${ENG_PARTITION}: ${eng_res} (${#ENG_IDS[@]} replicate jobs, spec=${eng_spec})"
        fi

        if (( ${#IC_IDS[@]} > 0 )); then
          ic_spec="$(IFS=,; echo "${IC_IDS[*]}")%${ic_cap}"
          ic_res=$(submit_job "${TRAIN_SBATCH}" "${IC_ACCOUNT}" "${IC_PARTITION}" "${ic_spec}" "${JOB_NAME}" "${TARGET_SCRIPT}" "${CONFIG_PATH}")
          echo "Submitted overflow to ${IC_PARTITION}: ${ic_res} (${#IC_IDS[@]} replicate jobs, spec=${ic_spec})"
        fi
      else
        end_id="$((REPLICATE_START_ID + NUM_REPLICATES - 1))"
        array_arg="${ARRAY_SPEC:-${REPLICATE_START_ID}-${end_id}%${CONCURRENCY}}"
        result=$(submit_job "${TRAIN_SBATCH}" "${ACCOUNT}" "${PARTITION}" "${array_arg}" "${JOB_NAME}" "${TARGET_SCRIPT}" "${CONFIG_PATH}")
        echo "Submitted train replicate job array: ${result} (partition=${PARTITION}, job_name=${JOB_NAME}, replicates=${NUM_REPLICATES}, array=${array_arg})"
      fi
    else
      result=$(submit_job "${TRAIN_SBATCH}" "${ACCOUNT}" "${PARTITION}" "${ARRAY_SPEC}" "${JOB_NAME}" "${TARGET_SCRIPT}" "${CONFIG_PATH}")
      echo "Submitted train job: ${result} (partition=${PARTITION}, job_name=${JOB_NAME}, script=${TARGET_SCRIPT}, config=${CONFIG_PATH})"
    fi
    ;;

  tune)
    if [[ "${#POSITIONAL_ARGS[@]}" -lt 1 ]]; then
      echo "Error: SCRIPT argument is required for 'tune'." >&2
      echo "Usage: $0 tune <SCRIPT> [OPTIONS]" >&2
      exit 1
    fi

    TARGET_SCRIPT="${POSITIONAL_ARGS[0]}"

    if [[ ! -f "${TARGET_SCRIPT}" ]]; then
      echo "Error: Tuning script '${TARGET_SCRIPT}' not found." >&2
      exit 1
    fi

    export TUNE_SCRIPT="${TARGET_SCRIPT}"
    export ENV_NAME="${TUNE_ENV}"
    export DEVICE="${DEVICE}"
    if [[ -n "${TUNE_TRIALS}" ]]; then export TRIALS="${TUNE_TRIALS}"; fi
    if [[ -n "${TUNE_TIMESTEPS}" ]]; then export TIMESTEPS="${TUNE_TIMESTEPS}"; fi
    if (( TUNE_TRAIN_REPLICATES == 1 )); then export TRAIN_REPLICATES="true"; fi
    if [[ "${#EXTRA_PYTHON_ARGS[@]}" -gt 0 ]]; then
      export EXTRA_ARGS="${EXTRA_PYTHON_ARGS[*]}"
    fi

    SCRIPT_NAME="$(basename "${TARGET_SCRIPT}" .py)"
    mkdir -p "${WORKSPACE_DIR}/cluster/logs/tune/${SCRIPT_NAME}"
    JOB_NAME="${JOB_NAME_OVERRIDE:-${SCRIPT_NAME}}"

    result=$(submit_job "${TUNE_SBATCH}" "${ACCOUNT}" "${PARTITION}" "" "${JOB_NAME}" "${TARGET_SCRIPT}")
    echo "Submitted tune job: ${result} (partition=${PARTITION}, job_name=${JOB_NAME}, script=${TARGET_SCRIPT}, env=${TUNE_ENV})"
    ;;

  campaign)
    if [[ "${#POSITIONAL_ARGS[@]}" -lt 1 ]]; then
      echo "Error: Manifest file path is required." >&2
      echo "Usage: $0 campaign <MANIFEST_FILE> [OPTIONS]" >&2
      exit 1
    fi

    MANIFEST_FILE="${POSITIONAL_ARGS[0]}"
    if [[ ! -f "${MANIFEST_FILE}" ]]; then
      echo "Error: Manifest file '${MANIFEST_FILE}' not found." >&2
      exit 1
    fi

    LINE_COUNT="$(grep -c -v '^[[:space:]]*$' "${MANIFEST_FILE}" || true)"
    if (( LINE_COUNT < 1 )); then
      echo "Error: Manifest file '${MANIFEST_FILE}' is empty." >&2
      exit 1
    fi

    echo "Campaign manifest '${MANIFEST_FILE}' contains ${LINE_COUNT} experiment(s)."
    export EXP_MANIFEST="${MANIFEST_FILE}"
    export DEVICE="${DEVICE}"

    # Scan manifest for scripts and ensure log directories exist
    CAMPAIGN_SCRIPTS=()
    while IFS= read -r line || [[ -n "${line}" ]]; do
      line="$(echo "${line}" | xargs)"
      if [[ -z "${line}" || "${line}" =~ ^# ]]; then
        continue
      fi
      read -r c_item1 _ <<< "${line}"
      if [[ "${c_item1}" == *.py ]]; then
        sname="$(basename "${c_item1}" .py)"
        mkdir -p "${WORKSPACE_DIR}/cluster/logs/train/${sname}"
        CAMPAIGN_SCRIPTS+=("${sname}")
      fi
    done < "${MANIFEST_FILE}"

    MANIFEST_BASENAME="$(basename "${MANIFEST_FILE%.*}")"
    mkdir -p "${WORKSPACE_DIR}/cluster/logs/train/${MANIFEST_BASENAME}"

    if [[ "${#CAMPAIGN_SCRIPTS[@]}" -gt 0 ]]; then
      all_same=1
      first_s="${CAMPAIGN_SCRIPTS[0]}"
      for s in "${CAMPAIGN_SCRIPTS[@]}"; do
        if [[ "${s}" != "${first_s}" ]]; then
          all_same=0
          break
        fi
      done
      if (( all_same == 1 )); then
        AUTO_JOB_NAME="${first_s}"
      else
        AUTO_JOB_NAME="${MANIFEST_BASENAME}"
      fi
    else
      AUTO_JOB_NAME="${MANIFEST_BASENAME}"
    fi

    JOB_NAME="${JOB_NAME_OVERRIDE:-${AUTO_JOB_NAME}}"

    if [[ "${PARTITION}" == "all" ]]; then
      # Priority: use eng first up to capacity, then ic for overflow
      eng_cap="${CONCURRENCY_OVERRIDE:-${ENG_CONCURRENCY}}"
      if (( eng_cap > ENG_CONCURRENCY_HARD_CAP )); then eng_cap="${ENG_CONCURRENCY_HARD_CAP}"; fi
      ic_cap="${IC_CONCURRENCY_HARD_CAP}"

      if (( LINE_COUNT <= eng_cap )); then
        eng_count="${LINE_COUNT}"
        ic_count=0
      else
        eng_count="${eng_cap}"
        ic_count="$((LINE_COUNT - eng_cap))"
        if (( ic_count > ic_cap )); then
          echo "Note: Campaign items (${LINE_COUNT}) exceed combined concurrency (${eng_cap} on ENG + ${ic_cap} on IC). Clamping IC concurrency to ${ic_cap}."
        fi
      fi

      ENG_IDS=()
      for ((i=0; i<eng_count; i++)); do ENG_IDS+=("${i}"); done
      IC_IDS=()
      for ((i=0; i<ic_count; i++)); do IC_IDS+=("$((eng_count + i))"); done

      if (( ${#ENG_IDS[@]} > 0 )); then
        eng_spec="$(IFS=,; echo "${ENG_IDS[*]}")%${eng_cap}"
        eng_res=$(submit_job "${TRAIN_SBATCH}" "${ENG_ACCOUNT}" "${ENG_PARTITION}" "${eng_spec}" "${JOB_NAME}")
        echo "Submitted to ${ENG_PARTITION}: ${eng_res} (${#ENG_IDS[@]} jobs, spec=${eng_spec})"
      fi

      if (( ${#IC_IDS[@]} > 0 )); then
        ic_spec="$(IFS=,; echo "${IC_IDS[*]}")%${ic_cap}"
        ic_res=$(submit_job "${TRAIN_SBATCH}" "${IC_ACCOUNT}" "${IC_PARTITION}" "${ic_spec}" "${JOB_NAME}")
        echo "Submitted overflow to ${IC_PARTITION}: ${ic_res} (${#IC_IDS[@]} jobs, spec=${ic_spec})"
      fi
    else
      # Single partition array
      array_arg="${ARRAY_SPEC:-0-$((LINE_COUNT - 1))%${CONCURRENCY}}"
      result=$(submit_job "${TRAIN_SBATCH}" "${ACCOUNT}" "${PARTITION}" "${array_arg}" "${JOB_NAME}")
      echo "Submitted campaign to ${PARTITION}: ${result} (array=${array_arg})"
    fi
    ;;

  *)
    echo "Unknown command: ${COMMAND}" >&2
    print_usage
    exit 1
    ;;
esac
