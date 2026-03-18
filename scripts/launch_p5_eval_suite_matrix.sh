#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=${ENV_FILE:-/mnt/preserved/world-modality-vla/p5_cosmos_env.sh}
if [[ -f "${ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
cd "${REPO_ROOT}"

export MUJOCO_GL=${MUJOCO_GL:-egl}

OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/train/p5_cogvideo_screen"}
LOG_DIR=${LOG_DIR:-"logs/cogvideo_suite_eval"}
EVAL_TASK=${EVAL_TASK:-"libero_object"}
EVAL_EPISODES=${EVAL_EPISODES:-200}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-10}
EVAL_N_ACTION_STEPS=${EVAL_N_ACTION_STEPS:-10}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
STEPS=${STEPS:-20000}
DRY_RUN=${DRY_RUN:-0}
EXP_NAMES=${EXP_NAMES:-"C0_front_seed0 C0_front_seed1 C0_wrist_seed0 C0_wrist_seed1 C1_f2_front_seed0 C1_f2_front_seed1 C1_f2_wrist_seed0 C1_f2_wrist_seed1"}

mkdir -p "${LOG_DIR}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
read -r -a EXP_ARR <<< "${EXP_NAMES}"

if (( ${#GPU_ARR[@]} < ${#EXP_ARR[@]} )); then
  echo "Need at least ${#EXP_ARR[@]} GPUs, got ${#GPU_ARR[@]}" >&2
  exit 2
fi

detect_eval_action_flag() {
  local help_txt
  help_txt="$(lerobot-wm-eval --help 2>&1 || true)"
  if echo "${help_txt}" | grep -q -- "--eval.n_action_steps"; then
    echo "--eval.n_action_steps=${EVAL_N_ACTION_STEPS}"
  elif echo "${help_txt}" | grep -q -- "--env.n_action_steps"; then
    echo "--env.n_action_steps=${EVAL_N_ACTION_STEPS}"
  elif echo "${help_txt}" | grep -q -- "--policy.n_action_steps"; then
    echo "--policy.n_action_steps=${EVAL_N_ACTION_STEPS}"
  fi
}

resolve_checkpoint_dir() {
  local base_dir="$1"
  local steps="$2"
  local padded
  padded="$(printf "%06d" "${steps}")"
  if [[ -d "${base_dir}/checkpoints/${padded}/pretrained_model" ]]; then
    echo "${base_dir}/checkpoints/${padded}/pretrained_model"
  elif [[ -d "${base_dir}/checkpoints/${steps}/pretrained_model" ]]; then
    echo "${base_dir}/checkpoints/${steps}/pretrained_model"
  else
    return 1
  fi
}

run_eval() {
  local gpu_id="$1"
  local exp_name="$2"
  local ckpt_dir="$3"
  local log_file="${LOG_DIR}/${exp_name}.${EVAL_TASK}.eval.log"
  local action_flag
  action_flag="$(detect_eval_action_flag || true)"
  local -a eval_extra=()
  if [[ -n "${action_flag}" ]]; then
    eval_extra+=("${action_flag}")
  fi
  echo "=== eval ${exp_name} on GPU ${gpu_id} suite=${EVAL_TASK} ===" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "ckpt=${ckpt_dir}" >&2
    LAST_LAUNCH_PID=""
    return
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    lerobot-wm-eval \
      --policy.path="${ckpt_dir}" \
      --policy.device=cuda \
      --env.type=libero \
      --env.task="${EVAL_TASK}" \
      --eval.n_episodes="${EVAL_EPISODES}" \
      --eval.batch_size="${EVAL_BATCH_SIZE}" \
      "${eval_extra[@]}" \
      2>&1 | tee "${log_file}"
  ) &
  LAST_LAUNCH_PID=$!
}

declare -a pids=()

for i in "${!EXP_ARR[@]}"; do
  exp_name="${EXP_ARR[$i]}"
  gpu_id="${GPU_ARR[$i]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    ckpt_dir="${OUTPUT_ROOT}/${exp_name}/checkpoints/$(printf "%06d" "${STEPS}")/pretrained_model"
  else
    ckpt_dir="$(resolve_checkpoint_dir "${OUTPUT_ROOT}/${exp_name}" "${STEPS}")"
  fi
  run_eval "${gpu_id}" "${exp_name}" "${ckpt_dir}"
  if [[ -n "${LAST_LAUNCH_PID:-}" ]]; then
    pids+=("${LAST_LAUNCH_PID}")
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

exit "${fail}"
