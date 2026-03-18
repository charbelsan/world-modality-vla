#!/usr/bin/env bash
set -euo pipefail

# Use the last ~2h before the capacity-block restart for short eval-heavy runs.
# The script starts background watchers: if a target GPU is free now it launches
# immediately, otherwise it waits until the GPU becomes idle.

ROOT_DIR=${ROOT_DIR:-$(pwd)}
VENV_PATH=${VENV_PATH:-/opt/dlami/nvme/.venvs/world-modality-vla-ss/bin/activate}
LOG_DIR=${LOG_DIR:-logs}

CHECK_INTERVAL=${CHECK_INTERVAL:-20}
IDLE_POLLS=${IDLE_POLLS:-2}
IDLE_MAX_UTIL=${IDLE_MAX_UTIL:-5}
IDLE_MAX_MEM_MB=${IDLE_MAX_MEM_MB:-1024}
DRY_RUN=${DRY_RUN:-0}

SPRINT_EPISODES=${SPRINT_EPISODES:-100}
SPRINT_BATCH_SIZE=${SPRINT_BATCH_SIZE:-5}
SPRINT_TASK=${SPRINT_TASK:-libero_spatial}
OBJECT_TASK=${OBJECT_TASK:-libero_object}
GOAL_TASK=${GOAL_TASK:-libero_goal}
N_ACTION_STEPS=${N_ACTION_STEPS:-10}

HF_ROOT=${HF_ROOT:-${ROOT_DIR}/.hf_cache}

E0_CKPT=${E0_CKPT:-mi300x_sync/checkpoints/E0_smolvla_baseline_seed0/checkpoints/050000/pretrained_model}
E2_FRONT_CKPT=${E2_FRONT_CKPT:-mi300x_sync/checkpoints/E2_world_pred_seed0/checkpoints/050000/pretrained_model}

resolve_checkpoint_dir() {
  local base_dir="$1"
  local steps="$2"
  local padded
  padded="$(printf "%06d" "${steps}")"
  if [[ -d "${base_dir}/checkpoints/${padded}/pretrained_model" ]]; then
    echo "${base_dir}/checkpoints/${padded}/pretrained_model"
  else
    echo "${base_dir}/checkpoints/${steps}/pretrained_model"
  fi
}

E2_WRIST_BASE=${E2_WRIST_BASE:-outputs/train/p5_h100/E2_wrist_seed0_bs192_amp_h100}
E2_WRIST_CKPT=${E2_WRIST_CKPT:-$(resolve_checkpoint_dir "${E2_WRIST_BASE}" 25000)}

mkdir -p "${ROOT_DIR}/${LOG_DIR}"

prepare_env() {
  cd "${ROOT_DIR}"
  # shellcheck disable=SC1090
  source "${VENV_PATH}"
  export MUJOCO_GL=${MUJOCO_GL:-osmesa}
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
  export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
  export HF_HOME=${HF_HOME:-${HF_ROOT}}
  export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-${HF_ROOT}/hub}
  export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_ROOT}/datasets}
  export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_ROOT}/transformers}
}

detect_eval_action_flag() {
  local help_txt
  help_txt="$(lerobot-wm-eval --help 2>&1 || true)"
  if echo "${help_txt}" | grep -q -- "--eval.n_action_steps"; then
    echo "--eval.n_action_steps=${N_ACTION_STEPS}"
  elif echo "${help_txt}" | grep -q -- "--env.n_action_steps"; then
    echo "--env.n_action_steps=${N_ACTION_STEPS}"
  elif echo "${help_txt}" | grep -q -- "--policy.n_action_steps"; then
    echo "--policy.n_action_steps=${N_ACTION_STEPS}"
  fi
}

wait_for_gpu_free() {
  local gpu_id="$1"
  local idle_count=0
  while true; do
    local util mem
    read -r util mem < <(
      nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits \
      | awk -F', ' -v gpu="${gpu_id}" '$1 == gpu {print $2, $3}'
    )
    util=${util:-100}
    mem=${mem:-999999}
    if (( util <= IDLE_MAX_UTIL && mem <= IDLE_MAX_MEM_MB )); then
      idle_count=$((idle_count + 1))
    else
      idle_count=0
    fi
    if (( idle_count >= IDLE_POLLS )); then
      break
    fi
    sleep "${CHECK_INTERVAL}"
  done
}

run_eval() {
  local ckpt="$1"
  local task="$2"
  local episodes="$3"
  shift 3
  local -a extra=("$@")
  local action_flag
  local -a args=()
  action_flag="$(detect_eval_action_flag || true)"
  if [[ -n "${action_flag}" ]]; then
    args+=("${action_flag}")
  fi
  lerobot-wm-eval \
    --policy.pretrained_path="${ckpt}" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task="${task}" \
    --eval.n_episodes="${episodes}" \
    --eval.batch_size="${SPRINT_BATCH_SIZE}" \
    "${args[@]}" \
    "${extra[@]}"
}

job_front_signflip() {
  run_eval "${E2_FRONT_CKPT}" "${SPRINT_TASK}" "${SPRINT_EPISODES}" \
    --policy.world_memory_mode_rollout=signflip
}

job_front_random_scaled() {
  run_eval "${E2_FRONT_CKPT}" "${SPRINT_TASK}" "${SPRINT_EPISODES}" \
    --policy.world_memory_mode_rollout=random_scaled
}

job_wrist_pred() {
  [[ -d "${E2_WRIST_CKPT}" ]] || { echo "Missing wrist checkpoint: ${E2_WRIST_CKPT}"; return 0; }
  run_eval "${E2_WRIST_CKPT}" "${SPRINT_TASK}" "${SPRINT_EPISODES}" \
    --policy.world_memory_mode_rollout=pred
}

job_wrist_signflip() {
  [[ -d "${E2_WRIST_CKPT}" ]] || { echo "Missing wrist checkpoint: ${E2_WRIST_CKPT}"; return 0; }
  run_eval "${E2_WRIST_CKPT}" "${SPRINT_TASK}" "${SPRINT_EPISODES}" \
    --policy.world_memory_mode_rollout=signflip
}

job_wrist_random_scaled() {
  [[ -d "${E2_WRIST_CKPT}" ]] || { echo "Missing wrist checkpoint: ${E2_WRIST_CKPT}"; return 0; }
  run_eval "${E2_WRIST_CKPT}" "${SPRINT_TASK}" "${SPRINT_EPISODES}" \
    --policy.world_memory_mode_rollout=random_scaled
}

job_object_compare() {
  run_eval "${E0_CKPT}" "${OBJECT_TASK}" "${SPRINT_EPISODES}"
  run_eval "${E2_FRONT_CKPT}" "${OBJECT_TASK}" "${SPRINT_EPISODES}" \
    --policy.world_memory_mode_rollout=pred
}

job_goal_compare() {
  run_eval "${E0_CKPT}" "${GOAL_TASK}" "${SPRINT_EPISODES}"
  run_eval "${E2_FRONT_CKPT}" "${GOAL_TASK}" "${SPRINT_EPISODES}" \
    --policy.world_memory_mode_rollout=pred
}

launch_job() {
  local gpu_id="$1"
  local name="$2"
  local fn_name="$3"
  local log_path="${ROOT_DIR}/${LOG_DIR}/${name}.log"
  local pid_path="${ROOT_DIR}/${LOG_DIR}/${name}.pid"

  if (( DRY_RUN )); then
    echo "[DRY_RUN] gpu=${gpu_id} name=${name} fn=${fn_name}"
    return 0
  fi

  (
    prepare_env
    wait_for_gpu_free "${gpu_id}"
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "Launching ${name} on GPU ${gpu_id} at $(date -Is)"
    "${fn_name}"
  ) >"${log_path}" 2>&1 &

  echo $! > "${pid_path}"
  echo "Started watcher ${name} (pid=$(cat "${pid_path}"))"
}

launch_job 4 "p5_sprint_front_signflip_gpu4" job_front_signflip
launch_job 5 "p5_sprint_front_random_scaled_gpu5" job_front_random_scaled
launch_job 6 "p5_sprint_wrist_signflip_gpu6" job_wrist_signflip
launch_job 0 "p5_sprint_object_compare_gpu0" job_object_compare
launch_job 1 "p5_sprint_goal_compare_gpu1" job_goal_compare
launch_job 2 "p5_sprint_wrist_pred_gpu2" job_wrist_pred
launch_job 3 "p5_sprint_wrist_random_scaled_gpu3" job_wrist_random_scaled

echo "Pre-reboot sprint watchers installed under ${ROOT_DIR}/${LOG_DIR}"
