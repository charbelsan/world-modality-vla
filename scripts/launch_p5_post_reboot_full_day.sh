#!/usr/bin/env bash
set -euo pipefail

# Full-day launcher for the renewed P5 capacity block.
# Intended to be started automatically at reboot.

ROOT_DIR=${ROOT_DIR:-$(pwd)}
VENV_PATH=${VENV_PATH:-/opt/dlami/nvme/.venvs/world-modality-vla-ss/bin/activate}
LOG_DIR=${LOG_DIR:-logs}
DRY_RUN=${DRY_RUN:-0}

HF_ROOT=${HF_ROOT:-${ROOT_DIR}/.hf_cache}
DATASET_REPO_ID=${DATASET_REPO_ID:-HuggingFaceVLA/libero}
INIT_POLICY_PATH=${INIT_POLICY_PATH:-HuggingFaceVLA/smolvla_libero}
CACHE_DIR=${CACHE_DIR:-cache}

WORLD_SOURCE=${WORLD_SOURCE:-vjepa}
WORLD_LATENT_DIM=${WORLD_LATENT_DIM:-1408}
CONTEXT_FRAMES=${CONTEXT_FRAMES:-4}
LAMBDA_WORLD=${LAMBDA_WORLD:-0.2}
BATCH_SIZE=${BATCH_SIZE:-192}
NUM_WORKERS=${NUM_WORKERS:-48}
STEPS=${STEPS:-25000}

EVAL_TASK=${EVAL_TASK:-libero_spatial}
OBJECT_TASK=${OBJECT_TASK:-libero_object}
GOAL_TASK=${GOAL_TASK:-libero_goal}
EVAL_EPISODES=${EVAL_EPISODES:-500}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-5}
N_ACTION_STEPS=${N_ACTION_STEPS:-10}

E0_CKPT=${E0_CKPT:-mi300x_sync/checkpoints/E0_smolvla_baseline_seed0/checkpoints/050000/pretrained_model}
E2_FRONT_CKPT=${E2_FRONT_CKPT:-mi300x_sync/checkpoints/E2_world_pred_seed0/checkpoints/050000/pretrained_model}
F3B_DIR=${F3B_DIR:-outputs/train/p5_h100/F3b_prefix_cross_seed0_bs192_amp_h100}
WRIST_DIR=${WRIST_DIR:-outputs/train/p5_h100/E2_wrist_seed0_bs192_amp_h100}

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

F2_BASE=${F2_BASE:-outputs/train/p5_h100/F2_suffix_in_seed0_bs192_amp_h100}
F2_CKPT=${F2_CKPT:-$(resolve_checkpoint_dir "${F2_BASE}" "${STEPS}")}
F3B_CKPT=${F3B_CKPT:-$(resolve_checkpoint_dir "${F3B_DIR}" "${STEPS}")}
WRIST_CKPT=${WRIST_CKPT:-$(resolve_checkpoint_dir "${WRIST_DIR}" "${STEPS}")}

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
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    "${args[@]}" \
    "${extra[@]}"
}

run_train_world() {
  local output_dir="$1"
  shift 1
  lerobot-wm-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --policy.type=smolvla_world \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.use_amp=true \
    --policy.init_from_policy_path="${INIT_POLICY_PATH}" \
    --policy.dataset_repo_id="${DATASET_REPO_ID}" \
    --policy.cache_dir="${CACHE_DIR}" \
    --policy.world_latents_source="${WORLD_SOURCE}" \
    --policy.world_latent_dim="${WORLD_LATENT_DIM}" \
    --policy.context_frames="${CONTEXT_FRAMES}" \
    --policy.lambda_world="${LAMBDA_WORLD}" \
    --policy.world_memory_mode_train=pred \
    --policy.enable_world_injection=true \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --output_dir="${output_dir}" \
    --seed=0 \
    --wandb.enable=false \
    "$@"
}

job_front_signflip() {
  run_eval "${E2_FRONT_CKPT}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=signflip
}

job_front_random_scaled() {
  run_eval "${E2_FRONT_CKPT}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=random_scaled
}

job_object_compare() {
  run_eval "${E0_CKPT}" "${OBJECT_TASK}" "${EVAL_EPISODES}"
  run_eval "${E2_FRONT_CKPT}" "${OBJECT_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=pred
}

job_goal_compare() {
  run_eval "${E0_CKPT}" "${GOAL_TASK}" "${EVAL_EPISODES}"
  run_eval "${E2_FRONT_CKPT}" "${GOAL_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=pred
}

job_k4() {
  local out_dir="outputs/train/p5_h100/E2_k4_seed0_bs${BATCH_SIZE}_amp_h100"
  local ckpt="${out_dir}/checkpoints/${STEPS}/pretrained_model"
  if [[ ! -d "${ckpt}" ]]; then
    run_train_world "${out_dir}" \
      --policy.latent_suffix=m4 \
      --policy.future_offset=4 \
      --policy.world_camera=front
  fi
  run_eval "${ckpt}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=pred
}

job_k2() {
  local out_dir="outputs/train/p5_h100/E2_k2_seed0_bs${BATCH_SIZE}_amp_h100"
  local ckpt="${out_dir}/checkpoints/${STEPS}/pretrained_model"
  if [[ ! -d "${ckpt}" ]]; then
    run_train_world "${out_dir}" \
      --policy.latent_suffix=m4 \
      --policy.future_offset=2 \
      --policy.world_camera=front
  fi
  run_eval "${ckpt}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=pred
}

job_wrist_branch() {
  if [[ ! -d "${WRIST_CKPT}" ]]; then
    run_train_world "${WRIST_DIR}" \
      --policy.latent_suffix=m4_wrist \
      --policy.future_offset=8 \
      --policy.world_camera=wrist
  fi
  run_eval "${WRIST_CKPT}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=signflip
  run_eval "${WRIST_CKPT}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
    --policy.world_memory_mode_rollout=random_scaled
}

job_f2_f3b_branch() {
  if [[ -d "${F2_CKPT}" ]]; then
    run_eval "${F2_CKPT}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
      --policy.world_memory_mode_rollout=pred
  fi

  if [[ ! -d "${F3B_CKPT}" ]]; then
    run_train_world "${F3B_DIR}" \
      --policy.latent_suffix=m4 \
      --policy.future_offset=8 \
      --policy.world_camera=front \
      --policy.world_prefix_cross_attn=true
  fi

  run_eval "${F3B_CKPT}" "${EVAL_TASK}" "${EVAL_EPISODES}" \
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
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    echo "Launching ${name} on GPU ${gpu_id} at $(date -Is)"
    "${fn_name}"
  ) >"${log_path}" 2>&1 &

  echo $! > "${pid_path}"
  echo "Started ${name} on GPU ${gpu_id} (pid=$(cat "${pid_path}"))"
}

launch_job 0 "p5_day1_front_signflip_gpu0" job_front_signflip
launch_job 1 "p5_day1_front_random_scaled_gpu1" job_front_random_scaled
launch_job 2 "p5_day1_object_compare_gpu2" job_object_compare
launch_job 3 "p5_day1_goal_compare_gpu3" job_goal_compare
launch_job 4 "p5_day1_k4_gpu4" job_k4
launch_job 5 "p5_day1_k2_gpu5" job_k2
launch_job 6 "p5_day1_wrist_branch_gpu6" job_wrist_branch
launch_job 7 "p5_day1_f2_f3b_branch_gpu7" job_f2_f3b_branch

echo "Post-reboot day-1 jobs launched under ${ROOT_DIR}/${LOG_DIR}"
