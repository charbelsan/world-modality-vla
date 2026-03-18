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

DATASET_REPO_ID=${DATASET_REPO_ID:-"HuggingFaceVLA/libero"}
INIT_POLICY_PATH=${INIT_POLICY_PATH:-"HuggingFaceVLA/smolvla_libero"}
CACHE_DIR=${CACHE_DIR:-"cache"}
WORLD_SOURCE=${WORLD_SOURCE:-"cogvideo"}
WORLD_VISION_MODEL_NAME=${WORLD_VISION_MODEL_NAME:-"cogvideo_2b_pool4_m4"}
WORLD_LATENT_DIM=${WORLD_LATENT_DIM:-256}
CONTEXT_FRAMES=${CONTEXT_FRAMES:-4}
LAMBDA_WORLD=${LAMBDA_WORLD:-0.2}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
FRONT_SUFFIX=${FRONT_SUFFIX:-"m4_cogvideo_front"}
WRIST_SUFFIX=${WRIST_SUFFIX:-"m4_cogvideo_wrist"}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-192}
NUM_WORKERS=${NUM_WORKERS:-24}
EVAL_TASK=${EVAL_TASK:-"libero_spatial"}
EVAL_EPISODES=${EVAL_EPISODES:-200}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-10}
EVAL_N_ACTION_STEPS=${EVAL_N_ACTION_STEPS:-10}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/train/p5_cogvideo_kscan"}
LOG_DIR=${LOG_DIR:-"logs/cogvideo_kscan"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if (( ${#GPU_ARR[@]} != 8 )); then
  echo "This k-scan script expects exactly 8 GPUs, got: ${GPU_IDS}" >&2
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
  else
    echo "${base_dir}/checkpoints/${steps}/pretrained_model"
  fi
}

run_variant() {
  local gpu_id="$1"
  local exp_name="$2"
  local latent_suffix="$3"
  local world_camera="$4"
  local seed="$5"
  local future_offset="$6"
  local out_dir="${OUTPUT_ROOT}/${exp_name}"
  local train_log="${LOG_DIR}/${exp_name}.train.log"
  local eval_log="${LOG_DIR}/${exp_name}.eval.log"
  local action_flag
  action_flag="$(detect_eval_action_flag || true)"
  local -a eval_extra=()
  if [[ -n "${action_flag}" ]]; then
    eval_extra+=("${action_flag}")
  fi
  echo "=== ${exp_name} on GPU ${gpu_id} ===" >&2
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "train -> ${out_dir}" >&2
    LAST_LAUNCH_PID=""
    return
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1

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
      --policy.world_vision_model_name="${WORLD_VISION_MODEL_NAME}" \
      --policy.latent_suffix="${latent_suffix}" \
      --policy.world_latent_dim="${WORLD_LATENT_DIM}" \
      --policy.context_frames="${CONTEXT_FRAMES}" \
      --policy.future_offset="${future_offset}" \
      --policy.lambda_world="${LAMBDA_WORLD}" \
      --policy.world_memory_mode_train=pred \
      --policy.world_camera="${world_camera}" \
      --policy.enable_world_injection=true \
      --policy.world_inject_suffix_in=true \
      --batch_size="${BATCH_SIZE}" \
      --num_workers="${NUM_WORKERS}" \
      --steps="${STEPS}" \
      --output_dir="${out_dir}" \
      --seed="${seed}" \
      --wandb.enable=false \
      2>&1 | tee "${train_log}"

    ckpt_dir="$(resolve_checkpoint_dir "${out_dir}" "${STEPS}")"
    lerobot-wm-eval \
      --policy.path="${ckpt_dir}" \
      --policy.device=cuda \
      --env.type=libero \
      --env.task="${EVAL_TASK}" \
      --eval.n_episodes="${EVAL_EPISODES}" \
      --eval.batch_size="${EVAL_BATCH_SIZE}" \
      "${eval_extra[@]}" \
      2>&1 | tee "${eval_log}"
  ) &
  LAST_LAUNCH_PID=$!
}

declare -a pids=()

run_variant "${GPU_ARR[0]}" "G2_k4_front_seed0" "${FRONT_SUFFIX}" "front" 0 4
pids+=("${LAST_LAUNCH_PID}")
run_variant "${GPU_ARR[1]}" "G2_k4_front_seed1" "${FRONT_SUFFIX}" "front" 1 4
pids+=("${LAST_LAUNCH_PID}")
run_variant "${GPU_ARR[2]}" "G2_k4_wrist_seed0" "${WRIST_SUFFIX}" "wrist" 0 4
pids+=("${LAST_LAUNCH_PID}")
run_variant "${GPU_ARR[3]}" "G2_k4_wrist_seed1" "${WRIST_SUFFIX}" "wrist" 1 4
pids+=("${LAST_LAUNCH_PID}")
run_variant "${GPU_ARR[4]}" "G3_k2_front_seed0" "${FRONT_SUFFIX}" "front" 0 2
pids+=("${LAST_LAUNCH_PID}")
run_variant "${GPU_ARR[5]}" "G3_k2_front_seed1" "${FRONT_SUFFIX}" "front" 1 2
pids+=("${LAST_LAUNCH_PID}")
run_variant "${GPU_ARR[6]}" "G3_k2_wrist_seed0" "${WRIST_SUFFIX}" "wrist" 0 2
pids+=("${LAST_LAUNCH_PID}")
run_variant "${GPU_ARR[7]}" "G3_k2_wrist_seed1" "${WRIST_SUFFIX}" "wrist" 1 2
pids+=("${LAST_LAUNCH_PID}")

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
