#!/usr/bin/env bash
set -euo pipefail

# Launch an 8-GPU Cosmos world-modality screening matrix on P5.
#
# Matrix:
# - GPU 0-1: C0-front, seeds 0-1
# - GPU 2-3: C0-wrist, seeds 0-1
# - GPU 4-5: C1-F2-front, seeds 0-1
# - GPU 6-7: C1-F2-wrist, seeds 0-1
#
# Usage:
#   source /mnt/preserved/world-modality-vla/p5_cosmos_env.sh
#   bash scripts/launch_p5_cosmos_screen.sh

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
WORLD_SOURCE=${WORLD_SOURCE:-"cosmos"}
WORLD_VISION_MODEL_NAME=${WORLD_VISION_MODEL_NAME:-"cosmos_cv8x8x8_pool4_m4"}
WORLD_LATENT_DIM=${WORLD_LATENT_DIM:-256}
CONTEXT_FRAMES=${CONTEXT_FRAMES:-4}
FUTURE_OFFSET=${FUTURE_OFFSET:-8}
LAMBDA_WORLD=${LAMBDA_WORLD:-0.2}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
FRONT_SUFFIX=${FRONT_SUFFIX:-"m4_cosmos_front"}
WRIST_SUFFIX=${WRIST_SUFFIX:-"m4_cosmos_wrist"}
STEPS=${STEPS:-20000}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-16}
EVAL_TASK=${EVAL_TASK:-"libero_spatial"}
EVAL_EPISODES=${EVAL_EPISODES:-200}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-10}
EVAL_N_ACTION_STEPS=${EVAL_N_ACTION_STEPS:-10}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/train/p5_cosmos_screen"}
LOG_DIR=${LOG_DIR:-"logs/cosmos_screen"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
if (( ${#GPU_ARR[@]} != 8 )); then
  echo "This screening script expects exactly 8 GPUs, got: ${GPU_IDS}" >&2
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

ensure_cache_exists() {
  local suffix="$1"
  local path="${CACHE_DIR}/${DATASET_REPO_ID}/train_world_latents_${WORLD_SOURCE}_${suffix}.fp16.npy"
  if [[ ! -f "${path}" ]]; then
    echo "Missing cache: ${path}" >&2
    exit 3
  fi
}

run_variant() {
  local gpu_id="$1"
  local exp_name="$2"
  local latent_suffix="$3"
  local world_camera="$4"
  local seed="$5"
  shift 5
  local extra_args=("$@")
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
      --policy.future_offset="${FUTURE_OFFSET}" \
      --policy.lambda_world="${LAMBDA_WORLD}" \
      --policy.world_memory_mode_train=pred \
      --policy.world_camera="${world_camera}" \
      --policy.enable_world_injection=true \
      --batch_size="${BATCH_SIZE}" \
      --num_workers="${NUM_WORKERS}" \
      --steps="${STEPS}" \
      --output_dir="${out_dir}" \
      --seed="${seed}" \
      --wandb.enable=false \
      "${extra_args[@]}" \
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
  echo $!
}

if [[ "${DRY_RUN}" == "1" ]]; then
  run_variant "${GPU_ARR[0]}" "C0_front_seed0" "${FRONT_SUFFIX}" "front" 0
  run_variant "${GPU_ARR[1]}" "C0_front_seed1" "${FRONT_SUFFIX}" "front" 1
  run_variant "${GPU_ARR[2]}" "C0_wrist_seed0" "${WRIST_SUFFIX}" "wrist" 0
  run_variant "${GPU_ARR[3]}" "C0_wrist_seed1" "${WRIST_SUFFIX}" "wrist" 1
  run_variant "${GPU_ARR[4]}" "C1_f2_front_seed0" "${FRONT_SUFFIX}" "front" 0 --policy.world_inject_suffix_in=true
  run_variant "${GPU_ARR[5]}" "C1_f2_front_seed1" "${FRONT_SUFFIX}" "front" 1 --policy.world_inject_suffix_in=true
  run_variant "${GPU_ARR[6]}" "C1_f2_wrist_seed0" "${WRIST_SUFFIX}" "wrist" 0 --policy.world_inject_suffix_in=true
  run_variant "${GPU_ARR[7]}" "C1_f2_wrist_seed1" "${WRIST_SUFFIX}" "wrist" 1 --policy.world_inject_suffix_in=true
  exit 0
fi

ensure_cache_exists "${FRONT_SUFFIX}"
ensure_cache_exists "${WRIST_SUFFIX}"

declare -a pids

pids+=("$(run_variant "${GPU_ARR[0]}" "C0_front_seed0" "${FRONT_SUFFIX}" "front" 0)")
pids+=("$(run_variant "${GPU_ARR[1]}" "C0_front_seed1" "${FRONT_SUFFIX}" "front" 1)")
pids+=("$(run_variant "${GPU_ARR[2]}" "C0_wrist_seed0" "${WRIST_SUFFIX}" "wrist" 0)")
pids+=("$(run_variant "${GPU_ARR[3]}" "C0_wrist_seed1" "${WRIST_SUFFIX}" "wrist" 1)")
pids+=("$(run_variant "${GPU_ARR[4]}" "C1_f2_front_seed0" "${FRONT_SUFFIX}" "front" 0 --policy.world_inject_suffix_in=true)")
pids+=("$(run_variant "${GPU_ARR[5]}" "C1_f2_front_seed1" "${FRONT_SUFFIX}" "front" 1 --policy.world_inject_suffix_in=true)")
pids+=("$(run_variant "${GPU_ARR[6]}" "C1_f2_wrist_seed0" "${WRIST_SUFFIX}" "wrist" 0 --policy.world_inject_suffix_in=true)")
pids+=("$(run_variant "${GPU_ARR[7]}" "C1_f2_wrist_seed1" "${WRIST_SUFFIX}" "wrist" 1 --policy.world_inject_suffix_in=true)")

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

exit "${fail}"
