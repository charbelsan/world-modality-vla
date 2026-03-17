#!/usr/bin/env bash
set -euo pipefail

# H100-friendly launcher for the immediate SmolVLA world-modality follow-ups.
#
# Supported queues:
#   ablation   -> signflip then random_scaled on the same checkpoint
#   wrist      -> precompute wrist latents, train E2-wrist, eval
#   fusion     -> F2 train+eval, then F3b train+eval
#   kscan      -> E2-K4 train+eval, then E2-K2 train+eval
#
# Example:
#   GPU_ID=6 QUEUE=ablation CKPT=outputs/train/.../pretrained_model \
#     ./scripts/launch_h100_followup_queue.sh

export MUJOCO_GL=${MUJOCO_GL:-osmesa}

QUEUE=${QUEUE:-""}
GPU_ID=${GPU_ID:-0}

DATASET_REPO_ID=${DATASET_REPO_ID:-"HuggingFaceVLA/libero"}
INIT_POLICY_PATH=${INIT_POLICY_PATH:-"HuggingFaceVLA/smolvla_libero"}
CACHE_DIR=${CACHE_DIR:-"cache"}
WORLD_SOURCE=${WORLD_SOURCE:-"vjepa"}
LATENT_SUFFIX=${LATENT_SUFFIX:-"m4"}
WORLD_CAMERA=${WORLD_CAMERA:-"front"}
WORLD_LATENT_DIM=${WORLD_LATENT_DIM:-1408}
CONTEXT_FRAMES=${CONTEXT_FRAMES:-4}
FUTURE_OFFSET=${FUTURE_OFFSET:-8}
LAMBDA_WORLD=${LAMBDA_WORLD:-0.2}

STEPS=${STEPS:-25000}
BATCH_SIZE=${BATCH_SIZE:-192}
NUM_WORKERS=${NUM_WORKERS:-24}
SEED=${SEED:-0}

EVAL_TASK=${EVAL_TASK:-"libero_spatial"}
EVAL_EPISODES=${EVAL_EPISODES:-500}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-5}
EVAL_N_ACTION_STEPS=${EVAL_N_ACTION_STEPS:-10}

OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/train/p5_h100"}
LOG_DIR=${LOG_DIR:-"logs"}

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

if [[ -d ".hf_cache" ]]; then
  export HF_HOME="${HF_HOME:-$(pwd)/.hf_cache}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$(pwd)/.hf_cache/hub}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$(pwd)/.hf_cache/datasets}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(pwd)/.hf_cache/transformers}"
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

run_eval_mode() {
  local ckpt="$1"
  local mode="$2"
  local tag="$3"
  local action_flag
  action_flag="$(detect_eval_action_flag || true)"
  local -a extra=()
  if [[ -n "${action_flag}" ]]; then
    extra+=("${action_flag}")
  fi
  lerobot-wm-eval \
    --policy.path="${ckpt}" \
    --policy.device=cuda \
    --policy.type=smolvla_world \
    --policy.world_memory_mode_rollout="${mode}" \
    --env.type=libero \
    --env.task="${EVAL_TASK}" \
    --eval.n_episodes="${EVAL_EPISODES}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    "${extra[@]}" \
    2>&1 | tee "${LOG_DIR}/${tag}.log"
}

train_world_variant() {
  local exp_name="$1"
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
    --policy.latent_suffix="${LATENT_SUFFIX}" \
    --policy.world_latent_dim="${WORLD_LATENT_DIM}" \
    --policy.context_frames="${CONTEXT_FRAMES}" \
    --policy.future_offset="${FUTURE_OFFSET}" \
    --policy.lambda_world="${LAMBDA_WORLD}" \
    --policy.world_memory_mode_train="pred" \
    --policy.world_camera="${WORLD_CAMERA}" \
    --policy.enable_world_injection=true \
    --batch_size="${BATCH_SIZE}" \
    --num_workers="${NUM_WORKERS}" \
    --steps="${STEPS}" \
    --output_dir="${OUTPUT_ROOT}/${exp_name}" \
    --seed="${SEED}" \
    --wandb.enable=false \
    "$@" \
    2>&1 | tee "${LOG_DIR}/${exp_name}.log"
}

eval_trained_variant() {
  local exp_name="$1"
  local ckpt="${OUTPUT_ROOT}/${exp_name}/checkpoints/${STEPS}/pretrained_model"
  run_eval_mode "${ckpt}" pred "${exp_name}_eval"
}

case "${QUEUE}" in
  ablation)
    : "${CKPT:?Set CKPT=/path/to/pretrained_model for QUEUE=ablation}"
    run_eval_mode "${CKPT}" signflip "ablation_signflip"
    run_eval_mode "${CKPT}" random_scaled "ablation_random_scaled"
    ;;
  wrist)
    python -m world_modality.precompute_world_latents \
      --dataset_name "${DATASET_REPO_ID}" \
      --image_key observation.images.image2 \
      --cache_dir "${CACHE_DIR}" \
      --world_latents_source "${WORLD_SOURCE}" \
      --temporal_window 4 \
      --latent_suffix m4_wrist \
      --device cuda \
      --batch_size 512 \
      --num_workers 32 \
      --resume 2>&1 | tee "${LOG_DIR}/precompute_wrist.log"
    LATENT_SUFFIX=m4_wrist WORLD_CAMERA=wrist train_world_variant "E2_wrist_seed${SEED}_bs${BATCH_SIZE}"
    LATENT_SUFFIX=m4_wrist WORLD_CAMERA=wrist eval_trained_variant "E2_wrist_seed${SEED}_bs${BATCH_SIZE}"
    ;;
  fusion)
    train_world_variant "F2_suffix_in_seed${SEED}_bs${BATCH_SIZE}" \
      --policy.world_inject_suffix_in=true
    eval_trained_variant "F2_suffix_in_seed${SEED}_bs${BATCH_SIZE}"
    train_world_variant "F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}" \
      --policy.world_prefix_cross_attn=true
    eval_trained_variant "F3b_prefix_cross_seed${SEED}_bs${BATCH_SIZE}"
    ;;
  kscan)
    FUTURE_OFFSET=4 train_world_variant "E2_k4_seed${SEED}_bs${BATCH_SIZE}"
    FUTURE_OFFSET=4 eval_trained_variant "E2_k4_seed${SEED}_bs${BATCH_SIZE}"
    FUTURE_OFFSET=2 train_world_variant "E2_k2_seed${SEED}_bs${BATCH_SIZE}"
    FUTURE_OFFSET=2 eval_trained_variant "E2_k2_seed${SEED}_bs${BATCH_SIZE}"
    ;;
  *)
    echo "Usage: GPU_ID=<id> QUEUE=ablation|wrist|fusion|kscan $0"
    echo "For ablation also set: CKPT=/path/to/pretrained_model"
    exit 2
    ;;
esac
