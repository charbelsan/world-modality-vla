#!/usr/bin/env bash
set -euo pipefail

# MI300X-friendly launcher for the minimal LIBERO matrix:
#   E0: smolvla baseline
#   E1: smolvla_world (zero memory; capacity control)
#   E2: smolvla_world (pred memory; main hypothesis)
#
# Assumes:
# - `lerobot-wm-train` / `lerobot-wm-eval` are on PATH (installed by this repo)
# - this repo is installed (`pip install -e .`) so `smolvla_world` is registered
#
# Customize via env vars below.

export MUJOCO_GL=${MUJOCO_GL:-osmesa}

DATASET_REPO_ID=${DATASET_REPO_ID:-"HuggingFaceVLA/libero"}
CACHE_DIR=${CACHE_DIR:-"cache"}
WORLD_SOURCE=${WORLD_SOURCE:-"vjepa"}     # vjepa|dino
LATENT_SUFFIX=${LATENT_SUFFIX:-"m4"}      # m4 recommended for vjepa temporal
WORLD_LATENT_DIM=${WORLD_LATENT_DIM:-1408}
CONTEXT_FRAMES=${CONTEXT_FRAMES:-4}
FUTURE_OFFSET=${FUTURE_OFFSET:-8}
LAMBDA_WORLD=${LAMBDA_WORLD:-0.2}

STEPS=${STEPS:-200000}
BATCH_SIZE=${BATCH_SIZE:-64}
SEEDS=${SEEDS:-"0 1"}

OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/train/libero_smolvla_world_matrix"}
INIT_POLICY_PATH=${INIT_POLICY_PATH:-"lerobot/smolvla_base"}
RENAME_MAP_E0=${RENAME_MAP_E0:-'{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'}

# LIBERO eval
EVAL_TASK=${EVAL_TASK:-"libero_spatial"}
EVAL_EPISODES=${EVAL_EPISODES:-50}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-10}

mkdir -p "${OUTPUT_ROOT}"

LATENTS_PATH="${CACHE_DIR}/${DATASET_REPO_ID}/train_world_latents_${WORLD_SOURCE}_${LATENT_SUFFIX}.fp16.npy"
if [[ ! -f "${LATENTS_PATH}" ]]; then
  echo "Missing latents cache: ${LATENTS_PATH}"
  echo "Run precompute first (see docs/MI300X_LIBERO_SMOLVLA_WORLD.md)."
  exit 2
fi

run_train () {
  local exp_name="$1"
  local seed="$2"
  shift 2
  local out_dir="${OUTPUT_ROOT}/${exp_name}_seed${seed}"

  echo "=== Train ${exp_name} seed=${seed} -> ${out_dir} ==="
  if [[ -n "${LEROBOT_WM_RENAME_MAP_JSON:-}" ]]; then
    env LEROBOT_WM_RENAME_MAP_JSON="${LEROBOT_WM_RENAME_MAP_JSON}" \
      lerobot-wm-train \
      --dataset.repo_id="${DATASET_REPO_ID}" \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --batch_size="${BATCH_SIZE}" \
      --steps="${STEPS}" \
      --output_dir "${out_dir}" \
      --seed="${seed}" \
      --wandb.enable=false \
      "$@"
    return
  fi

  lerobot-wm-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --output_dir "${out_dir}" \
    --seed="${seed}" \
    --wandb.enable=false \
    "$@"
}

run_eval () {
  local out_dir="$1"
  shift 1
  local ckpt_dir="${out_dir}/checkpoints/${STEPS}/pretrained_model"
  if [[ ! -d "${ckpt_dir}" ]]; then
    echo "Checkpoint not found: ${ckpt_dir}"
    exit 3
  fi
  echo "=== Eval ${ckpt_dir} on ${EVAL_TASK} (${EVAL_EPISODES} eps) ==="
  if [[ -n "${LEROBOT_WM_RENAME_MAP_JSON:-}" ]]; then
    env LEROBOT_WM_RENAME_MAP_JSON="${LEROBOT_WM_RENAME_MAP_JSON}" \
      lerobot-wm-eval \
      --policy.path "${ckpt_dir}" \
      --policy.device=cuda \
      --env.type=libero \
      --env.task="${EVAL_TASK}" \
      --eval.n_episodes="${EVAL_EPISODES}" \
      --eval.batch_size="${EVAL_BATCH_SIZE}" \
      "$@"
    return
  fi

  lerobot-wm-eval \
    --policy.path "${ckpt_dir}" \
    --policy.device=cuda \
    --env.type=libero \
    --env.task="${EVAL_TASK}" \
    --eval.n_episodes="${EVAL_EPISODES}" \
    --eval.batch_size="${EVAL_BATCH_SIZE}" \
    "$@"
}

for seed in ${SEEDS}; do
  # E0 baseline: fine-tune from the same pretrained weights for a fair comparison.
  LEROBOT_WM_RENAME_MAP_JSON="${RENAME_MAP_E0}" run_train "E0_smolvla_baseline" "${seed}" \
    --policy.path="${INIT_POLICY_PATH}"

  run_train "E1_world_zero" "${seed}" \
    --policy.type="smolvla_world" \
    --policy.init_from_policy_path="${INIT_POLICY_PATH}" \
    --policy.dataset_repo_id="${DATASET_REPO_ID}" \
    --policy.cache_dir="${CACHE_DIR}" \
    --policy.world_latents_source="${WORLD_SOURCE}" \
    --policy.latent_suffix="${LATENT_SUFFIX}" \
    --policy.world_latent_dim="${WORLD_LATENT_DIM}" \
    --policy.context_frames="${CONTEXT_FRAMES}" \
    --policy.future_offset="${FUTURE_OFFSET}" \
    --policy.lambda_world="${LAMBDA_WORLD}" \
    --policy.world_memory_mode_train="zero" \
    --policy.enable_world_injection=true

  run_train "E2_world_pred" "${seed}" \
    --policy.type="smolvla_world" \
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
    --policy.enable_world_injection=true

  # Optional: evaluate each run after training completes.
  if [[ "${DO_EVAL:-0}" == "1" ]]; then
    LEROBOT_WM_RENAME_MAP_JSON="${RENAME_MAP_E0}" run_eval "${OUTPUT_ROOT}/E0_smolvla_baseline_seed${seed}"
    run_eval "${OUTPUT_ROOT}/E1_world_zero_seed${seed}"
    run_eval "${OUTPUT_ROOT}/E2_world_pred_seed${seed}"
  fi
done
