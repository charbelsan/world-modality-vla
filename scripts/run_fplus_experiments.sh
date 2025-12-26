#!/usr/bin/env bash
set -euo pipefail

# Common settings
DATASET=${DATASET:-"HuggingFaceVLA/libero"}
IMAGE_KEY=${IMAGE_KEY:-"observation.images.image"}
INSTRUCTION_KEY=${INSTRUCTION_KEY:-"task"}
EPISODE_ID_KEY=${EPISODE_ID_KEY:-"episode_index"}
CACHE_DIR=${CACHE_DIR:-"cache"}
WORLD_SOURCE=${WORLD_SOURCE:-"vjepa"}  # dino or vjepa
BACKBONE=${BACKBONE:-"qwen3_vl_3b_instruct"}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_EPOCHS=${MAX_EPOCHS:-5}
LOG_EVERY=${LOG_EVERY:-50}
ACTION_HEAD=${ACTION_HEAD:-"mse"}
FLOW_STEPS_EVAL=${FLOW_STEPS_EVAL:-8}
TEMPORAL_WINDOW=${TEMPORAL_WINDOW:-1}     # vjepa only (1=single-frame, 4=recommended)
LATENT_SUFFIX=${LATENT_SUFFIX:-""}        # e.g., "m4" to match TEMPORAL_WINDOW=4
DELTA_PREDICTION=${DELTA_PREDICTION:-1}   # 1=recommended for V-JEPA (predict z_{t+k}-z_t)

# Optional CoC JSONL (required for E4)
COC_JSONL=${COC_JSONL:-""}

if [[ "${WORLD_SOURCE}" == "dino" && "${TEMPORAL_WINDOW}" != "1" ]]; then
  echo "ERROR: TEMPORAL_WINDOW>1 is only supported for WORLD_SOURCE=vjepa (not dino)."
  exit 2
fi

if [[ -n "${LATENT_SUFFIX}" && "${LATENT_SUFFIX}" =~ ^m[0-9]+$ ]]; then
  SUFFIX_M=${LATENT_SUFFIX#m}
  if [[ "${TEMPORAL_WINDOW}" == "1" ]]; then
    TEMPORAL_WINDOW="${SUFFIX_M}"
  elif [[ "${TEMPORAL_WINDOW}" != "${SUFFIX_M}" ]]; then
    echo "ERROR: LATENT_SUFFIX=${LATENT_SUFFIX} implies TEMPORAL_WINDOW=${SUFFIX_M}, but TEMPORAL_WINDOW=${TEMPORAL_WINDOW}."
    exit 2
  fi
fi

if [[ "${TEMPORAL_WINDOW}" != "1" && -z "${LATENT_SUFFIX}" ]]; then
  LATENT_SUFFIX="m${TEMPORAL_WINDOW}"
fi

TAG="${WORLD_SOURCE}"
if [[ -n "${LATENT_SUFFIX}" ]]; then
  TAG="${TAG}_${LATENT_SUFFIX}"
fi
if [[ "${DELTA_PREDICTION}" == "1" ]]; then
  TAG="${TAG}_delta"
fi
TAG="${TAG}_${ACTION_HEAD}"

COMMON_ARGS=(
  --vlm_backbone "${BACKBONE}"
  --trust_remote_code
  --dataset_name "${DATASET}"
  --image_key "${IMAGE_KEY}"
  --instruction_key "${INSTRUCTION_KEY}"
  --episode_id_key "${EPISODE_ID_KEY}"
  --cache_dir "${CACHE_DIR}"
  --world_latents_source "${WORLD_SOURCE}"
  --latent_suffix "${LATENT_SUFFIX}"
  --future_memory_source "scheduled"
  --freeze_backbone
  --use_lora
  --batch_size "${BATCH_SIZE}"
  --max_epochs "${MAX_EPOCHS}"
  --log_every "${LOG_EVERY}"
  --action_head "${ACTION_HEAD}"
  --flow_steps_eval "${FLOW_STEPS_EVAL}"
)

EXPECTED_LATENTS="${CACHE_DIR}/${DATASET}/train_world_latents_${WORLD_SOURCE}"
if [[ -n "${LATENT_SUFFIX}" ]]; then
  EXPECTED_LATENTS="${EXPECTED_LATENTS}_${LATENT_SUFFIX}"
fi
EXPECTED_LATENTS="${EXPECTED_LATENTS}.fp16.npy"

if [[ -f "${EXPECTED_LATENTS}" ]]; then
  echo "=== Found cached latents: ${EXPECTED_LATENTS} ==="
else
  echo "=== Precompute world latents ==="
  python -m world_modality.precompute_world_latents \
    --dataset_name "${DATASET}" \
    --image_key "${IMAGE_KEY}" \
    --cache_dir "${CACHE_DIR}" \
    --world_latents_source "${WORLD_SOURCE}" \
    --temporal_window "${TEMPORAL_WINDOW}"
fi

echo "=== E0: VLM-BC baseline (no world loss, no injection) ==="
python -m world_modality.train_llm_vla \
  "${COMMON_ARGS[@]}" \
  --output_dir "logs_llm/E0_baseline_${TAG}" \
  --disable_future_injection \
  --lambda_world 0.0

echo "=== E2: Model F (world memory injection) ==="
E2_ARGS=(
  "${COMMON_ARGS[@]}"
  --output_dir "logs_llm/E2_model_f_${TAG}"
  --lambda_world 0.2
)
if [[ "${DELTA_PREDICTION}" == "1" ]]; then
  E2_ARGS+=(--delta_prediction)
fi
python -m world_modality.train_llm_vla "${E2_ARGS[@]}"

if [[ -n "${COC_JSONL}" && -f "${COC_JSONL}" ]]; then
  echo "=== E4: Full F+ (Model F + CoC + FLARE alignment) ==="
  E4_ARGS=(
    "${COMMON_ARGS[@]}"
    --output_dir "logs_llm/E4_fplus_${TAG}"
    --lambda_world 0.2
    --lambda_text 0.1
    --text_loss_mode "joint_after_act"
    --coc_jsonl "${COC_JSONL}"
  )
  if [[ "${DELTA_PREDICTION}" == "1" ]]; then
    E4_ARGS+=(--delta_prediction)
  fi
  python -m world_modality.train_llm_vla "${E4_ARGS[@]}"
else
  echo "COC_JSONL not set or missing. Skipping E4."
fi
