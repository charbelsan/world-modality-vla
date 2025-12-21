#!/usr/bin/env bash
set -euo pipefail

# Common settings
DATASET=${DATASET:-"HuggingFaceVLA/libero"}
IMAGE_KEY=${IMAGE_KEY:-"observation.images.image"}
INSTRUCTION_KEY=${INSTRUCTION_KEY:-"instruction"}
CACHE_DIR=${CACHE_DIR:-"cache"}
WORLD_SOURCE=${WORLD_SOURCE:-"vjepa"}  # dino or vjepa
BACKBONE=${BACKBONE:-"qwen3_vl_3b_instruct"}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_EPOCHS=${MAX_EPOCHS:-5}
LOG_EVERY=${LOG_EVERY:-50}

# Optional CoC JSONL (required for E3/E4)
COC_JSONL=${COC_JSONL:-""}

COMMON_ARGS=(
  --vlm_backbone "${BACKBONE}"
  --trust_remote_code
  --dataset_name "${DATASET}"
  --image_key "${IMAGE_KEY}"
  --instruction_key "${INSTRUCTION_KEY}"
  --cache_dir "${CACHE_DIR}"
  --world_latents_source "${WORLD_SOURCE}"
  --future_memory_source "scheduled"
  --freeze_backbone
  --use_lora
  --batch_size "${BATCH_SIZE}"
  --max_epochs "${MAX_EPOCHS}"
  --log_every "${LOG_EVERY}"
)

echo "=== Precompute world latents (if not already cached) ==="
python -m world_modality.precompute_world_latents \
  --dataset_name "${DATASET}" \
  --image_key "${IMAGE_KEY}" \
  --cache_dir "${CACHE_DIR}" \
  --world_latents_source "${WORLD_SOURCE}"

echo "=== E0: VLM-BC baseline (no world loss, no injection) ==="
python -m world_modality.train_llm_vla \
  "${COMMON_ARGS[@]}" \
  --output_dir "logs_llm/E0_baseline" \
  --disable_future_injection \
  --lambda_world 0.0

echo "=== E1: Aux world loss only (no injection) ==="
python -m world_modality.train_llm_vla \
  "${COMMON_ARGS[@]}" \
  --output_dir "logs_llm/E1_aux_world" \
  --disable_future_injection \
  --lambda_world 0.2

echo "=== E2: Model F (world memory injection) ==="
python -m world_modality.train_llm_vla \
  "${COMMON_ARGS[@]}" \
  --output_dir "logs_llm/E2_model_f" \
  --lambda_world 0.2

if [[ -n "${COC_JSONL}" && -f "${COC_JSONL}" ]]; then
  echo "=== E3: CoC loss (talk while acting) ==="
  python -m world_modality.train_llm_vla \
    "${COMMON_ARGS[@]}" \
    --output_dir "logs_llm/E3_coc" \
    --lambda_world 0.2 \
    --lambda_text 0.1 \
    --coc_jsonl "${COC_JSONL}"

  echo "=== E4: Full F+ (Model F + CoC + FLARE alignment) ==="
  python -m world_modality.train_llm_vla \
    "${COMMON_ARGS[@]}" \
    --output_dir "logs_llm/E4_fplus" \
    --lambda_world 0.2 \
    --lambda_text 0.1 \
    --coc_jsonl "${COC_JSONL}"
else
  echo "COC_JSONL not set or missing. Skipping E3/E4."
fi
