#!/usr/bin/env bash
set -euo pipefail

# Launch the SmolVLA(+world) matrix in parallel background processes.
#
# Usage:
#   GPU_IDS="0,1,2,3" SEEDS="0 1 2" STEPS=200000 ./scripts/launch_parallel_mi300x_smolvla_world.sh
#
# Notes:
# - Uses `tmux`/`screen`-friendly plain background jobs and writes per-run logs.
# - Sets HIP/CUDA/ROCR visible device env vars for each job (round-robin over GPU_IDS).

export MUJOCO_GL=${MUJOCO_GL:-osmesa}

DATASET_REPO_ID=${DATASET_REPO_ID:-"HuggingFaceVLA/libero"}
CACHE_DIR=${CACHE_DIR:-"cache"}
WORLD_SOURCE=${WORLD_SOURCE:-"vjepa"}     # vjepa|dino
LATENT_SUFFIX=${LATENT_SUFFIX:-"m4"}
WORLD_LATENT_DIM=${WORLD_LATENT_DIM:-1408}
CONTEXT_FRAMES=${CONTEXT_FRAMES:-4}
FUTURE_OFFSET=${FUTURE_OFFSET:-8}
LAMBDA_WORLD=${LAMBDA_WORLD:-0.2}

STEPS=${STEPS:-200000}
BATCH_SIZE=${BATCH_SIZE:-64}
SEEDS=${SEEDS:-"0 1"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/train/libero_smolvla_world_parallel"}
INIT_POLICY_PATH=${INIT_POLICY_PATH:-"lerobot/smolvla_base"}

# Comma-separated GPU ids; if empty, runs everything on the default visible device.
GPU_IDS=${GPU_IDS:-""}

mkdir -p "${OUTPUT_ROOT}/logs"

LATENTS_PATH="${CACHE_DIR}/${DATASET_REPO_ID}/train_world_latents_${WORLD_SOURCE}_${LATENT_SUFFIX}.fp16.npy"
if [[ ! -f "${LATENTS_PATH}" ]]; then
  echo "Missing latents cache: ${LATENTS_PATH}"
  echo "Run precompute first (see docs/MI300X_LIBERO_SMOLVLA_WORLD.md)."
  exit 2
fi

IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
gpu_for_job () {
  local job_idx="$1"
  if [[ -z "${GPU_IDS}" ]]; then
    echo ""
    return
  fi
  local n="${#GPU_ARR[@]}"
  echo "${GPU_ARR[$((job_idx % n))]}"
}

run_job () {
  local job_idx="$1"
  local exp_name="$2"
  local seed="$3"
  shift 3

  local out_dir="${OUTPUT_ROOT}/${exp_name}_seed${seed}"
  local log_file="${OUTPUT_ROOT}/logs/${exp_name}_seed${seed}.log"
  local gpu
  gpu="$(gpu_for_job "${job_idx}")"

  echo "=== Launch ${exp_name} seed=${seed} gpu=${gpu:-default} -> ${out_dir} ==="

  (
    if [[ -n "${gpu}" ]]; then
      export HIP_VISIBLE_DEVICES="${gpu}"
      export ROCR_VISIBLE_DEVICES="${gpu}"
      export CUDA_VISIBLE_DEVICES="${gpu}"
    fi
    exec lerobot-wm-train \
      --dataset.repo_id="${DATASET_REPO_ID}" \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --batch_size="${BATCH_SIZE}" \
      --steps="${STEPS}" \
      --output_dir "${out_dir}" \
      --seed="${seed}" \
      --wandb.enable=true \
      --wandb.mode=offline \
      "$@"
  ) >"${log_file}" 2>&1 &

  echo $!  # pid
}

job_idx=0
pids=()

for seed in ${SEEDS}; do
  # E0 baseline: fine-tune from the same pretrained weights for a fair comparison.
  pids+=("$(run_job "${job_idx}" "E0_smolvla_baseline" "${seed}" \
    --policy.path="${INIT_POLICY_PATH}" \
  )"); job_idx=$((job_idx+1))

  pids+=("$(run_job "${job_idx}" "E1_world_zero" "${seed}" \
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
    --policy.enable_world_injection=true \
  )"); job_idx=$((job_idx+1))

  pids+=("$(run_job "${job_idx}" "E2_world_pred" "${seed}" \
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
    --policy.enable_world_injection=true \
  )"); job_idx=$((job_idx+1))
done

echo "Launched ${#pids[@]} jobs:"
printf '  pid=%s\n' "${pids[@]}"
echo "Logs: ${OUTPUT_ROOT}/logs/*.log"

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "Job failed: pid=${pid}"
    fail=1
  fi
done

exit "${fail}"
