#!/usr/bin/env bash
set -euo pipefail

# 8-GPU Cosmos feature precompute for LIBERO.
#
# Default layout:
# - first half of GPUs: front-view shards
# - second half of GPUs: wrist-view shards
# - merge both views after all shards complete
#
# Usage:
#   source /mnt/preserved/world-modality-vla/p5_cosmos_env.sh
#   bash scripts/launch_p5_cosmos_precompute.sh

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
CACHE_DIR=${CACHE_DIR:-"cache"}
VISION_MODEL_NAME=${VISION_MODEL_NAME:-"cosmos_cv8x8x8_pool4_m4"}
WORLD_SOURCE=${WORLD_SOURCE:-"cosmos"}
TEMPORAL_WINDOW=${TEMPORAL_WINDOW:-4}
GPU_IDS=${GPU_IDS:-"0,1,2,3,4,5,6,7"}
PRECOMP_BATCH_SIZE=${PRECOMP_BATCH_SIZE:-128}
PRECOMP_NUM_WORKERS=${PRECOMP_NUM_WORKERS:-16}
FRONT_SUFFIX=${FRONT_SUFFIX:-"m4_cosmos_front"}
WRIST_SUFFIX=${WRIST_SUFFIX:-"m4_cosmos_wrist"}
LOG_DIR=${LOG_DIR:-"logs/cosmos_precompute"}
DRY_RUN=${DRY_RUN:-0}

mkdir -p "${LOG_DIR}"
IFS=',' read -r -a GPU_ARR <<< "${GPU_IDS}"
GPU_COUNT=${#GPU_ARR[@]}
if (( GPU_COUNT < 2 || GPU_COUNT % 2 != 0 )); then
  echo "GPU_IDS must contain an even number of GPUs >= 2, got: ${GPU_IDS}" >&2
  exit 2
fi
SHARDS_PER_VIEW=$(( GPU_COUNT / 2 ))

launch_precompute() {
  local gpu_id="$1"
  local shard_index="$2"
  local image_key="$3"
  local suffix="$4"
  local tag="$5"
  local log_file="${LOG_DIR}/${tag}.log"
  local cmd=(
    python -m world_modality.precompute_world_latents
    --dataset_name "${DATASET_REPO_ID}"
    --image_key "${image_key}"
    --cache_dir "${CACHE_DIR}"
    --world_latents_source "${WORLD_SOURCE}"
    --vision_model_name "${VISION_MODEL_NAME}"
    --temporal_window "${TEMPORAL_WINDOW}"
    --latent_suffix "${suffix}"
    --device cuda
    --batch_size "${PRECOMP_BATCH_SIZE}"
    --num_workers "${PRECOMP_NUM_WORKERS}"
    --num_shards "${SHARDS_PER_VIEW}"
    --shard_index "${shard_index}"
    --resume
  )
  echo "=== ${tag} on GPU ${gpu_id} ==="
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "CUDA_VISIBLE_DEVICES=${gpu_id} ${cmd[*]}"
    return
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    "${cmd[@]}"
  ) > "${log_file}" 2>&1 &
  echo $!
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for ((i=0; i<SHARDS_PER_VIEW; i++)); do
    gpu="${GPU_ARR[$i]}"
    launch_precompute "${gpu}" "${i}" "observation.images.image" "${FRONT_SUFFIX}" "front_shard${i}"
  done
  for ((i=0; i<SHARDS_PER_VIEW; i++)); do
    gpu="${GPU_ARR[$((i + SHARDS_PER_VIEW))]}"
    launch_precompute "${gpu}" "${i}" "observation.images.image2" "${WRIST_SUFFIX}" "wrist_shard${i}"
  done
  exit 0
fi

pids=()
for ((i=0; i<SHARDS_PER_VIEW; i++)); do
  gpu="${GPU_ARR[$i]}"
  pids+=("$(launch_precompute "${gpu}" "${i}" "observation.images.image" "${FRONT_SUFFIX}" "front_shard${i}")")
done

for ((i=0; i<SHARDS_PER_VIEW; i++)); do
  gpu="${GPU_ARR[$((i + SHARDS_PER_VIEW))]}"
  pids+=("$(launch_precompute "${gpu}" "${i}" "observation.images.image2" "${WRIST_SUFFIX}" "wrist_shard${i}")")
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done
if [[ "${fail}" != "0" ]]; then
  echo "At least one shard precompute failed. See ${LOG_DIR}" >&2
  exit 3
fi

python -m world_modality.merge_world_latent_shards \
  --dataset_name "${DATASET_REPO_ID}" \
  --cache_dir "${CACHE_DIR}" \
  --world_latents_source "${WORLD_SOURCE}" \
  --latent_suffix "${FRONT_SUFFIX}" \
  --num_shards "${SHARDS_PER_VIEW}" \
  > "${LOG_DIR}/merge_front.log" 2>&1

python -m world_modality.merge_world_latent_shards \
  --dataset_name "${DATASET_REPO_ID}" \
  --cache_dir "${CACHE_DIR}" \
  --world_latents_source "${WORLD_SOURCE}" \
  --latent_suffix "${WRIST_SUFFIX}" \
  --num_shards "${SHARDS_PER_VIEW}" \
  > "${LOG_DIR}/merge_wrist.log" 2>&1

echo "Cosmos precompute finished successfully."
echo "Logs: ${LOG_DIR}"
