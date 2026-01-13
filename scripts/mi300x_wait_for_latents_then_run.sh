#!/usr/bin/env bash
set -euo pipefail

# Wait for world-latents precompute to finish (cache tail becomes non-zero),
# then run the minimal E0/E1/E2 matrix.
#
# This avoids a common failure mode where the .npy exists (correct size/shape)
# but the last rows are still zeros because the precompute job was killed.
#
# Usage:
#   source /path/to/venv/bin/activate
#   export MUJOCO_GL=osmesa
#   cd /root/world-modality-vla-git
#   STEPS=50000 SEEDS="0" ./scripts/mi300x_wait_for_latents_then_run.sh

DATASET_REPO_ID=${DATASET_REPO_ID:-"HuggingFaceVLA/libero"}
CACHE_DIR=${CACHE_DIR:-"cache"}
WORLD_SOURCE=${WORLD_SOURCE:-"vjepa"}
LATENT_SUFFIX=${LATENT_SUFFIX:-"m4"}

LATENTS_PATH="${CACHE_DIR}/${DATASET_REPO_ID}/train_world_latents_${WORLD_SOURCE}_${LATENT_SUFFIX}.fp16.npy"
META_PATH="${LATENTS_PATH/.npy/_metadata.json}"

WAIT_S=${WAIT_S:-30}
MAX_WAIT_S=${MAX_WAIT_S:-0} # 0 = wait forever

check_ready () {
  /usr/bin/env python3 - <<PY
import os, sys, numpy as np
p="${LATENTS_PATH}"
m="${META_PATH}"
if not os.path.exists(p):
    print("missing_latents")
    sys.exit(2)
try:
    a = np.load(p, mmap_mode="r")
except Exception as e:
    print("load_error", type(e).__name__, e)
    sys.exit(3)
ok = bool((a[-1] != 0).any())
meta_ok = os.path.exists(m)
print("ready" if (ok and meta_ok) else "not_ready", "last_row_nonzero="+str(ok), "meta="+str(meta_ok))
sys.exit(0 if (ok and meta_ok) else 1)
PY
}

start_ts=$(date +%s)
echo "Waiting for latents to be complete:"
echo "  ${LATENTS_PATH}"
echo "  ${META_PATH}"

while true; do
  if check_ready; then
    break
  fi
  now=$(date +%s)
  elapsed=$((now - start_ts))
  if [[ "${MAX_WAIT_S}" != "0" && "${elapsed}" -ge "${MAX_WAIT_S}" ]]; then
    echo "Timed out waiting for latents after ${elapsed}s"
    exit 4
  fi
  sleep "${WAIT_S}"
done

run_id=$(date +%Y-%m-%d_%H-%M-%S)
export OUTPUT_ROOT=${OUTPUT_ROOT:-"outputs/train/libero_smolvla_world_matrix_${run_id}"}

echo "Latents ready. Launching matrix:"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"

exec ./scripts/run_mi300x_smolvla_world_matrix.sh

