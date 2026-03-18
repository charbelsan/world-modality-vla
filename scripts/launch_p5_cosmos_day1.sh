#!/usr/bin/env bash
set -euo pipefail

# End-to-end day-1 launcher for the new P5 block.
#
# Phases:
# 1. bootstrap env + durable paths (optional)
# 2. front/wrist smoke test
# 3. 8-GPU Cosmos precompute
# 4. 8-GPU Cosmos screening matrix
#
# Usage:
#   nohup bash scripts/launch_p5_cosmos_day1.sh > /mnt/preserved/world-modality-vla/logs/p5_cosmos_day1.log 2>&1 &

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PRESERVED_ROOT=${PRESERVED_ROOT:-/mnt/preserved/world-modality-vla}
ENV_FILE=${ENV_FILE:-${PRESERVED_ROOT}/p5_cosmos_env.sh}
RUN_BOOTSTRAP=${RUN_BOOTSTRAP:-1}
LOG_DIR=${LOG_DIR:-${PRESERVED_ROOT}/logs/p5_cosmos_day1}
SMOKE_DIR=${SMOKE_DIR:-${LOG_DIR}/smoke}
GPU_FRONT_SMOKE=${GPU_FRONT_SMOKE:-0}
GPU_WRIST_SMOKE=${GPU_WRIST_SMOKE:-1}

mkdir -p "${LOG_DIR}" "${SMOKE_DIR}"

cd "${REPO_ROOT}"

if [[ "${RUN_BOOTSTRAP}" == "1" || ! -f "${ENV_FILE}" ]]; then
  bash scripts/p5_bootstrap_cosmos.sh | tee "${LOG_DIR}/bootstrap.log"
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

(
  export CUDA_VISIBLE_DEVICES="${GPU_FRONT_SMOKE}"
  python scripts/smoke_test_cosmos_pipeline.py \
    --device cuda \
    --sample-index 0 \
    --save-dir "${SMOKE_DIR}/front" \
    2>&1 | tee "${LOG_DIR}/smoke_front.log"
) &
PID_FRONT=$!

(
  export CUDA_VISIBLE_DEVICES="${GPU_WRIST_SMOKE}"
  python scripts/smoke_test_cosmos_pipeline.py \
    --device cuda \
    --sample-index 8 \
    --save-dir "${SMOKE_DIR}/wrist" \
    2>&1 | tee "${LOG_DIR}/smoke_wrist.log"
) &
PID_WRIST=$!

wait "${PID_FRONT}"
wait "${PID_WRIST}"

bash scripts/launch_p5_cosmos_precompute.sh 2>&1 | tee "${LOG_DIR}/precompute.log"
bash scripts/launch_p5_cosmos_screen.sh 2>&1 | tee "${LOG_DIR}/screen.log"
