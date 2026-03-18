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

echo "Starting post-chain suite sweep..."

EVAL_EPISODES_OBJECT=${EVAL_EPISODES_OBJECT:-200}
EVAL_EPISODES_GOAL=${EVAL_EPISODES_GOAL:-200}

EVAL_TASK=libero_object \
EVAL_EPISODES="${EVAL_EPISODES_OBJECT}" \
LOG_DIR=logs/cogvideo_suite_eval/object \
bash scripts/launch_p5_eval_suite_matrix.sh

EVAL_TASK=libero_goal \
EVAL_EPISODES="${EVAL_EPISODES_GOAL}" \
LOG_DIR=logs/cogvideo_suite_eval/goal \
bash scripts/launch_p5_eval_suite_matrix.sh

echo "Starting post-suite horizon scan..."
bash scripts/launch_p5_cogvideo_kscan.sh
