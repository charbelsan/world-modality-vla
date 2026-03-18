#!/usr/bin/env bash
set -euo pipefail

export WORLD_SOURCE=${WORLD_SOURCE:-cogvideo}
export WORLD_VISION_MODEL_NAME=${WORLD_VISION_MODEL_NAME:-cogvideo_2b_pool4_m4}
export WORLD_LATENT_DIM=${WORLD_LATENT_DIM:-256}
export FRONT_SUFFIX=${FRONT_SUFFIX:-m4_cogvideo_front}
export WRIST_SUFFIX=${WRIST_SUFFIX:-m4_cogvideo_wrist}
export OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/train/p5_cogvideo_screen}
export LOG_DIR=${LOG_DIR:-logs/cogvideo_screen}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${SCRIPT_DIR}/launch_p5_cosmos_screen.sh" "$@"
