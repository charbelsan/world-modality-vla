#!/usr/bin/env bash
set -euo pipefail

export WORLD_SOURCE=${WORLD_SOURCE:-cogvideo}
export VISION_MODEL_NAME=${VISION_MODEL_NAME:-cogvideo_2b_pool4_m4}
export FRONT_SUFFIX=${FRONT_SUFFIX:-m4_cogvideo_front}
export WRIST_SUFFIX=${WRIST_SUFFIX:-m4_cogvideo_wrist}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/launch_p5_cosmos_precompute.sh" "$@"
