#!/usr/bin/env bash
set -euo pipefail

# Pull logs/results from the MI300X VM so we don't lose them if the VM/credits end.
#
# Examples:
#   # Minimal (no model weights)
#   ./ops/pull_mi300x_artifacts.sh root@165.245.137.3 /root/world-modality-vla-git ./mi300x_artifacts --mode minimal
#
#   # Full (includes pretrained_model weights / checkpoints)
#   ./ops/pull_mi300x_artifacts.sh root@165.245.137.3 /root/world-modality-vla-git ./mi300x_artifacts --mode full

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <user@host> <remote_repo_dir> <local_out_dir> [--mode minimal|full]"
  exit 2
fi

REMOTE="$1"
REMOTE_REPO="$2"
LOCAL_OUT="$3"
shift 3

MODE="minimal"
if [[ $# -ge 2 && "$1" == "--mode" ]]; then
  MODE="$2"
  shift 2
fi

if [[ "${MODE}" != "minimal" && "${MODE}" != "full" ]]; then
  echo "Invalid --mode: ${MODE} (expected minimal|full)"
  exit 2
fi

ts="$(date +%Y%m%d_%H%M%S)"
dest="${LOCAL_OUT%/}/${ts}_${MODE}"
mkdir -p "${dest}"

echo "Remote: ${REMOTE}"
echo "Remote repo: ${REMOTE_REPO}"
echo "Local dest: ${dest}"
echo "Mode: ${MODE}"

echo
echo "== Pull /tmp logs (train/eval stdout) =="
rsync -av --partial --progress \
  "${REMOTE}:/tmp/" "${dest}/tmp/" \
  --include='*/' \
  --include='*.log' \
  --include='*.pid' \
  --include='*.json' \
  --exclude='*'

echo
echo "== Pull repo-run artifacts =="
# Always pull lightweight metadata and outputs; by default, exclude large weights.
rsync_args=(
  -av --partial --progress
  "${REMOTE}:${REMOTE_REPO%/}/" "${dest}/repo/"
  --include='*/'
  --include='outputs/**'
  --include='eval_libero_results/**'
  --include='logs_llm/**'
  --include='RESEARCH_ANALYSIS.md'
  --include='flare_study.md'
  --include='docs/**'
  --include='scripts/**'
  --include='lerobot_policy_world_modality/**'
  --include='world_modality/**'
  --include='pyproject.toml'
  --include='requirements.txt'
  --include='environment.yml'
  --exclude='*'
)

if [[ "${MODE}" == "minimal" ]]; then
  # Drop heavy model weights, keep configs/processors/results.
  rsync "${rsync_args[@]}" \
    --exclude='**/*.safetensors' \
    --exclude='**/*.bin' \
    --exclude='**/*.pt' \
    --exclude='**/*.ckpt' \
    --exclude='**/*.pth' \
    --exclude='**/*.npy'
else
  rsync "${rsync_args[@]}"
fi

echo
echo "Done. Artifacts saved under: ${dest}"
echo "Tip: to see disk usage: du -sh \"${dest}\""

