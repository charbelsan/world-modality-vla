#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR=""
REMOTE_DIR=""
INTERVAL_SEC="60"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash ops/run_with_rclone_sync.sh --local <dir> --remote <remote:path> [--interval 60] -- <command...>

Example:
  bash ops/run_with_rclone_sync.sh \
    --local logs_c \
    --remote gdrive:world-modality-vla/logs_c \
    --interval 60 \
    -- python train_model_c.py --log_dir logs_c ...
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local)
      LOCAL_DIR="$2"; shift 2 ;;
    --remote)
      REMOTE_DIR="$2"; shift 2 ;;
    --interval)
      INTERVAL_SEC="$2"; shift 2 ;;
    --)
      shift; break ;;
    -*)
      echo "Unknown flag: $1" >&2; usage; exit 2 ;;
    *)
      break ;;
  esac
done

if [[ -z "${LOCAL_DIR}" || -z "${REMOTE_DIR}" ]]; then
  echo "--local and --remote are required." >&2
  usage
  exit 2
fi

if [[ $# -lt 1 ]]; then
  echo "Missing command after --" >&2
  usage
  exit 2
fi

mkdir -p "${LOCAL_DIR}"

echo "[run_with_rclone_sync] Starting background sync: ${LOCAL_DIR} -> ${REMOTE_DIR}"
bash ops/sync_rclone_dir.sh "${LOCAL_DIR}" "${REMOTE_DIR}" "${INTERVAL_SEC}" &
SYNC_PID=$!

cleanup() {
  echo "[run_with_rclone_sync] Stopping sync (pid=${SYNC_PID})"
  kill "${SYNC_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[run_with_rclone_sync] Running command: $*"
"$@"

