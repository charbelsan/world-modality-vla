#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="${1:-}"
REMOTE_DIR="${2:-}"
INTERVAL_SEC="${3:-60}"

if [[ -z "${LOCAL_DIR}" || -z "${REMOTE_DIR}" ]]; then
  echo "Usage: $0 <local_dir> <remote_dir> [interval_sec]" >&2
  exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found. Install it first (see ops/rclone_gdrive_setup.md)." >&2
  exit 2
fi

echo "[sync_rclone_dir] Copying '${LOCAL_DIR}' -> '${REMOTE_DIR}' every ${INTERVAL_SEC}s"

while true; do
  ts="$(date -Is)"
  echo "[sync_rclone_dir] ${ts} rclone copy start"
  # copy (not sync) so we never delete remote artifacts if local disappears.
  rclone copy "${LOCAL_DIR}" "${REMOTE_DIR}" \
    --create-empty-src-dirs \
    --checksum \
    --transfers 8 \
    --checkers 16 \
    --stats-one-line \
    --stats 10s || true
  echo "[sync_rclone_dir] ${ts} rclone copy done"
  sleep "${INTERVAL_SEC}"
done

