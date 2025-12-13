#!/usr/bin/env bash
set -euo pipefail

LOCAL_FILE="${1:-}"
REMOTE_FILE="${2:-}"
INTERVAL_SEC="${3:-60}"

if [[ -z "${LOCAL_FILE}" || -z "${REMOTE_FILE}" ]]; then
  echo "Usage: $0 <local_file> <remote_file> [interval_sec]" >&2
  exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone not found. Install it first (see ops/rclone_gdrive_setup.md)." >&2
  exit 2
fi

echo "[sync_rclone_file] Copying '${LOCAL_FILE}' -> '${REMOTE_FILE}' every ${INTERVAL_SEC}s"

while true; do
  ts="$(date -Is)"
  if [[ -f "${LOCAL_FILE}" ]]; then
    echo "[sync_rclone_file] ${ts} rclone copyto start"
    rclone copyto "${LOCAL_FILE}" "${REMOTE_FILE}" \
      --checksum \
      --stats-one-line \
      --stats 10s || true
    echo "[sync_rclone_file] ${ts} rclone copyto done"
  else
    echo "[sync_rclone_file] ${ts} local file not found yet: ${LOCAL_FILE}"
  fi
  sleep "${INTERVAL_SEC}"
done

