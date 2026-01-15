#!/usr/bin/env bash
set -euo pipefail

# Wait for a PID to exit, then run a command (or a script).
#
# Example:
#   # Launch eval and save PID
#   nohup bash -lc 'MUJOCO_GL=osmesa lerobot-wm-eval ...' > /tmp/e2_eval.log 2>&1 & echo $! > /tmp/e2_eval.pid
#   # Queue next jobs
#   ./scripts/wait_for_pid_then_run.sh /tmp/e2_eval.pid "./scripts/run_next_jobs.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <pid_file|pid> <command...>"
  exit 2
fi

pid_arg="$1"
shift 1

pid=""
if [[ -f "${pid_arg}" ]]; then
  pid="$(cat "${pid_arg}")"
else
  pid="${pid_arg}"
fi

if ! [[ "${pid}" =~ ^[0-9]+$ ]]; then
  echo "Invalid PID: ${pid}"
  exit 2
fi

echo "Waiting for pid=${pid} to exit..."
while kill -0 "${pid}" 2>/dev/null; do
  sleep 30
done

echo "pid=${pid} exited. Running: $*"
exec "$@"

