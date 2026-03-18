#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-$(pwd)}
VENV_PATH=${VENV_PATH:-/opt/dlami/nvme/.venvs/world-modality-vla-ss/bin/activate}
LOG_DIR=${LOG_DIR:-logs}

mkdir -p "${ROOT_DIR}/${LOG_DIR}"

cron_line="@reboot cd ${ROOT_DIR} && ROOT_DIR=${ROOT_DIR} VENV_PATH=${VENV_PATH} nohup ${ROOT_DIR}/scripts/launch_p5_post_reboot_full_day.sh > ${ROOT_DIR}/${LOG_DIR}/p5_post_reboot_full_day.log 2>&1 &"

tmp_cron="$(mktemp)"
crontab -l 2>/dev/null | grep -v 'launch_p5_post_reboot_full_day.sh' > "${tmp_cron}" || true
echo "${cron_line}" >> "${tmp_cron}"
crontab "${tmp_cron}"
rm -f "${tmp_cron}"

echo "Installed reboot cron:"
crontab -l | grep 'launch_p5_post_reboot_full_day.sh'
