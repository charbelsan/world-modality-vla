#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOS_DIR="${ROOT_DIR}/repos"

mkdir -p "${REPOS_DIR}"

clone() {
  local url="$1"
  local name="$2"
  local target="${REPOS_DIR}/${name}"

  if [[ -d "${target}/.git" ]]; then
    echo "[skip] ${name} already exists: ${target}"
    return
  fi

  echo "[clone] ${name}"
  git clone --depth 1 "${url}" "${target}"
}

# JEPA / V-JEPA2
clone "https://github.com/facebookresearch/jepa.git" "jepa"
clone "https://github.com/facebookresearch/vjepa2.git" "vjepa2"

# NVIDIA GR00T
clone "https://github.com/NVIDIA/Isaac-GR00T.git" "isaac-groot"

# NVIDIA Cosmos (reasoning and prediction)
clone "https://github.com/nvidia-cosmos/cosmos-reason1.git" "cosmos-reason1"
clone "https://github.com/nvidia-cosmos/cosmos-predict2.5.git" "cosmos-predict2.5"

echo
echo "Done. External repos cloned into:"
echo "  ${REPOS_DIR}"

