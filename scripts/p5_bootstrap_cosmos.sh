#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a fresh P5 instance for the Cosmos-feature branch.
#
# Goals:
# - keep the repo on fast local disk if desired
# - keep caches, logs, outputs, and the venv on /mnt/preserved
# - install the Cosmos Predict package with its CUDA extra
# - install this repo + LeRobot into the same environment
#
# Usage:
#   cd /opt/dlami/nvme/world-modality-vla
#   bash scripts/p5_bootstrap_cosmos.sh

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PRESERVED_ROOT=${PRESERVED_ROOT:-/mnt/preserved/world-modality-vla}
VENV_DIR=${VENV_DIR:-${PRESERVED_ROOT}/venvs/world-modality-vla-cosmos}
HF_CACHE_ROOT=${HF_CACHE_ROOT:-${PRESERVED_ROOT}/hf_cache}
COSMOS_ROOT=${COSMOS_ROOT:-${REPO_ROOT}/coc_vla/external/repos/cosmos-predict2.5}
COSMOS_REPO_URL=${COSMOS_REPO_URL:-https://github.com/nvidia-cosmos/cosmos-predict2.5.git}
CUDA_EXTRA=${CUDA_EXTRA:-cu128}
INSTALL_SYSTEM_DEPS=${INSTALL_SYSTEM_DEPS:-1}
ENV_FILE=${ENV_FILE:-${PRESERVED_ROOT}/p5_cosmos_env.sh}

if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "Expected to run from the world-modality-vla repo root, got: ${REPO_ROOT}" >&2
  exit 2
fi

ensure_dir_writable() {
  local path="$1"
  if mkdir -p "${path}" 2>/dev/null; then
    return
  fi
  sudo mkdir -p "${path}"
  sudo chown -R "$(id -u):$(id -g)" "${path}"
}

ensure_dir_writable "${PRESERVED_ROOT}"
ensure_dir_writable "${PRESERVED_ROOT}/outputs"
ensure_dir_writable "${PRESERVED_ROOT}/cache"
ensure_dir_writable "${PRESERVED_ROOT}/logs"
ensure_dir_writable "${PRESERVED_ROOT}/eval_libero_results"
ensure_dir_writable "${PRESERVED_ROOT}/artifacts"
ensure_dir_writable "${PRESERVED_ROOT}/venvs"
ensure_dir_writable "${HF_CACHE_ROOT}"

link_dir() {
  local local_name="$1"
  local target_path="$2"
  local local_path="${REPO_ROOT}/${local_name}"
  mkdir -p "${target_path}"

  if [[ -L "${local_path}" ]]; then
    return
  fi
  if [[ -e "${local_path}" ]]; then
    if [[ -d "${local_path}" && -z "$(ls -A "${local_path}")" ]]; then
      rmdir "${local_path}"
    else
      echo "Refusing to replace non-empty path: ${local_path}" >&2
      exit 3
    fi
  fi
  ln -s "${target_path}" "${local_path}"
}

link_dir "outputs" "${PRESERVED_ROOT}/outputs"
link_dir "cache" "${PRESERVED_ROOT}/cache"
link_dir "logs" "${PRESERVED_ROOT}/logs"
link_dir "eval_libero_results" "${PRESERVED_ROOT}/eval_libero_results"
link_dir ".hf_cache" "${HF_CACHE_ROOT}"

if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]]; then
  sudo apt-get update
  sudo apt-get install -y curl ffmpeg git-lfs libegl1 libosmesa6 libx11-dev mesa-utils tree wget
  git lfs install
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

mkdir -p "$(dirname "${COSMOS_ROOT}")"
if [[ ! -d "${COSMOS_ROOT}/.git" ]]; then
  git clone "${COSMOS_REPO_URL}" "${COSMOS_ROOT}"
else
  git -C "${COSMOS_ROOT}" pull --ff-only
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install -U pip setuptools wheel

export HF_HOME="${HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
export COSMOS_PREDICT2_ROOT="${COSMOS_ROOT}"

uv sync --project "${COSMOS_ROOT}" --group libero --extra="${CUDA_EXTRA}" --active --inexact
python -m pip install -e ".[lerobot]"

cat > "${ENV_FILE}" <<EOF
#!/usr/bin/env bash
export REPO_ROOT="${REPO_ROOT}"
export PRESERVED_ROOT="${PRESERVED_ROOT}"
export VENV_DIR="${VENV_DIR}"
export HF_HOME="${HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/datasets"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/transformers"
export COSMOS_PREDICT2_ROOT="${COSMOS_ROOT}"
export MUJOCO_GL="\${MUJOCO_GL:-egl}"
source "${VENV_DIR}/bin/activate"
EOF
chmod +x "${ENV_FILE}"

python - <<'PY'
import importlib
mods = ["torch", "lerobot", "lerobot_policy_world_modality"]
for name in mods:
    m = importlib.import_module(name)
    print(f"{name}: ok ({getattr(m, '__file__', 'built-in')})")
PY

echo
echo "Bootstrap complete."
echo "Activate with:"
echo "  source ${ENV_FILE}"
echo
echo "Then smoke-test:"
echo "  python scripts/smoke_test_cosmos_pipeline.py --device cuda --save-dir logs/cosmos_smoke"
