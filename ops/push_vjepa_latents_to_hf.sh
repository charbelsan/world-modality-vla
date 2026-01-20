#!/usr/bin/env bash
set -euo pipefail

# Upload the precomputed V-JEPA LIBERO world latents to the Hugging Face Hub (as a dataset repo).
#
# Requires:
# - `hf` CLI (from `huggingface_hub`) installed
# - authentication already set up via `hf auth login` OR a write-capable token in env `HF_TOKEN`
#
# Example:
#   HF_TOKEN=hf_*** ./ops/push_vjepa_latents_to_hf.sh charbelsan/libero_world_latents_vjepa_m4
#
# Optional:
#   PRIVATE=1 HF_TOKEN=hf_*** ./ops/push_vjepa_latents_to_hf.sh charbelsan/libero_world_latents_vjepa_m4

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <repo_id>  (e.g. username/libero_world_latents_vjepa_m4)"
  exit 2
fi

REPO_ID="$1"

LATENTS_PATH="cache/HuggingFaceVLA/libero/train_world_latents_vjepa_m4.fp16.npy"
if [[ ! -f "${LATENTS_PATH}" ]]; then
  echo "Missing latents file: ${LATENTS_PATH}"
  exit 2
fi

if ! command -v hf >/dev/null; then
  echo "hf CLI not found. Install via: pip install -U huggingface_hub"
  exit 2
fi

PRIVATE_FLAG=""
if [[ "${PRIVATE:-0}" == "1" ]]; then
  PRIVATE_FLAG="--private"
fi

echo "Repo: ${REPO_ID} (dataset) ${PRIVATE_FLAG}"
echo "Latents: ${LATENTS_PATH}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "${HF_TOKEN}" >/dev/null
else
  if ! hf auth whoami >/dev/null 2>&1; then
    echo "Not logged in. Run this first (interactive, token won't be echoed):"
    echo "  hf auth login"
    echo "Or set HF_TOKEN=hf_*** when running this script."
    exit 2
  fi
fi

mkdir -p ops/_tmp_latents_readme
README_TMP="ops/_tmp_latents_readme/README.md"
cat > "${README_TMP}" <<'MD'
# LIBERO world latents (V-JEPA v2, temporal m=4)

This dataset contains **precomputed world latents** for `HuggingFaceVLA/libero`, used by the `smolvla_world`
policy in `world-modality-vla` as “world modality” external memory.

## Contents

- `train_world_latents_vjepa_m4.fp16.npy`

## How these latents were generated

- Source dataset: `HuggingFaceVLA/libero`
- Image key: `observation.images.image` (front camera)
- World encoder: `facebook/vjepa2-vitg-fpc64-256`
- Temporal encoding: `m=4` (4-frame temporal embedding)
- Dtype: float16
- Shape (expected): `[273465, 1408]`

## Licenses / attribution

These are *derived features* computed from:
- the source dataset (`HuggingFaceVLA/libero`)
- the world encoder weights (`facebook/vjepa2-vitg-fpc64-256`)

Please follow the licenses/terms of the upstream dataset and model when using or redistributing these latents.

## Intended use

Offline training caches world latents and indexes them by dataset global `index`. During closed-loop rollouts, the
policy uses an online world encoder + Prophet predictor; rollout should use **temporal encoding consistent with m=4**
to avoid distribution mismatch.
MD

META_TMP="ops/_tmp_latents_readme/metadata.json"
python3 - <<'PY' > "${META_TMP}"
import json
meta = {
  "source_dataset": "HuggingFaceVLA/libero",
  "image_key": "observation.images.image",
  "world_encoder": "facebook/vjepa2-vitg-fpc64-256",
  "latent_suffix": "m4",
  "temporal_window": 4,
  "dtype": "fp16",
}
print(json.dumps(meta, indent=2, sort_keys=True))
PY

echo "Creating repo (exist-ok)..."
hf repo create "${REPO_ID}" --repo-type dataset --exist-ok ${PRIVATE_FLAG}

echo "Uploading README + metadata..."
hf upload "${REPO_ID}" "${README_TMP}" "README.md" \
  --repo-type dataset --commit-message "Add dataset card"
hf upload "${REPO_ID}" "${META_TMP}" "metadata.json" \
  --repo-type dataset --commit-message "Add metadata"

echo "Uploading latents (this can take a few minutes)..."
hf upload "${REPO_ID}" "${LATENTS_PATH}" "train_world_latents_vjepa_m4.fp16.npy" \
  --repo-type dataset --commit-message "Add V-JEPA m4 world latents"

echo "Done."
echo "Pull with: hf_hub_download(repo_id='${REPO_ID}', repo_type='dataset', filename='train_world_latents_vjepa_m4.fp16.npy')"
