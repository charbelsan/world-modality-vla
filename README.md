# World Modality VLA

**Goal:** validate the hypothesis **“world modality as external memory improves VLA control”**.

We treat predicted future world representations as an **extra modality** (like vision/language/state), not as extra
tokens competing in the main sequence. The core principle is **Model‑F do‑no‑harm**:

- Predict futures with a separate module (“Prophet”): `z_{t-T+1:t} -> ẑ_{t+1:t+K}`
- Inject predicted memory **only into the action path** via **gated cross‑attention**
- Initialize the gate to **0**, so the policy starts identical to the baseline

This repo currently focuses on a **working LIBERO baseline** (SmolVLA) and adds world modality as a controlled
intervention.

---

## Start here (recommended path)

- Recommended branch for current experiments: `phaseC-flow-head`
- Research status + conclusions: `RESEARCH_ANALYSIS.md`
- MI300X runbook (setup + commands): `docs/MI300X_LIBERO_SMOLVLA_WORLD.md`
- Fusion ablations (F1–F3: late vs early world fusion): `docs/SMOLVLA_WORLD_FUSION_ABLATIONS.md`

---

## Quickstart: reproduce the core LIBERO matrix (E0/E1/E2)

### Run on a fresh machine (what your colleague should do)

1) Clone the repos (use the current branch):
```bash
git clone https://github.com/charbelsan/world-modality-vla.git
cd world-modality-vla
git checkout phaseC-flow-head
```

2) Create a Python env and install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install this repo (adds `lerobot-wm-train` / `lerobot-wm-eval` and `smolvla_world`)
pip install -e .

# Install LeRobot (either from pip or editable from source)
pip install lerobot
```

3) Install LIBERO simulator deps:
- System deps (Ubuntu example): `libosmesa6`, `libegl1`, `mesa-utils`, build tools
- Python deps: `mujoco`, `libero`

Then set headless rendering:
```bash
export MUJOCO_GL=osmesa
```

If anything is ambiguous for the environment setup, follow the runbook:
`docs/MI300X_LIBERO_SMOLVLA_WORLD.md`.

### 0) Install + env

You need:
- LeRobot installed (editable recommended)
- This repo installed (registers `smolvla_world` policy + wrapper CLIs)
- LIBERO simulator deps (`mujoco`, `libero`, render backend)

```bash
# In your Python venv
pip install -e .
pip install -e /path/to/lerobot

# Headless rollouts (most reliable default)
export MUJOCO_GL=osmesa
```

### 1) Precompute world latents (offline training only)

Training uses cached V‑JEPA latents indexed by dataset `index`.
Rollouts do **not** require this cache, but do require an online world encoder with matching settings.

```bash
python -m world_modality.precompute_world_latents \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --cache_dir cache \
  --world_latents_source vjepa \
  --temporal_window 4 \
  --device cuda
```

If you already have a precomputed file (e.g. from a previous VM), place it here (exact path matters):
- `cache/HuggingFaceVLA/libero/train_world_latents_vjepa_m4.fp16.npy`

Quick sanity check (should exist and be non-trivial size, ~735MB for LIBERO train):
```bash
ls -lh cache/HuggingFaceVLA/libero/train_world_latents_vjepa_m4.fp16.npy
```

### 2) Run the minimal experiment matrix

Definitions:
- **E0**: baseline SmolVLA fine‑tune
- **E1**: `smolvla_world` with **zero memory** (capacity control; should match E0)
- **E2**: `smolvla_world` with **predicted memory** (main hypothesis)

```bash
STEPS=50000 SEEDS="0" DO_EVAL=1 \
EVAL_TASK=libero_spatial EVAL_EPISODES=500 EVAL_BATCH_SIZE=10 EVAL_N_ACTION_STEPS=10 \
./scripts/run_mi300x_smolvla_world_matrix.sh
```

Notes:
- For apples‑to‑apples comparisons, keep **episode count identical** across E0/E1/E2 (don’t mix 20 and 200).
- If you precompute latents with `latent_suffix=m4`, keep rollout encoding consistent:
  - default `policy.world_rollout_temporal_window=0` infers `m=4` from `latent_suffix=m4`
  - forcing `policy.world_rollout_temporal_window=1` creates an embedding distribution mismatch (can hurt SR)
 - The launcher runs E0 first; E1/E2 require cached latents. If latents are missing, it will stop before E1.

### 3) Evaluate a trained checkpoint (direct)

After training, checkpoints live under:
- `outputs/train/libero_smolvla_world_matrix/<EXP_NAME>_seed<S>/checkpoints/<STEPS>/pretrained_model`

Example:
```bash
lerobot-wm-eval \
  --policy.path outputs/train/libero_smolvla_world_matrix/E2_world_pred_seed0/checkpoints/50000/pretrained_model \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=500 \
  --eval.batch_size=10
```

To run rollout ablations on the same checkpoint:
```bash
# Remove world information (do-no-harm at inference)
lerobot-wm-eval ... --policy.type=smolvla_world --policy.world_memory_mode_rollout=zero

# Corrupt memory (tests whether memory *quality* matters)
lerobot-wm-eval ... --policy.type=smolvla_world --policy.world_memory_mode_rollout=random
```

---

## How we interpret “world helps”

We only claim the hypothesis is supported if:

1) **E1 ≈ E0** (do‑no‑harm; capacity control passes), and
2) **E2 > E0** with the same eval protocol, and
3) Rollout ablations show **memory content matters**:
   - `world_memory_mode_rollout=pred` > `zero`
   - `random` < `pred`

Rollout modes:
- `pred`: Prophet memory (hypothesis mode)
- `zero`: zero memory (do‑no‑harm at inference)
- `random`: noise memory (tests quality vs “extra input”)

---

## What’s inside this repo (relevant folders)

- `lerobot_policy_world_modality/`: LeRobot policy plugin (`--policy.type=smolvla_world`)
- `world_modality/`: world encoder + Prophet + latent precompute utilities
- `scripts/`: launchers (matrix + parallel) and analysis helpers
- `docs/`: runbooks + experiment matrix + fusion ablations

---

## Legacy / exploratory pipelines

This repo also contains earlier/experimental pipelines (kept for research continuity, not the recommended baseline):

- Qwen‑based VLM + `<ACT_i>` action readout (“F+”): `docs/LLM_VLA_FPLUS.md`, `docs/L40S_RUNBOOK.md`
- Earlier model variants (A/B/C/F) in `world_modality/` used for offline studies and ablations

If your goal is **closed-loop LIBERO success rate**, start from SmolVLA + `smolvla_world` as above.

Optional:
- CoC label generation (interpretability experiments): `coc_vla/` (see `docs/LLM_VLA_FPLUS.md`)

---

## Sharing results / not losing VM logs

If you’re running on a rented VM, pull logs + outputs locally periodically:
- `ops/pull_mi300x_artifacts.sh` (see `docs/MI300X_LIBERO_SMOLVLA_WORLD.md`)
