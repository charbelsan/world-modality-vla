# MI300X Runbook: LIBERO + SmolVLA + World Modality (Plugin Policy)

Goal: run a clean, reproducible experiment matrix where **SmolVLA** is the working baseline and we add
**world modality memory** via **gated cross-attention into the action expert only** (Model‑F “do‑no‑harm”).

This repo provides a LeRobot plugin policy:
- `--policy.type=smolvla_world`

Newer LeRobot versions may auto-discover packages named `lerobot_policy_*`. Some versions (e.g. LeRobot `0.4.2`)
do **not** reliably auto-discover, so this repo also installs wrapper entrypoints:
- `lerobot-wm-train`
- `lerobot-wm-eval`

---

## 0) One-time setup

### Install
From this repo:
```bash
pip install -e .
```

Make sure LeRobot is installed too (editable recommended):
```bash
pip install -e /path/to/lerobot
```

### Environment (MI300X / ROCm)
Recommended (adjust for your system):
```bash
export HF_HOME=/mnt/fast/hf_cache
export HF_HUB_CACHE=/mnt/fast/hf_cache
export TRANSFORMERS_CACHE=/mnt/fast/hf_cache
#
# LIBERO headless rendering:
# - On MI300X/ROCm machines, `MUJOCO_GL=osmesa` is the most reliable default.
# - If EGL is supported on your driver stack, you can try `MUJOCO_GL=egl` for faster rendering.
export MUJOCO_GL=osmesa
```

### LIBERO simulator deps (Ubuntu)
LIBERO closed-loop rollouts require `libero` + `robosuite` + `mujoco`, plus a few system deps.

On Ubuntu, install:
```bash
sudo apt-get update
sudo apt-get install -y cmake build-essential libosmesa6 libegl1 mesa-utils
```

Then in your Python env:
```bash
pip install mujoco libero
```

Non-interactive note: the `libero` package creates `~/.libero/config.yaml` on first import and can prompt via `input()`.
`lerobot-wm-train` / `lerobot-wm-eval` auto-create this config if missing. Set `LEROBOT_WM_SKIP_LIBERO_CONFIG=1` to disable.

---

## 1) Precompute world latents (offline training only)

Offline training uses cached latents indexed by the dataset global `index`. Rollouts do **not** require this cache.

Recommended (V‑JEPA temporal, `m=4`):
```bash
python -m world_modality.precompute_world_latents \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --cache_dir cache \
  --world_latents_source vjepa \
  --temporal_window 4 \
  --device cuda
```

This produces (example):
`cache/HuggingFaceVLA/libero/train_world_latents_vjepa_m4.fp16.npy`

---

## 2) Train baseline vs world-modality

Note: some LeRobot versions do not auto-discover policy plugins. This repo installs wrapper entrypoints:
- `lerobot-wm-train` (imports the plugin, then runs LeRobot train)
- `lerobot-wm-eval` (imports the plugin, then runs LeRobot eval)

### 2.1 Baseline (SmolVLA)
Important: for a fair comparison against `smolvla_world`, baseline should start from the same pretrained
weights (`lerobot/smolvla_base`).
```bash
lerobot-wm-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --batch_size=64 \
  --steps=200000 \
  --output_dir outputs/train/libero_smolvla_baseline_seed0 \
  --seed=0 \
  --wandb.enable=false
```

### 2.2 World modality (SmolVLA + gated world cross-attn)
Key knobs:
- `policy.context_frames` = T_ctx (default 4)
- `policy.future_offset` = K (default 8)
- `policy.lambda_world` = weight on world predictor loss
- `policy.world_memory_mode_train` = `pred|oracle|zero|shuffle|random` (oracle is **training-only**)

```bash
lerobot-wm-train \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.type=smolvla_world \
  --policy.device=cuda \
  --policy.init_from_policy_path=lerobot/smolvla_base \
  --policy.dataset_repo_id=HuggingFaceVLA/libero \
  --policy.cache_dir=cache \
  --policy.world_latents_source=vjepa \
  --policy.latent_suffix=m4 \
  --policy.world_latent_dim=1024 \
  --policy.context_frames=4 \
  --policy.future_offset=8 \
  --policy.lambda_world=0.2 \
  --policy.world_memory_mode_train=pred \
  --policy.enable_world_injection=true \
  --batch_size=64 \
  --steps=200000 \
  --output_dir outputs/train/libero_smolvla_world_seed0 \
  --seed=0 \
  --wandb.enable=false
```

Notes:
- If cached latents are missing, training will fail as soon as the first batch includes `index`.
- The action expert is unchanged; world memory is injected **only** into the action expert hidden states.
- World modules (Prophet + world injector) are created at policy init so the optimizer sees them; set `policy.world_latent_dim` consistently with your cached latents backend.

---

## 3) Evaluate closed-loop LIBERO success rate

Use LeRobot’s built-in LIBERO env:
```bash
lerobot-wm-eval \
  --policy.path outputs/train/libero_smolvla_world_seed0/checkpoints/200000/pretrained_model \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=50 \
  --eval.batch_size=10
```

---

## 4) Recommended experiment matrix (minimal, high-signal)

Run with 2–3 seeds each:

- **E0**: `smolvla` baseline
- **E1**: `smolvla_world` with `world_memory_mode_train=zero` (capacity control)
- **E2**: `smolvla_world` with `world_memory_mode_train=pred` (main hypothesis)

Optional plumbing checks (training-only):
- **Oracle**: `world_memory_mode_train=oracle` (upper bound; not a closed-loop mode)
- **Shuffle/Random**: should not help; if they do, you’re likely seeing a bug or a regularization effect.

---

## 5) What to watch in logs

LeRobot logs policy outputs from `forward()`; `smolvla_world` adds:
- `world_loss` (aux loss, masked near episode end)
- `world_cos` (diagnostic cosine similarity)
- `world_gate` (tanh(gate), should move off 0 if memory is used)
- `world_attn_entropy`, `world_attn_pmax`, `world_ctx_norm`, `world_act_norm` (if `policy.log_attn_stats=true`)
- `grad_world_inject`, `grad_prophet` (previous-step grad norms; if `policy.log_grad_stats=true`)
- `loss_total` (action + lambda_world*world_loss)

---

## 6) Parallel launch (MI300X)

If you want to saturate a multi-GPU MI300X node, use:
`scripts/launch_parallel_mi300x_smolvla_world.sh`.

Example (4 GPUs, 3 seeds):
```bash
GPU_IDS="0,1,2,3" SEEDS="0 1 2" STEPS=200000 \
  ./scripts/launch_parallel_mi300x_smolvla_world.sh
```

This writes per-run logs to:
`outputs/train/libero_smolvla_world_parallel/logs/*.log`
