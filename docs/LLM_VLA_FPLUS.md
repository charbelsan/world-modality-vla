## Model F+ (Qwen3-VL + World Memory + CoC)

This document defines the F+ model and the exact experiments to run on L40S.
F+ extends Model F with optional CoC language loss and FLARE-style latent alignment.

### Core principle (do-no-harm)
Actions are always read from `<ACT_i>` hidden states. CoC “talk” is optional and
**must not change the action path**.

Default implementation (`--text_loss_mode joint_after_act`) appends CoC tokens
*after* `<ACT>` tokens in the same autoregressive sequence, so causality ensures
CoC tokens cannot influence `<ACT>` hidden states. You can force a fully
separate forward pass with `--text_loss_mode separate` if you want the simplest
mental model.

### Architecture summary
1) VLM backbone: Qwen3-VL-3B-Instruct.
2) Action interface:
   - Prompt: "<image> <instruction> <ACT_0> ... <ACT_{H-1}>"
   - Hidden states at ACT tokens -> ActionHead -> action chunk.
3) World memory (Model F):
   - Prophet predicts future latents z_pred from z_hist.
   - Cross-attend z_pred into ACT hidden states only.
   - Gated residual with gate initialized to 0.
4) CoC (Phase 2):
   - Default: **single forward pass** where CoC target tokens come after `<ACT>`.
   - Optional: **separate forward pass** for CoC loss.
   - In both cases: actions are computed from `<ACT>` hidden states and are not
     conditioned on generated CoC tokens.

### Relation to Alpamayo‑R1
Alpamayo‑R1 trains a VLM on a *unified autoregressive token stream* where
**reasoning and trajectories share a common token space** and are optimized with
standard next‑token cross‑entropy. In that setup, reasoning is an explicit
conditioning signal for the behavior/trajectory prediction.

F+ intentionally differs: we want a “do‑no‑harm” robotics policy where control
does not depend on any text generation (to avoid feedback loops and brittle
shortcuts). If you want to explore Alpamayo‑style coupling later, do it as an
explicit ablation (e.g., generate reasoning first, then act), not as the default.

Reference:
- Alpamayo‑R1 (NVIDIA, 2025): arXiv:2511.00088

### Losses
- L_action: MSE on action chunk.
- L_world: cosine loss between z_pred and z_future (FLARE-style alignment).
- L_text: text loss on CoC labels (optional).

Action head variants:
- `--action_head mse` (default): predicts continuous actions with MSE loss.
- `--action_head flow`: rectified flow matching over action chunks. Loss is
  MSE on predicted velocity field; validation still reports action MSE by
  sampling the flow (controlled by `--flow_steps_eval`).

Total:
  L = L_action + lambda_world * L_world + lambda_text * L_text

### Data requirements
- LIBERO dataset in LeRobot format.
- For `HuggingFaceVLA/libero`, the task language string is provided under the key `task` (not `instruction`).
- For closed-loop LIBERO, the baseline is much more likely to work if you also condition on:
  - `observation.state` (8-dim proprio; LeRobot’s flattened `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`)
  - `observation.images.image2` (wrist camera)
  The current recommended path is to **concatenate agentview + wrist into a single image** (`--wrist_mode concat`)
  and inject proprio into `<ACT>` states via a gated residual (`--use_proprio`).
- Precomputed world latents (DINO or V-JEPA).
- Optional CoC JSONL with {episode_index, coc_text} (or {episode_id, coc_text} for backward-compat).
  - Missing CoC entries are skipped by default; use `--require_coc` to drop episodes without labels.

### Precompute world latents
```
python -m world_modality.precompute_world_latents \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --world_latents_source vjepa \
  --cache_dir cache \
  --temporal_window 4
```
Then train with `--latent_suffix m4` so the trainer loads the `_m4` cache file.

Recommended for V-JEPA: also pass `--delta_prediction` so Prophet learns
`z_{t+k} - z_t` rather than copying near-invariant latents.

### Experiment matrix (E0-E4)
Default runs (in launcher):
- E0: VLM-BC baseline (no world loss, no injection)
- E2: Model F (world memory injection)
- E4: Full F+ (Model F + CoC + FLARE alignment)

Optional diagnostics (not run by default):
- E1: Auxiliary world loss only (no injection)
- E3: CoC loss without F+ (talk while acting)

### Corruption evaluation
Use validation corruption to verify reliance on future memory:
--corruption_mode {none, zero, random, shuffle, oracle}

Expected:
- Oracle helps early.
- Corruption hurts as gate opens.

### LIBERO success-rate evaluation
Use `world_modality/eval_libero.py` to run closed-loop rollouts and report success rate:
```
python -m world_modality.eval_libero \
  --checkpoint logs_llm/E2_model_f_vjepa_m4_delta_mse/llm_vla_epoch4.pt \
  --suite libero_spatial \
  --n_episodes 10 \
  --libero_root /home/ubuntu/LIBERO
```

### Notes from SmolVLA to borrow (principles)
- Keep the model small and efficient.
- Reduce token pressure and avoid bloated prompts.
- Separate policy and language outputs (avoid feedback loops).
- Prefer inference strategies that decouple prediction and execution (RTC-style).

### Quick launch
Use `scripts/run_fplus_experiments.sh` to run the default experiments (E0, E2, E4).
For a zero-ambiguity setup, see `docs/L40S_RUNBOOK.md`.

Environment variables (optional):
- DATASET, IMAGE_KEY, INSTRUCTION_KEY, EPISODE_ID_KEY, CACHE_DIR
- USE_PROPRIO (0/1), PROPRIO_KEY (default: observation.state)
- WRIST_MODE (none|concat), WRIST_IMAGE_KEY (default: observation.images.image2)
- WORLD_SOURCE (dino or vjepa)
- TEMPORAL_WINDOW (vjepa only), LATENT_SUFFIX (e.g., m4), DELTA_PREDICTION (0/1)
- BACKBONE (default: qwen3_vl_3b_instruct)
- BATCH_SIZE, MAX_EPOCHS, LOG_EVERY
- ACTION_HEAD (mse or flow), FLOW_STEPS_EVAL
- COC_JSONL (required for E4)
