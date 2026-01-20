# Wrist Camera World Modality Experiment Plan

**Status:** PLANNED (not yet executing)
**Trigger:** Run this if E2-fixed still underperforms on pickup tasks (0-1) after fair eval + ablations

## Hypothesis

The current world modality uses only the front camera (`observation.images.image`) for V-JEPA encoding. Pickup tasks (Tasks 0-1) may benefit more from wrist camera dynamics (`observation.images.image2`) because:
- Wrist camera captures gripper approach/contact timing
- Front camera may miss fine-grained gripper-object interactions
- Prophet predicting wrist-view futures could provide more actionable foresight for grasp

## Current Setup

```python
# world_modality/train_llm_vla.py
p.add_argument("--image_key", type=str, default="observation.images.image")      # Front camera (used)
p.add_argument("--wrist_image_key", type=str, default="observation.images.image2")  # Wrist camera (unused for world)
```

## Experiment Options

### Option A: Recompute latents from wrist camera

**Pros:** Clean experiment, no architecture changes
**Cons:** Requires full re-training of E2

Steps:
1. Modify `scripts/precompute_world_latents.py` to use `observation.images.image2`
2. Recompute world latents dataset
3. Retrain E2 with wrist-camera world latents
4. Eval on libero_spatial

```bash
# Hypothetical command
python scripts/precompute_world_latents.py \
  --dataset HuggingFaceVLA/libero \
  --image_key observation.images.image2 \  # CHANGED: wrist camera
  --output outputs/world_latents_wrist.parquet
```

### Option B: Online wrist encoding at rollout (no retrain)

**Pros:** Quick test, no retraining
**Cons:** Distribution shift (trained on front, evaluated on wrist)

Steps:
1. Add `--policy.world_image_key_rollout=observation.images.image2` flag
2. Modify rollout code to use wrist camera for V-JEPA encoding
3. Eval E2 checkpoint with wrist camera world inputs

**Warning:** This will likely fail due to distribution mismatch (same issue as m=1 vs m=4), but could provide a sanity check.

### Option C: Multi-camera world fusion

**Pros:** Uses all available information
**Cons:** Architecture changes, longer training

Steps:
1. Encode both front and wrist cameras with V-JEPA
2. Concatenate or fuse the latent streams
3. Prophet predicts fused future
4. Retrain and eval

## Decision Tree

```
                    E2-fixed eval complete
                           │
                           ▼
              ┌─────────────────────────────┐
              │ E2-fixed vs E0 on Tasks 0-1 │
              └─────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
    E2 ≥ E0 on pickup              E2 < E0 on pickup
    (world helps or neutral)       (world hurts pickup)
           │                               │
           ▼                               ▼
    ✓ Success!                     Run wrist experiment
    Consider multi-camera          (Option A preferred)
    for further gains
```

## Files to Modify

| File | Change |
|------|--------|
| `scripts/precompute_world_latents.py` | Add `--image_key` argument |
| `world_modality/train_llm_vla.py` | Use `--image_key` for world latents |
| `lerobot_policy_world_modality/modeling_smolvla_world.py` | Add `world_image_key_rollout` config |

## Estimated Time

- Option A: ~8 hours (2h latent compute + 6h training)
- Option B: ~4 hours (just eval)
- Option C: ~12 hours (architecture + training)

## Notes

- V-JEPA temporal window (m=4) should remain the same
- Keep all other hyperparameters identical for fair comparison
- Log per-K attention to see if wrist-based world changes attention patterns
