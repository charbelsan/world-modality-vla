# Claude Context for World Modality VLA

**Last updated:** 2026-01-16
**Branch:** `phaseC-flow-head`

---

## Project Goal

Test whether **world modality** (predicted future V-JEPA representations) improves VLA robot control on LIBERO benchmark.

**Core hypothesis:** Injecting predicted future world states as external memory helps the policy make better decisions.

---

## Experiment Matrix

| Experiment | Description | Purpose |
|------------|-------------|---------|
| **E0** | SmolVLA baseline (no world modality) | Baseline performance |
| **E1** | SmolVLA + world arch, **zero memory** | Capacity control (extra params shouldn't help) |
| **E2** | SmolVLA + world arch, **predicted memory** | **Main hypothesis test** |

**Critical:** E0 ≈ E1 means capacity doesn't confound. E2 > E0 means world modality helps.

---

## Current Results (as of 2026-01-16)

### E0 Baseline @50 eps/task (COMPLETE)
| Task | SR% |
|------|-----|
| 0 | 76.0 |
| 1 | 96.0 |
| 2 | 84.0 |
| 3 | 94.0 |
| 4 | 82.0 |
| 5 | 70.0 |
| 6 | 84.0 |
| 7 | 92.0 |
| 8 | 72.0 |
| 9 | 86.0 |
| **Overall** | **83.6%** |

### E1 Zero-Memory @50 eps/task (PARTIAL - 6/10 tasks)
| Task | SR% |
|------|-----|
| 0 | 80.0 |
| 1 | 94.0 |
| 2 | 86.0 |
| 3 | 94.0 |
| 4 | 94.0 |
| 5 | 66.0 |
| **Partial** | **85.7%** |

### E2 Pred-Memory (PARTIAL - 2/10 tasks, from earlier run @200 eps/task)
| Task | SR% |
|------|-----|
| 0 | 78.5 |
| 1 | 89.5 |

---

## YOUR FOCUS: EVALUATION ONLY

**Do NOT run training.** Checkpoints already exist. Your job is to run evals and collect results.

The team will handle training iterations based on your eval results.

---

## What Still Needs to Be Done

### Priority 1: E2 Full Evaluation (MOST IMPORTANT)
Run E2 (predicted memory) on all 10 tasks with fair episode count.

```bash
MUJOCO_GL=osmesa lerobot-wm-eval \
  --policy.path=<path_to_E2_checkpoint>/pretrained_model \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=200 \
  --eval.batch_size=10
```

### Priority 2: Causal Ablations (same E2 checkpoint)
These prove whether **content** of predicted memory matters:

```bash
# Zero memory ablation (inject zeros instead of predictions)
lerobot-wm-eval ... --policy.world_memory_mode_rollout=zero

# Random memory ablation (inject random vectors)
lerobot-wm-eval ... --policy.world_memory_mode_rollout=random
```

**Expected results:**
- If `pred > zero`: predicted content helps
- If `pred ≈ zero > random`: any structured input helps (not specifically predictions)
- If `pred ≈ zero ≈ random`: world modality doesn't matter

### Priority 3: Complete E1 (4 remaining tasks)

---

## Critical Technical Details

### Episode Counts (IMPORTANT!)
- **For fair comparison:** All evals must use the **same episode count**
- **Required:** `--eval.n_episodes=200` (200 eps per task × 10 tasks = 2000 total)
- **This gives tight confidence intervals (~±4%) for reliable conclusions**
- **Previous issue:** E0/E1 were run @20 eps/task, E2 @200 eps/task → unfair comparison
- **Do NOT use fewer episodes** - we need statistical power to detect differences

### Temporal Encoding (CRITICAL!)
- Training uses **m=4** temporal V-JEPA embeddings
- Rollout **MUST** also use m=4: `--policy.world_rollout_temporal_window=4`
- **Bug we found:** Using m=1 (single-frame) at rollout caused 20-40% performance drop
- This is now fixed in the code, but be aware of it

### World Latents
- **Location:** `cache/HuggingFaceVLA/libero/train_world_latents_vjepa_m4.fp16.npy`
- **Size:** 735MB, shape [273465, 1408]
- **Download from HF:** `Adjimavo/libero_world_latents_vjepa_m4`
- Only needed for **training**, not eval (eval uses online encoder)

### Checkpoints
Checkpoints should be at:
```
outputs/train/libero_smolvla_world_matrix/
├── E0_smolvla_baseline_seed0/checkpoints/050000/pretrained_model/
├── E1_world_zero_seed0/checkpoints/050000/pretrained_model/
└── E2_world_pred_seed0/checkpoints/050000/pretrained_model/
```

---

## Key Findings So Far

1. **Temporal mismatch was a major bug:** Training with m=4, evaluating with m=1 caused severe degradation (Task 1: 58% vs 89.5% fixed)

2. **Prophet learns meaningful predictions:** Delta prediction cosine similarity = 0.68 overall (K=1: 0.44, K=8: 0.79)

3. **Attention is focused:** Entropy ratio 0.86 (not uniform), model selectively attends to certain future steps

4. **Gate opens slightly:** 1.53% gate value shows model learned to use world memory (started at 0)

5. **E2 vs E0 unclear:** Early results show E2 ≈ E0 on Task 0, slightly worse on Task 1. Need full eval to conclude.

---

## Potential Next Experiments (if E2 doesn't help)

1. **Wrist camera latents:** Current world modality uses front camera only. Pickup tasks might benefit from wrist camera dynamics. See `docs/WRIST_CAMERA_EXPERIMENT_PLAN.md`

2. **Improve near-future prediction:** K=1 accuracy is only 0.44 (far future K=8 is 0.79). Pickup tasks need precise immediate-future predictions.

3. **Increase gate scale:** Current gate is 1.53%, might be too conservative.

---

## Important Files

| File | Purpose |
|------|---------|
| `README.md` | Setup instructions, experiment matrix |
| `RESEARCH_ANALYSIS.md` | Detailed research log and conclusions |
| `docs/MI300X_LIBERO_SMOLVLA_WORLD.md` | VM setup runbook |
| `docs/WRIST_CAMERA_EXPERIMENT_PLAN.md` | Future experiment plan |
| `docs/VM_CONTINUATION_GUIDE.md` | How to continue on new VM |
| `scripts/run_mi300x_smolvla_world_matrix.sh` | Main training script |
| `world_modality/model.py` | Prophet + GatedCrossAttention |
| `lerobot_policy_world_modality/modeling_smolvla_world.py` | SmolVLA-World policy |

---

## Quick Commands

### Download world latents
```python
from huggingface_hub import hf_hub_download
import os
os.makedirs("cache/HuggingFaceVLA/libero", exist_ok=True)
hf_hub_download(
    repo_id="Adjimavo/libero_world_latents_vjepa_m4",
    repo_type="dataset",
    filename="train_world_latents_vjepa_m4.fp16.npy",
    local_dir="cache/HuggingFaceVLA/libero"
)
```

### Run E2 eval
```bash
MUJOCO_GL=osmesa lerobot-wm-eval \
  --policy.path=outputs/train/libero_smolvla_world_matrix/E2_world_pred_seed0/checkpoints/050000/pretrained_model \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=200 \
  --eval.batch_size=10
```

### Run ablations
```bash
# Zero memory
MUJOCO_GL=osmesa lerobot-wm-eval \
  --policy.path=outputs/train/libero_smolvla_world_matrix/E2_world_pred_seed0/checkpoints/050000/pretrained_model \
  --policy.device=cuda \
  --policy.world_memory_mode_rollout=zero \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=200 \
  --eval.batch_size=10

# Random memory
MUJOCO_GL=osmesa lerobot-wm-eval \
  --policy.path=outputs/train/libero_smolvla_world_matrix/E2_world_pred_seed0/checkpoints/050000/pretrained_model \
  --policy.device=cuda \
  --policy.world_memory_mode_rollout=random \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=200 \
  --eval.batch_size=10
```

---

## Summary for Next Claude

**Your mission:** Run EVALS ONLY (no training) to answer:
1. **Does E2 (pred memory) beat E0 (baseline)?**
2. **Do ablations show causality?** (pred > zero, zero > random)
3. **Which tasks benefit/hurt from world modality?**

### CRITICAL REQUIREMENTS:
- **Use `--eval.n_episodes=200` for ALL evals** (200 per task, 2000 total)
- **EVAL ONLY** - do not run training, checkpoints already exist
- **Run E0, E1, E2 all at 200 eps** for fair comparison
- **Then run E2 ablations** (zero, random) at 200 eps each

### Eval queue (run sequentially):
1. E0 @200 eps → baseline
2. E1 @200 eps → capacity control
3. E2 @200 eps → **main hypothesis**
4. E2-zero @200 eps → ablation
5. E2-random @200 eps → ablation

Each eval takes ~3-4 hours. Total: ~17-18 hours. Queue overnight if needed.

### After all evals complete:
Report per-task success rates for E0, E1, E2, E2-zero, E2-random in a comparison table.
