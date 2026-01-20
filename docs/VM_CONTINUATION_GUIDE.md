# VM Continuation Guide

**Last updated:** 2026-01-16 09:30 UTC

## What We Have Locally

### Checkpoints (ready to use)
```
mi300x_sync/checkpoints/
├── E0_smolvla_baseline_seed0/   # Baseline SmolVLA
├── E1_world_zero_seed0/         # Zero memory control
└── E2_world_pred_seed0/         # Prophet-predicted memory (MAIN)
```

### Logs & Analysis
```
mi300x_artifacts/
├── *.log                        # All eval logs
├── eval_manifest.json           # Experiment tracking
└── analysis/                    # Prophet quality, attention patterns
```

### Code
- All code committed to `phaseC-flow-head` branch
- Per-K attention logging added
- Wrist camera plan documented

---

## Results So Far

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
| **Avg** | **83.6%** |

### E1 Zero-Mem @50 eps/task (PARTIAL - 6/10)
| Task | SR% |
|------|-----|
| 0 | 80.0 |
| 1 | 94.0 |
| 2 | 86.0 |
| 3 | 94.0 |
| 4 | 94.0 |
| 5 | 66.0 |
| **Partial** | **85.7%** |

### E2 Fixed (m=4) @200 eps/task (PARTIAL - 2/10)
| Task | SR% |
|------|-----|
| 0 | 78.5 |
| 1 | 89.5 |

---

## What's Still Needed

### Priority 1: E2 Full Eval
```bash
lerobot-wm-eval \
  --policy.path=<E2_checkpoint_path> \
  --policy.device=cuda \
  --env.type=libero \
  --env.task=libero_spatial \
  --eval.n_episodes=20 \
  --eval.batch_size=10
```

### Priority 2: Causal Ablations (same E2 checkpoint)
```bash
# Zero memory ablation
lerobot-wm-eval ... --policy.world_memory_mode_rollout=zero

# Random memory ablation
lerobot-wm-eval ... --policy.world_memory_mode_rollout=random
```

### Priority 3: Complete E1
Finish remaining 4 tasks of E1 @50 eps/task

---

## To Continue on New VM

1. **Clone repo:**
   ```bash
   git clone https://github.com/charbelsan/world-modality-vla.git
   cd world-modality-vla
   git checkout phaseC-flow-head
   ```

2. **Install:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e .
   pip install lerobot
   ```

3. **Download checkpoints from HuggingFace:**
   ```python
   from huggingface_hub import snapshot_download

   # E0: Baseline SmolVLA
   snapshot_download(repo_id="Adjimavo/smolvla_world_E0_baseline",
                     local_dir="checkpoints/E0_smolvla_baseline_seed0")

   # E1: Zero memory control
   snapshot_download(repo_id="Adjimavo/smolvla_world_E1_zero",
                     local_dir="checkpoints/E1_world_zero_seed0")

   # E2: Predicted memory (MAIN)
   snapshot_download(repo_id="Adjimavo/smolvla_world_E2_pred",
                     local_dir="checkpoints/E2_world_pred_seed0")
   ```

4. **Download world latents (only needed for training):**
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

5. **Run evals (200 eps/task for fair comparison):**
   ```bash
   # E0 baseline
   MUJOCO_GL=osmesa lerobot-wm-eval \
     --policy.path=checkpoints/E0_smolvla_baseline_seed0 \
     --policy.device=cuda \
     --env.type=libero \
     --env.task=libero_spatial \
     --eval.n_episodes=200 \
     --eval.batch_size=10

   # E1 zero-memory
   MUJOCO_GL=osmesa lerobot-wm-eval \
     --policy.path=checkpoints/E1_world_zero_seed0 \
     --policy.device=cuda \
     --env.type=libero \
     --env.task=libero_spatial \
     --eval.n_episodes=200 \
     --eval.batch_size=10

   # E2 pred-memory (MAIN)
   MUJOCO_GL=osmesa lerobot-wm-eval \
     --policy.path=checkpoints/E2_world_pred_seed0 \
     --policy.device=cuda \
     --env.type=libero \
     --env.task=libero_spatial \
     --eval.n_episodes=200 \
     --eval.batch_size=10
   ```

---

## Key Questions to Answer

1. **Is E2(pred, m=4) better than E0?** - Need E2 full eval
2. **Does pred memory content matter?** - Need zero/random ablations
3. **Which tasks hurt most?** - Compare per-task E0 vs E2

## Backup on Old VM

14GB tarball at `/root/wm_artifacts_20260115_200421.tgz` contains:
- All logs
- Training outputs
- Eval results
