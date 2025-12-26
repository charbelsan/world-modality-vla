## L40S Runbook (Model F+)

This is the zero-ambiguity launch guide for the L40S box.

### 0) Preflight
```
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

### 1) Update repo
```
git pull
```

### 2) Environment setup (pick one)

Option A: conda
```
conda env create -f environment.yml
conda activate world-vla
pip install -e ".[lerobot]"
```

Option B: existing env
```
pip install -e ".[lerobot]"
```

If Qwen3-VL fails to load, upgrade transformers:
```
pip install -U "transformers>=4.56" "accelerate" "peft"
```

### 3) HF login (needed to pull Qwen3-VL weights)
```
huggingface-cli login
```

Optional cache paths (recommended):
```
export HF_HOME=/mnt/fast/hf_cache
export HF_DATASETS_CACHE=/mnt/fast/hf_cache/datasets
export TRANSFORMERS_CACHE=/mnt/fast/hf_cache/models
```

### 4) CoC labels (Phase 2 only)
Set this if you want E4:
```
export COC_JSONL=/path/to/coc_labels.jsonl
```
Notes:
- JSONL must contain objects with `episode_id` and `coc_text`.
- By default, episodes without CoC just get `L_text=0` (skipped). If you want to
  drop such episodes entirely, pass `--require_coc`.
- CoC loss is computed in the same forward pass by default (`--text_loss_mode joint_after_act`).

### 5) Launch default experiments (E0, E2, E4)
```
scripts/run_fplus_experiments.sh
```

Environment overrides (optional):
```
export BACKBONE=qwen3_vl_3b_instruct
export WORLD_SOURCE=vjepa   # or dino
export BATCH_SIZE=8
export MAX_EPOCHS=5
export LOG_EVERY=50
```

Action head (optional):
```
# Use rectified flow action head instead of MSE regression.
export ACTION_HEAD=flow
export FLOW_STEPS_EVAL=8
```

### 6) Outputs
- Latents: `cache/<dataset>/train_world_latents_<source>.fp16.npy`
- Checkpoints: `logs_llm/E*/llm_vla_epoch*.pt`

### 7) Common blockers
- Qwen3-VL load fails: upgrade transformers or set `BACKBONE=qwen2_5_vl_3b_instruct`.
- CoC JSONL missing: E4 is skipped (E0/E2 still run).
- V-JEPA slow: use `WORLD_SOURCE=dino` to validate plumbing first.
