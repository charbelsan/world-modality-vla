# World Modality VLA

Treating world state as a first-class token modality for Vision-Language-Action policies.

## Key Insight

Standard VLAs map observations directly to actions. We add an explicit **world modality** that represents scene state as discrete or continuous tokens. This enables:
- **Multi-step future prediction**: Model learns what the world *will* look like
- **Causal reasoning**: Model understands *why* actions lead to outcomes
- **Better generalization**: World representations transfer across tasks

## Model Architectures

| Model | Input | Auxiliary Task | Key Idea |
|-------|-------|----------------|----------|
| **A** | Obs, State | None | Baseline BC |
| **B** | Obs, State | Predict future world tokens | Multi-task learning |
| **B_cont** | Obs, State | Predict future embeddings | No VQ quantization noise |
| **C** | Obs, State, World Token | Predict future world tokens | World as input modality |
| **F** | Obs, State + Prophet memory | Predict future embeddings | External memory via cross-attention |

All models share the same transformer backbone. They differ in:
- **What goes into the sequence** (C adds world token, F adds predicted future via cross-attention)
- **What auxiliary task is trained** (discrete tokens vs continuous embeddings)

### Model F Architecture (Best Performing)

```
Prophet Module:
  History [B, T_ctx, D] → Future Predictions [B, K_fut, D]

Policy Transformer:
  Obs + State → Action Queries → GatedCrossAttention(Query, Prophet Output) → Actions
```

The gate is initialized to 0 ("do-no-harm"), allowing the model to gradually learn to use future predictions.

## Vision Encoder

We use a **frozen vision encoder** to create frame embeddings:
- **DINOv2** (default): Widely available, strong semantic features
- **V-JEPA-v2** (upgrade): Better dynamics priors from video pretraining

The encoder is pluggable via `--vision_model_name`. Both produce embeddings that get quantized into discrete world tokens or used directly as continuous targets.

## Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **LIBERO** | Language-conditioned manipulation | [HuggingFaceVLA/libero](https://huggingface.co/datasets/HuggingFaceVLA/libero) |
| **MetaWorld MT50** | Multi-task manipulation benchmark | [HuggingFaceVLA/metaworld_mt50](https://huggingface.co/datasets/HuggingFaceVLA/metaworld_mt50) |

## Chain-of-Causality (CoC)

CoC provides textual explanations of robot behavior:
- **What**: "Robot moves gripper toward red block, then closes gripper to grasp"
- **Why**: Generated offline using VLMs (Qwen3-VL) on episode keyframes
- **Purpose**: Interpretability, debugging, and potential cross-embodiment transfer

## Quick Start

```bash
# 1. Install
pip install -e ".[lerobot]"

# 2. Precompute embeddings and world tokens
python -m world_modality.precompute_world_tokens \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --cache_dir cache

# 3. Train any model
python -m world_modality.train --model_type A --max_epochs 5
```

## Qwen VLM + World Modality (Model F+)

This is the "generalist VLM + action head + Model-F world memory" pipeline.
Model F+ adds CoC text loss (optional) and a FLARE-style future latent alignment loss.
See `docs/LLM_VLA_FPLUS.md` for the full experiment matrix and launch script.
See `docs/L40S_RUNBOOK.md` for a zero-ambiguity L40S setup.

Quick run on L40S:
```bash
export COC_JSONL=/path/to/coc_labels.jsonl
scripts/run_fplus_experiments.sh
```

Precompute continuous world latents (DINO or V-JEPA):

```bash
python -m world_modality.precompute_world_latents \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --world_latents_source dino \
  --cache_dir cache
```

E0 (VLM-BC baseline):

```bash
python -m world_modality.train_llm_vla \
  --vlm_backbone qwen3_vl_3b_instruct \
  --trust_remote_code \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --instruction_key task \
  --world_latents_source dino \
  --future_memory_source predicted \
  --disable_future_injection \
  --lambda_world 0.0
```

E2 (Model-F world memory injection):

```bash
python -m world_modality.train_llm_vla \
  --vlm_backbone qwen3_vl_3b_instruct \
  --trust_remote_code \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --instruction_key task \
  --world_latents_source dino \
  --future_memory_source scheduled \
  --lambda_world 0.2
```

Notes:
- Use `--freeze_backbone --use_lora` for LoRA tuning.
- Switch to `--world_latents_source vjepa` once V-JEPA latents are available.
- Action decoding uses `<ACT_i>` tokens and reads hidden states (no text decoding).

## SmolVLA + World Modality (LeRobot plugin)

This repo also ships a LeRobot policy plugin (`lerobot_policy_world_modality`) that registers:
- `--policy.type=smolvla_world`

Runbook (MI300X-friendly): `docs/MI300X_LIBERO_SMOLVLA_WORLD.md`

### Phase 2: Talk while acting (CoC loss)

This keeps control independent: actions are computed from <ACT> hidden states, and language
loss is computed in a separate forward pass.

```bash
python -m world_modality.train_llm_vla \
  --vlm_backbone qwen3_vl_3b_instruct \
  --trust_remote_code \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --instruction_key task \
  --world_latents_source dino \
  --future_memory_source scheduled \
  --lambda_world 0.2 \
  --lambda_text 0.1 \
  --coc_jsonl /path/to/coc_labels.jsonl
```

## Full Experiment Commands

### Phase 0: Sanity Check (K=1)

Quick verification that training works:

```bash
python -m world_modality.train --model_type A --future_offset 1 --max_epochs 3
python -m world_modality.train --model_type B --future_offset 1 --max_epochs 3
```

### Phase 1: Scientific Comparison (K=4)

Compare all model variants:

```bash
for model in A B B_cont C F; do
  python -m world_modality.train \
    --model_type $model \
    --future_offset 4 \
    --max_epochs 5 \
    --log_dir logs/${model}_k4
done
```

### Phase 2: Model F Ablations

Test whether Model F actually uses future predictions:

```bash
# Train Model F
python -m world_modality.train \
  --model_type F \
  --future_offset 4 \
  --max_epochs 5 \
  --log_dir logs/F_k4

# Corruption intervention tests
for mode in shuffle random zero oracle; do
  python -m world_modality.intervention_corrupt_future \
    --checkpoint logs/F_k4/model_F_best.pt \
    --corruption_mode $mode
done
```

**Interpretation:**
- `shuffle/random/zero`: ratio > 1.0 means model relies on future memory
- `oracle`: ratio < 1.0 means better predictions would improve actions

### Phase 3: Horizon Scaling (K=8)

Test with longer prediction horizon:

```bash
for model in A B_cont F; do
  python -m world_modality.train \
    --model_type $model \
    --future_offset 8 \
    --max_epochs 5 \
    --log_dir logs/${model}_k8
done
```

### CoC Generation

Generate Chain-of-Causality labels:

```bash
python -m coc_vla.coc_generation \
  --dataset_name HuggingFaceVLA/libero \
  --backend qwen3-vl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --output_jsonl coc_outputs/libero_coc.jsonl \
  --max_episodes 100
```

## Project Structure

```
world_modality/
  model.py                 # WorldPolicyTransformer, Prophet, GatedCrossAttention
  train.py                 # Unified training script
  train_utils.py           # Loss functions, schedulers
  data_sr100.py            # Data loading with caching
  precompute_world_tokens.py  # VQ tokenization
  intervention_corrupt_future.py  # Model F ablation tests

coc_vla/
  coc_generation.py        # Generate CoC with VLMs
  model.py                 # Two-head model (actions + CoC)
  train.py                 # Training with CoC loss

scripts/
  test_all_models.py       # Verify all model types work
```

## Verify Installation

```bash
# Test all models on CPU
python scripts/test_all_models.py --device cpu

# Test on GPU
python scripts/test_all_models.py --device cuda
```

## Environment Options

**uv (fastest):**
```bash
uv pip install -e ".[lerobot]"
```

**conda:**
```bash
conda env create -f environment.yml
conda activate world-modality
```

**Docker:**
```bash
docker build -t world-modality .
docker run --gpus all -v $(pwd):/workspace world-modality \
  python -m world_modality.train --model_type F
```
