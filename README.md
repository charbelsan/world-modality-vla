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
