# Onboarding: World Modality VLA + CoC Line

This repo tests the research bet:

> **Make the world model a first-class modality inside a VLA**, via a discrete world-token stream that the transformer both *consumes* and *predicts*.

It includes two parallel lines:

1) `world_modality/`: A/B/C (+ ablation) world-token VLA experiments.
2) `coc_vla/`: Chain-of-Causality (CoC) label generation (Qwen3-VL) and a two-head model scaffold.

If you only have time for one thing: generate high-quality CoC labels with `coc_vla/coc_generation.py` and make them crash-proof.

---

## 0) Clone + install

```bash
git clone https://github.com/charbelsan/world-modality-vla.git
cd world-modality-vla
pip install -r requirements.txt
```

On AMD MI300X (ROCm) you may need a ROCm PyTorch build; follow your cluster/VM instructions.

---

## 1) World-modality experiments (A/B/C/C_no_world_input)

### Models

- **A**: standard BC (actions only).
- **B**: BC + future world token prediction loss (auxiliary only; no world token input).
- **C**: BC + world token input modality + future token loss (our method).
- **C_no_world_input**: ablation (same as C but no `w_t` input; still has world loss).

Key comparisons:

- A vs B: does future prediction help at all?
- B vs C: does treating world as an input modality help beyond aux loss?
- C vs C_no_world_input: causal ablation for “world modality matters”.

### 1.1 Pick a first dataset

Start with:

- `HuggingFaceVLA/libero`

Typical keys:

- `--image_key observation.images.image`
- `--proprio_key observation.state`
- `--action_key action`

### 1.2 Precompute world tokens (one-time)

```bash
python -m world_modality.precompute_world_tokens \
  --dataset_name HuggingFaceVLA/libero \
  --split train \
  --image_key observation.images.image \
  --vision_model_name facebook/dinov2-base \
  --vq_num_tokens 1024 \
  --l2_normalize \
  --cache_dir cache
```

This writes:

- `cache/<dataset>/train_embeddings.fp16.npy`
- `cache/<dataset>/train_world_tokens.int.npy`
- `cache/<dataset>/train_codebook_centroids.f32.npy`

### 1.2b (Optional) Precompute instruction embeddings (VLA)

If you want a **true VLA** (instruction-conditioned policy), you must provide `instruction_embeddings` per frame.

Two common cases:

**Case A: dataset has an instruction string field**

```bash
python -m world_modality.precompute_instruction_embeddings \
  --dataset_name <DATASET> \
  --cache_dir cache \
  --instruction_key instruction \
  --episode_id_key episode_id \
  --text_model_name distilbert-base-uncased \
  --max_length 64
```

**Case B: dataset only has `task_index`, but ships `meta/tasks.jsonl` (e.g. MetaWorld)**

Example for `HuggingFaceVLA/metaworld_mt50`:

```bash
python -m world_modality.precompute_instruction_embeddings \
  --dataset_name HuggingFaceVLA/metaworld_mt50 \
  --cache_dir cache \
  --episode_id_key episode_index \
  --tasks_jsonl https://huggingface.co/datasets/HuggingFaceVLA/metaworld_mt50/raw/main/meta/tasks.jsonl \
  --task_index_key task_index \
  --task_text_field task \
  --text_model_name distilbert-base-uncased \
  --max_length 64
```

Important: rollouts must use the **same** `text_model_name` / `max_length` used here (the training script saves them in the checkpoint meta).

**LIBERO note:** `HuggingFaceVLA/libero` provides `task_index` but does not ship `meta/tasks.jsonl`. You can export a compatible mapping from the LIBERO benchmark via `benchmarks/libero/export_libero_tasks_jsonl.py` (requires LIBERO installed), then pass that JSONL to the command above.

### 1.3 Train A/B/C/C_no_world_input

All training runs write `config.json` into the `--log_dir`.

```bash
python train_baseline_a.py --dataset_name HuggingFaceVLA/libero --cache_dir cache --log_dir logs_a ...
python train_baseline_b.py --dataset_name HuggingFaceVLA/libero --cache_dir cache --log_dir logs_b ...
python train_model_c.py    --dataset_name HuggingFaceVLA/libero --cache_dir cache --log_dir logs_c ...
python train_model_c.py    --model_type C_no_world_input --dataset_name HuggingFaceVLA/libero --cache_dir cache --log_dir logs_c_no_input ...
```

### 1.3b If Model C hurts action quality

In some settings we observe:

- **B / C_no_world_input** improves action MSE (aux future loss helps)
- **C** improves world-token accuracy but can **hurt** action MSE

This usually means the discrete `w_t` input is acting as a shortcut for world prediction and/or injecting quantization noise into the action path.

Try these knobs (Model C only):

```bash
--world_input_scale 0.1
--world_input_dropout 0.5
--world_input_layernorm
```

Diagnostic ablation:

```bash
--block_world_to_action
```

If `--block_world_to_action` recovers B-like action MSE, you’ve confirmed the degradation comes from ACT queries attending to WORLD_CUR.

### 1.4 Offline eval + corruption test

```bash
python -m world_modality.eval_offline --checkpoint logs_c/model_C_epoch9.pt --dataset_name HuggingFaceVLA/libero --cache_dir cache

python -m world_modality.intervention_corrupt_world --checkpoint logs_c/model_C_epoch9.pt --dataset_name HuggingFaceVLA/libero --cache_dir cache
```

`eval_offline` prints action MSE and world-token Top‑1/Top‑5 accuracy (per horizon).
`intervention_corrupt_world` prints clean vs corrupted MSE and a ratio (corrupt/clean).

### 1.5 If world accuracy is ~0: analyze token predictability

Run:

```bash
python -m world_modality.analyze_world_tokens \
  --dataset_name HuggingFaceVLA/libero \
  --cache_dir cache \
  --split train \
  --max_k 8
```

If `P(w[t]==w[t+1])` and the bigram baseline are near random, your tokenization is too noisy:

- try `--l2_normalize` (enabled above),
- reduce `--vq_num_tokens` (e.g. 256),
- and start with `--future_offset 1` during training (only next-step prediction).

To visually sanity-check whether tokens are semantically coherent, generate token grids:

```bash
python -m world_modality.inspect_token_images \
  --dataset_name HuggingFaceVLA/libero \
  --cache_dir cache \
  --split train \
  --image_key observation.images.image \
  --out_dir token_inspection \
  --num_tokens 10 \
  --samples_per_token 12
```

Open the saved `token_inspection/token_*.jpg` files and verify that frames within a token look related.

---

## 2) Crash-proofing: do not lose checkpoints

VM restarts are common; do not rely on local disk.

See `ops/README.md` for `rclone` Google Drive continuous sync.

---

## 3) CoC label generation (Qwen3‑VL)

See `coc_vla/README.md`.

Minimal smoke test:

```bash
pip install -U "transformers>=4.56" accelerate
mkdir -p coc_outputs
python -m coc_vla.coc_generation \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --instruction_key instruction \
  --output_jsonl coc_outputs/libero_train_coc.jsonl \
  --backend qwen3-vl \
  --model_name Qwen/Qwen3-VL-8B-Instruct \
  --max_episodes 5
```

Use `--resume` and sync the JSONL if the machine is unstable.

---

## 4) Closed-loop benchmark rollouts (success rate)

For publication-grade claims you usually want at least one **closed-loop** success-rate metric.

This repo includes:

- `benchmarks/libero/` (LIBERO tasks; robosuite + MuJoCo)
- `benchmarks/metaworld/` (MetaWorld MT50; MuJoCo)

These deps are heavier than the base training stack; install them in a dedicated env.

Examples:

```bash
python -m benchmarks.libero.eval_libero_success --help
python -m benchmarks.metaworld.eval_metaworld_success --help
```
