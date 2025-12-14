# MetaWorld Rollout Evaluation (Success Rate)

This integration evaluates a trained checkpoint **in closed-loop** on MetaWorld tasks and reports **success rate**.

It uses the open-source `metaworld` benchmark (MuJoCo).

## 1) Install MetaWorld + MuJoCo

MetaWorld is not in the base `requirements.txt` because it pulls heavy simulation deps.

Create a separate env for benchmark rollouts, then:

```bash
pip install metaworld mujoco
```

If you hit rendering issues (headless servers), follow MuJoCo/EGL setup for your machine.

## 2) Train on the public LeRobot MetaWorld dataset (optional, recommended)

Dataset: `HuggingFaceVLA/metaworld_mt50`

Suggested keys:

- `--image_key observation.image`
- `--proprio_key observation.state`
- `--action_key action`
- `--episode_id_key episode_index`

Precompute world tokens:

```bash
python -m world_modality.precompute_world_tokens \
  --dataset_name HuggingFaceVLA/metaworld_mt50 \
  --split train \
  --image_key observation.image \
  --vision_model_name facebook/dinov2-base \
  --vq_num_tokens 1024 \
  --l2_normalize \
  --cache_dir cache
```

To condition on language, MetaWorld includes task descriptions in `meta/tasks.jsonl`. Precompute instruction embeddings from that mapping:

```bash
python -m world_modality.precompute_instruction_embeddings \
  --dataset_name HuggingFaceVLA/metaworld_mt50 \
  --cache_dir cache \
  --episode_id_key episode_index \
  --tasks_jsonl https://huggingface.co/datasets/HuggingFaceVLA/metaworld_mt50/raw/main/meta/tasks.jsonl \
  --task_index_key task_index \
  --task_text_field task
```

Then train with `--use_language`:

```bash
python train_model_c.py \
  --dataset_name HuggingFaceVLA/metaworld_mt50 \
  --image_key observation.image \
  --proprio_key observation.state \
  --action_key action \
  --episode_id_key episode_index \
  --cache_dir cache \
  --use_language \
  --learning_rate 3e-4 \
  --future_offset 1 \
  --log_dir logs_mw_c_lang
```

## 3) Run rollouts (success rate)

From this repo root:

```bash
python -m benchmarks.metaworld.eval_metaworld_success \
  --checkpoint logs_mw_c_lang/model_C_epoch9.pt \
  --codebook_centroids cache/HuggingFaceVLA/metaworld_mt50/train_codebook_centroids.f32.npy \
  --tasks_jsonl https://huggingface.co/datasets/HuggingFaceVLA/metaworld_mt50/raw/main/meta/tasks.jsonl \
  --n_trials 10 \
  --max_steps 200 \
  --device cuda
```

### Notes

- If your training used a different state key/dimension, pass `--state_dim_override` or retrain consistently.
- The default mapping from `task_index -> env_name` is a best-effort heuristic. For exact mapping, pass `--task_map_json`.

