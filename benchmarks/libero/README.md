# LIBERO Rollout Evaluation (Success Rate)

This integration evaluates a trained checkpoint **in closed-loop** on LIBERO tasks and reports **success rate**.

It uses LIBERO's official env wrappers (robosuite + MuJoCo).

## 1) Install LIBERO + dependencies

LIBERO is not in the base `requirements.txt` because it pulls heavy simulation deps.

Recommended: create a separate env for benchmark rollouts.

Example (Linux):

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
```

You'll also need MuJoCo + robosuite. Follow LIBERO's README for your platform.

## 2) Run rollouts

From this repo root:

```bash
python -m benchmarks.libero.eval_libero_success \
  --checkpoint logs_c_fixed/model_C_epoch9.pt \
  --codebook_centroids cache/HuggingFaceVLA/libero/train_codebook_centroids.f32.npy \
  --vision_model_name facebook/dinov2-base \
  --benchmark libero_10 \
  --env_num 10 \
  --n_trials 10 \
  --max_steps 300 \
  --camera_key agentview_image \
  --state_key robot0_proprio-state \
  --device cuda
```

## 3) (Optional) Train with language on `HuggingFaceVLA/libero`

The public LeRobot LIBERO dataset provides `task_index` but does not ship a `tasks.jsonl` in the repo.

To train an instruction-conditioned policy (`--use_language`), first export task language strings from the LIBERO benchmark:

```bash
python -m benchmarks.libero.export_libero_tasks_jsonl \
  --benchmark libero_10 \
  --out benchmarks/libero/tasks_libero10.jsonl
```

Then precompute per-frame instruction embeddings using that mapping:

```bash
python -m world_modality.precompute_instruction_embeddings \
  --dataset_name HuggingFaceVLA/libero \
  --cache_dir cache \
  --episode_id_key episode_index \
  --tasks_jsonl benchmarks/libero/tasks_libero10.jsonl \
  --task_index_key task_index \
  --task_text_field task \
  --text_model_name distilbert-base-uncased \
  --max_length 64
```

Finally train with `--use_language`:

```bash
python train_model_c.py \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --proprio_key observation.state \
  --action_key action \
  --episode_id_key episode_index \
  --cache_dir cache \
  --use_language \
  --future_offset 1 \
  --learning_rate 3e-4 \
  --log_dir logs_libero_c_lang
```

### Notes

- `--benchmark` can be `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.
- `--camera_key` and `--state_key` must match keys returned by LIBERO env observations.
- If LIBERO prompts you about config paths, set `--libero_config_path` or run once interactively to create `~/.libero/config.yaml`.
