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

### Notes

- `--benchmark` can be `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.
- `--camera_key` and `--state_key` must match keys returned by LIBERO env observations.
- If LIBERO prompts you about config paths, set `--libero_config_path` or run once interactively to create `~/.libero/config.yaml`.

