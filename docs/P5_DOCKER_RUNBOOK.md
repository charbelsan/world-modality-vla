# P5 Docker Runbook

Use this when running `world-modality-vla` on the shared `p5.48xlarge` without touching the existing workload on GPUs `0-3`.

The current recommendation is:

- use **GPU 6** for eval / ablations
- use **GPU 7** for training or latent precompute
- keep everything under `/opt/dlami/nvme` on the P5

---

## 1) Stage the repo on the P5

On the P5 host:

```bash
cd /opt/dlami/nvme
git clone https://github.com/charbelsan/world-modality-vla.git
cd world-modality-vla
git checkout phaseC-flow-head
```

If you only need the current committed code, clone is enough.
If you also need local-only artifacts (e.g. cached latents or synced checkpoints), copy those separately after cloning.

---

## 2) Build the image

From `/opt/dlami/nvme/world-modality-vla` on the P5 host:

```bash
docker build -t world-modality-vla:smolvla .
```

The repo includes a `.dockerignore`, so local caches and old artifacts are not sent into the build context.

---

## 3) Common run flags

Use these flags for every run:

```bash
--rm \
--ipc=host \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
--shm-size=32g \
--runtime=nvidia
```

Use host-mounted directories so results survive container exit:

```bash
-v /opt/dlami/nvme/world-modality-vla:/workspace \
-v /opt/dlami/nvme/world-modality-vla/cache:/workspace/cache \
-v /opt/dlami/nvme/world-modality-vla/outputs:/workspace/outputs \
-v /opt/dlami/nvme/world-modality-vla/eval_libero_results:/workspace/eval_libero_results \
-v /opt/dlami/nvme/world-modality-vla/logs:/workspace/logs
```

---

## 4) Smoke test

Before launching long jobs:

```bash
docker run \
  --gpus '"device=6"' \
  --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=32g \
  --runtime=nvidia \
  -v /opt/dlami/nvme/world-modality-vla:/workspace \
  -w /workspace \
  world-modality-vla:smolvla \
  python -c "import torch, lerobot, mujoco; print(torch.cuda.get_device_name(0))"
```

---

## 5) Eval on GPU 6

Example: run `signflip` on the current E2 checkpoint.

```bash
docker run \
  --gpus '"device=6"' \
  --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=32g \
  --runtime=nvidia \
  -e MUJOCO_GL=egl \
  -v /opt/dlami/nvme/world-modality-vla:/workspace \
  -w /workspace \
  world-modality-vla:smolvla \
  bash -lc '
    lerobot-wm-eval \
      --policy.path mi300x_sync/checkpoints/E2_world_pred_seed0/checkpoints/050000/pretrained_model \
      --policy.device=cuda \
      --policy.type=smolvla_world \
      --policy.world_memory_mode_rollout=signflip \
      --env.type=libero \
      --env.task=libero_spatial \
      --eval.n_episodes=500 \
      --eval.batch_size=5
  '
```

Swap the rollout mode for:

- `random_scaled`
- `pred`
- `zero`

---

## 6) Precompute wrist latents on GPU 7

```bash
docker run \
  --gpus '"device=7"' \
  --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=32g \
  --runtime=nvidia \
  -v /opt/dlami/nvme/world-modality-vla:/workspace \
  -w /workspace \
  world-modality-vla:smolvla \
  bash -lc '
    python -m world_modality.precompute_world_latents \
      --dataset_name HuggingFaceVLA/libero \
      --image_key observation.images.image2 \
      --cache_dir cache \
      --world_latents_source vjepa \
      --temporal_window 4 \
      --latent_suffix m4_wrist \
      --device cuda
  '
```

---

## 7) Train on GPU 7

Example: wrist-memory E2 variant.

```bash
docker run \
  --gpus '"device=7"' \
  --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size=32g \
  --runtime=nvidia \
  -e MUJOCO_GL=egl \
  -v /opt/dlami/nvme/world-modality-vla:/workspace \
  -w /workspace \
  world-modality-vla:smolvla \
  bash -lc '
    lerobot-wm-train \
      --dataset.repo_id=HuggingFaceVLA/libero \
      --policy.type=smolvla_world \
      --policy.device=cuda \
      --policy.push_to_hub=false \
      --policy.init_from_policy_path=HuggingFaceVLA/smolvla_libero \
      --policy.dataset_repo_id=HuggingFaceVLA/libero \
      --policy.cache_dir=cache \
      --policy.world_latents_source=vjepa \
      --policy.latent_suffix=m4_wrist \
      --policy.world_camera=wrist \
      --policy.world_latent_dim=1408 \
      --policy.context_frames=4 \
      --policy.future_offset=8 \
      --policy.lambda_world=0.2 \
      --policy.world_memory_mode_train=pred \
      --policy.enable_world_injection=true \
      --batch_size=64 \
      --steps=50000 \
      --output_dir outputs/train/libero_smolvla_world_matrix/E2_world_pred_wrist_seed0 \
      --seed=0 \
      --wandb.enable=false
  '
```

For the F2 ablation, add:

```bash
--policy.world_inject_suffix_in=true
```

For F3b, add:

```bash
--policy.world_prefix_cross_attn=true
```

Run them separately first.

---

## 8) Backup

The P5 NVMe is ephemeral. Sync outputs to S3 or elsewhere during and after runs:

```bash
aws s3 sync /opt/dlami/nvme/world-modality-vla/outputs \
  s3://reflexion-robotics-research-data/world-modality-vla-p5/outputs
```

Also back up:

- `/opt/dlami/nvme/world-modality-vla/cache`
- `/opt/dlami/nvme/world-modality-vla/eval_libero_results`
- any custom logs under `/opt/dlami/nvme/world-modality-vla/logs`

If the P5 host has no AWS credentials, pull artifacts from your workstation instead:

```bash
./ops/pull_p5_artifacts.sh ./p5_artifacts --mode minimal
./ops/pull_p5_artifacts.sh ./p5_artifacts --mode full
```
