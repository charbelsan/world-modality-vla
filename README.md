# World Modality Experiments on LeRobot SR100

This repo implements the "world tokens as a new modality" experiments on the LeRobot SR100 platform.

Core idea: learn a VLA policy that, in addition to predicting action chunks, also predicts discrete **world tokens** representing future visual states, and (for model C) consumes current world tokens as a first-class input modality.

See `REPORT.md` for the hackathon report template and `world_modality/` for code.

## 1. Setup

Create a fresh environment on the MI300X VM and install deps:

```bash
pip install -r requirements.txt
```

Verify ROCm / GPU:

```bash
rocminfo | head
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 2. Recommended first dataset (LIBERO via LeRobot)

The code is designed to work with any `LeRobotDataset`, but a strong first choice is the preprocessed LIBERO dataset:

- Dataset: `HuggingFaceVLA/libero`
- Suggested keys:
  - `--image_key observation.images.image`
  - `--proprio_key observation.state`
  - `--action_key action`

You can inspect a sample locally to confirm:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("HuggingFaceVLA/libero", split="train")
print(ds[0].keys())
```

## 3. Precompute world tokens (embeddings + VQ)

This step builds the vision embedding cache and k-means VQ codebook, then quantizes all frames into discrete world tokens for a given dataset split.

```bash
python -m world_modality.precompute_world_tokens \
  --dataset_name HuggingFaceVLA/libero \
  --split train \
  --image_key observation.images.image \
  --vision_model_name facebook/dinov2-base \
  --vq_num_tokens 1024 \
  --cache_dir cache
```

Run again with `--split val` if you have a validation split.

Artifacts (per split) under `cache/<dataset_name>/`:

- `*_embeddings.fp16.npy` – per-frame pooled vision embeddings `e_t`
- `*_world_tokens.int.npy` – per-frame discrete world tokens `w_t`
- `*_codebook_centroids.f32.npy` – VQ centroids for inference-time tokenization

## 4. Train baselines A/B and world-modality model C

All three models share the same transformer backbone; they differ only in sequence layout and whether world tokens are an auxiliary target (B) or an input modality (C).

Baseline A – standard BC (no world tokens):

```bash
python train_baseline_a.py \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --proprio_key observation.state \
  --action_key action \
  --cache_dir cache \
  --batch_size 256 \
  --context_frames 3 \
  --action_horizon 8 \
  --future_offset 8
```

Baseline B – BC + auxiliary future world-token prediction:

```bash
python train_baseline_b.py \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --proprio_key observation.state \
  --action_key action \
  --cache_dir cache \
  --batch_size 256 \
  --context_frames 3 \
  --action_horizon 8 \
  --future_offset 8 \
  --world_vocab_size 1024 \
  --lambda_world_loss 0.2
```

Model C – world-modality (world token as first-class input + future prediction):

```bash
python train_model_c.py \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --proprio_key observation.state \
  --action_key action \
  --cache_dir cache \
  --batch_size 256 \
  --context_frames 3 \
  --action_horizon 8 \
  --future_offset 8 \
  --world_vocab_size 1024 \
  --lambda_world_loss 0.2
```

During training the script logs:

- training/validation action MSE
- world-token prediction loss and accuracy (for B/C)

Checkpoints are written to `logs/` as `model_<A|B|C>_epoch*.pt` and include minimal metadata for inference.

## 5. Offline evaluation

For quick offline comparison you can reuse the validation loop by running a short training run (few epochs) and inspecting:

- `[VAL] Epoch ... | action MSE ... | world acc ...`

For more systematic analysis, use:

```bash
python -m world_modality.eval_offline \
  --checkpoint logs/model_C_epoch10.pt \
  --dataset_name HuggingFaceVLA/libero \
  --split val \
  --image_key observation.images.image \
  --proprio_key observation.state \
  --action_key action \
  --cache_dir cache
```

This reports:

- action MSE on the chosen split,
- world-token accuracy per prediction horizon k (for models B/C).

## 6. Real robot inference on SR100

`world_modality/inference_sr100.py` provides a minimal real-time loop skeleton. It expects:

- a trained checkpoint (`model_C_*.pt` recommended)
- the VQ centroids file for the same dataset/split

Example:

```bash
python -m world_modality.inference_sr100 \
  --checkpoint logs/model_C_epoch10.pt \
  --codebook_centroids cache/HuggingFaceVLA/libero/train_codebook_centroids.f32.npy \
  --vision_model_name facebook/dinov2-base \
  --device cuda \
  --hz 10
```

Inside the loop you must integrate SR100-specific APIs:

- grab camera frame → numpy / tensor
- read proprio state vector
- send the first action in the predicted chunk to the SR100 controller

The script already:

- encodes the current frame to a pooled embedding `e_t`
- maps `e_t` to a world token `w_t` via the VQ codebook
- runs the transformer to get actions and predicted future world token

## 7. World-modality intervention test (C vs corrupted world)

To verify that model C actually *uses* the world token modality, you can corrupt the WORLD_CUR input at inference and measure how much action error increases:

```bash
python -m world_modality.intervention_corrupt_world \
  --checkpoint logs/model_C_epoch10.pt \
  --dataset_name HuggingFaceVLA/libero \
  --split val \
  --image_key observation.images.image \
  --proprio_key observation.state \
  --action_key action \
  --cache_dir cache
```

This reports action MSE in the clean vs corrupted setting and their ratio.

## 8. Hackathon report and assets

- Fill `REPORT.md` with:
  - dataset details and hyperparameters
  - training curves (export from W&B or logs)
  - offline metrics and real-robot success table
  - qualitative notes on stability / drift
- Save demo videos (camera + predicted future token NN frame + robot) alongside the report.

This is enough for a 48h end-to-end experiment to test “world tokens as a new modality” on SR100.
