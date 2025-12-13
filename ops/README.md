# Persistence / Crash-Proofing (VM Restarts)

If your VM can restart at any moment, assume **local disk is not durable**.

This repo keeps `logs*/` and `cache/` **out of git** (see `.gitignore`). Instead, use one of:

1. **Write logs/checkpoints directly to a persistent mount** (best).
2. **Continuously sync** `logs_*/` and key outputs to remote storage (Google Drive, S3, etc).
3. **Use an experiment tracker** (e.g. W&B artifacts) for checkpoints and metrics.

This folder provides simple `rclone`-based sync scripts for Google Drive.

## Option A (recommended): log_dir on a persistent mount

If you have `/mnt/persist` (NFS, attached disk, etc):

```bash
python train_model_c.py --log_dir /mnt/persist/logs_c ...
```

## Option B: `rclone` continuous sync to Google Drive

### 1) Install + configure `rclone`

See `ops/rclone_gdrive_setup.md`.

### 2) Sync a run directory (checkpoints + config.json)

In one terminal:

```bash
bash ops/sync_rclone_dir.sh logs_c gdrive:world-modality-vla/logs_c 60
```

This copies new checkpoints as they appear (and also the `config.json`).

### 3) Run training in another terminal

```bash
python train_model_c.py --log_dir logs_c ...
```

## Option C: Wrapper that runs training + sync together

```bash
bash ops/run_with_rclone_sync.sh \
  --local logs_c \
  --remote gdrive:world-modality-vla/logs_c \
  --interval 60 \
  -- python train_model_c.py --log_dir logs_c ...
```

## Sync a single file (e.g. CoC JSONL)

```bash
bash ops/sync_rclone_file.sh \
  coc_outputs/libero_train.jsonl \
  gdrive:world-modality-vla/coc_outputs/libero_train.jsonl \
  60
```

