from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_modality.vision import VisionEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test Cosmos world-feature extraction on LIBERO.")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceVLA/libero")
    parser.add_argument("--front-key", type=str, default="observation.images.image")
    parser.add_argument("--wrist-key", type=str, default="observation.images.image2")
    parser.add_argument("--vision-model-name", type=str, default="cosmos_cv8x8x8_pool4_m4")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--temporal-window", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="logs/cosmos_smoke")
    return parser.parse_args()


def to_chw_uint8(step_value) -> torch.Tensor:
    arr = np.asarray(step_value)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={arr.shape}")
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_batch(ds, *, image_key: str, start_index: int, batch_size: int) -> torch.Tensor:
    frames = [to_chw_uint8(ds[i][image_key]) for i in range(start_index, start_index + batch_size)]
    return torch.stack(frames, dim=0)


def build_temporal_batch(
    ds,
    *,
    image_key: str,
    start_index: int,
    batch_size: int,
    temporal_window: int,
) -> torch.Tensor:
    clips = []
    for idx in range(start_index, start_index + batch_size):
        clip = []
        for t in range(temporal_window):
            src_idx = max(0, idx - (temporal_window - 1) + t)
            clip.append(to_chw_uint8(ds[src_idx][image_key]))
        clips.append(torch.stack(clip, dim=0))
    return torch.stack(clips, dim=0)


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return out, t1 - t0


def main() -> None:
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(args.dataset_name)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    encoder = VisionEncoder(
        model_name=args.vision_model_name,
        device=args.device,
        dtype=args.dtype,
    )

    results = {}
    for name, image_key in (("front", args.front_key), ("wrist", args.wrist_key)):
        batch = build_batch(ds, image_key=image_key, start_index=args.sample_index, batch_size=args.batch_size)
        clip_batch = build_temporal_batch(
            ds,
            image_key=image_key,
            start_index=args.sample_index,
            batch_size=args.batch_size,
            temporal_window=args.temporal_window,
        )

        single_emb, single_s = timed(lambda: encoder.encode(batch))
        temporal_emb, temporal_s = timed(lambda: encoder.encode_temporal(clip_batch))

        single_np = single_emb.detach().cpu().float().numpy()
        temporal_np = temporal_emb.detach().cpu().float().numpy()
        if not np.isfinite(single_np).all() or not np.isfinite(temporal_np).all():
            raise RuntimeError(f"Non-finite embeddings detected for {name}")

        np.savez_compressed(
            save_dir / f"{name}_sample_features.npz",
            single=single_np[:1],
            temporal=temporal_np[:1],
        )

        results[name] = {
            "single_shape": list(single_np.shape),
            "temporal_shape": list(temporal_np.shape),
            "single_seconds": round(single_s, 4),
            "temporal_seconds": round(temporal_s, 4),
            "single_mean_abs": float(np.abs(single_np).mean()),
            "temporal_mean_abs": float(np.abs(temporal_np).mean()),
        }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
