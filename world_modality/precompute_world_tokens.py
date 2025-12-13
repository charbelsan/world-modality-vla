from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .config import DataConfig, VisionConfig, VQConfig
from .data_sr100 import build_cache_paths
from .vision import VisionEncoder
from .vq import VQCodebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute vision embeddings and world tokens.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_key", type=str, default="rgb", help="Key for image observation in LeRobotDataset.")
    parser.add_argument("--context_frames", type=int, default=3)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--future_offset", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--vq_num_tokens", type=int, default=1024)
    parser.add_argument("--vq_sample_frames", type=int, default=200_000)
    parser.add_argument("--vq_batch_size", type=int, default=4096)
    parser.add_argument("--vision_model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        cache_dir=args.cache_dir,
    )
    vision_cfg = VisionConfig(model_name=args.vision_model_name, device=args.device)
    vq_cfg = VQConfig(
        num_tokens=args.vq_num_tokens,
        sample_frames=args.vq_sample_frames,
        kmeans_batch_size=args.vq_batch_size,
    )

    # Underlying LeRobot dataset: iterate over all timesteps in this split.
    if torch.cuda.is_available() and "cuda" not in vision_cfg.device:
        device = "cuda"
    else:
        device = vision_cfg.device

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(data_cfg.dataset_name, split=args.split)
    cache_paths = build_cache_paths(data_cfg, args.split)

    encoder = VisionEncoder(vision_cfg.model_name, device=device)
    encoder.eval()

    # Collect embeddings for all timesteps in this split.
    all_embs: List[np.ndarray] = []
    for i in tqdm(range(len(ds)), desc="Encoding frames"):
        step = ds[i]
        img = step[data_cfg.image_key]  # expected [C, H, W] tensor/array
        img_t = torch.as_tensor(img)
        if img_t.dim() == 3:
            img_t = img_t.unsqueeze(0)  # [1, C, H, W]
        with torch.no_grad():
            emb = encoder.encode(img_t)[0].cpu().numpy().astype(np.float16)
        all_embs.append(emb)

    all_embs_np = np.stack(all_embs)  # [T, d_e]
    np.save(cache_paths.embeddings_path, all_embs_np)

    # Build VQ codebook using a random subset if needed.
    num_frames = all_embs_np.shape[0]
    if num_frames > vq_cfg.sample_frames:
        idx = np.random.choice(num_frames, vq_cfg.sample_frames, replace=False)
        sample_embs = all_embs_np[idx].astype(np.float32)
    else:
        sample_embs = all_embs_np.astype(np.float32)

    codebook = VQCodebook.from_embeddings(
        sample_embs, num_tokens=vq_cfg.num_tokens, batch_size=vq_cfg.kmeans_batch_size
    )

    # Quantize all embeddings
    tokens = codebook.encode(all_embs_np.astype(np.float32))
    np.save(cache_paths.tokens_path, tokens.astype(np.int32))
    # Save centroids for inference-time nearest-neighbor lookup.
    np.save(cache_paths.centroids_path, codebook.centroids.astype(np.float32))

    print(f"Saved embeddings to {cache_paths.embeddings_path}")
    print(f"Saved world tokens to {cache_paths.tokens_path}")
    print(f"Saved VQ centroids to {cache_paths.centroids_path}")


if __name__ == "__main__":
    main()
