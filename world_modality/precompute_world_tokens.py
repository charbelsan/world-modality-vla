from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import DataConfig, VisionConfig, VQConfig
from .data_sr100 import build_cache_paths
from .vision import VisionEncoder
from .vq import VQCodebook


class FrameDataset(Dataset):
    """Wrapper dataset for parallel frame loading."""
    def __init__(self, lerobot_ds, image_key: str):
        self.ds = lerobot_ds
        self.image_key = image_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        step = self.ds[idx]
        img = step[self.image_key]
        return torch.as_tensor(img)


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
    parser.add_argument("--vq_random_state", type=int, default=0, help="Random seed for k-means init.")
    parser.add_argument("--vision_model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--l2_normalize", action="store_true", help="L2-normalize embeddings before k-means + assignment.")
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

    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(data_cfg.dataset_name)
    cache_paths = build_cache_paths(data_cfg, args.split)

    encoder = VisionEncoder(vision_cfg.model_name, device=device)
    encoder.eval()

    # Wrap dataset for parallel loading
    frame_ds = FrameDataset(ds, data_cfg.image_key)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": True if args.num_workers > 0 else False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(frame_ds, **loader_kwargs)

    # Collect embeddings using parallel data loading + batched GPU encoding
    all_embs: List[np.ndarray] = []

    print(f"Processing {len(ds)} frames with batch_size={args.batch_size}, num_workers={args.num_workers}")

    with torch.no_grad():
        for batch_imgs in tqdm(loader, desc="Encoding frames"):
            # batch_imgs: [B, C, H, W]
            batch_imgs = batch_imgs.to(device)
            batch_embs = encoder.encode(batch_imgs).cpu().numpy().astype(np.float16)
            all_embs.append(batch_embs)

    all_embs_np = np.concatenate(all_embs, axis=0)  # [T, d_e]
    np.save(cache_paths.embeddings_path, all_embs_np)

    if args.l2_normalize:
        norms = np.linalg.norm(all_embs_np.astype(np.float32), axis=1, keepdims=True) + 1e-8
        all_embs_np = (all_embs_np.astype(np.float32) / norms).astype(np.float16)

    # Build VQ codebook using a random subset if needed.
    num_frames = all_embs_np.shape[0]
    if num_frames > vq_cfg.sample_frames:
        idx = np.random.choice(num_frames, vq_cfg.sample_frames, replace=False)
        sample_embs = all_embs_np[idx].astype(np.float32)
    else:
        sample_embs = all_embs_np.astype(np.float32)

    codebook = VQCodebook.from_embeddings(
        sample_embs,
        num_tokens=vq_cfg.num_tokens,
        batch_size=vq_cfg.kmeans_batch_size,
        random_state=args.vq_random_state,
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
