from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import DataConfig
from .device import resolve_device
from .llm_vla_dataset import build_latent_cache_paths
from .vision import VisionEncoder


class FrameDataset(Dataset):
    """Wrapper dataset for parallel frame loading."""

    def __init__(self, lerobot_ds, image_key: str):
        self.ds = lerobot_ds
        self.image_key = image_key

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        step = self.ds[idx]
        return step[self.image_key]


def load_vjepa_encoder(checkpoint_path: str, device: torch.device, dtype: torch.dtype):
    if not checkpoint_path:
        raise ValueError("V-JEPA requires --vjepa_checkpoint (torchscript encoder).")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"V-JEPA checkpoint not found: {checkpoint_path}")
    model = torch.jit.load(checkpoint_path, map_location=device)
    model.eval()

    def encode(images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model(images)
        return out.to(dtype=dtype)

    return encode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute world latents for VLM training.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--image_key", type=str, default="rgb")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--world_latents_source", type=str, default="vjepa", choices=["dino", "vjepa"])
    parser.add_argument("--vision_model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--vjepa_checkpoint", type=str, default="", help="Optional TorchScript encoder for V-JEPA.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        cache_dir=args.cache_dir,
    )

    # Underlying LeRobot dataset: iterate over all timesteps.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(data_cfg.dataset_name)
    cache_paths = build_latent_cache_paths(data_cfg, args.split, args.world_latents_source)

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

    def to_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x)
        try:
            from PIL import Image
        except Exception:
            Image = None  # type: ignore
        if Image is not None and isinstance(x, Image.Image):
            return torch.as_tensor(np.array(x))
        return torch.as_tensor(x)

    if args.world_latents_source == "dino":
        encoder = VisionEncoder(args.vision_model_name, device=str(device))

        def encode_batch(batch_imgs: List) -> torch.Tensor:
            if isinstance(batch_imgs, torch.Tensor):
                imgs = batch_imgs
            else:
                imgs = torch.stack([to_tensor(x) for x in batch_imgs])
            if imgs.dim() == 4 and imgs.shape[-1] == 3:
                imgs = imgs.permute(0, 3, 1, 2)
            imgs = imgs.to(device)
            return encoder.encode(imgs)

    else:
        if device.type != "cuda":
            raise RuntimeError("V-JEPA latents require CUDA for reasonable throughput.")
        if args.vjepa_checkpoint:
            encode_fn = load_vjepa_encoder(args.vjepa_checkpoint, device, torch.float16)

            def encode_batch(batch_imgs: List) -> torch.Tensor:
                if isinstance(batch_imgs, torch.Tensor):
                    imgs = batch_imgs
                else:
                    imgs = torch.stack([to_tensor(x) for x in batch_imgs])
                if imgs.dim() == 4 and imgs.shape[-1] == 3:
                    imgs = imgs.permute(0, 3, 1, 2)
                imgs = imgs.to(device)
                return encode_fn(imgs)

        else:
            model_name = args.vision_model_name
            if "vjepa" not in model_name.lower():
                model_name = "facebook/vjepa2-vitg-fpc64-256"
                print(f"[V-JEPA] Using default HF model: {model_name}")
            encoder = VisionEncoder(model_name, device=str(device))

            def encode_batch(batch_imgs: List) -> torch.Tensor:
                if isinstance(batch_imgs, torch.Tensor):
                    imgs = batch_imgs
                else:
                    imgs = torch.stack([to_tensor(x) for x in batch_imgs])
                if imgs.dim() == 4 and imgs.shape[-1] == 3:
                    imgs = imgs.permute(0, 3, 1, 2)
                imgs = imgs.to(device)
                return encoder.encode(imgs)

    # Probe embedding dimension with a single batch.
    first_batch = next(iter(loader))
    with torch.no_grad():
        first_emb = encode_batch(first_batch)
    emb_dim = int(first_emb.shape[-1])

    # Create memmap for incremental writing.
    total = len(ds)
    os.makedirs(os.path.dirname(cache_paths.latents_path), exist_ok=True)
    latents_mm = np.lib.format.open_memmap(
        cache_paths.latents_path,
        mode="w+",
        dtype=np.float16,
        shape=(total, emb_dim),
    )

    offset = 0
    print(
        f"Encoding {total} frames to {cache_paths.latents_path} "
        f"(source={args.world_latents_source}, dim={emb_dim})"
    )
    with torch.no_grad():
        for batch_imgs in tqdm(loader, desc="Encoding latents"):
            emb = encode_batch(batch_imgs).detach().cpu().numpy().astype(np.float16)
            bsz = emb.shape[0]
            latents_mm[offset : offset + bsz] = emb
            offset += bsz

    latents_mm.flush()
    print(f"Saved latents to {cache_paths.latents_path}")


if __name__ == "__main__":
    main()
