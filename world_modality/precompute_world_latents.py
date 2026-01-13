from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from typing import List, Tuple

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


class TemporalClipDataset(Dataset):
    """Dataset that returns m consecutive frames for temporal encoding.

    Handles episode boundaries with repeat-padding at the start.
    """

    def __init__(self, lerobot_ds, image_key: str, temporal_window: int):
        self.ds = lerobot_ds
        self.image_key = image_key
        self.m = temporal_window

        # Build episode index: maps global_idx -> (episode_id, local_idx, episode_start_global)
        self.episode_info = self._build_episode_info()

    def _build_episode_info(self) -> List[Tuple[int, int, int]]:
        """Build mapping from global index to episode info."""
        info = []
        # LeRobot datasets have episode_index field
        if hasattr(self.ds, 'episodes') and self.ds.episodes is not None:
            episodes = self.ds.episodes
        else:
            # Fallback: treat entire dataset as one episode
            return [(0, i, 0) for i in range(len(self.ds))]

        # Build from episode_data_index
        episode_starts = {}
        for ep_idx in range(len(episodes)):
            start = self.ds.episode_data_index['from'][ep_idx].item()
            end = self.ds.episode_data_index['to'][ep_idx].item()
            for local_idx, global_idx in enumerate(range(start, end)):
                info.append((ep_idx, local_idx, start))
        return info

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        """Return m frames ending at idx, with repeat-padding at episode start."""
        ep_idx, local_idx, ep_start = self.episode_info[idx]

        # Get m frame indices with repeat-padding at boundaries
        frame_indices = []
        for i in range(self.m):
            offset = -(self.m - 1) + i  # e.g., m=4: [-3, -2, -1, 0]
            global_idx = idx + offset
            # Clamp to episode start
            global_idx = max(global_idx, ep_start)
            frame_indices.append(global_idx)

        # Load frames
        frames = []
        for fidx in frame_indices:
            frame = self.ds[fidx][self.image_key]
            frames.append(frame)

        # Stack to [m, H, W, C] or [m, C, H, W] depending on format
        frames = torch.stack([torch.as_tensor(np.array(f)) for f in frames])
        return frames


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
    parser.add_argument("--temporal_window", type=int, default=1,
                        help="Number of frames per clip for temporal encoding (1=single-frame, 4=recommended)")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If output .npy exists, resume by appending from the last written row (assumes trailing zeros).",
    )
    return parser.parse_args()


def _resume_offset_from_trailing_zeros(latents: np.ndarray, *, block_rows: int = 8192) -> int:
    """Return the next row index to write, assuming the file is written sequentially and the unwritten tail is 0s."""
    total = int(latents.shape[0])
    if total == 0:
        return 0
    if np.any(latents[-1] != 0):
        return total
    end = total
    while end > 0:
        start = max(0, end - block_rows)
        blk = latents[start:end]
        nonzero = np.any(blk != 0, axis=1)
        if np.any(nonzero):
            last = int(start + np.where(nonzero)[0].max())
            return last + 1
        end = start
    return 0


def main():
    args = parse_args()
    device = resolve_device(args.device)
    temporal_window = args.temporal_window

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        cache_dir=args.cache_dir,
    )

    # Underlying LeRobot dataset: iterate over all timesteps.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    ds = LeRobotDataset(data_cfg.dataset_name)
    cache_paths = build_latent_cache_paths(data_cfg, args.split, args.world_latents_source)

    # Modify cache path for temporal encoding
    if temporal_window > 1:
        base_path = cache_paths.latents_path
        # Insert _m{window} before .fp16.npy
        if ".fp16.npy" in base_path:
            new_path = base_path.replace(".fp16.npy", f"_m{temporal_window}.fp16.npy")
        else:
            new_path = base_path.replace(".npy", f"_m{temporal_window}.npy")
        cache_paths = replace(cache_paths, latents_path=new_path)
        print(f"[Temporal] Using m={temporal_window} frames per embedding")

    # Choose dataset based on temporal_window
    if temporal_window > 1:
        frame_ds = TemporalClipDataset(ds, data_cfg.image_key, temporal_window)
    else:
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
        if temporal_window > 1:
            raise ValueError("Temporal encoding only supported for V-JEPA, not DINO")
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
            if temporal_window > 1:
                raise ValueError("TorchScript V-JEPA checkpoint doesn't support temporal encoding. Use HF model.")
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

            if temporal_window > 1:
                # Multi-frame temporal encoding
                def encode_batch(batch_clips: torch.Tensor) -> torch.Tensor:
                    # batch_clips: [B, m, H, W, C] from TemporalClipDataset
                    if batch_clips.dim() == 5 and batch_clips.shape[-1] == 3:
                        # [B, m, H, W, C] -> [B, m, C, H, W]
                        batch_clips = batch_clips.permute(0, 1, 4, 2, 3)
                    return encoder.encode_temporal(batch_clips)
            else:
                # Single-frame encoding (original behavior)
                def encode_batch(batch_imgs: List) -> torch.Tensor:
                    if isinstance(batch_imgs, torch.Tensor):
                        imgs = batch_imgs
                    else:
                        imgs = torch.stack([to_tensor(x) for x in batch_imgs])
                    if imgs.dim() == 4 and imgs.shape[-1] == 3:
                        imgs = imgs.permute(0, 3, 1, 2)
                    imgs = imgs.to(device)
                    return encoder.encode(imgs)

    total = len(ds)
    os.makedirs(os.path.dirname(cache_paths.latents_path), exist_ok=True)
    offset = 0
    latents_mm: np.ndarray

    if bool(args.resume) and os.path.exists(cache_paths.latents_path):
        latents_mm = np.load(cache_paths.latents_path, mmap_mode="r+")
        if int(latents_mm.shape[0]) != int(total):
            raise ValueError(
                f"Existing latents file has shape[0]={int(latents_mm.shape[0])} but dataset has {int(total)}. "
                "Delete the file and recompute with the same dataset revision/order."
            )
        offset = _resume_offset_from_trailing_zeros(latents_mm)
        if offset >= total:
            print(f"Latents already complete: {cache_paths.latents_path} (rows={total})")
            return
        print(f"Resuming latents write at row offset={offset}/{total}: {cache_paths.latents_path}")

        # Verify embedding dimension matches current encoder settings.
        first_batch = next(iter(loader))
        with torch.no_grad():
            first_emb = encode_batch(first_batch)
        emb_dim = int(first_emb.shape[-1])
        if int(latents_mm.shape[1]) != int(emb_dim):
            raise ValueError(
                f"Existing latents file dim={int(latents_mm.shape[1])} but encoder produces dim={int(emb_dim)}. "
                "Delete the file and recompute with consistent `--vision_model_name/--temporal_window`."
            )
    else:
        # Probe embedding dimension with a single batch.
        first_batch = next(iter(loader))
        with torch.no_grad():
            first_emb = encode_batch(first_batch)
        emb_dim = int(first_emb.shape[-1])

        # Create memmap for incremental writing.
        latents_mm = np.lib.format.open_memmap(
            cache_paths.latents_path,
            mode="w+",
            dtype=np.float16,
            shape=(total, emb_dim),
        )

    print(
        f"Encoding {total} frames to {cache_paths.latents_path} "
        f"(source={args.world_latents_source}, dim={int(latents_mm.shape[1])}, start_row={offset})"
    )
    start_row = int(offset)
    with torch.no_grad():
        for batch_imgs in tqdm(loader, desc="Encoding latents"):
            emb = encode_batch(batch_imgs).detach().cpu().numpy().astype(np.float16)
            bsz = emb.shape[0]
            latents_mm[offset : offset + bsz] = emb
            offset += bsz

    latents_mm.flush()
    print(f"Saved latents to {cache_paths.latents_path}")

    # Save metadata JSON for reproducibility (always).
    metadata = {
        "temporal_window": temporal_window,
        "model": model_name if args.world_latents_source == "vjepa" else args.vision_model_name,
        "pooling": "mean_spatial_then_temporal" if temporal_window > 1 else "mean_spatial",
        "embedding_dim": int(latents_mm.shape[1]),
        "total_frames": total,
        "dataset": args.dataset_name,
        "split": args.split,
        "source": args.world_latents_source,
        "resume_start_row": start_row,
        "final_written_rows": int(offset),
    }
    metadata_path = cache_paths.latents_path.replace(".npy", "_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
