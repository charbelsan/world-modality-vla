from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import DataConfig


try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None  # type: ignore


@dataclass
class WorldLatentCachePaths:
    latents_path: str


def build_latent_cache_paths(cfg: DataConfig, split: str, source: str) -> WorldLatentCachePaths:
    cache_root = os.path.join(cfg.cache_dir, cfg.dataset_name)
    os.makedirs(cache_root, exist_ok=True)
    latents_path = os.path.join(cache_root, f"{split}_world_latents_{source}.fp16.npy")
    return WorldLatentCachePaths(latents_path=latents_path)


def _to_pil_image(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, torch.Tensor):
        img = x
        if img.dim() == 3 and img.shape[0] in (1, 3):
            if img.dtype != torch.uint8:
                img = (img * 255).clamp(0, 255).to(torch.uint8)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(img_np)
    if isinstance(x, np.ndarray):
        if x.ndim == 3:
            return Image.fromarray(x.astype(np.uint8))
    raise ValueError("Unsupported image type for conversion to PIL.")


class LiberoVLADataset(Dataset):
    """Dataset for VLM action prediction with precomputed world latents."""

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        world_latents_source: str = "dino",
        coc_jsonl: Optional[str] = None,
        train_val_split: float = 0.9,
    ):
        if LeRobotDataset is None:
            raise ImportError("lerobot is not installed; please `pip install lerobot`.")

        self.cfg = cfg
        self.split = split
        self.train_val_split = train_val_split
        self.dataset = LeRobotDataset(cfg.dataset_name)

        self.context = cfg.context_frames
        self.horizon = cfg.action_horizon
        self.future_offset = cfg.future_offset

        # Latents are precomputed on the full dataset; we use the "train" cache by default.
        self.cache_paths = build_latent_cache_paths(cfg, "train", world_latents_source)
        self.latents = self._load_latents()
        self.episode_to_coc = self._load_coc_mapping(coc_jsonl) if coc_jsonl else {}

        self.episode_indices: List[List[int]] = self._build_episode_indices()
        self.indices: List[Tuple[int, int]] = self._compute_indices()
        self.episode_ids: List[int] = self._infer_episode_ids()

    def _load_latents(self) -> np.ndarray:
        if not os.path.exists(self.cache_paths.latents_path):
            raise FileNotFoundError(
                f"World-latent cache not found at {self.cache_paths.latents_path}. "
                "Run `python -m world_modality.precompute_world_latents ...` first."
            )
        latents = np.load(self.cache_paths.latents_path, mmap_mode="r")
        return latents.astype(np.float16)

    def _build_episode_indices(self) -> List[List[int]]:
        episode_to_indices: Dict[int, List[int]] = {}
        sample0 = self.dataset[0]
        ep_key = self.cfg.episode_id_key
        has_ep = ep_key in sample0

        if not has_ep:
            total_len = len(self.dataset)
            split_idx = int(total_len * self.train_val_split)
            if self.split == "train":
                return [list(range(split_idx))]
            return [list(range(split_idx, total_len))]

        for idx in range(len(self.dataset)):
            step = self.dataset[idx]
            ep_id = int(step[ep_key])
            episode_to_indices.setdefault(ep_id, []).append(idx)

        all_episodes = [
            sorted(idxs) for _, idxs in sorted(episode_to_indices.items(), key=lambda x: x[0])
        ]
        num_episodes = len(all_episodes)
        split_idx = int(num_episodes * self.train_val_split)
        if self.split == "train":
            return all_episodes[:split_idx]
        return all_episodes[split_idx:]

    def _infer_episode_ids(self) -> List[int]:
        if not self.episode_indices:
            return []
        sample0 = self.dataset[0]
        if self.cfg.episode_id_key not in sample0:
            return [0]

        episode_ids: List[int] = []
        for ep_indices in self.episode_indices:
            if not ep_indices:
                continue
            gidx0 = ep_indices[0]
            step0 = self.dataset[gidx0]
            episode_ids.append(int(step0[self.cfg.episode_id_key]))
        return episode_ids

    def _load_coc_mapping(self, coc_jsonl: Optional[str]) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        if not coc_jsonl:
            return mapping
        if not os.path.exists(coc_jsonl):
            raise FileNotFoundError(f"CoC JSONL file not found at {coc_jsonl}")
        import json

        with open(coc_jsonl, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    mapping[int(obj["episode_id"])] = str(obj["coc_text"])
                except Exception:
                    continue
        return mapping

    def _compute_indices(self) -> List[Tuple[int, int]]:
        indices: List[Tuple[int, int]] = []
        for ep_idx, ep_global_indices in enumerate(self.episode_indices):
            length = len(ep_global_indices)
            for t in range(length):
                t_start_ctx = t - self.context + 1
                t_end_act = t + self.horizon - 1
                t_future_max = t + self.future_offset
                if t_start_ctx < 0 or t_end_act >= length or t_future_max >= length:
                    continue
                indices.append((ep_idx, t))
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def _get_global_index(self, episode_idx: int, local_t: int) -> int:
        return self.episode_indices[episode_idx][local_t]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_idx, local_t = self.indices[idx]
        cfg = self.cfg

        gidx = self._get_global_index(ep_idx, local_t)
        step = self.dataset[gidx]
        image = _to_pil_image(step[cfg.image_key])
        instruction = step.get(cfg.instruction_key, "")

        # Actions
        actions = []
        for h in range(self.horizon):
            gi = self._get_global_index(ep_idx, local_t + h)
            actions.append(self.dataset[gi][cfg.action_key])
        actions_t = torch.as_tensor(np.stack(actions)).float()

        # History + future latents
        hist_idxs = [
            self._get_global_index(ep_idx, local_t - self.context + 1 + dt)
            for dt in range(self.context)
        ]
        fut_idxs = [
            self._get_global_index(ep_idx, local_t + k)
            for k in range(1, self.future_offset + 1)
        ]
        z_hist = torch.as_tensor(np.array(self.latents[hist_idxs])).float()
        z_future = torch.as_tensor(np.array(self.latents[fut_idxs])).float()
        coc_text = ""
        if self.episode_to_coc:
            ep_id = self.episode_ids[ep_idx] if ep_idx < len(self.episode_ids) else 0
            coc_text = self.episode_to_coc.get(ep_id, "")

        return {
            "image": image,
            "instruction": str(instruction),
            "actions": actions_t,
            "z_hist": z_hist,
            "z_future": z_future,
            "coc_text": coc_text,
        }
