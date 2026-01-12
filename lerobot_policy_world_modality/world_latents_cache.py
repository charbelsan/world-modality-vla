from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from world_modality.config import DataConfig
from world_modality.llm_vla_dataset import build_latent_cache_paths


@dataclass(frozen=True)
class WorldLatentsBatch:
    z_hist: torch.Tensor  # [B, T_ctx, D]
    z_future: torch.Tensor  # [B, K, D]
    future_valid: torch.Tensor  # [B, K] (bool)


class WorldLatentsCache:
    """Index world latents by LeRobotDataset global index, with episode-aware boundaries."""

    def __init__(
        self,
        *,
        dataset_repo_id: str,
        cache_dir: str,
        source: str,
        latent_suffix: str,
        context_frames: int,
        future_offset: int,
    ):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

        self.dataset_repo_id = str(dataset_repo_id)
        self.context_frames = int(context_frames)
        self.future_offset = int(future_offset)

        cfg = DataConfig(dataset_name=self.dataset_repo_id, cache_dir=str(cache_dir))
        cfg.latent_suffix = str(latent_suffix or "")
        # Latents are stored over the full dataset. We use "train" naming convention.
        paths = build_latent_cache_paths(cfg, "train", str(source))
        self.latents = np.load(paths.latents_path, mmap_mode="r")
        if self.latents.dtype != np.float16:
            # Keep mmap if possible; fall back to a cast only if needed.
            self.latents = self.latents.astype(np.float16)

        ds = LeRobotDataset(self.dataset_repo_id)
        if hasattr(ds, "episodes") and ds.episodes is not None and hasattr(ds, "episode_data_index"):
            episode_from = ds.episode_data_index["from"].cpu().numpy().astype(np.int64)
            episode_to = ds.episode_data_index["to"].cpu().numpy().astype(np.int64)
        else:
            # Fallback: treat as one episode.
            episode_from = np.array([0], dtype=np.int64)
            episode_to = np.array([len(ds)], dtype=np.int64)

        n = int(self.latents.shape[0])
        if n < len(ds):
            raise ValueError(
                f"Latents cache has {n} frames but dataset has {len(ds)}. "
                "Recompute latents with the same dataset version/order."
            )

        # Build global index -> (episode_idx, episode_start, episode_end)
        ep_idx_of = np.empty((len(ds),), dtype=np.int64)
        ep_start_of = np.empty((len(ds),), dtype=np.int64)
        ep_end_of = np.empty((len(ds),), dtype=np.int64)
        for ep, (start, end) in enumerate(zip(episode_from, episode_to, strict=False)):
            ep_idx_of[start:end] = ep
            ep_start_of[start:end] = start
            ep_end_of[start:end] = end
        self._ep_start_of = ep_start_of
        self._ep_end_of = ep_end_of

    def get_by_index(self, index: torch.Tensor) -> Optional[WorldLatentsBatch]:
        """Return (z_hist, z_future) for each global index in the batch.

        Args:
          index: [B] int64 tensor (global dataset indices)
        """
        if index is None:
            return None
        if not isinstance(index, torch.Tensor):
            index = torch.as_tensor(index)
        if index.numel() == 0:
            return None

        idx = index.detach().cpu().to(torch.int64).view(-1).numpy()
        B = int(idx.shape[0])
        T = self.context_frames
        K = self.future_offset

        z_hist = []
        z_future = []
        future_valid = []
        for g in idx.tolist():
            if g < 0 or g >= self._ep_start_of.shape[0]:
                raise IndexError(f"Global index out of range: {g}")
            ep_start = int(self._ep_start_of[g])
            ep_end = int(self._ep_end_of[g])  # exclusive

            # History: clamp to episode start (repeat-padding)
            hist_idx = [max(ep_start, g - (T - 1) + t) for t in range(T)]

            # Future: clamp to episode end-1, but track validity
            fut = []
            valid = []
            for k in range(1, K + 1):
                gi = g + k
                if gi < ep_end:
                    fut.append(gi)
                    valid.append(True)
                else:
                    fut.append(ep_end - 1)
                    valid.append(False)

            z_hist.append(self.latents[np.array(hist_idx, dtype=np.int64)])
            z_future.append(self.latents[np.array(fut, dtype=np.int64)])
            future_valid.append(np.array(valid, dtype=np.bool_))

        z_hist_t = torch.as_tensor(np.stack(z_hist), dtype=torch.float32)
        z_future_t = torch.as_tensor(np.stack(z_future), dtype=torch.float32)
        valid_t = torch.as_tensor(np.stack(future_valid), dtype=torch.bool)
        return WorldLatentsBatch(z_hist=z_hist_t, z_future=z_future_t, future_valid=valid_t)
