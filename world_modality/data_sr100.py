from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig


try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    LeRobotDataset = None  # type: ignore


@dataclass
class WorldTokenCachePaths:
    embeddings_path: str
    tokens_path: str
    centroids_path: str


def build_cache_paths(cfg: DataConfig, split: str) -> WorldTokenCachePaths:
    cache_root = os.path.join(cfg.cache_dir, cfg.dataset_name)
    os.makedirs(cache_root, exist_ok=True)
    emb_path = os.path.join(cache_root, f"{split}_embeddings.fp16.npy")
    tok_path = os.path.join(cache_root, f"{split}_world_tokens.int.npy")
    cen_path = os.path.join(cache_root, f"{split}_codebook_centroids.f32.npy")
    return WorldTokenCachePaths(
        embeddings_path=emb_path,
        tokens_path=tok_path,
        centroids_path=cen_path,
    )


class SR100SequenceDataset(Dataset):
    """
    Wrap a LeRobot SR100 dataset to produce imitation-learning sequences with
    cached world-token ids and (optionally) precomputed image embeddings.

    Each item corresponds to a timestep t and returns:
        - context proprio over [t-T_ctx+1, ..., t]
        - optional context image embeddings over [t-T_ctx+1, ..., t]
        - target actions over [t, ..., t+H-1]
        - current world token w_t
        - future world token w_{t+K}
    """

    def __init__(
        self,
        cfg: DataConfig,
        split: str = "train",
        load_embeddings: bool = True,
        train_val_split: float = 0.9,
    ):
        if LeRobotDataset is None:
            raise ImportError("lerobot is not installed; please `pip install lerobot`.")

        self.cfg = cfg
        self.split = split
        self.train_val_split = train_val_split
        # LeRobotDataset loads the full dataset; splitting is handled manually by episodes.
        self.dataset = LeRobotDataset(cfg.dataset_name)

        self.context = cfg.context_frames
        self.horizon = cfg.action_horizon
        self.future_offset = cfg.future_offset

        # Use "train" cache for both train/val since tokens are precomputed on full dataset.
        self.cache_paths = build_cache_paths(cfg, "train")
        self.world_tokens = self._load_world_tokens()
        self.embeddings = self._load_embeddings() if load_embeddings else None

        # Build mapping from episodes to global indices and valid (episode_idx, local_t)
        self.episode_indices: List[List[int]] = self._build_episode_indices()
        self.indices: List[Tuple[int, int]] = self._compute_indices()

    def _load_world_tokens(self) -> np.ndarray:
        if not os.path.exists(self.cache_paths.tokens_path):
            raise FileNotFoundError(
                f"World-token cache not found at {self.cache_paths.tokens_path}. "
                "Run the precompute script first."
            )
        tokens = np.load(self.cache_paths.tokens_path, mmap_mode="r")
        return tokens.astype(np.int64)

    def _load_embeddings(self) -> np.ndarray:
        if not os.path.exists(self.cache_paths.embeddings_path):
            raise FileNotFoundError(
                f"Embedding cache not found at {self.cache_paths.embeddings_path}. "
                "Run the precompute script first."
            )
        emb = np.load(self.cache_paths.embeddings_path, mmap_mode="r")
        return emb.astype(np.float16)

    def _build_episode_indices(self) -> List[List[int]]:
        """
        Build a list of episodes, each represented as an ordered list of global
        indices into the underlying LeRobotDataset.

        If the dataset exposes an `episode_id` field per step we use that to
        avoid crossing episode boundaries; otherwise the entire dataset is
        treated as a single episode.

        Episodes are split into train/val based on self.train_val_split ratio.
        """
        episode_to_indices: Dict[int, List[int]] = {}
        sample0 = self.dataset[0]
        has_ep = "episode_id" in sample0

        if not has_ep:
            # No episode info - treat entire dataset as one episode and split by timesteps.
            total_len = len(self.dataset)
            split_idx = int(total_len * self.train_val_split)
            if self.split == "train":
                return [list(range(split_idx))]
            else:
                return [list(range(split_idx, total_len))]

        for idx in range(len(self.dataset)):
            step = self.dataset[idx]
            ep_id = int(step["episode_id"])
            episode_to_indices.setdefault(ep_id, []).append(idx)

        # Sort episodes by id for reproducibility.
        all_episodes = [
            sorted(idxs) for _, idxs in sorted(episode_to_indices.items(), key=lambda x: x[0])
        ]

        # Split episodes into train/val.
        num_episodes = len(all_episodes)
        split_idx = int(num_episodes * self.train_val_split)

        if self.split == "train":
            return all_episodes[:split_idx]
        else:
            return all_episodes[split_idx:]

    def _compute_indices(self) -> List[Tuple[int, int]]:
        """
        Construct a flat list of (episode_idx, local_t) pairs that are valid
        given context, horizon, and future offset.
        """
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

        # Context indices in this episode and corresponding global indices.
        ctx_local_indices = [
            local_t - self.context + 1 + dt for dt in range(self.context)
        ]
        ctx_global_indices = [
            self._get_global_index(ep_idx, lt) for lt in ctx_local_indices
        ]

        # Context proprio and (optional) image embeddings.
        obs_states = []
        img_emb_ctx = []
        for gidx in ctx_global_indices:
            step = self.dataset[gidx]
            obs_states.append(step[cfg.proprio_key])
            if self.embeddings is not None:
                img_emb_ctx.append(self.embeddings[gidx])

        # Target actions
        actions = []
        for h in range(self.horizon):
            gidx = self._get_global_index(ep_idx, local_t + h)
            step = self.dataset[gidx]
            actions.append(step[cfg.action_key])

        # Current and future world tokens (global indices).
        cur_global_idx = self._get_global_index(ep_idx, local_t)
        current_token = self.world_tokens[cur_global_idx]

        # Predict tokens for w_{t+1}..w_{t+K}, where K = future_offset.
        future_idxs = [
            self._get_global_index(ep_idx, local_t + k)
            for k in range(1, self.future_offset + 1)
        ]
        future_tokens = self.world_tokens[future_idxs]

        obs_states_t = torch.as_tensor(np.stack(obs_states))  # [T_ctx, D_s]
        actions_t = torch.as_tensor(np.stack(actions))  # [H, D_a]
        current_token_t = torch.as_tensor(current_token, dtype=torch.long)
        future_tokens_t = torch.as_tensor(future_tokens, dtype=torch.long)

        out: Dict[str, torch.Tensor] = {
            "obs_states": obs_states_t,
            "actions": actions_t,
            "current_world_token": current_token_t,
            "future_world_tokens": future_tokens_t,
        }

        if img_emb_ctx:
            img_emb_ctx_t = torch.as_tensor(np.stack(img_emb_ctx))  # [T_ctx, d_e]
            out["img_embeddings"] = img_emb_ctx_t

        return out
