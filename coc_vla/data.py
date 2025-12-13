from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from world_modality.config import DataConfig
from world_modality.data_sr100 import SR100SequenceDataset, build_cache_paths


@dataclass
class EpisodeCoC:
    episode_id: int
    coc_text: str


class CoCSR100Dataset(Dataset):
    """
    Dataset that extends SR100SequenceDataset with chain-of-causality (CoC) text.

    It assumes:
      - A LeRobot-compatible dataset identified by `DataConfig.dataset_name`
      - Precomputed world tokens and embeddings in the `cache_dir`
      - A JSONL file mapping episode_id -> coc_text:
          {
            "episode_id": 42,
            "coc_text": "1. ... 2. ... 3. ..."
          }

    For now we attach the same CoC text to every timestep within an episode.
    This is sufficient for episode-level CoC conditioning, and can later be
    extended to chunk-level CoC per segment.
    """

    def __init__(
        self,
        data_cfg: DataConfig,
        coc_jsonl: str,
        split: str = "train",
        load_embeddings: bool = True,
    ):
        self.base = SR100SequenceDataset(data_cfg, split=split, load_embeddings=load_embeddings)
        self.episode_to_coc: Dict[int, str] = self._load_coc_mapping(coc_jsonl)

        # Map base indices (episode_idx, local_t) to an "episode_id" consistent with CoC JSONL.
        # Here we assume that SR100SequenceDataset episodes are ordered by episode_id.
        # For many LeRobot datasets, `episode_id` used inside SR100SequenceDataset's
        # _build_episode_indices comes from the underlying dataset.
        self.episode_ids: List[int] = self._infer_episode_ids()

    def _load_coc_mapping(self, coc_jsonl: str) -> Dict[int, str]:
        mapping: Dict[int, str] = {}
        if not os.path.exists(coc_jsonl):
            raise FileNotFoundError(f"CoC JSONL file not found at {coc_jsonl}")
        with open(coc_jsonl, "r") as f:
            for line in f:
                obj = json.loads(line)
                ep_id = int(obj["episode_id"])
                coc_text = obj["coc_text"]
                mapping[ep_id] = coc_text
        return mapping

    def _infer_episode_ids(self) -> List[int]:
        # SR100SequenceDataset.episode_indices is a list of lists of global indices.
        # We assume episodes are ordered by episode_id; map indices 0..N-1 accordingly.
        num_eps = len(self.base.episode_indices)
        return list(range(num_eps))

    def __len__(self) -> int:
        return len(self.base.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base[idx]
        ep_idx, _ = self.base.indices[idx]
        ep_id = self.episode_ids[ep_idx]
        coc_text = self.episode_to_coc.get(ep_id, "")
        sample["coc_text"] = coc_text
        return sample

