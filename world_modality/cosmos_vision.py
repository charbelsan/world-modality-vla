from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .device import resolve_device


def _default_cosmos_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "coc_vla" / "external" / "repos" / "cosmos-predict2.5"


def ensure_cosmos_importable(root: str | os.PathLike[str] | None = None) -> Path:
    env_root = os.environ.get("COSMOS_PREDICT2_ROOT", "")
    candidate = Path(root) if root else (Path(env_root) if env_root else _default_cosmos_root())
    candidate = candidate.expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Cosmos Predict repo not found at {candidate}. "
            "Set COSMOS_PREDICT2_ROOT or clone nvidia-cosmos/cosmos-predict2.5."
        )
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    return candidate


class CosmosLatentEncoder(torch.nn.Module):
    """Encode images/clips with the Cosmos tokenizer and pool latents to a fixed vector.

    v0 contract:
    - input: image or short clip
    - output: pooled latent vector [B, D]
    - D = 16 * pool_hw * pool_hw
    """

    def __init__(
        self,
        model_name: str = "cosmos_wan2pt1_pool4",
        device: str = "auto",
        dtype: str = "float32",
    ):
        super().__init__()
        self.device = resolve_device(device)
        self.model_name = model_name
        self.dtype = torch.float32 if dtype == "float32" else torch.float16
        self.pool_hw = self._parse_pool_hw(model_name)
        self.temporal_window = self._parse_temporal_window(model_name)
        cosmos_root = ensure_cosmos_importable()
        try:
            from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import Cosmos Predict tokenizer. Ensure the repo is present and the package is "
                "installed with its CUDA extra, e.g. `uv sync --project "
                f"{cosmos_root} --extra=cu128 --active --inexact`."
            ) from e

        self.tokenizer = Wan2pt1VAEInterface(temporal_window=self.temporal_window, load_mean_std=False)
        self.tokenizer_dtype = torch.float32

    @staticmethod
    def _parse_pool_hw(model_name: str) -> int:
        parts = str(model_name).replace("-", "_").split("_")
        for part in parts:
            if part.startswith("pool") and part[4:].isdigit():
                return max(1, int(part[4:]))
        return 4

    @staticmethod
    def _parse_temporal_window(model_name: str) -> int:
        parts = str(model_name).replace("-", "_").split("_")
        for part in parts:
            if part.startswith("m") and part[1:].isdigit():
                return max(1, int(part[1:]))
        return 4

    @property
    def embedding_dim(self) -> int:
        return int(self.tokenizer.latent_ch) * self.pool_hw * self.pool_hw

    def _normalize_to_n1p1(self, frames: torch.Tensor) -> torch.Tensor:
        frames = frames.float()
        maxv = float(frames.max().item())
        minv = float(frames.min().item())
        if minv >= -1.01 and maxv <= 1.01:
            return frames.clamp(-1.0, 1.0)
        if maxv > 1.0:
            frames = frames / 255.0
        return frames.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

    def _pool_latents(self, latents: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(latents.float(), output_size=(1, self.pool_hw, self.pool_hw))
        return pooled.flatten(start_dim=1)

    def _encode_bcthw(self, video: torch.Tensor) -> torch.Tensor:
        video = self._normalize_to_n1p1(video).to(self.device, dtype=self.tokenizer_dtype)
        latents = self.tokenizer.encode(video)
        return self._pool_latents(latents).to(dtype=self.dtype)

    @torch.no_grad()
    def encode(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            batch = images
            if batch.dim() != 4:
                raise ValueError(f"Expected [B, C, H, W] tensor, got shape={tuple(batch.shape)}")
        else:
            batch = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1) for img in list(images)])
        video = batch.unsqueeze(2)  # [B, C, T=1, H, W]
        return self._encode_bcthw(video)

    @torch.no_grad()
    def encode_temporal(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 5:
            raise ValueError(f"Expected [B, T, C, H, W] tensor, got shape={tuple(frames.shape)}")
        video = frames.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]
        return self._encode_bcthw(video)
