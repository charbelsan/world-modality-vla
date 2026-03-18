from __future__ import annotations

import os
import re
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image

from .device import resolve_device


class CosmosLatentEncoder(torch.nn.Module):
    """Encode images/clips with a public Cosmos tokenizer encoder and pool latents.

    v0 contract:
    - input: image or short clip
    - output: pooled latent vector [B, D]
    - D = 16 * pool_hw * pool_hw
    """

    def __init__(
        self,
        model_name: str = "cosmos_cv8x8x8_pool4",
        device: str = "auto",
        dtype: str = "float32",
    ):
        super().__init__()
        self.device = resolve_device(device)
        self.model_name = model_name
        self.dtype = torch.float32 if dtype == "float32" else torch.float16
        self.runtime_dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.pool_hw = self._parse_pool_hw(model_name)
        self.temporal_window = self._parse_temporal_window(model_name)
        self.repo_id = self._resolve_repo_id(model_name)
        self.temporal_align, self.spatial_align = self._parse_compression(self.repo_id)
        self.latent_ch = 16
        self.encoder = self._load_public_encoder(self.repo_id)

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

    @staticmethod
    def _resolve_repo_id(model_name: str) -> str:
        model_name = str(model_name)
        lower = model_name.lower()
        if model_name.startswith("nvidia/"):
            return model_name
        if "ci16x16" in lower:
            return "nvidia/Cosmos-0.1-Tokenizer-CI16x16"
        if "ci8x8" in lower or "cosmos_ci" in lower:
            return "nvidia/Cosmos-0.1-Tokenizer-CI8x8"
        if "cv4x8x8" in lower:
            return "nvidia/Cosmos-0.1-Tokenizer-CV4x8x8"
        if "cv8x16x16" in lower:
            return "nvidia/Cosmos-0.1-Tokenizer-CV8x16x16"
        if "cv8x8x8" in lower or "wan2pt1" in lower or "cosmos_" in lower:
            return "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8"
        return os.environ.get("COSMOS_HF_REPO_DEFAULT", "nvidia/Cosmos-1.0-Tokenizer-CV8x8x8")

    @staticmethod
    def _parse_compression(repo_id: str) -> tuple[int, int]:
        name = repo_id.rsplit("/", 1)[-1]
        m = re.search(r"(?:CV|DV)(\d+)x(\d+)x(\d+)", name, flags=re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
        m = re.search(r"(?:CI|DI)(\d+)x(\d+)", name, flags=re.IGNORECASE)
        if m:
            return 1, int(m.group(1))
        return 8, 8

    def _load_public_encoder(self, repo_id: str) -> torch.jit.ScriptModule:
        cache_dir = os.environ.get("COSMOS_TOKENIZER_CACHE_DIR") or os.environ.get("HF_HOME")
        encoder_path = hf_hub_download(
            repo_id=repo_id,
            filename="encoder.jit",
            cache_dir=cache_dir,
        )
        return torch.jit.load(encoder_path, map_location=self.device).eval().to(self.device)

    @property
    def embedding_dim(self) -> int:
        return int(self.latent_ch) * self.pool_hw * self.pool_hw

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

    def _pad_bcthw(self, video: torch.Tensor) -> torch.Tensor:
        _, _, t, h, w = video.shape
        h_to_pad = (self.spatial_align - h % self.spatial_align) % self.spatial_align
        w_to_pad = (self.spatial_align - w % self.spatial_align) % self.spatial_align
        if h_to_pad or w_to_pad:
            pad_top = h_to_pad >> 1
            pad_bottom = h_to_pad - pad_top
            pad_left = w_to_pad >> 1
            pad_right = w_to_pad - pad_left
            video = F.pad(video, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        frames_to_pad = 0
        if self.temporal_align > 1:
            frames_to_pad = (self.temporal_align - (t - 1) % self.temporal_align) % self.temporal_align
        if frames_to_pad:
            pad_before = frames_to_pad >> 1
            pad_after = frames_to_pad - pad_before
            left = video[:, :, :1].expand(-1, -1, pad_before, -1, -1) if pad_before else None
            right = video[:, :, -1:].expand(-1, -1, pad_after, -1, -1) if pad_after else None
            chunks = []
            if left is not None:
                chunks.append(left)
            chunks.append(video)
            if right is not None:
                chunks.append(right)
            video = torch.cat(chunks, dim=2)
        return video

    def _encode_bcthw(self, video: torch.Tensor) -> torch.Tensor:
        video = self._normalize_to_n1p1(video)
        video = self._pad_bcthw(video)
        video = video.to(self.device, dtype=self.runtime_dtype)
        latents = self.encoder(video)
        if isinstance(latents, (tuple, list)):
            latents = latents[0]
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
