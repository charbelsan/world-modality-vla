from typing import List, Union

import torch
from PIL import Image

from .device import resolve_device


class VisionEncoder(torch.nn.Module):
    """
    Wrapper around pretrained HF vision backbones.

    Supports:
    - V-JEPA-v2 (facebook/vjepa2-*) - video-native, 1024-dim embeddings
    - DINOv2 (facebook/dinov2-*) - image-native, 768-dim embeddings

    Exposes an `encode(images)` method that returns pooled embeddings [B, d_e].
    """

    def __init__(
        self,
        model_name: str = "facebook/vjepa2-vitg-fpc64-256",
        device: str = "auto",
        dtype: str = "float16",
    ):
        super().__init__()
        self.device = resolve_device(device)
        self.model_name = model_name
        self.is_vjepa = "vjepa" in model_name.lower()

        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        if self.is_vjepa:
            # V-JEPA-v2: Use AutoVideoProcessor for video-native model
            from transformers import AutoVideoProcessor

            try:
                # Newer transformers exposes a dedicated model class.
                from transformers import VJEPA2Model  # type: ignore

                backbone_cls = VJEPA2Model
            except Exception:  # pragma: no cover
                # Fall back to generic AutoModel for older versions.
                from transformers import AutoModel  # type: ignore

                backbone_cls = AutoModel

            self.processor = AutoVideoProcessor.from_pretrained(model_name)
            self.backbone = backbone_cls.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",  # Faster attention
                trust_remote_code=True,
            )
            print(f"[VisionEncoder] Loaded V-JEPA-v2: {model_name}")
            print(f"[VisionEncoder] Embedding dim: {self.backbone.config.hidden_size}")
        else:
            # DINOv2 or other image models
            from transformers import AutoImageProcessor, AutoModel

            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.backbone = AutoModel.from_pretrained(model_name)
            print(f"[VisionEncoder] Loaded image model: {model_name}")

        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension of the model."""
        if hasattr(self.backbone, "config"):
            return self.backbone.config.hidden_size
        return 768  # Default fallback

    @torch.no_grad()
    def encode(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: List of PIL images or tensor of shape [B, C, H, W]

        Returns:
            Tensor of shape [B, d_e] with pooled embeddings.
        """
        if self.is_vjepa:
            return self._encode_vjepa_batch(images)
        else:
            return self._encode_dino(images)

    @torch.no_grad()
    def encode_temporal(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode multiple consecutive frames with temporal context (V-JEPA only).

        Uses V-JEPA's native video capabilities to capture motion dynamics.
        Each clip of m frames produces a single embedding that encodes temporal info.

        Args:
            frames: [B, m, C, H, W] tensor of m consecutive frames
                    Values can be 0-255 uint8 or 0-1 float

        Returns:
            [B, D] embedding capturing temporal dynamics
        """
        import torch.nn.functional as F

        if not self.is_vjepa:
            raise ValueError("encode_temporal only supported for V-JEPA models")

        B, m, C, H, W = frames.shape

        # Normalize to float [0, 1]
        frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0

        # Resize each frame to 256x256
        frames = F.interpolate(
            frames.view(B * m, C, H, W),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        ).view(B, m, C, 256, 256)

        # Normalize with processor stats
        mean = torch.tensor(self.processor.image_mean, device=frames.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=frames.device).view(1, 1, 3, 1, 1)
        frames = (frames - mean) / std

        # Move to device and convert dtype
        frames = frames.to(self.device, dtype=self.backbone.dtype)

        # V-JEPA forward with multi-frame input
        # Output shape: [B, num_tokens, hidden_dim]
        # With tubelet_size=2: num_tokens = (m/2) * (256/16)^2 = (m/2) * 256
        outputs = self.backbone(pixel_values_videos=frames)

        # Pool over all tokens (temporal + spatial)
        pooled = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_dim]
        return pooled

    def _encode_vjepa(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """Encode using V-JEPA-v2 (treats each image as single-frame video)."""
        if isinstance(images, torch.Tensor):
            # Convert tensor to list of PIL images for processor
            images = [Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype('uint8'))
                      for img in images]
        else:
            images = list(images)

        # V-JEPA expects videos - treat each image as a 1-frame "video"
        # Process each image separately and stack
        embeddings = []
        for img in images:
            inputs = self.processor(
                videos=[[img]],  # Single frame as video
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.backbone(**inputs)

            # Pool spatial tokens: last_hidden_state is [B, num_patches, hidden_dim]
            # For single frame, take mean over patches
            pooled = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            embeddings.append(pooled)

        return torch.cat(embeddings, dim=0)  # [B, hidden_dim]

    def _encode_vjepa_batch(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """Batch encode using V-JEPA-v2 (more efficient for multiple images)."""
        import torch.nn.functional as F

        if isinstance(images, torch.Tensor):
            # Fast path: tensor input [B, C, H, W]
            batch = images.float()
            if batch.max() > 1.0:
                batch = batch / 255.0
        else:
            # Convert PIL images to tensor
            import numpy as np
            images = list(images)
            batch = torch.stack([torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                                for img in images])

        # Manual preprocessing: resize to 256x256 and normalize
        batch = F.interpolate(batch, size=(256, 256), mode='bilinear', align_corners=False)

        # Normalize using processor params
        mean = torch.tensor(self.processor.image_mean, device=batch.device).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std, device=batch.device).view(1, 3, 1, 1)
        batch = (batch - mean) / std

        # Add time dimension: [B, T=1, C, H, W]
        batch = batch.unsqueeze(1)

        # Move to device and convert dtype
        batch = batch.to(self.device, dtype=self.backbone.dtype)

        with torch.no_grad():
            outputs = self.backbone(pixel_values_videos=batch)

        # Pool spatial tokens
        pooled = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_dim]
        return pooled

    def _encode_dino(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """Encode using DINOv2 or similar image models."""
        if isinstance(images, torch.Tensor):
            inputs = self.processor(images=images, return_tensors="pt")
        else:
            inputs = self.processor(images=list(images), return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Fall back to CLS token
            pooled = outputs.last_hidden_state[:, 0]

        return pooled
