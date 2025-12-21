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
            from transformers import AutoVideoProcessor, VJepa2Model

            self.processor = AutoVideoProcessor.from_pretrained(model_name)
            self.backbone = VJepa2Model.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa",  # Faster attention
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
            return self._encode_vjepa(images)
        else:
            return self._encode_dino(images)

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

    def _encode_vjepa_batch(self, images: List[Image.Image]) -> torch.Tensor:
        """Batch encode using V-JEPA-v2 (more efficient for multiple images)."""
        # Process all images as single-frame videos in batch
        videos = [[img] for img in images]  # Each image is a 1-frame video

        inputs = self.processor(
            videos=videos,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.backbone(**inputs)

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
