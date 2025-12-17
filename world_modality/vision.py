from typing import List, Union

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from .device import resolve_device


class VisionEncoder(torch.nn.Module):
    """
    Thin wrapper around a pretrained HF vision backbone (e.g., DINOv2).

    Exposes an `encode(images)` method that returns a batch of pooled embeddings
    of shape [B, d_e].
    """

    def __init__(self, model_name: str = "facebook/dinov2-base", device: str = "auto"):
        super().__init__()
        self.device = resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images: Union[List[Image.Image], torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: list of PIL images or a float tensor acceptable by the processor.

        Returns:
            Tensor of shape [B, d_e] with pooled embeddings.
        """

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
