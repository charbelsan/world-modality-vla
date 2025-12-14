from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class TextEncoderConfig:
    model_name: str = "distilbert-base-uncased"
    max_length: int = 64
    device: str = "cuda"
    dtype: str = "float16"


class TextEncoder(torch.nn.Module):
    """
    Frozen HF text encoder producing a pooled embedding per string.
    """

    def __init__(self, cfg: TextEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(cfg.model_name)
        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Pick dtype.
        if self.device.type == "cuda":
            if cfg.dtype == "bfloat16" and torch.cuda.is_bf16_supported():
                self._dtype = torch.bfloat16
            elif cfg.dtype in ("float16", "fp16"):
                self._dtype = torch.float16
            else:
                self._dtype = torch.float32
        else:
            self._dtype = torch.float32

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.backbone(**enc)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            # Mean pool over tokens using attention mask.
            last = out.last_hidden_state  # [B, L, D]
            mask = enc.get("attention_mask", torch.ones(last.shape[:2], device=last.device)).unsqueeze(-1)
            summed = (last * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom

        return pooled.to(dtype=self._dtype)

