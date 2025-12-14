from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import nn

from .config import ModelType, TransformerConfig


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: TransformerConfig, seq_len: int):
        super().__init__()
        # Pre-norm (norm_first=True) is typically more stable for transformer training.
        # Some older torch versions may not support `norm_first`, so we fall back.
        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model,
                dropout=cfg.dropout,
                batch_first=True,
                norm_first=getattr(cfg, "norm_first", False),
            )
        except TypeError:  # pragma: no cover
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=4 * cfg.d_model,
                dropout=cfg.dropout,
                batch_first=True,
            )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, cfg.d_model))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, L, D]
        x = x + self.pos_emb[:, : x.size(1), :]
        return self.transformer(x, mask=attn_mask)


class WorldPolicyTransformer(nn.Module):
    """
    Unified transformer for Baseline A, Baseline B, and model C.

    Sequence layouts:
        A: [OBS] [ACT_Q1..ACT_QH]
        B: [OBS] [FUT_MASK] [ACT_Q1..ACT_QH]
        C: [OBS] [WORLD_CUR] [FUT_MASK] [ACT_Q1..ACT_QH]
    """

    def __init__(
        self,
        model_type: ModelType,
        cfg: TransformerConfig,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        world_vocab_size: int,
        horizon: int,
        future_horizon: int,
        use_language: bool = False,
        lang_dim: int = 0,
        world_input_scale: float = 1.0,
        world_input_dropout: float = 0.0,
        world_input_layernorm: bool = False,
        block_world_to_action: bool = False,
    ):
        super().__init__()
        assert model_type in ("A", "B", "C", "C_no_world_input")
        self.model_type: ModelType = model_type
        self.cfg = cfg
        self.horizon = horizon
        self.future_horizon = future_horizon
        self.use_language = use_language
        self.world_input_scale = float(world_input_scale)
        self.world_input_dropout = float(world_input_dropout)
        self.world_input_layernorm = bool(world_input_layernorm)
        self.block_world_to_action = bool(block_world_to_action)

        d_model = cfg.d_model

        # Per-modality projections
        self.proj_img = nn.Linear(obs_dim, d_model)
        self.proj_state = nn.Linear(state_dim, d_model)
        self.proj_lang = nn.Linear(lang_dim, d_model) if use_language else None

        # World token embedding (only consumed as input for C; shared with output head)
        self.emb_world = nn.Embedding(world_vocab_size, d_model)

        # Learnable tokens
        self.lang_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_language else None
        self.obs_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # One learnable query per future horizon step.
        self.future_queries = nn.Parameter(torch.zeros(1, future_horizon, d_model))
        self.action_queries = nn.Parameter(torch.zeros(1, horizon, d_model))

        # Determine sequence length
        seq_len = 1 + horizon  # OBS + ACT_Q
        if use_language:
            seq_len += 1  # LANG
        if model_type == "C":
            seq_len += 1  # WORLD_CUR
        if model_type in ("B", "C", "C_no_world_input"):
            seq_len += future_horizon  # FUT_QUERY × K

        self.backbone = TransformerBackbone(cfg, seq_len=seq_len)

        # Heads
        self.action_head = nn.Linear(d_model, action_dim)
        self.world_head = nn.Linear(d_model, world_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.lang_token is not None:
            nn.init.normal_(self.lang_token, mean=0.0, std=0.02)
        nn.init.normal_(self.obs_token, mean=0.0, std=0.02)
        nn.init.normal_(self.future_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)

    def forward(
        self,
        img_emb: torch.Tensor,
        state: torch.Tensor,
        current_world_token: Optional[torch.Tensor] = None,
        lang_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            img_emb: [B, d_e] pooled visual embeddings.
            state: [B, D_s] proprio state (last timestep in context).
            current_world_token: [B] int64, current world token id (for model C).

        Returns:
            actions: [B, H, D_a]
            world_logits: [B, V] or None, logits over future world token.
        """
        B = img_emb.size(0)
        d_model = self.cfg.d_model

        tokens = []

        if self.use_language:
            assert self.proj_lang is not None
            assert self.lang_token is not None
            assert lang_emb is not None, "use_language=True requires lang_emb."
            lang = self.proj_lang(lang_emb)  # [B, D]
            lang_tok = self.lang_token.expand(B, -1, -1) + lang.unsqueeze(1)
            tokens.append(lang_tok)

        obs = self.proj_img(img_emb) + self.proj_state(state)  # [B, D]
        obs_tok = self.obs_token.expand(B, -1, -1) + obs.unsqueeze(1)

        tokens.append(obs_tok)

        if self.model_type == "C":
            assert current_world_token is not None, "Model C requires current_world_token input."
            world_cur = self.emb_world(current_world_token)  # [B, D]
            if self.world_input_layernorm:
                world_cur = torch.nn.functional.layer_norm(world_cur, (d_model,))
            if self.world_input_scale != 1.0:
                world_cur = world_cur * self.world_input_scale
            if self.training and self.world_input_dropout > 0:
                keep = (torch.rand(B, 1, device=world_cur.device) > self.world_input_dropout).to(
                    dtype=world_cur.dtype
                )
                world_cur = world_cur * keep
            world_cur = world_cur.unsqueeze(1)  # [B, 1, D]
            tokens.append(world_cur)

        # B, C, and C_no_world_input all have future queries for world-token prediction.
        if self.model_type in ("B", "C", "C_no_world_input"):
            fut_q = self.future_queries.expand(B, -1, -1)
            tokens.append(fut_q)

        act_q = self.action_queries.expand(B, -1, -1)
        tokens.append(act_q)

        x = torch.cat(tokens, dim=1)  # [B, L, D]
        attn_mask = None
        if self.block_world_to_action and self.model_type == "C":
            # Block ACT_Q positions (targets) from attending to WORLD_CUR (source).
            L = x.shape[1]
            act_start = L - self.horizon
            world_pos = (1 if self.use_language else 0) + 1  # LANG? + OBS
            mask = torch.zeros((L, L), dtype=torch.bool, device=x.device)
            mask[act_start : act_start + self.horizon, world_pos] = True
            attn_mask = mask
        h = self.backbone(x, attn_mask=attn_mask)

        # Readout indices: ACT_Q are always appended last.
        act_start = h.shape[1] - self.horizon
        fut_start = (
            act_start - self.future_horizon
            if self.model_type in ("B", "C", "C_no_world_input")
            else None
        )

        act_h = h[:, act_start : act_start + self.horizon, :]
        actions = self.action_head(act_h)

        world_logits = None
        if self.model_type in ("B", "C", "C_no_world_input") and fut_start is not None:
            fut_h = h[:, fut_start : fut_start + self.future_horizon, :]  # [B, K, D]
            world_logits = self.world_head(fut_h)  # [B, K, V]

        return actions, world_logits
