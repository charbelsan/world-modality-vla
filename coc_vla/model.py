from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from world_modality.model import WorldPolicyTransformer
from world_modality.config import TransformerConfig, ModelType


class CoCDecoder(nn.Module):
    """
    Lightweight causal language decoder for CoC text.

    This is a simple causal transformer LM conditioned on a fixed context
    vector derived from the world-modality trunk. For real use you likely
    want to replace this with a stronger pretrained LM or adapter.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.context_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        coc_input_ids: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coc_input_ids: [B, L] token ids (teacher-forced input).
            context: [B, D] context vector from trunk (world + language).
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = coc_input_ids.shape
        x = self.token_emb(coc_input_ids)  # [B, L, D]
        x = x + self.pos_emb[:, :L, :]

        # Use context as a single "memory" token per batch.
        mem = self.context_proj(context).unsqueeze(1)  # [B, 1, D]

        # Create causal mask for decoder (no future tokens).
        tgt_mask = torch.triu(
            torch.ones(L, L, device=coc_input_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        h = self.decoder(
            tgt=x,
            memory=mem,
            tgt_mask=tgt_mask,
        )
        logits = self.lm_head(h)
        return logits


class WorldCoCTransformer(nn.Module):
    """
    Two-head model:
      - world-modality trunk (WorldPolicyTransformer): actions + world tokens
      - CoC decoder head: causal language generation conditioned on trunk state
    """

    def __init__(
        self,
        model_type: ModelType,
        trunk_cfg: TransformerConfig,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        world_vocab_size: int,
        horizon: int,
        future_horizon: int,
        coc_vocab_size: int,
        coc_d_model: int = 512,
        coc_n_layers: int = 4,
        coc_n_heads: int = 8,
        coc_dropout: float = 0.1,
        coc_max_len: int = 128,
    ):
        super().__init__()
        self.trunk = WorldPolicyTransformer(
            model_type=model_type,
            cfg=trunk_cfg,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            world_vocab_size=world_vocab_size,
            horizon=horizon,
            future_horizon=future_horizon,
        )

        self.coc_decoder = CoCDecoder(
            vocab_size=coc_vocab_size,
            d_model=coc_d_model,
            n_layers=coc_n_layers,
            n_heads=coc_n_heads,
            dropout=coc_dropout,
            max_len=coc_max_len,
        )

        # Simple pooling over trunk hidden states to create CoC context.
        # We assume trunk's last layer activations are accessible via a hook or
        # by re-running forward; for now we pool over action head inputs.
        self.context_pool = nn.Linear(trunk_cfg.d_model, coc_d_model)

    def forward(
        self,
        img_emb: torch.Tensor,
        state: torch.Tensor,
        current_world_token: Optional[torch.Tensor],
        coc_input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Args:
            img_emb: [B, d_e]
            state: [B, D_s]
            current_world_token: [B] or None
            coc_input_ids: [B, L] CoC tokens (teacher-forced input) or None

        Returns:
            actions: [B, H, D_a]
            world_logits: [B, K, V] or None
            coc_logits: [B, L, coc_vocab] or None
        """
        # For now, we re-use the existing trunk forward and approximate a context
        # as the mean of action-query hidden states via an extra forward hook.
        # To avoid refactoring the trunk, we call its forward and then reconstruct
        # a context from the action head's inputs (approximated by applying the
        # inverse of the action head). In practice you may want to modify the
        # trunk to return hidden states directly.

        actions, world_logits = self.trunk(
            img_emb=img_emb,
            state=state,
            current_world_token=current_world_token,
        )

        coc_logits = None
        if coc_input_ids is not None:
            # Approximate context: mean of projected image+state features.
            # This is a placeholder; in a real implementation you'd pool over
            # specific trunk hidden states (e.g. WORLD_CUR + FUT_QUERY).
            context_vec = self.context_pool(self.trunk.proj_img(img_emb) + self.trunk.proj_state(state))
            coc_logits = self.coc_decoder(coc_input_ids=coc_input_ids, context=context_vec)

        return actions, world_logits, coc_logits

