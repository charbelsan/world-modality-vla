from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .config import ModelType, TransformerConfig


class TransformerBackbone(nn.Module):
    def __init__(self, cfg: TransformerConfig, seq_len: int):
        super().__init__()
        # Pre-norm (norm_first=True) is typically more stable for transformer training.
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


class Prophet(nn.Module):
    """Predict future continuous embeddings from a short history.

    We use K learned 'future query slots' appended to the history, then run a small
    transformer encoder over [history || queries] and read out the last K slots.

    Shapes:
      z_hist: [B, T_ctx, emb_dim]
      z_fut_pred: [B, K_fut, emb_dim]
    """

    def __init__(
        self,
        emb_dim: int,
        hidden_dim: int,
        future_horizon: int,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.future_horizon = future_horizon

        self.input_proj = nn.Linear(emb_dim, hidden_dim)
        self.query_slots = nn.Parameter(torch.zeros(1, future_horizon, hidden_dim))

        # A small transformer encoder; pre-norm if available.
        try:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
        except TypeError:  # pragma: no cover
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                batch_first=True,
            )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, emb_dim)

        nn.init.normal_(self.query_slots, mean=0.0, std=0.02)

    def forward(self, z_hist: torch.Tensor) -> torch.Tensor:
        # z_hist: [B, T_ctx, emb_dim]
        B = z_hist.shape[0]
        h = self.input_proj(z_hist)  # [B, T_ctx, hidden]
        q = self.query_slots.expand(B, -1, -1)  # [B, K_fut, hidden]
        x = torch.cat([h, q], dim=1)  # [B, T_ctx + K_fut, hidden]
        y = self.encoder(x)
        fut_h = y[:, -self.future_horizon :, :]  # [B, K_fut, hidden]
        return self.output_proj(fut_h)  # [B, K_fut, emb_dim]


class GatedCrossAttention(nn.Module):
    """Cross-attend from policy action queries into an external future memory.

    This is used in Model F to avoid letting 'future tokens' compete in the main
    self-attention stream.

    Shapes:
      act_h: [B, H_act, d_model]
      future_memory: [B, K_fut, future_dim]
    """

    def __init__(self, d_model: int, n_heads: int, future_dim: int, track_stats: bool = False):
        super().__init__()
        self.future_proj = nn.Linear(future_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.track_stats = bool(track_stats)
        self._last_stats: dict[str, float] = {}

    def forward(self, act_h: torch.Tensor, future_memory: torch.Tensor) -> torch.Tensor:
        kv = self.future_proj(future_memory)  # [B, K, d_model]
        ctx, attn = self.cross_attn(
            query=act_h,
            key=kv,
            value=kv,
            need_weights=self.track_stats,
            average_attn_weights=False if self.track_stats else True,
        )
        gate = torch.tanh(self.gate)
        out = act_h + gate * ctx
        if self.track_stats and attn is not None:
            # attn: [B, heads, Q, K] when average_attn_weights=False
            with torch.no_grad():
                p = attn.float().clamp_min(1e-8)
                ent = float((-(p * p.log()).sum(dim=-1)).mean().cpu().item())
                pmax = float(p.max(dim=-1).values.mean().cpu().item())
                self._last_stats = {
                    "attn_entropy": ent,
                    "attn_pmax": pmax,
                    "gate": float(gate.detach().cpu().item()),
                    "ctx_norm": float(ctx.float().norm(dim=-1).mean().cpu().item()),
                    "act_norm": float(act_h.float().norm(dim=-1).mean().cpu().item()),
                }
        return out

    def gate_value(self) -> float:
        return float(torch.tanh(self.gate).detach().cpu().item())

    def last_stats(self) -> dict[str, float]:
        return dict(self._last_stats)


class WorldPolicyTransformer(nn.Module):
    """Unified transformer for A/B/C and Phase-2 variants (B_cont, F).

    Sequence layouts:
      A:        [OBS] [ACT_Q1..ACT_QH]
      B/B_cont: [OBS] [FUT_Q1..FUT_QK] [ACT_Q1..ACT_QH]
      C:        [OBS] [WORLD_CUR] [FUT_Q1..FUT_QK] [ACT_Q1..ACT_QH]
      F:        [OBS] [ACT_Q1..ACT_QH]  (+ external future memory injected via cross-attn)
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
        # Model C knobs (kept for backward-compat)
        world_input_scale: float = 1.0,
        world_input_dropout: float = 0.0,
        world_input_layernorm: bool = False,
        block_world_to_action: bool = False,
        # Phase-2 knobs
        continuous_world: bool = False,
        world_target_dim: int = 0,
        enable_future_injection: bool = False,
        future_memory_dim: int = 0,
    ):
        super().__init__()
        assert model_type in ("A", "B", "B_cont", "C", "C_no_world_input", "F")
        self.model_type: ModelType = model_type
        self.cfg = cfg
        self.horizon = horizon
        self.future_horizon = future_horizon
        self.use_language = use_language

        # C-specific knobs
        self.world_input_scale = float(world_input_scale)
        self.world_input_dropout = float(world_input_dropout)
        self.world_input_layernorm = bool(world_input_layernorm)
        self.block_world_to_action = bool(block_world_to_action)

        # Phase-2 knobs
        self.continuous_world = bool(continuous_world)
        self.world_target_dim = int(world_target_dim) if continuous_world else 0
        self.enable_future_injection = bool(enable_future_injection)
        self.future_memory_dim = int(future_memory_dim) if enable_future_injection else 0

        d_model = cfg.d_model

        # Per-modality projections
        self.proj_img = nn.Linear(obs_dim, d_model)
        self.proj_state = nn.Linear(state_dim, d_model)
        self.proj_lang = nn.Linear(lang_dim, d_model) if use_language else None

        # World token embedding (only consumed as input for C; still needed for C output head)
        self.emb_world = nn.Embedding(world_vocab_size, d_model)

        # Learnable tokens
        self.lang_token = nn.Parameter(torch.zeros(1, 1, d_model)) if use_language else None
        self.obs_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.future_queries = nn.Parameter(torch.zeros(1, future_horizon, d_model))
        self.action_queries = nn.Parameter(torch.zeros(1, horizon, d_model))

        # Determine sequence length
        seq_len = 1 + horizon  # OBS + ACT_Q
        if use_language:
            seq_len += 1  # LANG
        if model_type == "C":
            seq_len += 1  # WORLD_CUR
        if model_type in ("B", "B_cont", "C", "C_no_world_input"):
            seq_len += future_horizon  # FUT_QUERY × K
        # F intentionally keeps the sequence minimal to avoid token competition.

        self.backbone = TransformerBackbone(cfg, seq_len=seq_len)

        # Heads
        self.action_head = nn.Linear(d_model, action_dim)
        if self.continuous_world:
            assert self.world_target_dim > 0
            self.world_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.world_target_dim),
            )
        else:
            self.world_head = nn.Linear(d_model, world_vocab_size)

        self.future_inject: Optional[GatedCrossAttention] = None
        if self.enable_future_injection:
            assert self.future_memory_dim > 0, "future_memory_dim must be set when enable_future_injection=True"
            self.future_inject = GatedCrossAttention(
                d_model=d_model,
                n_heads=cfg.n_heads,
                future_dim=self.future_memory_dim,
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.lang_token is not None:
            nn.init.normal_(self.lang_token, mean=0.0, std=0.02)
        nn.init.normal_(self.obs_token, mean=0.0, std=0.02)
        nn.init.normal_(self.future_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.action_queries, mean=0.0, std=0.02)

    def get_gate_value(self) -> Optional[float]:
        if self.future_inject is None:
            return None
        return self.future_inject.gate_value()

    def forward(
        self,
        img_emb: torch.Tensor,
        state: torch.Tensor,
        current_world_token: Optional[torch.Tensor] = None,
        lang_emb: Optional[torch.Tensor] = None,
        future_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward.

        Args:
          img_emb: [B, obs_dim]
          state:   [B, state_dim]
          current_world_token: [B] (C only)
          lang_emb: [B, lang_dim] (optional)
          future_memory: [B, K_fut, future_memory_dim] (F only; external memory)

        Returns:
          actions: [B, H, action_dim]
          world_pred:
            - discrete: [B, K_fut, vocab]
            - continuous: [B, K_fut, world_target_dim]
            - None if model_type doesn't include future queries and continuous_world=False
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
            tokens.append(world_cur.unsqueeze(1))  # [B, 1, D]

        if self.model_type in ("B", "B_cont", "C", "C_no_world_input"):
            fut_q = self.future_queries.expand(B, -1, -1)
            tokens.append(fut_q)

        act_q = self.action_queries.expand(B, -1, -1)
        tokens.append(act_q)

        x = torch.cat(tokens, dim=1)  # [B, L, D]

        attn_mask = None
        if self.block_world_to_action and self.model_type == "C":
            # Block ACT_Q positions from attending to WORLD_CUR.
            L = x.shape[1]
            act_start = L - self.horizon
            world_pos = (1 if self.use_language else 0) + 1  # LANG? + OBS
            mask = torch.zeros((L, L), dtype=torch.bool, device=x.device)
            mask[act_start : act_start + self.horizon, world_pos] = True
            attn_mask = mask

        h = self.backbone(x, attn_mask=attn_mask)

        # ACT_Q are always appended last.
        act_start = h.shape[1] - self.horizon
        act_h = h[:, act_start : act_start + self.horizon, :]

        # Optional: inject external future memory into ACT_Q only (Model F).
        if self.future_inject is not None and future_memory is not None:
            act_h = self.future_inject(act_h, future_memory)

        actions = self.action_head(act_h)

        world_out = None
        if self.model_type in ("B", "B_cont", "C", "C_no_world_input"):
            fut_start = act_start - self.future_horizon
            fut_h = h[:, fut_start : fut_start + self.future_horizon, :]  # [B, K, D]
            world_out = self.world_head(fut_h)

        return actions, world_out
