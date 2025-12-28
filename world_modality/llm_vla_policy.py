from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .action_flow import FlowActionHead
from .model import GatedCrossAttention


def build_act_tokens(horizon: int) -> List[str]:
    return [f"<ACT_{i}>" for i in range(horizon)]


def find_act_positions(input_ids: torch.Tensor, act_token_ids: Sequence[int]) -> torch.Tensor:
    """Find positions of <ACT_i> tokens for each sequence.

    Returns:
        positions: [B, H] with token indices for each ACT token.
    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [B, L]")
    bsz, _ = input_ids.shape
    horizon = len(act_token_ids)
    positions = torch.full((bsz, horizon), -1, dtype=torch.long)
    for j, tok_id in enumerate(act_token_ids):
        matches = input_ids == int(tok_id)
        if not torch.all(matches.any(dim=1)):
            raise ValueError(f"ACT token id {tok_id} missing from at least one prompt.")
        pos = matches.float().argmax(dim=1)
        positions[:, j] = pos
    return positions


class ActionHeadMLP(nn.Module):
    def __init__(self, d_model: int, action_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        h = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, h),
            nn.GELU(),
            nn.Linear(h, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProprioConditioner(nn.Module):
    """Inject proprioception into <ACT> hidden states via gated residual.

    This mirrors the Model-F safety principle:
    - proprio is external conditioning
    - injected only into action computation path
    - gate starts at 0 (no effect) and learns to open
    """

    def __init__(self, proprio_dim: int, hidden_size: int, gate_init: float = 0.0):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(proprio_dim),
            nn.Linear(proprio_dim, hidden_size),
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, act_h: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        # act_h: [B, H, D], proprio: [B, P]
        ctx = self.proj(proprio).unsqueeze(1)
        return act_h + torch.tanh(self.gate) * ctx

    def gate_value(self) -> float:
        return float(torch.tanh(self.gate).detach().cpu().item())


class QwenVLAWrapper(nn.Module):
    """Wrap a VLM and expose action decoding + optional future-memory injection."""

    def __init__(
        self,
        vlm: nn.Module,
        hidden_size: int,
        num_attention_heads: int,
        action_dim: int,
        horizon: int,
        future_dim: int,
        enable_future_injection: bool = True,
        enable_proprio: bool = False,
        proprio_dim: int = 0,
        action_head_type: str = "mse",
        flow_steps: int = 8,
    ):
        super().__init__()
        self.vlm = vlm
        self.horizon = horizon
        self.action_head_type = action_head_type
        self.flow_steps = flow_steps
        if action_head_type == "flow":
            self.action_head = FlowActionHead(hidden_size, action_dim)
        else:
            self.action_head = ActionHeadMLP(hidden_size, action_dim)
        self.proprio_conditioner = (
            ProprioConditioner(proprio_dim, hidden_size, gate_init=0.0)
            if enable_proprio and proprio_dim > 0
            else None
        )
        self.future_injection = (
            GatedCrossAttention(hidden_size, num_attention_heads, future_dim)
            if enable_future_injection
            else None
        )

    def predict_actions(self, act_h: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        if self.action_head_type == "flow":
            return self.action_head.sample(act_h, steps=steps or self.flow_steps)
        return self.action_head(act_h)

    def forward(
        self,
        model_inputs: dict,
        act_positions: torch.Tensor,
        future_memory: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        disable_future_injection: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.vlm(**model_inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        # Gather ACT hidden states.
        bsz, horizon = act_positions.shape
        hidden_size = hidden.size(-1)
        gather_index = act_positions.unsqueeze(-1).expand(bsz, horizon, hidden_size)
        act_h = hidden.gather(dim=1, index=gather_index)

        if self.proprio_conditioner is not None and proprio is not None:
            act_h = self.proprio_conditioner(act_h, proprio)

        if (
            self.future_injection is not None
            and future_memory is not None
            and not disable_future_injection
        ):
            act_h = self.future_injection(act_h, future_memory)

        actions = self.predict_actions(act_h)
        return actions, act_h

    def proprio_gate_value(self) -> float:
        if self.proprio_conditioner is None:
            return 0.0
        return self.proprio_conditioner.gate_value()

    def gate_value(self) -> float:
        if self.future_injection is None:
            return 0.0
        return self.future_injection.gate_value()
