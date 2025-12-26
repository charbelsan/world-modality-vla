from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class FlowActionHead(nn.Module):
    """Rectified-flow action head conditioned on ACT hidden states.

    We model actions with a simple flow-matching objective:
      x_t = (1 - t) * eps + t * x_1
      v* = x_1 - eps
      v_theta = f(act_h, x_t, t)

    Training minimizes ||v_theta - v*||^2.
    Inference integrates x_t forward from noise.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim

        self.act_proj = nn.Linear(d_model, hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim + time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, act_h: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity given ACT context, noisy actions, and time.

        Args:
            act_h: [B, H, d_model]
            x_t: [B, H, action_dim]
            t: [B, H, 1] in [0,1]
        """
        h = self.act_proj(act_h)
        t_emb = self.time_mlp(t)
        inp = torch.cat([h, x_t, t_emb], dim=-1)
        return self.net(inp)

    @torch.no_grad()
    def sample(self, act_h: torch.Tensor, steps: int = 8) -> torch.Tensor:
        """Integrate rectified flow from noise to action."""
        if steps <= 0:
            raise ValueError("steps must be >= 1")
        bsz, horizon, _ = act_h.shape
        device = act_h.device
        dtype = act_h.dtype
        x = torch.randn(bsz, horizon, self.action_dim, device=device, dtype=dtype)
        dt = 1.0 / float(steps)
        for i in range(steps):
            t = torch.full((bsz, horizon, 1), (i + 1) / float(steps), device=device, dtype=dtype)
            v = self.forward(act_h, x, t)
            x = x + dt * v
        return x
