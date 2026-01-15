# World Modality as Mixture-of-Experts (MoE)

This document proposes treating world injection as a **Mixture-of-Experts** problem with **loss-free load balancing** inspired by DeepSeek.

---

## 1. Motivation

### 1.1 Current Limitation: Scalar Gate

The current world injection uses a **single scalar gate** for all tokens:

```python
output = action_hidden + tanh(gate) * context
#                        └────┬────┘
#                    same value for ALL tokens
```

**Problems:**

| Issue | Description |
|-------|-------------|
| **No per-token control** | All action tokens get the same world influence, but some tokens may benefit more than others |
| **Binary thinking** | Gate is either "open" or "closed" - no nuance about *which* world information to use |
| **Single hypothesis** | Only one predicted future is used, but the future is uncertain |

### 1.2 Observation: Different Tokens Need Different World Influence

Consider a robot action chunk with 10 tokens:

```
Token 0: gripper_x      → World prediction helps (where will object be?)
Token 1: gripper_y      → World prediction helps
Token 2: gripper_z      → World prediction helps
Token 3: gripper_open   → World prediction helps (when to grasp?)
Token 4: arm_rotation   → World prediction somewhat helps
Token 5: arm_extension  → World prediction somewhat helps
Token 6: wrist_angle    → World prediction marginal
Token 7: base_x         → World prediction less relevant
Token 8: base_y         → World prediction less relevant
Token 9: wait_flag      → World prediction irrelevant/harmful
```

A scalar gate of 0.3 applies equally to all - suboptimal.

### 1.3 Observation: Multiple Futures Are Possible

The Prophet predicts a single future trajectory, but reality is stochastic:

```
Current state: Robot sees cup on table

Possible futures:
  - Future A: Cup stays still (user doesn't intervene)
  - Future B: Cup is moved by user
  - Future C: Another object enters scene

Prophet outputs ONE prediction, which might not match what actually happens.
```

An **imagination bank** with multiple hypotheses could help, but how does the policy know which hypothesis to trust?

---

## 2. The Idea: World as MoE

### 2.1 Core Concept

Reframe world injection as a **routing problem**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   Instead of:  output = h + gate * world_context                    │
│                                                                     │
│   Think of it as:                                                   │
│                                                                     │
│   Expert 0: Identity      → output = h           (ignore world)     │
│   Expert 1: World-Aug     → output = h + ctx     (use world)        │
│                                                                     │
│   Router decides per-token: [w₀, w₁] = how much of each expert      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Extension to Imagination Bank

With N future hypotheses, we have N+1 experts:

```
Expert 0: Identity (baseline, no world)
Expert 1: Cross-attention to Hypothesis 1
Expert 2: Cross-attention to Hypothesis 2
...
Expert N: Cross-attention to Hypothesis N

Router learns: "For token t, which expert(s) are most useful?"
```

### 2.3 Why MoE Framing Helps

| Benefit | Explanation |
|---------|-------------|
| **Per-token routing** | Each action token can choose its own world influence |
| **Hypothesis selection** | Router selects most relevant predicted future |
| **Graceful degradation** | If world predictions are bad, route to Expert 0 |
| **Sparse computation** | Top-k routing avoids computing all experts |

---

## 3. The Solution

### 3.1 Architecture Overview

```
                         IMAGINATION BANK (from Prophet)
            ┌─────────────────────────────────────────────────┐
            │  Hyp 1 [B,K,D]   Hyp 2 [B,K,D]   Hyp 3 [B,K,D]  │
            └────────┬────────────────┬────────────────┬──────┘
                     │                │                │
                     ▼                ▼                ▼
              ┌───────────┐    ┌───────────┐    ┌───────────┐
              │  Expert 1 │    │  Expert 2 │    │  Expert 3 │
              │ CrossAttn │    │ CrossAttn │    │ CrossAttn │
              └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                    │                │                │
       ┌────────────┴────────────────┴────────────────┴────────────┐
       │                                                            │
       │    ┌───────────┐                                           │
       │    │  Expert 0 │  (Identity - baseline path)               │
       │    │  h → h    │                                           │
       │    └─────┬─────┘                                           │
       │          │                                                 │
       │          ▼                                                 │
       │    ┌─────────────────────────────────────────────────┐     │
       │    │                   ROUTER                        │     │
       │    │                                                 │     │
       │    │  action_hidden [B, H, D]                        │     │
       │    │         │                                       │     │
       │    │         ▼                                       │     │
       │    │  router_logits = Linear(h) + expert_bias        │     │
       │    │         │                                       │     │
       │    │         ▼                                       │     │
       │    │  weights = softmax(logits)  [B, H, N+1]         │     │
       │    │                                                 │     │
       │    └─────────────────────────────────────────────────┘     │
       │                         │                                  │
       └─────────────────────────┼──────────────────────────────────┘
                                 │
                                 ▼
                    output = Σᵢ wᵢ * expertᵢ(h)
```

### 3.2 Loss-Free Load Balancing (DeepSeek Style)

Traditional MoE uses auxiliary losses to prevent expert collapse:

```python
L_total = L_task + λ * L_balance  # λ is annoying to tune
```

**DeepSeek's insight**: Adjust router bias dynamically without gradient:

```python
# Track expert usage with EMA
usage_ema = 0.99 * usage_ema + 0.01 * current_usage

# Adjust bias to encourage balance (no gradient, just heuristic)
target_usage = 1.0 / n_experts
expert_bias += lr_bias * (target_usage - usage_ema)
```

This achieves load balancing **without interfering with the task loss**.

### 3.3 Do-No-Harm Initialization

To preserve the baseline behavior at initialization:

```python
# Expert 0 (identity) starts strongly favored
expert_bias = [1.0, 0.0, 0.0, 0.0]  # For N=3 hypotheses

# This means:
# - At step 0, router outputs ~[0.7, 0.1, 0.1, 0.1] (favors baseline)
# - Model starts behaving like E0 baseline
# - As training progresses, bias adjusts and world experts can be selected
```

### 3.4 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldMoE(nn.Module):
    """
    World injection as Mixture-of-Experts with loss-free load balancing.

    Expert 0: Identity (baseline, ignore world)
    Expert 1..N: Cross-attention to N different world hypotheses
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        future_dim: int,
        n_hypotheses: int = 1,
        top_k: int = 2,
        balance_lr: float = 0.001,
        baseline_bias_init: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_hypotheses = n_hypotheses
        self.n_experts = 1 + n_hypotheses  # Expert 0 = identity
        self.top_k = min(top_k, self.n_experts)
        self.balance_lr = balance_lr

        # Cross-attention for world experts (shared or separate)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            kdim=future_dim,
            vdim=future_dim,
            batch_first=True,
        )

        # Project future_dim to d_model if different
        self.future_proj = nn.Linear(future_dim, d_model) if future_dim != d_model else nn.Identity()

        # Router
        self.router = nn.Linear(d_model, self.n_experts)

        # Expert bias (for loss-free balancing)
        self.expert_bias = nn.Parameter(torch.zeros(self.n_experts))
        self.expert_bias.data[0] = baseline_bias_init  # Favor baseline initially

        # Usage tracking (not a parameter, just a buffer)
        self.register_buffer('usage_ema', torch.ones(self.n_experts) / self.n_experts)

    def forward(
        self,
        action_hidden: torch.Tensor,      # [B, H, D]
        world_hypotheses: torch.Tensor,   # [B, N, K, D_fut] or [B, K, D_fut] if N=1
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            action_hidden: Action token representations [B, H, D]
            world_hypotheses: Predicted future(s) [B, N, K, D_fut] or [B, K, D_fut]

        Returns:
            output: Routed output [B, H, D]
            metrics: Dict with routing statistics
        """
        B, H, D = action_hidden.shape

        # Handle single hypothesis case
        if world_hypotheses.dim() == 3:
            world_hypotheses = world_hypotheses.unsqueeze(1)  # [B, 1, K, D_fut]

        N = world_hypotheses.shape[1]
        assert N == self.n_hypotheses, f"Expected {self.n_hypotheses} hypotheses, got {N}"

        # Project futures to d_model
        world_hypotheses = self.future_proj(world_hypotheses)  # [B, N, K, D]

        # === Compute router weights ===
        router_logits = self.router(action_hidden) + self.expert_bias  # [B, H, n_experts]
        router_weights = F.softmax(router_logits, dim=-1)  # [B, H, n_experts]

        # === Compute expert outputs ===
        # Expert 0: Identity
        expert_outputs = [action_hidden]  # List of [B, H, D]

        # Expert 1..N: Cross-attention to each hypothesis
        for i in range(N):
            mem_i = world_hypotheses[:, i, :, :]  # [B, K, D]
            ctx_i, _ = self.cross_attn(
                query=action_hidden,
                key=mem_i,
                value=mem_i,
            )  # [B, H, D]
            expert_outputs.append(action_hidden + ctx_i)

        # Stack: [B, H, n_experts, D]
        expert_outputs = torch.stack(expert_outputs, dim=2)

        # === Weighted combination ===
        # output[b,h,d] = sum_e weights[b,h,e] * expert_outputs[b,h,e,d]
        output = torch.einsum('bhe,bhed->bhd', router_weights, expert_outputs)

        # === Loss-free load balancing ===
        metrics = self._update_balance(router_weights)

        return output, metrics

    @torch.no_grad()
    def _update_balance(self, weights: torch.Tensor) -> dict:
        """
        DeepSeek-style loss-free load balancing.
        Adjusts expert_bias to encourage balanced usage.
        """
        # Current usage: average weight per expert across batch and tokens
        current_usage = weights.mean(dim=[0, 1])  # [n_experts]

        # EMA update
        self.usage_ema = 0.99 * self.usage_ema + 0.01 * current_usage

        # Target: uniform usage
        target = 1.0 / self.n_experts

        # Compute adjustment
        adjustment = self.balance_lr * (target - self.usage_ema)

        # Slower adjustment for baseline expert (conservative)
        adjustment[0] *= 0.5

        # Apply adjustment
        self.expert_bias.data += adjustment

        # Compute metrics
        metrics = {
            'moe_usage_expert0': float(self.usage_ema[0].item()),
            'moe_usage_world_avg': float(self.usage_ema[1:].mean().item()) if self.n_experts > 1 else 0.0,
            'moe_bias_expert0': float(self.expert_bias[0].item()),
            'moe_entropy': float(-(self.usage_ema * (self.usage_ema + 1e-8).log()).sum().item()),
        }

        # Per-hypothesis usage
        for i in range(self.n_hypotheses):
            metrics[f'moe_usage_hyp{i+1}'] = float(self.usage_ema[i+1].item())

        return metrics


class WorldMoETopK(WorldMoE):
    """
    Top-K variant: only compute selected experts for efficiency.
    """

    def forward(
        self,
        action_hidden: torch.Tensor,
        world_hypotheses: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        B, H, D = action_hidden.shape

        if world_hypotheses.dim() == 3:
            world_hypotheses = world_hypotheses.unsqueeze(1)

        N = world_hypotheses.shape[1]
        world_hypotheses = self.future_proj(world_hypotheses)

        # === Router with Top-K ===
        router_logits = self.router(action_hidden) + self.expert_bias  # [B, H, n_experts]

        # Top-K selection
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B, H, K]
        topk_weights = F.softmax(topk_weights, dim=-1)  # Normalize among selected

        # === Sparse expert computation ===
        output = torch.zeros_like(action_hidden)

        # This is simplified - production code would batch by expert
        for k in range(self.top_k):
            expert_idx = topk_indices[:, :, k]  # [B, H]
            weight_k = topk_weights[:, :, k:k+1]  # [B, H, 1]

            # Compute expert output for each sample
            for e in range(self.n_experts):
                mask = (expert_idx == e)  # [B, H]
                if not mask.any():
                    continue

                if e == 0:
                    # Expert 0: Identity
                    expert_out = action_hidden
                else:
                    # Expert e: Cross-attention to hypothesis e-1
                    mem = world_hypotheses[:, e-1, :, :]
                    ctx, _ = self.cross_attn(action_hidden, mem, mem)
                    expert_out = action_hidden + ctx

                output = output + weight_k * expert_out * mask.unsqueeze(-1).float()

        # === Load balancing on full distribution ===
        full_weights = F.softmax(router_logits, dim=-1)
        metrics = self._update_balance(full_weights)

        return output, metrics
```

---

## 4. Experimental Plan

### 4.1 Phased Approach

```
Phase 1: Per-Token Gating (N=1, simplest MoE)
─────────────────────────────────────────────────────────────────────
Goal: Validate that per-token routing helps vs scalar gate
Config:
  - n_hypotheses = 1 (single Prophet prediction)
  - n_experts = 2 (Expert 0 = identity, Expert 1 = world)
  - top_k = 2 (compute both, soft routing)

Compare: E2 (scalar gate) vs E2-MoE (per-token routing)
Metrics: SR, per-token routing weights visualization


Phase 2: Imagination Bank (N=4)
─────────────────────────────────────────────────────────────────────
Goal: Test if multiple hypotheses help
Config:
  - n_hypotheses = 4
  - n_experts = 5
  - top_k = 2 (sparse for efficiency)
  - Generate hypotheses via dropout sampling in Prophet

Compare: E2-MoE-N1 vs E2-MoE-N4
Metrics: SR, which hypotheses get selected, usage entropy


Phase 3: Loss-Free Balancing Ablation
─────────────────────────────────────────────────────────────────────
Goal: Verify DeepSeek-style balancing works
Config:
  - Same as Phase 2
  - Ablate: balance_lr = 0 (no balancing) vs balance_lr = 0.001

Compare: With vs without load balancing
Metrics: Usage histogram, expert collapse detection
```

### 4.2 Configuration Options

Add to `SmolVLAWorldConfig`:

```python
# World MoE settings
world_injection_type: str = "gate"  # "gate" | "moe" | "moe_topk"
world_moe_n_hypotheses: int = 1
world_moe_top_k: int = 2
world_moe_balance_lr: float = 0.001
world_moe_baseline_bias_init: float = 1.0
```

### 4.3 Generating Multiple Hypotheses

Modify Prophet to output N samples:

```python
class Prophet(nn.Module):
    def forward(self, z_hist, n_samples=1, sample_dropout=True):
        """
        Generate N future hypotheses.

        If n_samples > 1 and sample_dropout=True:
          - Keep dropout enabled
          - Run forward N times with different dropout masks
          - Return [B, N, K, D]
        """
        if n_samples == 1:
            return self._forward_single(z_hist)  # [B, K, D]

        hypotheses = []
        for _ in range(n_samples):
            if sample_dropout:
                # Dropout creates diversity
                h = self._forward_single(z_hist)  # [B, K, D]
            else:
                # Add noise to query slots for diversity
                h = self._forward_single(z_hist, noise_scale=0.1)
            hypotheses.append(h)

        return torch.stack(hypotheses, dim=1)  # [B, N, K, D]
```

### 4.4 Metrics to Track

```python
# Add to training loop
metrics = {
    # Standard
    "loss_action": ...,
    "loss_world": ...,
    "world_gate": ...,  # For comparison with scalar gate

    # MoE-specific
    "moe_usage_expert0": ...,      # How often baseline is chosen
    "moe_usage_world_avg": ...,    # Average world expert usage
    "moe_entropy": ...,            # Routing entropy (uniform = high)
    "moe_bias_expert0": ...,       # Current baseline bias

    # Per-hypothesis (if N > 1)
    "moe_usage_hyp1": ...,
    "moe_usage_hyp2": ...,
    ...
}
```

### 4.5 Expected Outcomes

```
If MoE routing helps:
─────────────────────────────────────────────────────────────────────
✓ SR improves vs scalar gate E2
✓ Usage shows differentiation (some tokens use world more)
✓ Entropy decreases over training (routing becomes decisive)
✓ Expert 0 usage < 1.0 (world is being used)

If imagination bank helps:
─────────────────────────────────────────────────────────────────────
✓ SR improves with N=4 vs N=1
✓ Different hypotheses get selected for different samples
✓ Attention patterns show "spiky" selection (pmax high)

If it fails:
─────────────────────────────────────────────────────────────────────
✗ Expert 0 usage ≈ 1.0 (world ignored despite routing)
✗ All hypotheses have equal usage (no meaningful selection)
✗ SR unchanged or worse vs E2
```

---

## 5. Comparison with Alternatives

| Approach | Per-Token | Multi-Hyp | Load Balance | Complexity |
|----------|-----------|-----------|--------------|------------|
| Scalar gate | No | No | N/A | Lowest |
| Vector gate [D] | Partial | No | N/A | Low |
| Vector gate [H] | Yes | No | N/A | Low |
| **MoE (this doc)** | Yes | Yes | Loss-free | Medium |
| Full attention | Yes | Yes | N/A | High |

---

## 6. References

- DeepSeek-MoE: Towards Ultimate Expert Specialization (2024)
- Switch Transformers: Scaling to Trillion Parameter Models (2021)
- Mixture-of-Experts Meets Instruction Tuning (2023)
- FLARE: World Models with Efficient Multimodal Transformers (2024)

---

## 7. Next Steps

1. [ ] Implement `WorldMoE` module in `world_modality/model.py`
2. [ ] Add config options to `SmolVLAWorldConfig`
3. [ ] Modify Prophet for multi-hypothesis generation
4. [ ] Run Phase 1 experiments (N=1 per-token routing)
5. [ ] Analyze routing patterns across action tokens
6. [ ] If Phase 1 positive, proceed to Phase 2 (imagination bank)
