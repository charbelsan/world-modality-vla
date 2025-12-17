#!/usr/bin/env python3
"""
Synthetic training test for all model types.
Run this to verify the training loop works for all models.

Usage:
    # CPU test
    python scripts/test_all_models.py --device cpu

    # GPU test
    python scripts/test_all_models.py --device cuda

    # Auto (CUDA if healthy, else CPU)
    python scripts/test_all_models.py --device auto

    # Docker GPU test
    docker run --rm --gpus all -v $(pwd):/workspace -w /workspace \
        pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime \
        python scripts/test_all_models.py --device cuda
"""

import argparse
import gc
import os
import sys

# Allow running as `python scripts/test_all_models.py` without installing the package.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from world_modality.model import Prophet, WorldPolicyTransformer
from world_modality.config import TransformerConfig
from world_modality.device import resolve_device
from world_modality.train_utils import (
    compute_action_loss,
    compute_world_loss,
    compute_world_loss_continuous,
)


def test_model(model_type: str, device: torch.device, cfg: TransformerConfig, dims: dict):
    """Test a single model type with forward + backward pass."""
    B = dims["batch"]
    obs_dim = dims["obs"]
    state_dim = dims["state"]
    action_dim = dims["action"]
    horizon = dims["horizon"]
    future_horizon = dims["future_horizon"]
    world_vocab = dims["world_vocab"]

    # Create test data
    img = torch.randn(B, obs_dim, device=device)
    state = torch.randn(B, state_dim, device=device)
    actions_gt = torch.randn(B, horizon, action_dim, device=device)
    future_tokens = torch.randint(0, world_vocab, (B, future_horizon), device=device)
    future_emb = torch.randn(B, future_horizon, obs_dim, device=device)
    current_token = torch.randint(0, world_vocab, (B,), device=device)

    # Create model based on type
    if model_type == "A":
        model = WorldPolicyTransformer(
            model_type="A", cfg=cfg, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=action_dim, world_vocab_size=world_vocab,
            horizon=horizon, future_horizon=future_horizon
        ).to(device)
        pred, _ = model(img, state)
        loss = compute_action_loss(pred, actions_gt)

    elif model_type == "B":
        model = WorldPolicyTransformer(
            model_type="B", cfg=cfg, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=action_dim, world_vocab_size=world_vocab,
            horizon=horizon, future_horizon=future_horizon
        ).to(device)
        pred, wl = model(img, state)
        loss = compute_action_loss(pred, actions_gt) + 0.2 * compute_world_loss(wl, future_tokens)

    elif model_type == "B_cont":
        model = WorldPolicyTransformer(
            model_type="B_cont", cfg=cfg, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=action_dim, world_vocab_size=world_vocab,
            horizon=horizon, future_horizon=future_horizon,
            continuous_world=True, world_target_dim=obs_dim
        ).to(device)
        pred, wp = model(img, state)
        loss = compute_action_loss(pred, actions_gt) + 0.2 * compute_world_loss_continuous(wp, future_emb)

    elif model_type == "C":
        model = WorldPolicyTransformer(
            model_type="C", cfg=cfg, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=action_dim, world_vocab_size=world_vocab,
            horizon=horizon, future_horizon=future_horizon
        ).to(device)
        pred, wl = model(img, state, current_world_token=current_token)
        loss = compute_action_loss(pred, actions_gt) + 0.2 * compute_world_loss(wl, future_tokens)

    elif model_type == "F":
        model = WorldPolicyTransformer(
            model_type="F", cfg=cfg, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=action_dim, world_vocab_size=world_vocab,
            horizon=horizon, future_horizon=future_horizon,
            enable_future_injection=True, future_memory_dim=obs_dim
        ).to(device)
        prophet = Prophet(
            emb_dim=obs_dim, hidden_dim=cfg.d_model,
            future_horizon=future_horizon, n_layers=1, n_heads=cfg.n_heads
        ).to(device)
        z_hist = torch.randn(B, 3, obs_dim, device=device)
        z_pred = prophet(z_hist)
        pred, _ = model(img, state, future_memory=z_pred)
        loss = compute_action_loss(pred, actions_gt) + 0.2 * compute_world_loss_continuous(z_pred, future_emb)
        gate_val = model.get_gate_value()

        # Backward
        loss.backward()
        print(f"Model F: OK (loss={loss.item():.4f}, gate={gate_val:.4f})")
        del model, prophet
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return True
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Backward pass
    loss.backward()
    print(f"Model {model_type}: OK (loss={loss.item():.4f})")

    # Cleanup
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return True


def main():
    parser = argparse.ArgumentParser(description="Test all model types")
    parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"=== Testing All Models on {device.type.upper()} ===")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Small config for limited GPU
    cfg = TransformerConfig(d_model=128, n_layers=2, n_heads=4, dropout=0.0, norm_first=True)
    dims = {
        "batch": args.batch,
        "obs": 384,
        "state": 14,
        "action": 7,
        "horizon": 4,
        "future_horizon": 2,
        "world_vocab": 128,
    }

    model_types = ["A", "B", "B_cont", "C", "F"]
    all_passed = True

    for model_type in model_types:
        try:
            test_model(model_type, device, cfg, dims)
        except Exception as e:
            print(f"Model {model_type}: FAILED ({e})")
            all_passed = False

    print()
    if all_passed:
        print("=== ALL TESTS PASSED ===")
    else:
        print("=== SOME TESTS FAILED ===")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
