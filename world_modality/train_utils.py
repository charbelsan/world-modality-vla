from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ExperimentConfig, TrainingConfig
from .data_sr100 import SR100SequenceDataset
from .model import WorldPolicyTransformer


def create_dataloaders(
    data_cfg: DataConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders.

    If preload_to_gpu=True, all data is loaded into GPU memory at init time,
    which can significantly speed up training for small-ish datasets.
    """
    train_ds = SR100SequenceDataset(
        data_cfg, split=data_cfg.train_split, preload_to_gpu=data_cfg.preload_to_gpu
    )
    val_ds = SR100SequenceDataset(
        data_cfg, split=data_cfg.val_split, preload_to_gpu=data_cfg.preload_to_gpu
    )

    num_workers = 0 if data_cfg.preload_to_gpu else data_cfg.num_workers
    pin_memory = not data_cfg.preload_to_gpu  # No pinning needed if already on GPU

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model(
    exp_cfg: ExperimentConfig,
    img_emb_dim: int,
    state_dim: int,
    action_dim: int,
    world_vocab_size: int,
    use_language: bool = False,
    lang_dim: int = 0,
    world_input_scale: float = 1.0,
    world_input_dropout: float = 0.0,
    world_input_layernorm: bool = False,
    block_world_to_action: bool = False,
) -> WorldPolicyTransformer:
    return WorldPolicyTransformer(
        model_type=exp_cfg.training.model_type,
        cfg=exp_cfg.transformer,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=world_vocab_size,
        horizon=exp_cfg.data.action_horizon,
        future_horizon=exp_cfg.data.future_offset,
        use_language=use_language,
        lang_dim=lang_dim,
        world_input_scale=world_input_scale,
        world_input_dropout=world_input_dropout,
        world_input_layernorm=world_input_layernorm,
        block_world_to_action=block_world_to_action,
    )


def compute_action_loss(pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred_actions, target_actions)


def compute_world_loss(
    logits: Optional[torch.Tensor], target_tokens: Optional[torch.Tensor]
) -> torch.Tensor:
    if logits is None or target_tokens is None:
        return torch.tensor(0.0, device=logits.device if logits is not None else "cpu")
    # logits: [B, K, V], target_tokens: [B, K]
    if logits.dim() == 2:
        return torch.nn.functional.cross_entropy(logits, target_tokens)
    b, k, v = logits.shape
    logits_flat = logits.reshape(b * k, v)
    targets_flat = target_tokens.reshape(b * k)
    return torch.nn.functional.cross_entropy(logits_flat, targets_flat)


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_world_loss_continuous(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine loss for continuous future embeddings.

    pred/target: [B, K_fut, D] (float16/float32 ok)

    We compute the loss in float32 for numerical stability even if inputs are float16.
    """
    pred_f = pred.float()
    tgt_f = target.float()
    # Normalize to unit vectors to match cosine similarity semantics.
    pred_n = pred_f / (pred_f.norm(dim=-1, keepdim=True).clamp_min(eps))
    tgt_n = tgt_f / (tgt_f.norm(dim=-1, keepdim=True).clamp_min(eps))
    cos = (pred_n * tgt_n).sum(dim=-1)  # [B, K]
    return (1.0 - cos).mean()


@torch.no_grad()
def compute_world_cosine(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    pred_f = pred.float()
    tgt_f = target.float()
    pred_n = pred_f / (pred_f.norm(dim=-1, keepdim=True).clamp_min(eps))
    tgt_n = tgt_f / (tgt_f.norm(dim=-1, keepdim=True).clamp_min(eps))
    cos = (pred_n * tgt_n).sum(dim=-1)
    return float(cos.mean().cpu().item())


@torch.no_grad()
def compute_world_cosine_per_step(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> list[float]:
    """Compute cosine similarity per horizon step.

    This diagnostic shows if Prophet accuracy degrades for distant future steps.
    If step 1 is good but step K collapses, we know the horizon is too long.

    Args:
        pred: [B, K, D] predicted future embeddings
        target: [B, K, D] ground truth future embeddings

    Returns:
        List of K floats, one cosine similarity per step [step_1, step_2, ..., step_K]
    """
    pred_f = pred.float()
    tgt_f = target.float()
    pred_n = pred_f / (pred_f.norm(dim=-1, keepdim=True).clamp_min(eps))
    tgt_n = tgt_f / (tgt_f.norm(dim=-1, keepdim=True).clamp_min(eps))
    cos = (pred_n * tgt_n).sum(dim=-1)  # [B, K]
    # Average over batch, return per-step
    cos_per_step = cos.mean(dim=0)  # [K]
    return [float(c.cpu().item()) for c in cos_per_step]


def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create a linear warmup scheduler.

    Learning rate increases linearly from 0 to base_lr over warmup_steps,
    then remains constant.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)

