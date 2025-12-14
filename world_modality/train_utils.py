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
    train_ds = SR100SequenceDataset(data_cfg, split=data_cfg.train_split)
    val_ds = SR100SequenceDataset(data_cfg, split=data_cfg.val_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
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
