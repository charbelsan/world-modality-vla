from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig
from .data_sr100 import SR100SequenceDataset
from .model import WorldPolicyTransformer
from .train_utils import compute_action_loss
from .config import TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation of world-modality models.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--image_key", type=str, default="rgb")
    parser.add_argument("--proprio_key", type=str, default="proprio")
    parser.add_argument("--action_key", type=str, default="action")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    meta = ckpt.get("meta", {})

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
        cache_dir=args.cache_dir,
        train_split=cfg.get("train_split", "train"),
        val_split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ds = SR100SequenceDataset(data_cfg, split=args.split)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    transformer_cfg = TransformerConfig()
    img_emb_dim = int(meta.get("img_emb_dim", 768))
    state_dim = int(meta.get("state_dim", ds[0]["obs_states"].shape[-1]))
    action_dim = int(meta.get("action_dim", ds[0]["actions"].shape[-1]))
    horizon = int(meta.get("action_horizon", data_cfg.action_horizon))
    future_horizon = int(meta.get("future_offset", data_cfg.future_offset))
    world_vocab_size = int(meta.get("world_vocab_size", 1024))
    model_type = meta.get("model_type", cfg.get("model_type", "A"))

    model = WorldPolicyTransformer(
        model_type=model_type,  # type: ignore
        cfg=transformer_cfg,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=world_vocab_size,
        horizon=horizon,
        future_horizon=future_horizon,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mse_list: List[float] = []
    world_correct = None
    world_total = None

    with torch.no_grad():
        for batch in loader:
            actions_gt = batch["actions"].to(device)
            states = batch["obs_states"][:, -1].to(device)
            img_ctx = batch["img_embeddings"][:, -1].to(device)
            future_tokens = batch["future_world_tokens"].to(device)
            current_tokens = batch["current_world_token"].to(device)

            current_world = current_tokens if model_type == "C" else None

            pred_actions, world_logits = model(
                img_emb=img_ctx, state=states, current_world_token=current_world
            )

            act_loss = compute_action_loss(pred_actions, actions_gt)
            mse_list.append(act_loss.item())

            if world_logits is not None:
                preds = world_logits.argmax(dim=-1)  # [B, K]
                if world_correct is None:
                    k = preds.shape[1]
                    world_correct = torch.zeros(k, dtype=torch.long)
                    world_total = torch.zeros(k, dtype=torch.long)
                eq = (preds == future_tokens).cpu()
                world_correct += eq.sum(dim=0)
                world_total += torch.tensor(eq.shape[0], dtype=torch.long).expand_as(world_total)

    mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
    print(f"Offline eval on split={args.split}: action MSE = {mean_mse:.6f}")

    if world_correct is not None and world_total is not None:
        acc_per_k = (world_correct.float() / world_total.float()).numpy()
        for k, acc in enumerate(acc_per_k, start=1):
            print(f"World-token top-1 accuracy at horizon k={k}: {acc:.4f}")


if __name__ == "__main__":
    main()

