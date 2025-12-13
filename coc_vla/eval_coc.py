from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from world_modality.config import DataConfig, TransformerConfig
from coc_vla.data import CoCSR100Dataset
from coc_vla.model import WorldCoCTransformer
from world_modality.train_utils import compute_action_loss, compute_world_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-head CoC VLA model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--coc_jsonl", type=str, required=True)
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

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    ds = CoCSR100Dataset(data_cfg, coc_jsonl=args.coc_jsonl, split="train")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    transformer_cfg = TransformerConfig()

    # Infer dims.
    batch = next(iter(loader))
    obs_states = batch["obs_states"]
    actions = batch["actions"]
    img_emb = batch["img_embeddings"]
    state_dim = obs_states.shape[-1]
    action_dim = actions.shape[-1]
    img_emb_dim = img_emb.shape[-1]

    world_vocab_size = int(cfg.get("world_vocab_size", 1024))
    horizon = int(cfg.get("action_horizon", 8))
    future_horizon = int(cfg.get("future_offset", 8))
    coc_vocab_size = int(cfg.get("coc_vocab_size", 32000))

    model = WorldCoCTransformer(
        model_type="C",  # full world-modality
        trunk_cfg=transformer_cfg,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=world_vocab_size,
        horizon=horizon,
        future_horizon=future_horizon,
        coc_vocab_size=coc_vocab_size,
        coc_d_model=transformer_cfg.d_model,
        coc_n_layers=4,
        coc_n_heads=transformer_cfg.n_heads,
        coc_dropout=transformer_cfg.dropout,
        coc_max_len=128,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    act_losses = []
    world_losses = []

    with torch.no_grad():
        for batch in loader:
            actions_gt = batch["actions"].to(device).float()
            states = batch["obs_states"][:, -1].to(device).float()
            img_ctx = batch["img_embeddings"][:, -1].to(device).float()
            future_tokens = batch["future_world_tokens"].to(device)
            current_tokens = batch["current_world_token"].to(device)

            # CoC is not evaluated quantitatively here yet; this is a place-holder.
            actions_pred, world_logits, _ = model(
                img_emb=img_ctx,
                state=states,
                current_world_token=current_tokens,
                coc_input_ids=None,
            )

            act_loss = compute_action_loss(actions_pred, actions_gt)
            world_loss = compute_world_loss(world_logits, future_tokens)

            act_losses.append(act_loss.item())
            world_losses.append(world_loss.item())

    print("=== CoC VLA Evaluation ===")
    print(f"Action MSE:  {float(np.mean(act_losses)):.6f}")
    print(f"World CE:    {float(np.mean(world_losses)):.6f}")
    print("(CoC quality needs an LLM-as-judge or human eval; not reported here.)")


if __name__ == "__main__":
    main()

