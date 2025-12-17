from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, TransformerConfig
from .data_sr100 import SR100SequenceDataset
from .model import WorldPolicyTransformer
from .train_utils import compute_action_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation of world-modality models.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--image_key", type=str, default="rgb")
    parser.add_argument("--proprio_key", type=str, default="proprio")
    parser.add_argument("--action_key", type=str, default="action")
    parser.add_argument("--use_language", action="store_true", help="Enable instruction conditioning.")
    parser.add_argument("--instruction_key", type=str, default="instruction")
    parser.add_argument("--episode_id_key", type=str, default="episode_id")
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

    # Use checkpoint config for context_frames, action_horizon, future_offset
    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
        instruction_key=args.instruction_key,
        episode_id_key=args.episode_id_key,
        use_language=bool(meta.get("use_language", cfg.get("use_language", args.use_language))),
        cache_dir=args.cache_dir,
        train_split=cfg.get("train_split", "train"),
        val_split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        context_frames=cfg.get("context_frames", 3),
        action_horizon=cfg.get("action_horizon", 8),
        future_offset=cfg.get("future_offset", 8),
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
    use_language = bool(meta.get("use_language", cfg.get("use_language", args.use_language)))
    lang_dim = int(meta.get("lang_dim", 0))
    state_dim = int(meta.get("state_dim", ds[0]["obs_states"].shape[-1]))
    action_dim = int(meta.get("action_dim", ds[0]["actions"].shape[-1]))
    horizon = int(meta.get("action_horizon", data_cfg.action_horizon))
    future_horizon = int(meta.get("future_offset", data_cfg.future_offset))
    world_vocab_size = int(meta.get("world_vocab_size", 1024))
    model_type = meta.get("model_type", cfg.get("model_type", "A"))
    world_input_scale = float(meta.get("world_input_scale", cfg.get("world_input_scale", 1.0)))
    world_input_dropout = float(meta.get("world_input_dropout", cfg.get("world_input_dropout", 0.0)))
    world_input_layernorm = bool(meta.get("world_input_layernorm", cfg.get("world_input_layernorm", False)))
    block_world_to_action = bool(meta.get("block_world_to_action", cfg.get("block_world_to_action", False)))

    model = WorldPolicyTransformer(
        model_type=model_type,  # type: ignore
        cfg=transformer_cfg,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=world_vocab_size,
        horizon=horizon,
        future_horizon=future_horizon,
        use_language=use_language,
        lang_dim=lang_dim,
        world_input_scale=world_input_scale,
        world_input_dropout=world_input_dropout,
        world_input_layernorm=world_input_layernorm,
        block_world_to_action=block_world_to_action,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mse_list: List[float] = []
    # Top-1 and Top-5 accuracy per horizon.
    world_top1_correct = None
    world_top5_correct = None
    world_total = None

    with torch.no_grad():
        for batch in loader:
            actions_gt = batch["actions"].to(device).float()
            states = batch["obs_states"][:, -1].to(device).float()
            img_ctx = batch["img_embeddings"][:, -1].to(device).float()
            lang_emb = batch.get("instruction_embeddings")
            if use_language:
                assert lang_emb is not None
                lang_emb = lang_emb.to(device).float()
            future_tokens = batch["future_world_tokens"].to(device)
            current_tokens = batch["current_world_token"].to(device)

            # Only C uses world token as input.
            current_world = current_tokens if model_type == "C" else None

            pred_actions, world_logits = model(
                img_emb=img_ctx,
                state=states,
                current_world_token=current_world,
                lang_emb=lang_emb if use_language else None,
            )

            act_loss = compute_action_loss(pred_actions, actions_gt)
            mse_list.append(act_loss.item())

            if world_logits is not None:
                # world_logits: [B, K, V]
                B, K, V = world_logits.shape

                if world_top1_correct is None:
                    world_top1_correct = torch.zeros(K, dtype=torch.long)
                    world_top5_correct = torch.zeros(K, dtype=torch.long)
                    world_total = torch.zeros(K, dtype=torch.long)

                # Top-1 accuracy.
                preds_top1 = world_logits.argmax(dim=-1)  # [B, K]
                eq_top1 = (preds_top1 == future_tokens).cpu()
                world_top1_correct += eq_top1.sum(dim=0)

                # Top-5 accuracy.
                _, preds_top5 = world_logits.topk(min(5, V), dim=-1)  # [B, K, 5]
                future_tokens_expanded = future_tokens.unsqueeze(-1)  # [B, K, 1]
                eq_top5 = (preds_top5 == future_tokens_expanded).any(dim=-1).cpu()  # [B, K]
                world_top5_correct += eq_top5.sum(dim=0)

                world_total += torch.tensor(B, dtype=torch.long).expand(K)

    # Print results.
    print("\n" + "=" * 50)
    print(f"=== Model {model_type} Evaluation ===")
    print("=" * 50)

    mean_mse = float(np.mean(mse_list)) if mse_list else 0.0
    print(f"Action MSE: {mean_mse:.6f}")

    if world_top1_correct is not None and world_total is not None:
        # Overall accuracy (across all horizons).
        overall_top1 = world_top1_correct.sum().float() / world_total.sum().float()
        overall_top5 = world_top5_correct.sum().float() / world_total.sum().float()

        print(f"\nWorld Token Top-1 Accuracy: {overall_top1.item():.4f}")
        print(f"World Token Top-5 Accuracy: {overall_top5.item():.4f}")

        # Per-horizon accuracy.
        top1_per_k = (world_top1_correct.float() / world_total.float()).numpy()
        top5_per_k = (world_top5_correct.float() / world_total.float()).numpy()

        print(
            "\nPer-horizon Top-1: "
            + ", ".join([f"k={k+1}: {acc:.3f}" for k, acc in enumerate(top1_per_k)])
        )
        print(
            "Per-horizon Top-5: "
            + ", ".join([f"k={k+1}: {acc:.3f}" for k, acc in enumerate(top5_per_k)])
        )

    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()