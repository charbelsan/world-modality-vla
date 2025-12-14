from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .config import (
    DataConfig,
    ExperimentConfig,
    TrainingConfig,
    TransformerConfig,
    VQConfig,
    VisionConfig,
)
from .train_utils import (
    build_model,
    compute_action_loss,
    compute_world_loss,
    create_dataloaders,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train world-modality models (A/B/C) on SR100.")
    parser.add_argument("--model_type", type=str, choices=["A", "B", "C", "C_no_world_input"], default="A")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--image_key", type=str, default="rgb", help="Key for image observation in LeRobotDataset.")
    parser.add_argument("--proprio_key", type=str, default="proprio", help="Key for proprio state in LeRobotDataset.")
    parser.add_argument("--action_key", type=str, default="action", help="Key for action in LeRobotDataset.")
    parser.add_argument("--use_language", action="store_true", help="Enable instruction conditioning (VLA).")
    parser.add_argument("--instruction_key", type=str, default="instruction", help="Key for instruction text.")
    parser.add_argument("--episode_id_key", type=str, default="episode_id", help="Key for episode id grouping.")
    parser.add_argument(
        "--text_model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Text encoder used to precompute instruction embeddings (saved into checkpoint meta).",
    )
    parser.add_argument(
        "--text_max_length",
        type=int,
        default=64,
        help="Max token length for the text encoder used in precompute (saved into checkpoint meta).",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--context_frames", type=int, default=3)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument(
        "--future_offset",
        type=int,
        default=8,
        help="Max future horizon K for world tokens (predict w_{t+1..t+K}).",
    )
    parser.add_argument("--world_vocab_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lambda_world_loss", type=float, default=0.2)
    parser.add_argument("--warmup_steps", type=int, default=0, help="Linear warmup steps (0 = no warmup)")
    parser.add_argument("--gradient_clip", type=float, default=0.0, help="Max gradient norm (0 = no clipping)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="val")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="world-modality-sr100")
    return parser.parse_args()


def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create a linear warmup scheduler."""

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def save_config_json(args: argparse.Namespace, log_dir: str):
    """Save experiment config to JSON for reproducibility."""
    config = {
        "model_type": args.model_type,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "warmup_steps": args.warmup_steps,
        "gradient_clip": args.gradient_clip,
        "lambda_world_loss": args.lambda_world_loss,
        "world_vocab_size": args.world_vocab_size,
        "context_frames": args.context_frames,
        "action_horizon": args.action_horizon,
        "future_offset": args.future_offset,
        "dataset_name": args.dataset_name,
        "image_key": args.image_key,
        "proprio_key": args.proprio_key,
        "action_key": args.action_key,
        "use_language": args.use_language,
        "instruction_key": args.instruction_key,
        "episode_id_key": args.episode_id_key,
        "text_model_name": args.text_model_name,
        "text_max_length": args.text_max_length,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = os.path.join(log_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
        instruction_key=args.instruction_key,
        episode_id_key=args.episode_id_key,
        use_language=args.use_language,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        action_horizon=args.action_horizon,
        future_offset=args.future_offset,
        train_split=args.train_split,
        val_split=args.val_split,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )

    vision_cfg = VisionConfig()
    vq_cfg = VQConfig(num_tokens=args.world_vocab_size)
    transformer_cfg = TransformerConfig()
    training_cfg = TrainingConfig(
        model_type=args.model_type,  # type: ignore
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        lambda_world_loss=args.lambda_world_loss,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        log_wandb=args.log_wandb,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )

    exp_cfg = ExperimentConfig(
        data=data_cfg,
        vision=vision_cfg,
        vq=vq_cfg,
        transformer=transformer_cfg,
        training=training_cfg,
    )

    os.makedirs(args.log_dir, exist_ok=True)

    # Save config JSON for reproducibility.
    save_config_json(args, args.log_dir)

    train_loader, val_loader = create_dataloaders(exp_cfg.data)

    # Infer dimensionalities from a single batch.
    batch = next(iter(train_loader))
    # Expect shapes: obs_states [B, T_ctx, D_s], actions [B, H, D_a],
    # img_embeddings [B, T_ctx, d_e], current_world_token [B], future_world_tokens [B, K]
    obs_states = batch["obs_states"]
    actions = batch["actions"]
    state_dim = obs_states.shape[-1]
    action_dim = actions.shape[-1]
    if "img_embeddings" not in batch:
        raise ValueError(
            "SR100SequenceDataset must be created with load_embeddings=True to provide "
            "precomputed image embeddings."
        )
    img_emb_dim = batch["img_embeddings"].shape[-1]
    lang_dim = 0
    if args.use_language:
        if "instruction_embeddings" not in batch:
            raise ValueError(
                "use_language=True but dataset batch has no instruction_embeddings. "
                "Run `python -m world_modality.precompute_instruction_embeddings ...` first."
            )
        lang_dim = int(batch["instruction_embeddings"].shape[-1])

    model = build_model(
        exp_cfg,
        img_emb_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=args.world_vocab_size,
        use_language=args.use_language,
        lang_dim=lang_dim,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=training_cfg.learning_rate, weight_decay=1e-4)

    # Warmup scheduler.
    total_steps = len(train_loader) * training_cfg.max_epochs
    scheduler = None
    if training_cfg.warmup_steps > 0:
        scheduler = get_linear_warmup_scheduler(optimizer, training_cfg.warmup_steps, total_steps)
        print(f"Using linear warmup for {training_cfg.warmup_steps} steps")

    if training_cfg.gradient_clip > 0:
        print(f"Using gradient clipping with max_norm={training_cfg.gradient_clip}")

    if training_cfg.log_wandb:
        import wandb

        wandb.init(project=training_cfg.wandb_project, config=vars(args))

    global_step = 0
    for epoch in range(training_cfg.max_epochs):
        model.train()
        for batch in train_loader:
            actions_gt = batch["actions"].to(device).float()
            # Aggregate over context by taking the last timestep.
            states = batch["obs_states"][:, -1].to(device).float()
            img_ctx = batch["img_embeddings"][:, -1].to(device).float()
            lang_emb = batch.get("instruction_embeddings")
            if args.use_language:
                assert lang_emb is not None
                lang_emb = lang_emb.to(device).float()
            future_tokens = batch["future_world_tokens"].to(device)
            current_tokens = batch["current_world_token"].to(device)

            # C uses world token as input; C_no_world_input does not.
            current_world = current_tokens if training_cfg.model_type == "C" else None

            pred_actions, world_logits = model(
                img_emb=img_ctx,
                state=states,
                current_world_token=current_world,
                lang_emb=lang_emb if args.use_language else None,
            )

            act_loss = compute_action_loss(pred_actions, actions_gt)
            # B, C, and C_no_world_input all have world loss.
            world_loss = (
                compute_world_loss(world_logits, future_tokens)
                if training_cfg.model_type in ("B", "C", "C_no_world_input")
                else torch.tensor(0.0, device=device)
            )
            loss = act_loss + training_cfg.lambda_world_loss * world_loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping.
            if training_cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_cfg.gradient_clip)

            optimizer.step()

            # Step scheduler.
            if scheduler is not None:
                scheduler.step()

            if global_step % 50 == 0:
                msg = f"Epoch {epoch} step {global_step} | loss {loss.item():.4f} | act {act_loss.item():.4f}"
                if training_cfg.model_type in ("B", "C", "C_no_world_input"):
                    msg += f" | world {world_loss.item():.4f}"
                print(msg, flush=True)

                if training_cfg.log_wandb:
                    import wandb

                    log_dict = {
                        "train/loss": loss.item(),
                        "train/action_loss": act_loss.item(),
                    }
                    if training_cfg.model_type in ("B", "C", "C_no_world_input"):
                        log_dict["train/world_loss"] = world_loss.item()
                    wandb.log(log_dict, step=global_step)

            global_step += 1

        # Simple validation loop (action MSE + world accuracy).
        model.eval()
        val_action_losses = []
        val_world_accuracies = []
        with torch.no_grad():
            for batch in val_loader:
                actions_gt = batch["actions"].to(device).float()
                states = batch["obs_states"][:, -1].to(device).float()
                img_ctx = batch["img_embeddings"][:, -1].to(device).float()
                lang_emb = batch.get("instruction_embeddings")
                if args.use_language:
                    assert lang_emb is not None
                    lang_emb = lang_emb.to(device).float()
                future_tokens = batch["future_world_tokens"].to(device)
                current_tokens = batch["current_world_token"].to(device)

                # Only C uses world token as input in validation.
                current_world = current_tokens if training_cfg.model_type == "C" else None

                pred_actions, world_logits = model(
                    img_emb=img_ctx,
                    state=states,
                    current_world_token=current_world,
                    lang_emb=lang_emb if args.use_language else None,
                )

                act_loss = compute_action_loss(pred_actions, actions_gt)
                val_action_losses.append(act_loss.item())

                # B, C, and C_no_world_input predict world tokens.
                if training_cfg.model_type in ("B", "C", "C_no_world_input") and world_logits is not None:
                    preds = world_logits.argmax(dim=-1)
                    acc = (preds == future_tokens).float().mean().item()
                    val_world_accuracies.append(acc)

        mean_val_act = sum(val_action_losses) / max(len(val_action_losses), 1)
        mean_val_world = (
            sum(val_world_accuracies) / max(len(val_world_accuracies), 1)
            if val_world_accuracies
            else 0.0
        )

        print(
            f"[VAL] Epoch {epoch} | action MSE {mean_val_act:.4f} | world acc {mean_val_world:.4f}",
            flush=True,
        )

        if training_cfg.log_wandb:
            import wandb

            wandb.log(
                {
                    "val/action_mse": mean_val_act,
                    "val/world_acc": mean_val_world,
                },
                step=global_step,
            )

        # Save a checkpoint per epoch.
        ckpt_path = os.path.join(args.log_dir, f"model_{args.model_type}_epoch{epoch}.pt")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "meta": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "img_emb_dim": img_emb_dim,
                "lang_dim": lang_dim,
                "use_language": args.use_language,
                "text_model_name": args.text_model_name,
                "text_max_length": args.text_max_length,
                "world_vocab_size": args.world_vocab_size,
                "model_type": args.model_type,
                "action_horizon": args.action_horizon,
                "future_offset": args.future_offset,
            },
        }
        torch.save(ckpt, ckpt_path)


if __name__ == "__main__":
    main()
