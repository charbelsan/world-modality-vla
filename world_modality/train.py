from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
from torch.optim import AdamW
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
    parser.add_argument("--model_type", type=str, choices=["A", "B", "C"], default="A")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--image_key", type=str, default="rgb", help="Key for image observation in LeRobotDataset.")
    parser.add_argument("--proprio_key", type=str, default="proprio", help="Key for proprio state in LeRobotDataset.")
    parser.add_argument("--action_key", type=str, default="action", help="Key for action in LeRobotDataset.")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--context_frames", type=int, default=3)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--future_offset", type=int, default=8, help="Max future horizon K for world tokens (predict w_{t+1..t+K}).")
    parser.add_argument("--world_vocab_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lambda_world_loss", type=float, default=0.2)
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


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
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

    model = build_model(
        exp_cfg,
        img_emb_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=args.world_vocab_size,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=training_cfg.learning_rate, weight_decay=1e-4)

    if training_cfg.log_wandb:
        import wandb

        wandb.init(project=training_cfg.wandb_project, config=vars(args))

    global_step = 0
    for epoch in range(training_cfg.max_epochs):
        model.train()
        for batch in train_loader:
            actions_gt = batch["actions"].to(device)
            # Aggregate over context by taking the last timestep.
            states = batch["obs_states"][:, -1].to(device)
            img_ctx = batch["img_embeddings"][:, -1].to(device)
            future_tokens = batch["future_world_tokens"].to(device)
            current_tokens = batch["current_world_token"].to(device)

            current_world = current_tokens if training_cfg.model_type == "C" else None

            pred_actions, world_logits = model(
                img_emb=img_ctx, state=states, current_world_token=current_world
            )

            act_loss = compute_action_loss(pred_actions, actions_gt)
            world_loss = (
                compute_world_loss(world_logits, future_tokens)
                if training_cfg.model_type in ("B", "C")
                else torch.tensor(0.0, device=device)
            )
            loss = act_loss + training_cfg.lambda_world_loss * world_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_step % 50 == 0:
                msg = f"Epoch {epoch} step {global_step} | loss {loss.item():.4f} | act {act_loss.item():.4f}"
                if training_cfg.model_type in ("B", "C"):
                    msg += f" | world {world_loss.item():.4f}"
                print(msg, flush=True)

                if training_cfg.log_wandb:
                    import wandb

                    log_dict = {
                        "train/loss": loss.item(),
                        "train/action_loss": act_loss.item(),
                    }
                    if training_cfg.model_type in ("B", "C"):
                        log_dict["train/world_loss"] = world_loss.item()
                    wandb.log(log_dict, step=global_step)

            global_step += 1

        # Simple validation loop (action MSE + world accuracy)
        model.eval()
        val_action_losses = []
        val_world_accuracies = []
        with torch.no_grad():
            for batch in val_loader:
                actions_gt = batch["actions"].to(device)
                states = batch["obs_states"][:, -1].to(device)
                img_ctx = batch["img_embeddings"][:, -1].to(device)
                future_tokens = batch["future_world_tokens"].to(device)
                current_tokens = batch["current_world_token"].to(device)

                current_world = current_tokens if training_cfg.model_type == "C" else None

                pred_actions, world_logits = model(
                    img_emb=img_ctx, state=states, current_world_token=current_world
                )

                act_loss = compute_action_loss(pred_actions, actions_gt)
                val_action_losses.append(act_loss.item())

                if training_cfg.model_type in ("B", "C") and world_logits is not None:
                    preds = world_logits.argmax(dim=-1)
                    # Compare over all horizons.
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

        # Save a simple checkpoint per epoch
        ckpt_path = os.path.join(args.log_dir, f"model_{args.model_type}_epoch{epoch}.pt")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "meta": {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "img_emb_dim": img_emb_dim,
                "world_vocab_size": args.world_vocab_size,
                "model_type": args.model_type,
                "action_horizon": args.action_horizon,
                "future_offset": args.future_offset,
            },
        }
        torch.save(ckpt, ckpt_path)


if __name__ == "__main__":
    main()
