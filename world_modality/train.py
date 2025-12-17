from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Optional

import torch
from torch.optim import AdamW

from .config import DatasetConfig, ModelType, TrainingConfig, TransformerConfig
from .data_sr100 import get_dataloaders
from .model import Prophet, WorldPolicyTransformer
from .train_utils import (
    compute_action_loss,
    compute_world_cosine,
    compute_world_loss,
    compute_world_loss_continuous,
    get_linear_warmup_scheduler,
)


def choose_future_memory(
    source: str,
    z_oracle: torch.Tensor,  # [B, K, D]
    z_pred: torch.Tensor,  # [B, K, D]
    p_oracle: float,
    training: bool,
) -> torch.Tensor:
    """Select which future memory to inject for Model F.

    Note: we always return float32 to avoid dtype mismatches in Linear layers.
    """
    z_oracle_f = z_oracle.float()
    z_pred_f = z_pred.float()

    if source == "oracle":
        return z_oracle_f
    if source == "predicted":
        return z_pred_f
    # scheduled
    if not training:
        return z_pred_f
    mask = (torch.rand(z_oracle.size(0), 1, 1, device=z_oracle.device) < p_oracle)
    return torch.where(mask, z_oracle_f, z_pred_f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset_name",
        type=str,
        default="HuggingFaceVLA/libero",
        help="Dataset name (HuggingFace dataset path).",
    )
    p.add_argument(
        "--model_type",
        type=str,
        default="A",
        choices=["A", "B", "B_cont", "C", "C_no_world_input", "F"],
    )
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--lambda_world_loss", type=float, default=0.2)
    p.add_argument("--world_vocab_size", type=int, default=1024)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--future_offset", type=int, default=8)
    p.add_argument("--log_dir", type=str, default="logs")

    # Optional language conditioning.
    p.add_argument("--use_language", action="store_true")
    p.add_argument("--text_model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--text_max_length", type=int, default=32)

    # C-model stability knobs.
    p.add_argument("--world_input_scale", type=float, default=1.0)
    p.add_argument("--world_input_dropout", type=float, default=0.0)
    p.add_argument("--world_input_layernorm", action="store_true")
    p.add_argument("--block_world_to_action", action="store_true")

    # F-model knobs.
    p.add_argument(
        "--future_memory_source",
        type=str,
        default="scheduled",
        choices=["scheduled", "oracle", "predicted"],
        help="For Model F: source of future memory injected into cross-attn.",
    )
    p.add_argument(
        "--disable_future_injection",
        action="store_true",
        help="For Model F: disable cross-attn injection into ACT_Q (prophet still trains).",
    )
    p.add_argument("--scheduled_sampling_start", type=float, default=0.9)
    p.add_argument("--scheduled_sampling_end", type=float, default=0.1)

    # WandB
    p.add_argument("--log_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="world-modality")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_type: ModelType = args.model_type  # type: ignore

    dataset_cfg = DatasetConfig(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        action_horizon=args.action_horizon,
        future_offset=args.future_offset,
        use_language=args.use_language,
        text_model_name=args.text_model_name,
        text_max_length=args.text_max_length,
        preload_to_gpu=True,
    )

    transformer_cfg = TransformerConfig(
        d_model=512,
        n_layers=4,
        n_heads=8,
        dropout=0.0,
        norm_first=True,
    )

    training_cfg = TrainingConfig(
        model_type=model_type,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        lambda_world_loss=args.lambda_world_loss,
        log_wandb=args.log_wandb,
        wandb_project=args.wandb_project,
    )

    print("Dataset config:", asdict(dataset_cfg))
    print("Transformer config:", asdict(transformer_cfg))
    print("Training config:", asdict(training_cfg))

    train_loader, val_loader, meta = get_dataloaders(dataset_cfg)
    state_dim = meta["state_dim"]
    action_dim = meta["action_dim"]
    img_emb_dim = meta["img_emb_dim"]
    lang_dim = meta.get("lang_dim", 0)

    # Build model
    continuous_world = model_type == "B_cont"
    enable_future_injection = model_type == "F"
    model = WorldPolicyTransformer(
        model_type=model_type,
        cfg=transformer_cfg,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=args.world_vocab_size,
        horizon=args.action_horizon,
        future_horizon=args.future_offset,
        use_language=args.use_language,
        lang_dim=lang_dim,
        world_input_scale=args.world_input_scale,
        world_input_dropout=args.world_input_dropout,
        world_input_layernorm=args.world_input_layernorm,
        block_world_to_action=args.block_world_to_action,
        continuous_world=continuous_world,
        world_target_dim=img_emb_dim,
        enable_future_injection=enable_future_injection,
        future_memory_dim=img_emb_dim,
    ).to(device)

    prophet: Optional[Prophet] = None
    if model_type == "F":
        prophet = Prophet(
            emb_dim=img_emb_dim,
            hidden_dim=transformer_cfg.d_model,
            future_horizon=args.future_offset,
            n_layers=2,
            n_heads=transformer_cfg.n_heads,
            dropout=transformer_cfg.dropout,
        ).to(device)

    # Optimizer (model + prophet if present)
    params = list(model.parameters()) + (list(prophet.parameters()) if prophet is not None else [])
    optimizer = AdamW(params, lr=training_cfg.learning_rate, weight_decay=1e-4)

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

    def scheduled_p_oracle(epoch: int) -> float:
        if training_cfg.max_epochs <= 1:
            return float(args.scheduled_sampling_end)
        t = epoch / float(training_cfg.max_epochs - 1)
        return float(args.scheduled_sampling_start + t * (args.scheduled_sampling_end - args.scheduled_sampling_start))

    global_step = 0
    for epoch in range(training_cfg.max_epochs):
        model.train()
        if prophet is not None:
            prophet.train()

        p_oracle = scheduled_p_oracle(epoch)

        for batch in train_loader:
            actions_gt = batch["actions"].to(device).float()
            states = batch["obs_states"][:, -1].to(device).float()
            img_ctx = batch["img_embeddings"][:, -1].to(device).float()

            lang_emb = batch.get("instruction_embeddings")
            if args.use_language:
                assert lang_emb is not None
                lang_emb = lang_emb.to(device).float()

            # Discrete tokens (legacy)
            future_tokens = batch["future_world_tokens"].to(device)
            current_tokens = batch["current_world_token"].to(device)
            current_world = current_tokens if model_type == "C" else None

            # Continuous embeddings (Phase-2)
            future_emb = batch.get("future_world_embeddings")
            if model_type in ("B_cont", "F"):
                assert future_emb is not None, "future_world_embeddings missing. Re-run precompute_world_tokens.py."
                future_emb = future_emb.to(device)  # keep fp16

            future_memory = None
            prophet_pred = None
            prophet_loss = torch.tensor(0.0, device=device)

            if model_type == "F":
                assert prophet is not None
                z_hist = batch["img_embeddings"].to(device)  # [B, T_ctx, D] (fp16)
                prophet_pred = prophet(z_hist)  # [B, K, D] (fp32 output)
                # choose memory for injection (unless disabled)
                if not args.disable_future_injection:
                    future_memory = choose_future_memory(
                        args.future_memory_source,
                        z_oracle=future_emb,
                        z_pred=prophet_pred,
                        p_oracle=p_oracle,
                        training=True,
                    )
                prophet_loss = compute_world_loss_continuous(prophet_pred, future_emb)

            pred_actions, world_out = model(
                img_emb=img_ctx,
                state=states,
                current_world_token=current_world,
                lang_emb=lang_emb if args.use_language else None,
                future_memory=future_memory,
            )

            act_loss = compute_action_loss(pred_actions, actions_gt)

            world_loss = torch.tensor(0.0, device=device)
            if model_type in ("B", "C", "C_no_world_input"):
                world_loss = compute_world_loss(world_out, future_tokens)
            elif model_type == "B_cont":
                assert future_emb is not None
                assert world_out is not None
                world_loss = compute_world_loss_continuous(world_out, future_emb)
            elif model_type == "F":
                world_loss = prophet_loss  # prophet supervision

            loss = act_loss + training_cfg.lambda_world_loss * world_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if training_cfg.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, training_cfg.gradient_clip)

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if global_step % 50 == 0:
                msg = f"Epoch {epoch} step {global_step} | loss {loss.item():.4f} | act {act_loss.item():.4f}"
                if model_type in ("B", "B_cont", "C", "C_no_world_input", "F"):
                    msg += f" | world {world_loss.item():.4f}"
                if model_type == "F":
                    gate = model.get_gate_value()
                    msg += f" | p_oracle {p_oracle:.2f} | gate {gate if gate is not None else 0.0:.3f}"
                print(msg, flush=True)

                if training_cfg.log_wandb:
                    import wandb

                    log_dict = {
                        "train/loss": loss.item(),
                        "train/action_loss": act_loss.item(),
                    }
                    if model_type in ("B", "B_cont", "C", "C_no_world_input", "F"):
                        log_dict["train/world_loss"] = world_loss.item()
                    if model_type == "F":
                        log_dict["train/p_oracle"] = p_oracle
                        gate = model.get_gate_value()
                        if gate is not None:
                            log_dict["train/gate"] = gate
                    wandb.log(log_dict, step=global_step)

            global_step += 1

        # Validation
        model.eval()
        if prophet is not None:
            prophet.eval()

        val_action_losses = []
        val_world_metrics = []
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
                current_world = current_tokens if model_type == "C" else None

                future_emb = batch.get("future_world_embeddings")
                if model_type in ("B_cont", "F"):
                    assert future_emb is not None
                    future_emb = future_emb.to(device)

                future_memory = None
                if model_type == "F":
                    assert prophet is not None
                    z_hist = batch["img_embeddings"].to(device)
                    prophet_pred = prophet(z_hist)
                    if not args.disable_future_injection:
                        future_memory = choose_future_memory(
                            args.future_memory_source,
                            z_oracle=future_emb,
                            z_pred=prophet_pred,
                            p_oracle=0.0,
                            training=False,
                        )

                pred_actions, world_out = model(
                    img_emb=img_ctx,
                    state=states,
                    current_world_token=current_world,
                    lang_emb=lang_emb if args.use_language else None,
                    future_memory=future_memory,
                )

                act_loss = compute_action_loss(pred_actions, actions_gt)
                val_action_losses.append(act_loss.item())

                if model_type in ("B", "C", "C_no_world_input") and world_out is not None:
                    preds = world_out.argmax(dim=-1)
                    acc = (preds == future_tokens).float().mean().item()
                    val_world_metrics.append(acc)
                elif model_type == "B_cont":
                    assert world_out is not None
                    val_world_metrics.append(compute_world_cosine(world_out, future_emb))
                elif model_type == "F":
                    assert prophet is not None
                    val_world_metrics.append(compute_world_cosine(prophet_pred, future_emb))

        mean_val_act = sum(val_action_losses) / max(len(val_action_losses), 1)
        mean_val_world = sum(val_world_metrics) / max(len(val_world_metrics), 1) if val_world_metrics else 0.0

        if model_type in ("B", "C", "C_no_world_input"):
            print(f"[VAL] Epoch {epoch} | action MSE {mean_val_act:.4f} | world acc {mean_val_world:.4f}", flush=True)
        else:
            print(f"[VAL] Epoch {epoch} | action MSE {mean_val_act:.4f} | world cos {mean_val_world:.4f}", flush=True)

        if training_cfg.log_wandb:
            import wandb

            wandb.log(
                {
                    "val/action_mse": mean_val_act,
                    "val/world_metric": mean_val_world,
                },
                step=global_step,
            )

        # Save checkpoint per epoch
        ckpt_path = os.path.join(args.log_dir, f"model_{args.model_type}_epoch{epoch}.pt")
        ckpt = {
            "model_state_dict": model.state_dict(),
            "prophet_state_dict": prophet.state_dict() if prophet is not None else None,
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
                "future_memory_source": args.future_memory_source,
                "disable_future_injection": args.disable_future_injection,
            },
        }
        torch.save(ckpt, ckpt_path)


if __name__ == "__main__":
    main()
