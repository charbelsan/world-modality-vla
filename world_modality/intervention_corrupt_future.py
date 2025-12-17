from __future__ import annotations

import argparse

import torch

from .config import DatasetConfig, TransformerConfig
from .data_sr100 import get_dataloaders
from .model import Prophet, WorldPolicyTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="HuggingFaceVLA/libero")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument(
        "--corruption_mode",
        type=str,
        default="shuffle",
        choices=["shuffle", "random", "zero", "oracle"],
        help="How to corrupt the injected future memory (baseline is always predicted).",
    )
    p.add_argument("--max_batches", type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def eval_clean_and_corrupt(
    model: WorldPolicyTransformer,
    prophet: Prophet,
    loader,
    device: torch.device,
    corruption_mode: str,
    max_batches: int,
):
    model.eval()
    prophet.eval()

    clean_sum = 0.0
    corrupt_sum = 0.0
    n = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        actions_gt = batch["actions"].to(device).float()
        states = batch["obs_states"][:, -1].to(device).float()
        img_ctx = batch["img_embeddings"][:, -1].to(device).float()

        future_oracle = batch.get("future_world_embeddings")
        assert future_oracle is not None, "future_world_embeddings missing in batch."
        future_oracle = future_oracle.to(device).float()

        z_hist = batch["img_embeddings"].to(device)
        future_pred = prophet(z_hist)

        # Baseline: predicted future memory.
        pred_actions, _ = model(img_emb=img_ctx, state=states, future_memory=future_pred)
        clean_mse = torch.nn.functional.mse_loss(pred_actions, actions_gt, reduction="sum")

        # Corrupt future memory.
        if corruption_mode == "oracle":
            future_mem = future_oracle
        elif corruption_mode == "zero":
            future_mem = torch.zeros_like(future_pred)
        elif corruption_mode == "random":
            future_mem = torch.randn_like(future_pred)
        elif corruption_mode == "shuffle":
            perm = torch.randperm(future_pred.size(0), device=device)
            future_mem = future_pred[perm]
        else:
            raise ValueError(corruption_mode)

        pred_actions_corrupt, _ = model(img_emb=img_ctx, state=states, future_memory=future_mem)
        corrupt_mse = torch.nn.functional.mse_loss(pred_actions_corrupt, actions_gt, reduction="sum")

        clean_sum += float(clean_mse.item())
        corrupt_sum += float(corrupt_mse.item())
        n += int(actions_gt.numel())

    clean = clean_sum / max(n, 1)
    corrupt = corrupt_sum / max(n, 1)
    ratio = corrupt / max(clean, 1e-12)
    return clean, corrupt, ratio


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    meta = ckpt.get("meta", {})
    cfg_args = ckpt.get("config", {})

    model_type = meta.get("model_type", cfg_args.get("model_type", None))
    if model_type != "F":
        raise ValueError(f"This script expects a Model F checkpoint. Got: {model_type!r}")

    action_horizon = int(meta.get("action_horizon", 8))
    future_offset = int(meta.get("future_offset", 8))
    world_vocab_size = int(meta.get("world_vocab_size", 1024))

    dataset_cfg = DatasetConfig(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        action_horizon=action_horizon,
        future_offset=future_offset,
        use_language=False,
        preload_to_gpu=True,
    )

    transformer_cfg = TransformerConfig(
        d_model=512,
        n_layers=4,
        n_heads=8,
        dropout=0.0,
        norm_first=True,
    )

    _, val_loader, dl_meta = get_dataloaders(dataset_cfg)
    state_dim = dl_meta["state_dim"]
    action_dim = dl_meta["action_dim"]
    img_emb_dim = dl_meta["img_emb_dim"]

    model = WorldPolicyTransformer(
        model_type="F",
        cfg=transformer_cfg,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=world_vocab_size,
        horizon=action_horizon,
        future_horizon=future_offset,
        use_language=False,
        lang_dim=0,
        continuous_world=False,
        enable_future_injection=True,
        future_memory_dim=img_emb_dim,
    ).to(device)

    prophet = Prophet(
        emb_dim=img_emb_dim,
        hidden_dim=transformer_cfg.d_model,
        future_horizon=future_offset,
        n_layers=2,
        n_heads=transformer_cfg.n_heads,
        dropout=transformer_cfg.dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if ckpt.get("prophet_state_dict") is None:
        raise ValueError("Checkpoint is missing prophet_state_dict.")
    prophet.load_state_dict(ckpt["prophet_state_dict"], strict=True)

    clean, corrupt, ratio = eval_clean_and_corrupt(
        model=model,
        prophet=prophet,
        loader=val_loader,
        device=device,
        corruption_mode=args.corruption_mode,
        max_batches=args.max_batches,
    )

    print(f"Clean (predicted) action MSE:   {clean:.6f}")
    print(f"Corrupt ({args.corruption_mode}) action MSE: {corrupt:.6f}")
    print(f"Corruption ratio (corrupt/clean): {ratio:.4f}")


if __name__ == "__main__":
    main()
