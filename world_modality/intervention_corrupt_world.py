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
    parser = argparse.ArgumentParser(
        description="Intervention test: corrupt WORLD_CUR tokens at inference."
    )
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

    model_type = meta.get("model_type", cfg.get("model_type", "A"))
    if model_type not in ("C", "C_no_world_input"):
        raise ValueError(
            "Intervention test is only meaningful for model type C (world modality) "
            "or C_no_world_input (ablation). For A and B, corruption has no effect."
        )

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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    mse_clean: List[float] = []
    mse_corrupt: List[float] = []

    rng = np.random.default_rng(seed=0)

    # For C_no_world_input, world token is not used as input, so corruption should have no effect.
    uses_world_input = model_type == "C"

    with torch.no_grad():
        for batch in loader:
            actions_gt = batch["actions"].to(device).float()
            states = batch["obs_states"][:, -1].to(device).float()
            img_ctx = batch["img_embeddings"][:, -1].to(device).float()
            lang_emb = batch.get("instruction_embeddings")
            if use_language:
                assert lang_emb is not None
                lang_emb = lang_emb.to(device).float()
            current_tokens = batch["current_world_token"].to(device)

            # Clean run: use true WORLD_CUR (only for model C).
            current_world_clean = current_tokens if uses_world_input else None
            pred_actions_clean, _ = model(
                img_emb=img_ctx,
                state=states,
                current_world_token=current_world_clean,
                lang_emb=lang_emb if use_language else None,
            )
            mse_c = compute_action_loss(pred_actions_clean, actions_gt).item()
            mse_clean.append(mse_c)

            # Corrupted run: replace WORLD_CUR with random tokens (only for model C).
            if uses_world_input:
                bsz = current_tokens.shape[0]
                random_tokens = rng.integers(
                    low=0,
                    high=world_vocab_size,
                    size=(bsz,),
                    endpoint=False,
                )
                random_tokens_t = torch.as_tensor(
                    random_tokens, dtype=torch.long, device=device
                )
                pred_actions_corrupt, _ = model(
                    img_emb=img_ctx,
                    state=states,
                    current_world_token=random_tokens_t,
                    lang_emb=lang_emb if use_language else None,
                )
                mse_r = compute_action_loss(pred_actions_corrupt, actions_gt).item()
            else:
                # C_no_world_input: corruption has no effect, MSE same as clean.
                mse_r = mse_c
            mse_corrupt.append(mse_r)

    mean_clean = float(np.mean(mse_clean)) if mse_clean else 0.0
    mean_corrupt = float(np.mean(mse_corrupt)) if mse_corrupt else 0.0

    ratio = mean_corrupt / mean_clean if mean_clean > 0 else 0.0

    print("\n" + "=" * 50)
    print("=== World Corruption Intervention Test ===")
    print("=" * 50)
    print(f"Model: {model_type}")
    print(f"Split: {args.split}")
    print("-" * 50)
    print(f"Clean Action MSE:     {mean_clean:.6f}")
    print(f"Corrupted Action MSE: {mean_corrupt:.6f}")
    print("-" * 50)
    print(f"CORRUPTION RATIO:     {ratio:.2f}x")
    print("-" * 50)

    # Interpretation.
    if model_type == "C":
        if ratio > 1.1:
            interpretation = "Model C uses world tokens as modality (ratio > 1)"
        elif ratio > 1.0:
            interpretation = "Model C may weakly use world tokens (ratio slightly > 1)"
        else:
            interpretation = "Model C does not appear to use world tokens (ratio <= 1)"
    else:
        interpretation = "C_no_world_input: world tokens not used as input (ratio should be ~1.0)"

    print(f"Interpretation: {interpretation}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
