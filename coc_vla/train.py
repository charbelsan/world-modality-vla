from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from world_modality.config import DataConfig, TransformerConfig
from world_modality.device import resolve_device
from world_modality.train_utils import (
    compute_action_loss,
    compute_world_loss,
    set_seed,
)
from coc_vla.data import CoCSR100Dataset
from coc_vla.model import WorldCoCTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train two-head CoC VLA model.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--coc_jsonl", type=str, required=True)
    parser.add_argument("--image_key", type=str, default="rgb")
    parser.add_argument("--proprio_key", type=str, default="proprio")
    parser.add_argument("--action_key", type=str, default="action")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--context_frames", type=int, default=3)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--future_offset", type=int, default=8)
    parser.add_argument("--world_vocab_size", type=int, default=1024)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lambda_world_loss", type=float, default=0.2)
    parser.add_argument("--alpha_coc_loss", type=float, default=0.2)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs_coc")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--coc_tokenizer_name", type=str, default="Qwen/Qwen2.5-0.5B", help="HF tokenizer name for CoC.")
    parser.add_argument("--coc_max_len", type=int, default=128, help="Max CoC tokens (after tokenization).")
    return parser.parse_args()


def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def save_config(args: argparse.Namespace, log_dir: str):
    cfg = {
        "dataset_name": args.dataset_name,
        "coc_jsonl": args.coc_jsonl,
        "image_key": args.image_key,
        "proprio_key": args.proprio_key,
        "action_key": args.action_key,
        "cache_dir": args.cache_dir,
        "batch_size": args.batch_size,
        "context_frames": args.context_frames,
        "action_horizon": args.action_horizon,
        "future_offset": args.future_offset,
        "world_vocab_size": args.world_vocab_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "lambda_world_loss": args.lambda_world_loss,
        "alpha_coc_loss": args.alpha_coc_loss,
        "warmup_steps": args.warmup_steps,
        "gradient_clip": args.gradient_clip,
        "device": args.device,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "coc_tokenizer_name": args.coc_tokenizer_name,
        "coc_max_len": args.coc_max_len,
        "timestamp": datetime.now().isoformat(),
    }
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def make_collate_with_coc(tokenizer, max_len: int):
    """
    Create a collate_fn that tokenizes and pads CoC text.

    We train the CoC decoder as a causal LM with teacher forcing:
      - decoder_input_ids = tokens[:-1]
      - labels = tokens[1:]
    """

    pad_id = tokenizer.pad_token_id

    def collate(batch: Dict) -> Dict[str, torch.Tensor]:
        collated: Dict[str, torch.Tensor] = {}
        keys = batch[0].keys()
        for k in keys:
            if k == "coc_text":
                continue
            collated[k] = torch.stack([b[k] for b in batch], dim=0)

        coc_texts = [b.get("coc_text", "") for b in batch]
        enc = tokenizer(
            coc_texts,
            padding=True,
            truncation=True,
            max_length=max_len + 1,  # +1 because we shift for LM
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]  # [B, L]
        attn = enc.get("attention_mask", torch.ones_like(input_ids))

        if input_ids.shape[1] < 2:
            # Ensure we can shift.
            input_ids = torch.cat([input_ids, torch.full_like(input_ids, pad_id)], dim=1)
            attn = torch.cat([attn, torch.zeros_like(attn)], dim=1)

        collated["coc_input_ids"] = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()
        label_attn = attn[:, 1:].contiguous()
        labels = labels.masked_fill(label_attn == 0, -100)
        collated["coc_labels"] = labels
        return collated

    return collate


def main():
    args = parse_args()

    device = resolve_device(args.device)
    set_seed(args.seed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.coc_tokenizer_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        # Many LLM tokenizers have no pad token; using eos as pad is standard.
        tokenizer.pad_token = tokenizer.eos_token

    # Build configs.
    data_cfg_world = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        proprio_key=args.proprio_key,
        action_key=args.action_key,
        batch_size=args.batch_size,
        context_frames=args.context_frames,
        action_horizon=args.action_horizon,
        future_offset=args.future_offset,
        cache_dir=args.cache_dir,
    )

    # CoC data wrapper.
    train_ds = CoCSR100Dataset(data_cfg_world, coc_jsonl=args.coc_jsonl, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=make_collate_with_coc(tokenizer, max_len=args.coc_max_len),
    )

    # Infer dimensions from one batch.
    batch = next(iter(train_loader))
    obs_states = batch["obs_states"]
    actions = batch["actions"]
    img_emb = batch["img_embeddings"]

    state_dim = obs_states.shape[-1]
    action_dim = actions.shape[-1]
    img_emb_dim = img_emb.shape[-1]

    trunk_cfg = TransformerConfig()

    model = WorldCoCTransformer(
        model_type="C",  # full world-modality as input
        trunk_cfg=trunk_cfg,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=args.world_vocab_size,
        horizon=args.action_horizon,
        future_horizon=args.future_offset,
        coc_vocab_size=int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        coc_d_model=trunk_cfg.d_model,
        coc_n_layers=4,
        coc_n_heads=trunk_cfg.n_heads,
        coc_dropout=trunk_cfg.dropout,
        coc_max_len=args.coc_max_len,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    total_steps = len(train_loader) * args.max_epochs
    scheduler = get_linear_warmup_scheduler(optimizer, args.warmup_steps, total_steps) if args.warmup_steps > 0 else None

    save_config(args, args.log_dir)

    global_step = 0
    for epoch in range(args.max_epochs):
        model.train()
        epoch_act_losses = []
        epoch_world_losses = []
        epoch_coc_losses = []

        for batch in train_loader:
            actions_gt = batch["actions"].to(device).float()
            states = batch["obs_states"][:, -1].to(device).float()
            img_ctx = batch["img_embeddings"][:, -1].to(device).float()
            future_tokens = batch["future_world_tokens"].to(device)
            current_tokens = batch["current_world_token"].to(device)

            coc_input_ids = batch["coc_input_ids"].to(device)
            coc_labels = batch["coc_labels"].to(device)

            optimizer.zero_grad()

            actions_pred, world_logits, coc_logits = model(
                img_emb=img_ctx,
                state=states,
                current_world_token=current_tokens,
                coc_input_ids=coc_input_ids,
            )

            act_loss = compute_action_loss(actions_pred, actions_gt)
            world_loss = compute_world_loss(world_logits, future_tokens)

            coc_loss = torch.tensor(0.0, device=device)
            if coc_logits is not None:
                # coc_logits: [B, L, V], coc_labels: [B, L]
                num_valid = int((coc_labels != -100).sum().item())
                if num_valid > 0:
                    coc_loss = F.cross_entropy(
                        coc_logits.reshape(-1, coc_logits.shape[-1]),
                        coc_labels.reshape(-1),
                        ignore_index=-100,
                    )

            loss = act_loss + args.lambda_world_loss * world_loss + args.alpha_coc_loss * coc_loss

            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_act_losses.append(act_loss.item())
            epoch_world_losses.append(world_loss.item())
            epoch_coc_losses.append(coc_loss.item())

            if global_step % 50 == 0:
                print(
                    f"[CoC TRAIN] Epoch {epoch} step {global_step} | "
                    f"loss {loss.item():.4f} | act {act_loss.item():.4f} | "
                    f"world {world_loss.item():.4f} | coc {coc_loss.item():.4f}",
                    flush=True,
                )

            global_step += 1

        mean_act = float(np.mean(epoch_act_losses)) if epoch_act_losses else 0.0
        mean_world = float(np.mean(epoch_world_losses)) if epoch_world_losses else 0.0
        mean_coc = float(np.mean(epoch_coc_losses)) if epoch_coc_losses else 0.0

        print(
            f"[CoC EPOCH] {epoch} | act {mean_act:.4f} | world {mean_world:.4f} | coc {mean_coc:.4f}",
            flush=True,
        )

        # Save checkpoint per epoch.
        os.makedirs(args.log_dir, exist_ok=True)
        ckpt_path = os.path.join(args.log_dir, f"world_coc_epoch{epoch}.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()
