from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoModelForVision2Seq, AutoProcessor

from .config import DataConfig
from .device import resolve_device
from .llm_vla_dataset import LiberoVLADataset
from .llm_vla_policy import QwenVLAWrapper, build_act_tokens, find_act_positions
from .model import Prophet
from .train_utils import compute_world_loss_continuous, get_linear_warmup_scheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Qwen VLM + action head + world memory (Model F-style).")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceVLA/libero")
    p.add_argument("--image_key", type=str, default="rgb")
    p.add_argument("--instruction_key", type=str, default="instruction")
    p.add_argument("--action_key", type=str, default="action")
    p.add_argument("--episode_id_key", type=str, default="episode_id")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--context_frames", type=int, default=3)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--future_offset", type=int, default=8)

    p.add_argument("--vlm_backbone", type=str, default="qwen3_vl_3b_instruct")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    p.add_argument("--lora_layers", type=int, default=8)

    p.add_argument(
        "--future_memory_source",
        type=str,
        default="scheduled",
        choices=["scheduled", "oracle", "predicted"],
    )
    p.add_argument("--disable_future_injection", action="store_true")
    p.add_argument(
        "--world_latents_source",
        type=str,
        default="vjepa",
        choices=["dino", "vjepa"],
    )
    p.add_argument("--lambda_world", type=float, default=0.2)
    p.add_argument("--lambda_text", type=float, default=0.0)
    p.add_argument("--coc_jsonl", type=str, default="")
    p.add_argument("--scheduled_sampling_start", type=float, default=0.9)
    p.add_argument("--scheduled_sampling_end", type=float, default=0.1)

    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--gradient_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--output_dir", type=str, default="logs_llm")
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--corruption_mode",
        type=str,
        default="none",
        choices=["none", "zero", "random", "shuffle", "oracle"],
        help="Apply corruption to future memory during validation.",
    )
    return p.parse_args()


def resolve_backbone_name(name: str) -> str:
    key = name.lower().strip()
    mapping = {
        "qwen2_5_vl_3b_instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen3_vl_3b_instruct": "Qwen/Qwen3-VL-3B-Instruct",
    }
    return mapping.get(key, name)


def choose_future_memory(
    source: str,
    z_oracle: torch.Tensor,
    z_pred: torch.Tensor,
    p_oracle: float,
    training: bool,
) -> torch.Tensor:
    if source == "oracle":
        return z_oracle
    if source == "predicted":
        return z_pred
    if not training:
        return z_pred
    mask = (torch.rand(z_oracle.size(0), 1, 1, device=z_oracle.device) < p_oracle)
    return torch.where(mask, z_oracle, z_pred)


def build_prompt(instruction: str, act_tokens: List[str]) -> str:
    act_text = " ".join(act_tokens)
    instr = instruction.strip() if instruction else ""
    if instr:
        return f"<image>\n{instr}\n{act_text}"
    return f"<image>\n{act_text}"


def build_coc_prompt(instruction: str) -> str:
    instr = instruction.strip() if instruction else ""
    if instr:
        return f"<image>\nInstruction: {instr}\nExplain the action sequence:"
    return "<image>\nExplain the action sequence:"


def get_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device.type != "cuda":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_config_attr(model, name: str, fallback: str) -> int:
    if hasattr(model.config, name):
        return int(getattr(model.config, name))
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, fallback):
        return int(getattr(text_cfg, fallback))
    raise AttributeError(f"Model config missing {name}/{fallback}.")


def build_dataloaders(args, processor, act_tokens: List[str], use_text_loss: bool):
    cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        instruction_key=args.instruction_key,
        action_key=args.action_key,
        episode_id_key=args.episode_id_key,
        cache_dir=args.cache_dir,
        context_frames=args.context_frames,
        action_horizon=args.action_horizon,
        future_offset=args.future_offset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_ds = LiberoVLADataset(
        cfg,
        split="train",
        world_latents_source=args.world_latents_source,
        coc_jsonl=args.coc_jsonl if use_text_loss else None,
    )
    val_ds = LiberoVLADataset(
        cfg,
        split="val",
        world_latents_source=args.world_latents_source,
        coc_jsonl=args.coc_jsonl if use_text_loss else None,
    )

    act_token_ids = processor.tokenizer.convert_tokens_to_ids(act_tokens)

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        prompts = [build_prompt(b["instruction"], act_tokens) for b in batch]
        model_inputs = processor(
            text=prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        act_positions = find_act_positions(model_inputs["input_ids"], act_token_ids)
        actions = torch.stack([b["actions"] for b in batch], dim=0)
        z_hist = torch.stack([b["z_hist"] for b in batch], dim=0)
        z_future = torch.stack([b["z_future"] for b in batch], dim=0)
        out = {
            "model_inputs": model_inputs,
            "act_positions": act_positions,
            "actions": actions,
            "z_hist": z_hist,
            "z_future": z_future,
        }
        if use_text_loss:
            coc_texts = [b.get("coc_text", "") for b in batch]
            text_prompts = [build_coc_prompt(b["instruction"]) for b in batch]
            full_texts = [f"{p} {t}" if t else p for p, t in zip(text_prompts, coc_texts)]
            text_inputs = processor(
                text=full_texts,
                images=images,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            # Mask prompt tokens.
            prefix_inputs = processor(
                text=text_prompts,
                images=images,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            labels = text_inputs["input_ids"].clone()
            for i in range(labels.size(0)):
                prefix_len = int(prefix_inputs["input_ids"][i].ne(processor.tokenizer.pad_token_id).sum())
                labels[i, :prefix_len] = -100
            out["text_inputs"] = text_inputs
            out["text_labels"] = labels
        return out

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.num_workers > 0 else False,
        collate_fn=collate,
    )
    return train_loader, val_loader, act_token_ids


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    dtype = get_dtype(args.dtype, device)

    backbone = resolve_backbone_name(args.vlm_backbone)
    processor = AutoProcessor.from_pretrained(backbone, trust_remote_code=args.trust_remote_code)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.lambda_text > 0 and not args.coc_jsonl:
        raise ValueError("--lambda_text > 0 requires --coc_jsonl")

    act_tokens = build_act_tokens(args.action_horizon)
    tokenizer.add_special_tokens({"additional_special_tokens": act_tokens})

    model = AutoModelForVision2Seq.from_pretrained(
        backbone,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as e:
            raise RuntimeError("peft is required for --use_lora") from e
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        num_layers = get_config_attr(model, "num_hidden_layers", "num_hidden_layers")
        layers = list(range(max(0, num_layers - args.lora_layers), num_layers)) if num_layers else None
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            layers_to_transform=layers,
            layers_pattern="model.layers",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)

    # Build dataloaders after tokenizer update.
    use_text_loss = args.lambda_text > 0
    train_loader, val_loader, act_token_ids = build_dataloaders(
        args, processor, act_tokens, use_text_loss=use_text_loss
    )

    # Infer dimensions from a batch.
    sample = next(iter(train_loader))
    actions = sample["actions"]
    z_hist = sample["z_hist"]
    action_dim = int(actions.shape[-1])
    latent_dim = int(z_hist.shape[-1])

    # Prophet + policy wrapper.
    prophet = Prophet(
        emb_dim=latent_dim,
        hidden_dim=latent_dim,
        future_horizon=args.future_offset,
        n_layers=2,
        n_heads=8,
        dropout=0.0,
    ).to(device)

    num_heads = get_config_attr(model, "num_attention_heads", "num_attention_heads")
    wrapper = QwenVLAWrapper(
        vlm=model,
        hidden_size=get_config_attr(model, "hidden_size", "hidden_size"),
        num_attention_heads=num_heads,
        action_dim=action_dim,
        horizon=args.action_horizon,
        future_dim=latent_dim,
        enable_future_injection=not args.disable_future_injection,
    ).to(device)

    params = list(wrapper.action_head.parameters()) + list(prophet.parameters())
    if wrapper.future_injection is not None:
        params += list(wrapper.future_injection.parameters())
    if not args.freeze_backbone:
        params += list(wrapper.vlm.parameters())
    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.max_epochs
    scheduler = None
    if args.warmup_steps > 0:
        scheduler = get_linear_warmup_scheduler(optimizer, args.warmup_steps, total_steps)

    def scheduled_p_oracle(epoch: int) -> float:
        if args.max_epochs <= 1:
            return float(args.scheduled_sampling_end)
        t = epoch / float(args.max_epochs - 1)
        return float(args.scheduled_sampling_start + t * (args.scheduled_sampling_end - args.scheduled_sampling_start))

    global_step = 0
    for epoch in range(args.max_epochs):
        wrapper.train()
        prophet.train()
        p_oracle = scheduled_p_oracle(epoch)

        for step, batch in enumerate(train_loader):
            model_inputs = {k: v.to(device) for k, v in batch["model_inputs"].items()}
            act_positions = batch["act_positions"].to(device)
            actions_gt = batch["actions"].to(device)
            z_hist = batch["z_hist"].to(device)
            z_future = batch["z_future"].to(device)

            z_pred = prophet(z_hist)
            future_mem = choose_future_memory(
                args.future_memory_source, z_future, z_pred, p_oracle, training=True
            )

            actions_pred, _ = wrapper(
                model_inputs=model_inputs,
                act_positions=act_positions,
                future_memory=future_mem,
                disable_future_injection=args.disable_future_injection,
            )

            loss_action = torch.nn.functional.mse_loss(actions_pred, actions_gt)
            loss_world = compute_world_loss_continuous(z_pred, z_future)
            loss_text = torch.tensor(0.0, device=device)
            if use_text_loss:
                text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
                text_labels = batch["text_labels"].to(device)
                text_out = wrapper.vlm(**text_inputs, labels=text_labels, return_dict=True)
                loss_text = text_out.loss

            loss = loss_action + args.lambda_world * loss_world + args.lambda_text * loss_text
            loss = loss / max(1, args.grad_accum_steps)
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.gradient_clip)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            if global_step % args.log_every == 0:
                gate = wrapper.gate_value()
                print(
                    f"[Epoch {epoch} Step {global_step}] "
                    f"loss={loss.item():.4f} action={loss_action.item():.4f} "
                    f"world={loss_world.item():.4f} text={loss_text.item():.4f} gate={gate:.4f}"
                )
            global_step += 1

        # Simple validation.
        wrapper.eval()
        prophet.eval()
        val_action_losses = []
        val_text_losses = []
        with torch.no_grad():
            for batch in val_loader:
                model_inputs = {k: v.to(device) for k, v in batch["model_inputs"].items()}
                act_positions = batch["act_positions"].to(device)
                actions_gt = batch["actions"].to(device)
                z_hist = batch["z_hist"].to(device)
                z_future = batch["z_future"].to(device)

                z_pred = prophet(z_hist)
                future_mem = z_pred
                if args.corruption_mode == "oracle":
                    future_mem = z_future
                elif args.corruption_mode == "zero":
                    future_mem = torch.zeros_like(z_pred)
                elif args.corruption_mode == "random":
                    future_mem = torch.randn_like(z_pred)
                elif args.corruption_mode == "shuffle":
                    perm = torch.randperm(z_pred.size(0), device=z_pred.device)
                    future_mem = z_pred[perm]

                actions_pred, _ = wrapper(
                    model_inputs=model_inputs,
                    act_positions=act_positions,
                    future_memory=future_mem,
                    disable_future_injection=args.disable_future_injection,
                )
                loss_action = torch.nn.functional.mse_loss(actions_pred, actions_gt)
                val_action_losses.append(loss_action.item())
                if use_text_loss:
                    text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
                    text_labels = batch["text_labels"].to(device)
                    text_out = wrapper.vlm(**text_inputs, labels=text_labels, return_dict=True)
                    val_text_losses.append(float(text_out.loss.item()))

        mean_val = float(sum(val_action_losses) / max(1, len(val_action_losses)))
        gate = wrapper.gate_value()
        if use_text_loss:
            mean_text = float(sum(val_text_losses) / max(1, len(val_text_losses)))
            print(
                f"[Epoch {epoch}] VAL action MSE={mean_val:.6f} text={mean_text:.6f} gate={gate:.4f}"
            )
        else:
            print(f"[Epoch {epoch}] VAL action MSE={mean_val:.6f} gate={gate:.4f}")

        # Save checkpoint per epoch.
        ckpt = {
            "config": vars(args),
            "act_tokens": act_tokens,
            "action_head_state_dict": wrapper.action_head.state_dict(),
            "prophet_state_dict": prophet.state_dict(),
            "future_injection_state_dict": (
                wrapper.future_injection.state_dict() if wrapper.future_injection is not None else None
            ),
        }
        ckpt_path = os.path.join(args.output_dir, f"llm_vla_epoch{epoch}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
