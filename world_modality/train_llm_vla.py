from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

from .config import DataConfig
from .device import resolve_device
from .llm_vla_dataset import LiberoVLADataset
from .llm_vla_policy import QwenVLAWrapper, build_act_tokens, find_act_positions
from .model import Prophet
from .train_utils import compute_world_loss_continuous, get_linear_warmup_scheduler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Qwen VLM + action head + world memory (Model F-style).")
    p.add_argument("--dataset_name", type=str, default="HuggingFaceVLA/libero")
    p.add_argument("--image_key", type=str, default="observation.images.image")
    p.add_argument("--wrist_image_key", type=str, default="observation.images.image2")
    p.add_argument(
        "--wrist_mode",
        type=str,
        default="none",
        choices=["none", "concat"],
        help="How to use wrist camera during training. 'concat' concatenates agentview and wrist into one image.",
    )
    p.add_argument("--proprio_key", type=str, default="observation.state")
    p.add_argument(
        "--use_proprio",
        action="store_true",
        help="Condition the policy on proprioception (LIBERO: observation.state, 8-dim).",
    )
    p.add_argument("--instruction_key", type=str, default="task")
    p.add_argument("--action_key", type=str, default="action")
    p.add_argument("--episode_id_key", type=str, default="episode_index")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--context_frames", type=int, default=3)
    p.add_argument("--action_horizon", type=int, default=8)
    p.add_argument("--future_offset", type=int, default=8)
    p.add_argument(
        "--action_head",
        type=str,
        default="mse",
        choices=["mse", "flow"],
        help="Action head type: mse (regression) or flow (rectified flow matching).",
    )
    p.add_argument("--flow_steps_eval", type=int, default=8, help="Sampling steps for flow head during eval.")

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
    p.add_argument(
        "--latent_suffix",
        type=str,
        default="",
        help="Suffix for latent file (e.g., 'm4' for temporal latents)",
    )
    p.add_argument("--lambda_world", type=float, default=0.2)
    p.add_argument("--lambda_text", type=float, default=0.0)
    p.add_argument(
        "--delta_prediction",
        action="store_true",
        help="Train Prophet to predict delta (z_future - z_current) instead of z_future. "
             "This makes prediction non-trivial when cos(z_t, z_{t+k}) is high.",
    )
    p.add_argument("--coc_jsonl", type=str, default="")
    p.add_argument(
        "--require_coc",
        action="store_true",
        help="If set (and --lambda_text > 0), drop episodes without CoC labels. "
        "Otherwise, text loss is skipped for samples with missing CoC.",
    )
    p.add_argument(
        "--text_loss_mode",
        type=str,
        default="joint_after_act",
        choices=["joint_after_act", "separate"],
        help="How to compute CoC text loss. joint_after_act uses a single forward pass where CoC tokens come after <ACT>; "
        "separate runs an independent forward pass for CoC loss.",
    )
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
    p.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint path")
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


def build_prompt(instruction: str, act_tokens: List[str], use_chat_template: bool = False) -> str:
    """Build prompt text. If use_chat_template=True, returns raw text without <image> placeholder."""
    act_text = " ".join(act_tokens)
    instr = instruction.strip() if instruction else ""
    if use_chat_template:
        # For Qwen2.5-VL: just return text, image placeholder added by chat template
        if instr:
            return f"{instr}\n{act_text}"
        return act_text
    else:
        # For Qwen3-VL: use <image> placeholder
        if instr:
            return f"<image>\n{instr}\n{act_text}"
        return f"<image>\n{act_text}"


def build_coc_prompt(instruction: str, use_chat_template: bool = False) -> str:
    instr = instruction.strip() if instruction else ""
    if use_chat_template:
        if instr:
            return f"Instruction: {instr}\nExplain the action sequence:"
        return "Explain the action sequence:"
    else:
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


def build_dataloaders(args, processor, act_tokens: List[str], use_text_loss: bool, use_chat_template: bool = False):
    cfg = DataConfig(
        dataset_name=args.dataset_name,
        image_key=args.image_key,
        proprio_key=args.proprio_key if args.use_proprio else "",
        instruction_key=args.instruction_key,
        action_key=args.action_key,
        episode_id_key=args.episode_id_key,
        cache_dir=args.cache_dir,
        context_frames=args.context_frames,
        action_horizon=args.action_horizon,
        future_offset=args.future_offset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        latent_suffix=args.latent_suffix,
    )

    train_ds = LiberoVLADataset(
        cfg,
        split="train",
        world_latents_source=args.world_latents_source,
        wrist_image_key=args.wrist_image_key if args.wrist_mode != "none" else None,
        require_proprio=bool(args.use_proprio),
        require_wrist=bool(args.wrist_mode != "none"),
        coc_jsonl=args.coc_jsonl if use_text_loss else None,
        require_coc=bool(use_text_loss and args.require_coc),
    )
    val_ds = LiberoVLADataset(
        cfg,
        split="val",
        world_latents_source=args.world_latents_source,
        wrist_image_key=args.wrist_image_key if args.wrist_mode != "none" else None,
        require_proprio=bool(args.use_proprio),
        require_wrist=bool(args.wrist_mode != "none"),
        coc_jsonl=args.coc_jsonl if use_text_loss else None,
        require_coc=bool(use_text_loss and args.require_coc),
    )

    act_token_ids = processor.tokenizer.convert_tokens_to_ids(act_tokens)

    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        def _concat_images(left, right):
            if left.size[1] != right.size[1]:
                right = right.resize((right.size[0], left.size[1]))
            out = Image.new("RGB", (left.size[0] + right.size[0], left.size[1]))
            out.paste(left, (0, 0))
            out.paste(right, (left.size[0], 0))
            return out

        images = [b["image"] for b in batch]
        if args.wrist_mode == "concat":
            wrist_images = [b["wrist_image"] for b in batch]
            if any(w is None for w in wrist_images):
                raise ValueError("wrist_mode=concat but at least one sample is missing wrist_image.")
            images = [_concat_images(img, wrist) for img, wrist in zip(images, wrist_images)]
        if use_text_loss and args.text_loss_mode == "joint_after_act":
            base_texts = [
                build_prompt(b["instruction"], act_tokens, use_chat_template=use_chat_template) for b in batch
            ]
            coc_prefix_texts = [f"{p}\nExplain the action sequence:" for p in base_texts]
            coc_texts = [b.get("coc_text", "") for b in batch]
            full_texts = [
                f"{pref} {t}".strip() if t else pref for pref, t in zip(coc_prefix_texts, coc_texts)
            ]

            if use_chat_template:
                prompts = []
                prefix_prompts = []
                for full_text, prefix_text in zip(full_texts, coc_prefix_texts):
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": full_text}],
                        }
                    ]
                    prompts.append(
                        processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )
                    prefix_messages = [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": prefix_text}],
                        }
                    ]
                    prefix_prompts.append(
                        processor.apply_chat_template(
                            prefix_messages, tokenize=False, add_generation_prompt=True
                        )
                    )
            else:
                prompts = full_texts
                prefix_prompts = coc_prefix_texts

            model_inputs = processor(
                text=prompts,
                images=images,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            prefix_inputs = processor(
                text=prefix_prompts,
                images=images,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            labels = model_inputs["input_ids"].clone()
            labels[model_inputs["attention_mask"] == 0] = -100
            for i in range(labels.size(0)):
                prefix_len = int(prefix_inputs["attention_mask"][i].sum())
                labels[i, :prefix_len] = -100
        else:
            if use_chat_template:
                prompts = []
                for b in batch:
                    text_content = build_prompt(b["instruction"], act_tokens, use_chat_template=True)
                    messages = [
                        {
                            "role": "user",
                            "content": [{"type": "image"}, {"type": "text", "text": text_content}],
                        }
                    ]
                    prompts.append(
                        processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )
            else:
                prompts = [
                    build_prompt(b["instruction"], act_tokens, use_chat_template=False) for b in batch
                ]

            model_inputs = processor(
                text=prompts,
                images=images,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            labels = None
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
        if args.use_proprio:
            proprios = [b["proprio"] for b in batch]
            if any(p is None for p in proprios):
                raise ValueError("--use_proprio set but at least one sample is missing proprio.")
            out["proprio"] = torch.stack([p for p in proprios], dim=0)
        if use_text_loss:
            if args.text_loss_mode == "joint_after_act":
                out["text_labels"] = labels
            else:
                coc_texts = [b.get("coc_text", "") for b in batch]
                if use_chat_template:
                    full_texts = []
                    text_prompts = []
                    for b, coc in zip(batch, coc_texts):
                        prompt_text = build_coc_prompt(b["instruction"], use_chat_template=True)
                        full_text = f"{prompt_text} {coc}" if coc else prompt_text
                        messages = [
                            {
                                "role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": full_text}],
                            }
                        ]
                        full_texts.append(
                            processor.apply_chat_template(
                                messages, tokenize=False, add_generation_prompt=True
                            )
                        )
                        prefix_messages = [
                            {
                                "role": "user",
                                "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
                            }
                        ]
                        text_prompts.append(
                            processor.apply_chat_template(
                                prefix_messages, tokenize=False, add_generation_prompt=True
                            )
                        )
                else:
                    text_prompts = [
                        build_coc_prompt(b["instruction"], use_chat_template=False) for b in batch
                    ]
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
                text_labels = text_inputs["input_ids"].clone()
                text_labels[text_inputs["attention_mask"] == 0] = -100
                for i in range(text_labels.size(0)):
                    prefix_len = int(prefix_inputs["attention_mask"][i].sum())
                    text_labels[i, :prefix_len] = -100
                out["text_inputs"] = text_inputs
                out["text_labels"] = text_labels
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
        # Determine layers_pattern based on backbone architecture
        if "qwen2_5" in args.vlm_backbone.lower() or "qwen2.5" in args.vlm_backbone.lower():
            layers_pattern = "language_model.layers"
        else:
            layers_pattern = "model.layers"
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM,
            layers_to_transform=layers,
            layers_pattern=layers_pattern,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)

    # Build dataloaders after tokenizer update.
    print("[Main] Building dataloaders...", flush=True)
    use_text_loss = args.lambda_text > 0
    # Use chat template for Qwen2.5-VL (required for proper image token handling)
    use_chat_template = "qwen2_5" in args.vlm_backbone.lower() or "qwen2.5" in args.vlm_backbone.lower()
    train_loader, val_loader, act_token_ids = build_dataloaders(
        args, processor, act_tokens, use_text_loss=use_text_loss, use_chat_template=use_chat_template
    )
    print(f"[Main] Dataloaders built: train={len(train_loader)} batches, val={len(val_loader)} batches", flush=True)

    # Infer dimensions from a batch.
    print("[Main] Fetching first batch to infer dimensions...", flush=True)
    sample = next(iter(train_loader))
    print("[Main] First batch fetched", flush=True)
    actions = sample["actions"]
    z_hist = sample["z_hist"]
    action_dim = int(actions.shape[-1])
    latent_dim = int(z_hist.shape[-1])
    proprio_dim = 0
    if args.use_proprio:
        if "proprio" not in sample:
            raise ValueError("--use_proprio was set but the dataloader did not provide 'proprio'.")
        proprio_dim = int(sample["proprio"].shape[-1])

    # Prophet + policy wrapper.
    prophet = Prophet(
        emb_dim=latent_dim,
        hidden_dim=latent_dim,
        future_horizon=args.future_offset,
        n_layers=2,
        n_heads=8,
        dropout=0.0,
    ).to(device, dtype=dtype)

    num_heads = get_config_attr(model, "num_attention_heads", "num_attention_heads")
    wrapper = QwenVLAWrapper(
        vlm=model,
        hidden_size=get_config_attr(model, "hidden_size", "hidden_size"),
        num_attention_heads=num_heads,
        action_dim=action_dim,
        horizon=args.action_horizon,
        future_dim=latent_dim,
        enable_future_injection=not args.disable_future_injection,
        enable_proprio=bool(args.use_proprio),
        proprio_dim=proprio_dim,
        action_head_type=args.action_head,
        flow_steps=args.flow_steps_eval,
    ).to(device, dtype=dtype)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        print(f"[Resume] Loading checkpoint from {args.resume_from}", flush=True)
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)

        # Load LoRA weights if present
        if "lora_state_dict" in resume_ckpt and args.use_lora:
            model.load_state_dict(resume_ckpt["lora_state_dict"], strict=False)
            print("[Resume] Loaded LoRA weights", flush=True)

        # Load ACT embeddings if present
        if "act_embeddings" in resume_ckpt:
            num_act = len(act_tokens)
            embed_layer = model.get_base_model().get_input_embeddings() if args.use_lora else model.get_input_embeddings()
            with torch.no_grad():
                embed_layer.weight[-num_act:] = resume_ckpt["act_embeddings"].to(embed_layer.weight.dtype)
            print("[Resume] Loaded ACT embeddings", flush=True)

        # Load action head
        wrapper.action_head.load_state_dict(resume_ckpt["action_head_state_dict"])
        print("[Resume] Loaded action head", flush=True)

        # Load prophet
        prophet.load_state_dict(resume_ckpt["prophet_state_dict"])
        print("[Resume] Loaded prophet", flush=True)

        # Load future injection if present
        if wrapper.future_injection is not None and "future_injection_state_dict" in resume_ckpt:
            if resume_ckpt["future_injection_state_dict"] is not None:
                wrapper.future_injection.load_state_dict(resume_ckpt["future_injection_state_dict"])
                print("[Resume] Loaded future injection", flush=True)

        # Load proprio conditioner if present
        if wrapper.proprio_conditioner is not None and "proprio_conditioner_state_dict" in resume_ckpt:
            if resume_ckpt["proprio_conditioner_state_dict"] is not None:
                wrapper.proprio_conditioner.load_state_dict(resume_ckpt["proprio_conditioner_state_dict"])
                print("[Resume] Loaded proprio conditioner", flush=True)

        # Determine start epoch from checkpoint filename (e.g., llm_vla_epoch0.pt -> start at epoch 1)
        import re
        match = re.search(r"epoch(\d+)\.pt$", args.resume_from)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"[Resume] Starting from epoch {start_epoch}", flush=True)

    params = list(wrapper.action_head.parameters()) + list(prophet.parameters())
    if wrapper.future_injection is not None:
        params += list(wrapper.future_injection.parameters())
    if wrapper.proprio_conditioner is not None:
        params += list(wrapper.proprio_conditioner.parameters())
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

    joint_text_loss = bool(use_text_loss and args.text_loss_mode == "joint_after_act")
    print(f"[Training] Starting training for {args.max_epochs} epochs (from epoch {start_epoch})", flush=True)
    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, args.max_epochs):
        wrapper.train()
        prophet.train()
        p_oracle = scheduled_p_oracle(epoch)
        print(f"[Training] Epoch {epoch} started, p_oracle={p_oracle:.2f}", flush=True)

        for step, batch in enumerate(train_loader):
            model_inputs = {k: v.to(device) for k, v in batch["model_inputs"].items()}
            act_positions = batch["act_positions"].to(device)
            actions_gt = batch["actions"].to(device, dtype=dtype)
            z_hist = batch["z_hist"].to(device, dtype=dtype)
            z_future = batch["z_future"].to(device, dtype=dtype)
            proprio = None
            if args.use_proprio:
                proprio = batch["proprio"].to(device, dtype=dtype)

            # z_current: last frame of history for delta prediction
            z_current = z_hist[:, -1:, :]  # [B, 1, D]

            z_pred = prophet(z_hist)

            # For delta prediction: Prophet outputs delta, convert to absolute for future_mem
            if args.delta_prediction:
                z_pred_absolute = z_current + z_pred  # [B, K, D]
                delta_target = z_future - z_current  # [B, K, D]
            else:
                z_pred_absolute = z_pred
                delta_target = z_future

            future_mem = choose_future_memory(
                args.future_memory_source, z_future, z_pred_absolute, p_oracle, training=True
            )

            loss_text = torch.tensor(0.0, device=device)
            is_flow = args.action_head == "flow"
            if is_flow:
                if joint_text_loss:
                    text_labels = batch["text_labels"].to(device)
                    has_text = bool((text_labels != -100).any().item())
                    if has_text:
                        outputs = wrapper.vlm(
                            **model_inputs,
                            labels=text_labels,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        loss_text = outputs.loss
                    else:
                        outputs = wrapper.vlm(
                            **model_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                else:
                    outputs = wrapper.vlm(
                        **model_inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden = outputs.hidden_states[-1]
                bsz, horizon = act_positions.shape
                gather_index = act_positions.unsqueeze(-1).expand(bsz, horizon, hidden.size(-1))
                act_h = hidden.gather(dim=1, index=gather_index)
                if wrapper.proprio_conditioner is not None and proprio is not None:
                    act_h = wrapper.proprio_conditioner(act_h, proprio)
                if (
                    wrapper.future_injection is not None
                    and future_mem is not None
                    and not args.disable_future_injection
                ):
                    act_h = wrapper.future_injection(act_h, future_mem)
                t = torch.rand(bsz, horizon, 1, device=device, dtype=actions_gt.dtype)
                eps = torch.randn_like(actions_gt)
                x_t = (1.0 - t) * eps + t * actions_gt
                v_target = actions_gt - eps
                v_pred = wrapper.action_head(act_h, x_t, t)
                loss_action = torch.nn.functional.mse_loss(v_pred, v_target)
            else:
                if joint_text_loss:
                    text_labels = batch["text_labels"].to(device)
                    has_text = bool((text_labels != -100).any().item())
                    if has_text:
                        outputs = wrapper.vlm(
                            **model_inputs,
                            labels=text_labels,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        loss_text = outputs.loss
                    else:
                        outputs = wrapper.vlm(
                            **model_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    hidden = outputs.hidden_states[-1]
                    bsz, horizon = act_positions.shape
                    gather_index = act_positions.unsqueeze(-1).expand(bsz, horizon, hidden.size(-1))
                    act_h = hidden.gather(dim=1, index=gather_index)
                    if wrapper.proprio_conditioner is not None and proprio is not None:
                        act_h = wrapper.proprio_conditioner(act_h, proprio)
                    if (
                        wrapper.future_injection is not None
                        and future_mem is not None
                        and not args.disable_future_injection
                    ):
                        act_h = wrapper.future_injection(act_h, future_mem)
                    actions_pred = wrapper.action_head(act_h)
                else:
                    actions_pred, _ = wrapper(
                        model_inputs=model_inputs,
                        act_positions=act_positions,
                        future_memory=future_mem,
                        proprio=proprio,
                        disable_future_injection=args.disable_future_injection,
                    )
                loss_action = torch.nn.functional.mse_loss(actions_pred, actions_gt)
            # Loss is on delta when delta_prediction, else on absolute.
            loss_world = compute_world_loss_continuous(z_pred, delta_target)
            if use_text_loss and not joint_text_loss:
                text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
                text_labels = batch["text_labels"].to(device)
                has_text = bool((text_labels != -100).any().item())
                if has_text:
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
                p_gate = wrapper.proprio_gate_value()
                if is_flow:
                    with torch.no_grad():
                        actions_sampled = wrapper.action_head.sample(act_h, steps=args.flow_steps_eval)
                        train_action_mse = torch.nn.functional.mse_loss(
                            actions_sampled, actions_gt
                        ).item()
                    print(
                        f"[Epoch {epoch} Step {global_step}] "
                        f"loss={loss.item():.4f} flow={loss_action.item():.4f} "
                        f"act_mse={train_action_mse:.4f} world={loss_world.item():.4f} "
                        f"text={loss_text.item():.4f} gate={gate:.4f} proprio_gate={p_gate:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"[Epoch {epoch} Step {global_step}] "
                        f"loss={loss.item():.4f} action={loss_action.item():.4f} "
                        f"world={loss_world.item():.4f} text={loss_text.item():.4f} "
                        f"gate={gate:.4f} proprio_gate={p_gate:.4f}",
                        flush=True,
                    )
            global_step += 1
            if global_step == 1:
                print(f"[Training] First step completed", flush=True)

        # Simple validation.
        wrapper.eval()
        prophet.eval()
        val_action_losses = []
        val_text_losses = []
        is_flow = args.action_head == "flow"
        with torch.no_grad():
            for batch in val_loader:
                model_inputs = {k: v.to(device) for k, v in batch["model_inputs"].items()}
                act_positions = batch["act_positions"].to(device)
                actions_gt = batch["actions"].to(device, dtype=dtype)
                z_hist = batch["z_hist"].to(device, dtype=dtype)
                z_future = batch["z_future"].to(device, dtype=dtype)
                proprio = None
                if args.use_proprio:
                    proprio = batch["proprio"].to(device, dtype=dtype)

                z_current = z_hist[:, -1:, :]  # [B, 1, D]
                z_pred = prophet(z_hist)

                # Convert delta prediction to absolute
                if args.delta_prediction:
                    z_pred_absolute = z_current + z_pred
                else:
                    z_pred_absolute = z_pred

                future_mem = z_pred_absolute
                if args.corruption_mode == "oracle":
                    future_mem = z_future
                elif args.corruption_mode == "zero":
                    future_mem = torch.zeros_like(z_pred_absolute)
                elif args.corruption_mode == "random":
                    future_mem = torch.randn_like(z_pred_absolute)
                elif args.corruption_mode == "shuffle":
                    perm = torch.randperm(z_pred_absolute.size(0), device=z_pred_absolute.device)
                    future_mem = z_pred_absolute[perm]

                loss_text = None
                if joint_text_loss:
                    text_labels = batch["text_labels"].to(device)
                    has_text = bool((text_labels != -100).any().item())
                    if has_text:
                        outputs = wrapper.vlm(
                            **model_inputs,
                            labels=text_labels,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        loss_text = outputs.loss
                    else:
                        outputs = wrapper.vlm(
                            **model_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    hidden = outputs.hidden_states[-1]
                    bsz, horizon = act_positions.shape
                    gather_index = act_positions.unsqueeze(-1).expand(bsz, horizon, hidden.size(-1))
                    act_h = hidden.gather(dim=1, index=gather_index)
                    if wrapper.proprio_conditioner is not None and proprio is not None:
                        act_h = wrapper.proprio_conditioner(act_h, proprio)
                    if (
                        wrapper.future_injection is not None
                        and future_mem is not None
                        and not args.disable_future_injection
                    ):
                        act_h = wrapper.future_injection(act_h, future_mem)
                    if is_flow:
                        actions_pred = wrapper.action_head.sample(act_h, steps=args.flow_steps_eval)
                    else:
                        actions_pred = wrapper.action_head(act_h)
                else:
                    if is_flow:
                        outputs = wrapper.vlm(
                            **model_inputs,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                        hidden = outputs.hidden_states[-1]
                        bsz, horizon = act_positions.shape
                        gather_index = act_positions.unsqueeze(-1).expand(
                            bsz, horizon, hidden.size(-1)
                        )
                        act_h = hidden.gather(dim=1, index=gather_index)
                        if wrapper.proprio_conditioner is not None and proprio is not None:
                            act_h = wrapper.proprio_conditioner(act_h, proprio)
                        if (
                            wrapper.future_injection is not None
                            and future_mem is not None
                            and not args.disable_future_injection
                        ):
                            act_h = wrapper.future_injection(act_h, future_mem)
                        actions_pred = wrapper.action_head.sample(act_h, steps=args.flow_steps_eval)
                    else:
                        actions_pred, _ = wrapper(
                            model_inputs=model_inputs,
                            act_positions=act_positions,
                            future_memory=future_mem,
                            proprio=proprio,
                            disable_future_injection=args.disable_future_injection,
                        )
                loss_action = torch.nn.functional.mse_loss(actions_pred, actions_gt)
                val_action_losses.append(loss_action.item())
                if use_text_loss and joint_text_loss and loss_text is not None:
                    val_text_losses.append(float(loss_text.item()))
                elif use_text_loss and not joint_text_loss:
                    text_inputs = {k: v.to(device) for k, v in batch["text_inputs"].items()}
                    text_labels = batch["text_labels"].to(device)
                    has_text = bool((text_labels != -100).any().item())
                    if has_text:
                        text_out = wrapper.vlm(**text_inputs, labels=text_labels, return_dict=True)
                        val_text_losses.append(float(text_out.loss.item()))

        mean_val = float(sum(val_action_losses) / max(1, len(val_action_losses)))
        gate = wrapper.gate_value()
        p_gate = wrapper.proprio_gate_value()
        if use_text_loss and val_text_losses:
            mean_text = float(sum(val_text_losses) / max(1, len(val_text_losses)))
            print(
                f"[Epoch {epoch}] VAL action MSE={mean_val:.6f} text={mean_text:.6f} "
                f"gate={gate:.4f} proprio_gate={p_gate:.4f}"
            )
        else:
            print(f"[Epoch {epoch}] VAL action MSE={mean_val:.6f} gate={gate:.4f} proprio_gate={p_gate:.4f}")

        # Save checkpoint per epoch.
        ckpt = {
            "config": vars(args),
            "act_tokens": act_tokens,
            "action_head_state_dict": wrapper.action_head.state_dict(),
            "prophet_state_dict": prophet.state_dict(),
            "future_injection_state_dict": (
                wrapper.future_injection.state_dict() if wrapper.future_injection is not None else None
            ),
            "proprio_conditioner_state_dict": (
                wrapper.proprio_conditioner.state_dict() if wrapper.proprio_conditioner is not None else None
            ),
        }

        # Save LoRA adapter weights if using LoRA
        if args.use_lora:
            # PEFT model's state_dict() returns only adapter weights
            ckpt["lora_state_dict"] = model.state_dict()

        # Save ACT token embeddings (only the new tokens, not full embedding table)
        # Get embedding layer from the model
        if args.use_lora:
            embed_layer = model.get_base_model().get_input_embeddings()
        else:
            embed_layer = model.get_input_embeddings()
        # ACT tokens are the last `len(act_tokens)` in the vocabulary
        num_act_tokens = len(act_tokens)
        act_embeddings = embed_layer.weight[-num_act_tokens:].detach().cpu()
        ckpt["act_embeddings"] = act_embeddings

        ckpt_path = os.path.join(args.output_dir, f"llm_vla_epoch{epoch}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
