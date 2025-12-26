"""LIBERO success-rate evaluation for LLM-VLA models (Qwen-VL + ACT tokens).

This evaluates *closed-loop* success rate by running the policy in the LIBERO
environment and executing actions in a receding-horizon manner (use the first
action from the predicted chunk at each step).

Example:
    python -m world_modality.eval_libero \\
        --checkpoint logs_llm/E2_model_f_vjepa_m4_delta_mse/llm_vla_epoch4.pt \\
        --suite libero_spatial \\
        --n_episodes 10 \\
        --libero_root /home/ubuntu/LIBERO
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from typing import Deque, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="LIBERO success-rate evaluation (LLM-VLA)")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to LLM-VLA checkpoint")
    p.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    p.add_argument("--n_episodes", type=int, default=10, help="Episodes per task")
    p.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--camera_size", type=int, default=256, help="Camera resolution")
    p.add_argument("--output_dir", type=str, default="eval_libero_results")
    p.add_argument(
        "--vlm_backbone",
        type=str,
        default="",
        help="Override backbone (otherwise use the checkpoint config).",
    )
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument(
        "--libero_root",
        type=str,
        default="",
        help="Optional path to a local LIBERO checkout to add to sys.path (e.g., /home/ubuntu/LIBERO).",
    )
    # World encoder args (for Prophet + future memory).
    p.add_argument(
        "--vision_model_name",
        type=str,
        default="",
        help="Override vision model for world latents (DINO or V-JEPA). If empty, use a sensible default.",
    )
    p.add_argument(
        "--temporal_window",
        type=int,
        default=0,
        help="If >0, override temporal window for V-JEPA encoding. If 0, infer from checkpoint latent_suffix (e.g., m4).",
    )
    # Corruption modes for reliance testing.
    p.add_argument(
        "--corruption_mode",
        type=str,
        default="none",
        choices=["none", "zero", "random", "shuffle"],
        help="Corrupt future memory to test reliance.",
    )
    p.add_argument("--disable_future_injection", action="store_true")
    p.add_argument(
        "--flow_steps_eval",
        type=int,
        default=0,
        help="If >0, override flow sampling steps at inference.",
    )
    return p.parse_args()


def resolve_backbone_name(name: str) -> str:
    key = name.lower().strip()
    mapping = {
        "qwen2_5_vl_3b_instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen3_vl_3b_instruct": "Qwen/Qwen3-VL-3B-Instruct",
    }
    return mapping.get(key, name)


def _get_config_attr(model, name: str) -> int:
    if hasattr(model.config, name):
        return int(getattr(model.config, name))
    text_cfg = getattr(model.config, "text_config", None)
    if text_cfg is not None and hasattr(text_cfg, name):
        return int(getattr(text_cfg, name))
    raise AttributeError(f"Model config missing {name}.")


def _infer_action_dim(action_head_state_dict: dict) -> int:
    for key in ("net.3.weight", "net.2.weight"):
        w = action_head_state_dict.get(key)
        if isinstance(w, torch.Tensor):
            return int(w.shape[0])
    raise ValueError("Unable to infer action_dim from action_head_state_dict.")


def _infer_latent_dim(prophet_state_dict: dict) -> int:
    w = prophet_state_dict.get("output_proj.weight")
    if isinstance(w, torch.Tensor):
        return int(w.shape[0])
    q = prophet_state_dict.get("query_slots")
    if isinstance(q, torch.Tensor):
        return int(q.shape[-1])
    raise ValueError("Unable to infer latent_dim from prophet_state_dict.")


def _infer_temporal_window(latent_suffix: str) -> int:
    s = (latent_suffix or "").strip().lower()
    if s.startswith("m") and s[1:].isdigit():
        return int(s[1:])
    return 1


def _build_prompt(instruction: str, act_tokens: list, processor, use_chat_template: bool) -> str:
    act_text = " ".join(act_tokens)
    instr = instruction.strip() if instruction else ""
    if use_chat_template:
        text_content = f"{instr}\n{act_text}" if instr else act_text
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_content}]}
        ]
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if instr:
        return f"<image>\n{instr}\n{act_text}"
    return f"<image>\n{act_text}"


def load_checkpoint(checkpoint_path: str) -> dict:
    # torch.load compatibility: some environments default weights_only=True
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_vlm_and_processor(
    backbone_name: str,
    trust_remote_code: bool,
    device: str,
    act_tokens: list,
    ckpt: dict,
):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    config = ckpt["config"]
    model_id = resolve_backbone_name(backbone_name)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": act_tokens})

    vlm = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=trust_remote_code,
    )
    vlm.resize_token_embeddings(len(processor.tokenizer))

    # Load ACT token embeddings.
    if "act_embeddings" in ckpt:
        num_act_tokens = len(act_tokens)
        embed_layer = vlm.get_input_embeddings()
        with torch.no_grad():
            embed_layer.weight[-num_act_tokens:] = ckpt["act_embeddings"].to(embed_layer.weight.dtype)
    else:
        print("WARNING: No ACT embeddings in checkpoint - using random init", flush=True)

    # Apply LoRA if adapter weights exist.
    if "lora_state_dict" in ckpt and config.get("use_lora", False):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError:
            print("WARNING: peft not available, skipping LoRA loading", flush=True)
        else:
            target_modules = [
                m.strip()
                for m in config.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj").split(",")
                if m.strip()
            ]
            num_layers = _get_config_attr(vlm, "num_hidden_layers")
            lora_layers = int(config.get("lora_layers", 8))
            layers = list(range(max(0, num_layers - lora_layers), num_layers)) if num_layers else None

            if "qwen2_5" in backbone_name.lower() or "qwen2.5" in backbone_name.lower():
                layers_pattern = "language_model.layers"
            else:
                layers_pattern = "model.layers"

            lora_cfg = LoraConfig(
                r=int(config.get("lora_r", 16)),
                lora_alpha=int(config.get("lora_alpha", 32)),
                lora_dropout=0.0,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
                layers_to_transform=layers,
                layers_pattern=layers_pattern,
            )
            vlm = get_peft_model(vlm, lora_cfg)
            vlm.load_state_dict(ckpt["lora_state_dict"], strict=False)

    vlm = vlm.to(device)
    vlm.eval()
    return vlm, processor


def load_wrapper_and_prophet(
    vlm,
    device: str,
    ckpt: dict,
    disable_future_injection: bool,
    flow_steps_eval: int,
):
    from world_modality.llm_vla_policy import QwenVLAWrapper
    from world_modality.model import Prophet

    config = ckpt["config"]
    action_dim = _infer_action_dim(ckpt["action_head_state_dict"])
    horizon = int(config.get("action_horizon", 8))
    latent_dim = _infer_latent_dim(ckpt["prophet_state_dict"])

    action_head_type = str(config.get("action_head", "mse"))
    flow_steps = int(flow_steps_eval) if flow_steps_eval > 0 else int(config.get("flow_steps_eval", 8))

    hidden_size = _get_config_attr(vlm, "hidden_size")
    num_heads = _get_config_attr(vlm, "num_attention_heads")

    wrapper = QwenVLAWrapper(
        vlm=vlm,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        action_dim=action_dim,
        horizon=horizon,
        future_dim=latent_dim,
        enable_future_injection=not disable_future_injection,
        action_head_type=action_head_type,
        flow_steps=flow_steps,
    ).to(device)
    wrapper.action_head.load_state_dict(ckpt["action_head_state_dict"])
    if not disable_future_injection and ckpt.get("future_injection_state_dict") is not None:
        wrapper.future_injection.load_state_dict(ckpt["future_injection_state_dict"])
    wrapper.eval()

    prophet = Prophet(
        emb_dim=latent_dim,
        hidden_dim=latent_dim,
        future_horizon=int(config.get("future_offset", 8)),
        n_layers=2,
        n_heads=8,
        dropout=0.0,
    ).to(device)
    prophet.load_state_dict(ckpt["prophet_state_dict"])
    prophet.eval()
    return wrapper, prophet


@torch.no_grad()
def run_episode(
    env,
    wrapper,
    prophet,
    world_encoder,
    processor,
    instruction: str,
    act_tokens: list,
    act_token_ids: list,
    init_state,
    device: str,
    max_steps: int,
    corruption_mode: str,
    disable_future_injection: bool,
    temporal_window: int,
    context_frames: int,
    delta_prediction: bool,
    use_chat_template: bool,
):
    from world_modality.llm_vla_policy import find_act_positions

    # Match LeRobot convention: set_init_state BEFORE reset.
    env.set_init_state(init_state)
    obs = env.reset()

    # Let physics settle after reset (LeRobot uses 10 no-op steps).
    dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)

    frame_buf: Deque[torch.Tensor] = deque(maxlen=max(1, temporal_window))
    latent_buf: Deque[torch.Tensor] = deque(maxlen=max(1, context_frames))

    def _latent_from_obs(img_np: np.ndarray) -> torch.Tensor:
        if world_encoder is None:
            raise RuntimeError("world_encoder is required when future injection is enabled.")
        if getattr(world_encoder, "is_vjepa", False):
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]
            frame_buf.append(img_t)
            if temporal_window > 1:
                while len(frame_buf) < temporal_window:
                    frame_buf.appendleft(frame_buf[0])
                clip = torch.stack(list(frame_buf), dim=0).unsqueeze(0).to(device)  # [1, m, 3, H, W]
                z_t = world_encoder.encode_temporal(clip)  # [1, D]
            else:
                batch = img_t.unsqueeze(0).to(device)  # [1, 3, H, W]
                z_t = world_encoder.encode(batch)  # [1, D]
        else:
            # DINO path: use PIL input to match HF processors reliably.
            z_t = world_encoder.encode([Image.fromarray(img_np)])  # [1, D]
        return z_t

    for step in range(max_steps):
        img = obs["agentview_image"]  # (H, W, 3) uint8
        # Match HuggingFaceVLA/libero camera convention: 180° flip (H and W).
        img = img[::-1, ::-1].copy()
        pil_img = Image.fromarray(img)
        prompt = _build_prompt(instruction, act_tokens, processor, use_chat_template=use_chat_template)
        inputs = processor(text=[prompt], images=[pil_img], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        act_positions = find_act_positions(inputs["input_ids"], act_token_ids).to(device)

        future_mem = None
        if not disable_future_injection:
            z_t = _latent_from_obs(img)  # [1, D]
            latent_buf.append(z_t.squeeze(0))
            while len(latent_buf) < context_frames:
                latent_buf.appendleft(latent_buf[0])
            z_hist = torch.stack(list(latent_buf), dim=0).unsqueeze(0)  # [1, T, D]

            z_pred = prophet(z_hist)  # [1, K, D] (delta or absolute)
            if delta_prediction:
                z_current = z_hist[:, -1:, :]
                z_pred = z_current + z_pred
            future_mem = z_pred

            if corruption_mode == "zero":
                future_mem = torch.zeros_like(future_mem)
            elif corruption_mode == "random":
                future_mem = torch.randn_like(future_mem)
            elif corruption_mode == "shuffle":
                perm = torch.randperm(future_mem.size(1), device=future_mem.device)
                future_mem = future_mem[:, perm, :]

        actions, _ = wrapper(
            model_inputs=inputs,
            act_positions=act_positions,
            future_memory=future_mem,
            disable_future_injection=disable_future_injection,
        )
        action = actions[0, 0].detach().cpu().numpy()
        action = np.clip(action, -1.0, 1.0)
        obs, _, done, info = env.step(action)
        # LIBERO does not reliably expose success in info; use check_success().
        if env.check_success():
            return True, step + 1
        if done:
            break
    return False, max_steps


def main():
    # Set EGL for headless rendering BEFORE importing mujoco/libero.
    os.environ.setdefault("MUJOCO_GL", "egl")

    args = parse_args()

    if args.libero_root:
        sys.path.insert(0, args.libero_root)

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    ckpt = load_checkpoint(args.checkpoint)
    config = ckpt["config"]
    act_tokens = ckpt["act_tokens"]

    backbone = args.vlm_backbone.strip() or str(config.get("vlm_backbone", "qwen3_vl_3b_instruct"))
    use_chat_template = "qwen2_5" in backbone.lower() or "qwen2.5" in backbone.lower()

    vlm, processor = load_vlm_and_processor(
        backbone_name=backbone,
        trust_remote_code=args.trust_remote_code,
        device=args.device,
        act_tokens=act_tokens,
        ckpt=ckpt,
    )
    wrapper, prophet = load_wrapper_and_prophet(
        vlm=vlm,
        device=args.device,
        ckpt=ckpt,
        disable_future_injection=args.disable_future_injection,
        flow_steps_eval=args.flow_steps_eval,
    )

    act_token_ids = processor.tokenizer.convert_tokens_to_ids(act_tokens)
    if any(i is None or i < 0 for i in act_token_ids):
        raise RuntimeError("ACT tokens were not added to the tokenizer correctly.")

    # World encoder (for predicted future memory).
    world_encoder = None
    temporal_window = 1
    if not args.disable_future_injection:
        from world_modality.vision import VisionEncoder

        world_source = str(config.get("world_latents_source", "vjepa"))
        if args.vision_model_name:
            vision_model = args.vision_model_name
        elif world_source == "vjepa":
            vision_model = "facebook/vjepa2-vitg-fpc64-256"
        else:
            vision_model = "facebook/dinov2-base"

        temporal_window = (
            int(args.temporal_window)
            if args.temporal_window > 0
            else _infer_temporal_window(str(config.get("latent_suffix", "")))
        )
        if world_source == "dino" and temporal_window > 1:
            raise ValueError("temporal_window>1 is only supported for world_latents_source=vjepa")

        world_encoder = VisionEncoder(
            model_name=vision_model,
            device=args.device,
            dtype="float16" if args.device == "cuda" else "float32",
        )

    TaskSuiteClass = benchmark.get_benchmark(args.suite)
    task_suite = TaskSuiteClass()
    n_tasks = task_suite.get_num_tasks()
    os.makedirs(args.output_dir, exist_ok=True)

    context_frames = int(config.get("context_frames", 3))
    delta_prediction = bool(config.get("delta_prediction", False))

    per_task = []
    for task_idx in range(n_tasks):
        task = task_suite.get_task(task_idx)
        task_name = task.name
        instruction = task.language
        bddl_path = task_suite.get_task_bddl_file_path(task_idx)
        init_states = task_suite.get_task_init_states(task_idx)

        env_args = {
            "bddl_file_name": bddl_path,
            "camera_heights": args.camera_size,
            "camera_widths": args.camera_size,
        }
        env = OffScreenRenderEnv(**env_args)

        successes = 0
        for ep_idx in tqdm(range(args.n_episodes), desc=f"Task {task_idx + 1}/{n_tasks}"):
            init_state = init_states[ep_idx % len(init_states)]
            ok, _ = run_episode(
                env=env,
                wrapper=wrapper,
                prophet=prophet,
                world_encoder=world_encoder,
                processor=processor,
                instruction=instruction,
                act_tokens=act_tokens,
                act_token_ids=act_token_ids,
                init_state=init_state,
                device=args.device,
                max_steps=args.max_steps,
                corruption_mode=args.corruption_mode,
                disable_future_injection=args.disable_future_injection,
                temporal_window=temporal_window,
                context_frames=context_frames,
                delta_prediction=delta_prediction,
                use_chat_template=use_chat_template,
            )
            successes += int(ok)

        env.close()
        sr = 100.0 * float(successes) / float(args.n_episodes)
        per_task.append({"task": task_name, "success_rate": sr, "successes": successes})
        print(f"[{task_idx + 1}/{n_tasks}] {task_name[:50]}... SR={sr:.1f}% ({successes}/{args.n_episodes})")

    overall = float(np.mean([x["success_rate"] for x in per_task])) if per_task else 0.0
    print(f"\nOverall SR ({args.suite}): {overall:.1f}%")

    out_path = os.path.join(args.output_dir, f"{args.suite}_results.json")
    import json

    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "overall_success_rate": overall, "per_task": per_task}, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
