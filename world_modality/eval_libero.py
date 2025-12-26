"""LIBERO success-rate evaluation for LLM-VLA models.

Usage:
    python -m world_modality.eval_libero \
        --checkpoint logs_llm/E2_m4_delta/llm_vla_epoch0.pt \
        --suite libero_spatial \
        --n_episodes 10
"""
from __future__ import annotations

import argparse
import os
import sys

# Set EGL for headless rendering BEFORE any imports
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Patch torch.load for LIBERO compatibility (PyTorch 2.6+ requires weights_only)
_original_torch_load = torch.load


def _patched_torch_load(f, *args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(f, *args, **kwargs)


torch.load = _patched_torch_load

# Add LIBERO to path AFTER patching torch.load
sys.path.insert(0, "/home/ubuntu/LIBERO")


def parse_args():
    p = argparse.ArgumentParser(description="LIBERO success-rate evaluation")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to LLM-VLA checkpoint")
    p.add_argument(
        "--suite",
        type=str,
        default="libero_spatial",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
    )
    p.add_argument("--n_episodes", type=int, default=10, help="Episodes per task")
    p.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--camera_size", type=int, default=256, help="Camera resolution")
    p.add_argument("--save_videos", action="store_true", help="Save rollout videos")
    p.add_argument("--output_dir", type=str, default="eval_libero_results")
    p.add_argument("--vlm_backbone", type=str, default="qwen2_5_vl_3b_instruct")
    p.add_argument("--trust_remote_code", action="store_true")
    # World latent args (for Prophet)
    p.add_argument("--world_latents_source", type=str, default="vjepa")
    p.add_argument("--latent_suffix", type=str, default="m4")
    # Corruption modes for reliance testing
    p.add_argument(
        "--corruption_mode",
        type=str,
        default="none",
        choices=["none", "zero", "random", "shuffle"],
        help="Corrupt future memory to test reliance",
    )
    p.add_argument("--disable_future_injection", action="store_true")
    return p.parse_args()


def load_vlm_and_processor(backbone_name: str, trust_remote_code: bool, device: str, act_tokens: list, ckpt: dict):
    """Load VLM backbone and processor with ACT tokens, LoRA, and embeddings."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    backbone_map = {
        "qwen2_5_vl_3b_instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    }
    model_id = backbone_map.get(backbone_name, backbone_name)
    config = ckpt["config"]

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    # Add ACT tokens as special tokens (must be done before model loading)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": act_tokens})

    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=trust_remote_code,
    )
    # Resize embeddings to include new tokens
    vlm.resize_token_embeddings(len(processor.tokenizer))

    # Load ACT token embeddings if available
    if "act_embeddings" in ckpt:
        print("Loading ACT token embeddings from checkpoint...")
        num_act_tokens = len(act_tokens)
        embed_layer = vlm.get_input_embeddings()
        with torch.no_grad():
            embed_layer.weight[-num_act_tokens:] = ckpt["act_embeddings"].to(embed_layer.weight.dtype)
    else:
        print("WARNING: No ACT embeddings in checkpoint - using random initialization")

    # Apply LoRA if checkpoint has LoRA weights
    if "lora_state_dict" in ckpt and config.get("use_lora", False):
        print("Loading LoRA weights from checkpoint...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType

            # Reconstruct LoRA config from training config
            target_modules = [m.strip() for m in config.get("lora_target_modules", "q_proj,k_proj,v_proj,o_proj").split(",")]
            num_layers = vlm.config.num_hidden_layers
            lora_layers = config.get("lora_layers", 8)
            layers = list(range(max(0, num_layers - lora_layers), num_layers)) if num_layers else None

            # Match training: use correct layers_pattern for Qwen2.5-VL
            if "qwen2_5" in backbone_name.lower() or "qwen2.5" in backbone_name.lower():
                layers_pattern = "language_model.layers"
            else:
                layers_pattern = "model.layers"

            lora_cfg = LoraConfig(
                r=config.get("lora_r", 16),
                lora_alpha=config.get("lora_alpha", 32),
                lora_dropout=0.0,  # No dropout at inference
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
                layers_to_transform=layers,
                layers_pattern=layers_pattern,
            )
            vlm = get_peft_model(vlm, lora_cfg)
            vlm.load_state_dict(ckpt["lora_state_dict"], strict=False)
            print(f"LoRA weights loaded successfully")
        except ImportError:
            print("WARNING: peft not available, skipping LoRA loading")
    else:
        print("WARNING: No LoRA weights in checkpoint - using base model")

    vlm = vlm.to(device)
    vlm.eval()

    return vlm, processor


def load_checkpoint_config(checkpoint_path: str):
    """Load checkpoint to get config and act_tokens before VLM loading."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["config"], ckpt["act_tokens"], ckpt


def load_vjepa_encoder(model_name: str, device: str):
    """Load V-JEPA encoder for computing world latents at eval time."""
    from world_modality.vision import VisionEncoder
    encoder = VisionEncoder(model_name=model_name, device=device, dtype="float16")
    return encoder


def load_policy(checkpoint_path: str, vlm, device: str, ckpt: dict, disable_future_injection: bool = False):
    """Load LLM-VLA policy from checkpoint."""
    from world_modality.llm_vla_policy import QwenVLAWrapper, ActionHeadMLP
    from world_modality.model import GatedCrossAttention, Prophet

    config = ckpt["config"]
    act_tokens = ckpt["act_tokens"]

    # Get dimensions from config
    action_dim = config.get("action_dim", 7)
    horizon = config.get("action_horizon", 8)
    hidden_size = vlm.config.hidden_size
    num_heads = vlm.config.num_attention_heads
    latent_dim = config.get("latent_dim", 1408)

    # Create wrapper
    wrapper = QwenVLAWrapper(
        vlm=vlm,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        action_dim=action_dim,
        horizon=horizon,
        future_dim=latent_dim,
        enable_future_injection=not disable_future_injection,
    ).to(device).half()  # Match VLM dtype (float16)

    # Load action head
    wrapper.action_head.load_state_dict(ckpt["action_head_state_dict"])

    # Load future injection if present and enabled
    if not disable_future_injection and ckpt.get("future_injection_state_dict") is not None:
        wrapper.future_injection.load_state_dict(ckpt["future_injection_state_dict"])

    # Load Prophet (hidden_dim = emb_dim for this checkpoint)
    prophet = Prophet(
        emb_dim=latent_dim,
        hidden_dim=latent_dim,  # Same as emb_dim in training
        future_horizon=config.get("future_offset", 8),
        n_layers=2,
        n_heads=8,
        dropout=0.0,
    ).to(device).half()  # Match VLM dtype
    prophet.load_state_dict(ckpt["prophet_state_dict"])
    prophet.eval()

    wrapper.eval()
    return wrapper, prophet, act_tokens, config


def build_prompt(instruction: str, act_tokens: list, processor) -> str:
    """Build prompt for VLM using same format as training."""
    act_str = " ".join(act_tokens)
    instr = instruction.strip() if instruction else ""

    # Match training: simple format wrapped by chat template
    text_content = f"{instr}\n{act_str}" if instr else act_str

    # Use processor's chat template (same as training)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text_content}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Append the ACT tokens as assistant response start (model predicts actions here)
    prompt = prompt + act_str

    return prompt


@torch.no_grad()
def run_episode(
    env,
    wrapper,
    prophet,
    processor,
    act_tokens,
    act_token_ids,
    init_state,
    instruction: str,
    device: str,
    max_steps: int,
    corruption_mode: str = "none",
    disable_future_injection: bool = False,
    vjepa_encoder=None,
    delta_prediction: bool = True,
    temporal_frames: int = 4,
):
    """Run single episode, return success.

    When future injection is enabled:
    1. Track last `temporal_frames` frames in history
    2. Encode with V-JEPA to get current world latent z_t
    3. Use Prophet to predict future delta
    4. Add z_t to get predicted future z_{t+k}
    5. Pass to wrapper as future_memory
    """
    from world_modality.llm_vla_policy import find_act_positions

    # CRITICAL: set_init_state BEFORE reset (LeRobot convention)
    env.set_init_state(init_state)
    obs = env.reset()

    # CRITICAL: No-op steps after reset to let physics settle (LeRobot does 10 steps)
    dummy_action = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float64)  # 7-dim, gripper closed
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)

    # Track frame history for temporal encoding
    frame_history = []

    for step in range(max_steps):
        # Get image observation
        img = obs["agentview_image"]  # (H, W, 3) uint8
        # CRITICAL: Flip 180° to match HuggingFaceVLA/libero dataset convention
        img = img[::-1, ::-1].copy()  # Flip both H and W
        pil_img = Image.fromarray(img)

        # Update frame history
        frame_tensor = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
        frame_history.append(frame_tensor)
        if len(frame_history) > temporal_frames:
            frame_history.pop(0)

        # Compute future memory if enabled
        future_memory = None
        if not disable_future_injection and vjepa_encoder is not None and len(frame_history) >= temporal_frames:
            # Stack last m frames: [1, m, C, H, W]
            frames = torch.stack(frame_history[-temporal_frames:], dim=0).unsqueeze(0)

            # Encode with V-JEPA temporal
            z_current = vjepa_encoder.encode_temporal(frames)  # [1, D]

            # Use Prophet to predict future
            z_current_expanded = z_current.unsqueeze(1)  # [1, 1, D] for Prophet
            z_pred = prophet(z_current_expanded)  # [1, K, D] delta prediction

            # For delta prediction, add z_current to get absolute
            # z_current is [1, D], z_pred is [1, K, D] - broadcast over K
            if delta_prediction:
                z_pred_absolute = z_current.unsqueeze(1) + z_pred  # [1, K, D]
            else:
                z_pred_absolute = z_pred  # [1, K, D]

            # Apply corruption if requested
            if corruption_mode == "zero":
                z_pred_absolute = torch.zeros_like(z_pred_absolute)
            elif corruption_mode == "random":
                z_pred_absolute = torch.randn_like(z_pred_absolute)
            elif corruption_mode == "shuffle":
                # Shuffle within batch (no effect for B=1, but valid)
                idx = torch.randperm(z_pred_absolute.shape[0])
                z_pred_absolute = z_pred_absolute[idx]

            future_memory = z_pred_absolute  # [1, K, D]

        # Build prompt (same format as training)
        prompt = build_prompt(instruction, act_tokens, processor)

        # Process with VLM processor
        inputs = processor(
            text=[prompt],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Find ACT token positions
        act_positions = find_act_positions(inputs["input_ids"], act_token_ids)
        act_positions = act_positions.to(device)

        # Forward through wrapper
        actions, _ = wrapper(
            model_inputs=inputs,
            act_positions=act_positions,
            future_memory=future_memory,
            disable_future_injection=disable_future_injection,
        )

        # Get first action (actions is [B, H, action_dim])
        action = actions[0, 0].cpu().numpy()

        # Debug: print action stats for first few steps
        if step < 3:
            fm_status = "with_future" if future_memory is not None else "no_future"
            print(f"  Step {step} ({fm_status}): action = {action}, min={action.min():.3f}, max={action.max():.3f}")

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Step environment
        obs, reward, done, info = env.step(action)

        # CRITICAL: Use env.check_success() - info dict doesn't reliably have is_success
        if env.check_success():
            return True, step + 1

        if done:
            break

    return False, max_steps


def main():
    args = parse_args()

    # Import LIBERO
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    # First load checkpoint to get act_tokens
    print(f"Loading checkpoint config: {args.checkpoint}")
    config, act_tokens, ckpt = load_checkpoint_config(args.checkpoint)

    # Load VLM with act_tokens, LoRA, and embeddings
    print(f"Loading VLM backbone: {args.vlm_backbone}")
    vlm, processor = load_vlm_and_processor(
        args.vlm_backbone, args.trust_remote_code, args.device, act_tokens, ckpt
    )

    # Load policy weights
    print(f"Loading policy from checkpoint...")
    wrapper, prophet, act_tokens, config = load_policy(
        args.checkpoint, vlm, args.device, ckpt, args.disable_future_injection
    )

    # Get ACT token IDs (should work now since tokens were added)
    act_token_ids = processor.tokenizer.convert_tokens_to_ids(act_tokens)
    print(f"ACT tokens: {act_tokens[:3]}... -> IDs: {act_token_ids[:3]}...")

    # Load V-JEPA encoder if future injection is enabled
    vjepa_encoder = None
    if not args.disable_future_injection:
        print(f"Loading V-JEPA encoder for future memory...")
        vjepa_model = "facebook/vjepa2-vitg-fpc64-256"
        vjepa_encoder = load_vjepa_encoder(vjepa_model, args.device)

    # Get delta_prediction and temporal_frames from config
    delta_prediction = config.get("delta_prediction", True)
    temporal_frames = 4 if "m4" in config.get("latent_suffix", "m4") else 1
    print(f"Delta prediction: {delta_prediction}, Temporal frames: {temporal_frames}")

    # Load benchmark
    print(f"Loading LIBERO suite: {args.suite}")
    TaskSuiteClass = benchmark.get_benchmark(args.suite)
    task_suite = TaskSuiteClass()
    n_tasks = task_suite.get_num_tasks()
    print(f"Number of tasks: {n_tasks}")

    # Results tracking
    all_results = []
    task_successes = {}

    os.makedirs(args.output_dir, exist_ok=True)

    for task_idx in range(n_tasks):
        task = task_suite.get_task(task_idx)
        task_name = task.name
        instruction = task_suite.get_task(task_idx).language
        bddl_path = task_suite.get_task_bddl_file_path(task_idx)
        init_states = task_suite.get_task_init_states(task_idx)

        print(f"\n=== Task {task_idx + 1}/{n_tasks}: {task_name[:50]}... ===")
        print(f"Instruction: {instruction}")

        # Create environment
        env_args = {
            "bddl_file_name": bddl_path,
            "camera_heights": args.camera_size,
            "camera_widths": args.camera_size,
        }
        env = OffScreenRenderEnv(**env_args)

        task_success_count = 0
        task_results = []

        for ep_idx in tqdm(range(args.n_episodes), desc=f"Task {task_idx + 1}"):
            # Use different init states for each episode
            init_state = init_states[ep_idx % len(init_states)]

            success, steps = run_episode(
                env=env,
                wrapper=wrapper,
                prophet=prophet,
                processor=processor,
                act_tokens=act_tokens,
                act_token_ids=act_token_ids,
                init_state=init_state,
                instruction=instruction,
                device=args.device,
                max_steps=args.max_steps,
                corruption_mode=args.corruption_mode,
                disable_future_injection=args.disable_future_injection,
                vjepa_encoder=vjepa_encoder,
                delta_prediction=delta_prediction,
                temporal_frames=temporal_frames,
            )

            task_results.append({"episode": ep_idx, "success": success, "steps": steps})
            if success:
                task_success_count += 1

        env.close()

        success_rate = task_success_count / args.n_episodes * 100
        task_successes[task_name] = success_rate
        all_results.append(
            {
                "task_idx": task_idx,
                "task_name": task_name,
                "success_rate": success_rate,
                "successes": task_success_count,
                "episodes": args.n_episodes,
            }
        )
        print(f"Success rate: {success_rate:.1f}% ({task_success_count}/{args.n_episodes})")

    # Summary
    print("\n" + "=" * 60)
    print("LIBERO EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Suite: {args.suite}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Corruption: {args.corruption_mode}")
    print(f"Future injection: {'disabled' if args.disable_future_injection else 'enabled'}")
    print()

    for result in all_results:
        print(f"  {result['task_name'][:40]:40s} {result['success_rate']:5.1f}%")

    overall_sr = np.mean([r["success_rate"] for r in all_results])
    print(f"\nOverall Success Rate: {overall_sr:.1f}%")
    print("=" * 60)

    # Save results
    import json

    results_path = os.path.join(args.output_dir, f"{args.suite}_results.json")
    with open(results_path, "w") as f:
        json.dump(
            {
                "args": vars(args),
                "overall_success_rate": overall_sr,
                "per_task": all_results,
            },
            f,
            indent=2,
        )
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
