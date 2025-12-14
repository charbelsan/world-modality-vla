from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from world_modality.model import WorldPolicyTransformer
from world_modality.vq import VQCodebook
from world_modality.vision import VisionEncoder
from world_modality.text import TextEncoder, TextEncoderConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop rollout evaluation on LIBERO (success rate).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--codebook_centroids", type=str, required=True, help="Path to *_codebook_centroids.f32.npy")
    p.add_argument("--vision_model_name", type=str, default="facebook/dinov2-base")
    p.add_argument("--text_model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--text_max_length", type=int, default=64)
    p.add_argument("--benchmark", type=str, default="libero_10")
    p.add_argument("--task_order_index", type=int, default=0)
    p.add_argument("--task_ids", type=str, default="", help="Comma-separated task indices to eval (default: all).")
    p.add_argument("--env_num", type=int, default=10)
    p.add_argument("--n_trials", type=int, default=10, help="Trials per task (each trial = one env rollout).")
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--camera_key", type=str, default="agentview_image")
    p.add_argument("--state_key", type=str, default="robot0_proprio-state")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--libero_config_path", type=str, default="", help="Set LIBERO_CONFIG_PATH for non-interactive runs.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _write_libero_config_if_missing(libero_config_path: str) -> None:
    """
    LIBERO's `libero.libero` module can prompt interactively if config.yaml doesn't exist.
    We avoid this by creating a config.yaml before importing LIBERO.
    """
    import yaml

    if not libero_config_path:
        libero_config_path = os.path.expanduser("~/.libero")
    os.makedirs(libero_config_path, exist_ok=True)
    config_file = os.path.join(libero_config_path, "config.yaml")
    if os.path.exists(config_file):
        return

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not locate 'libero.libero'. Install LIBERO first (see benchmarks/libero/README.md)."
        )
    benchmark_root = os.path.dirname(os.path.abspath(spec.origin))

    cfg = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "bddl_files"),
        "init_states": os.path.join(benchmark_root, "init_files"),
        "datasets": os.path.join(os.path.dirname(benchmark_root), "datasets"),
        "assets": os.path.join(benchmark_root, "assets"),
    }
    with open(config_file, "w") as f:
        yaml.safe_dump(cfg, f)


def load_codebook(centroids_path: str) -> VQCodebook:
    centroids = np.load(centroids_path).astype(np.float32)
    cb = VQCodebook(centroids=centroids)
    try:
        import faiss  # noqa: F401

        cb.faiss_index = None
        cb.faiss_index = __build_faiss_index(centroids)
    except Exception:
        cb.faiss_index = None
    return cb


def __build_faiss_index(centroids: np.ndarray):
    import faiss

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids.astype(np.float32))
    return index


def extract_batch(obs: Dict[str, Any], key: str) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Observation missing key '{key}'. Available keys: {list(obs.keys())[:50]}")
    arr = obs[key]
    return np.asarray(arr)


@torch.no_grad()
def policy_step(
    model: WorldPolicyTransformer,
    model_type: str,
    vision: VisionEncoder,
    codebook: VQCodebook,
    text_encoder: Optional[TextEncoder],
    instruction: str,
    obs: Dict[str, Any],
    camera_key: str,
    state_key: str,
    device: torch.device,
) -> np.ndarray:
    imgs = extract_batch(obs, camera_key)  # [B, H, W, 3]
    states = extract_batch(obs, state_key)  # [B, Ds]

    # Vision embedding.
    img_emb = vision.encode(list(imgs)).to(device)  # [B, d_e]

    # Current world token (for model C only).
    current_world = None
    if model_type == "C":
        w = codebook.encode(img_emb.detach().cpu().numpy().astype(np.float32))
        current_world = torch.as_tensor(w, dtype=torch.long, device=device)

    # Language embedding (optional).
    lang_emb = None
    if text_encoder is not None:
        lang = text_encoder.encode([instruction]).to(device)  # [1, d_lang]
        lang_emb = lang.expand(img_emb.shape[0], -1).contiguous()

    actions, _ = model(
        img_emb=img_emb,
        state=torch.as_tensor(states, dtype=torch.float32, device=device),
        current_world_token=current_world,
        lang_emb=lang_emb,
    )

    # Execute first action (MPC-style).
    a0 = actions[:, 0, :].detach().cpu().numpy()
    return a0


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path
        _write_libero_config_if_missing(args.libero_config_path)
    else:
        _write_libero_config_if_missing(os.path.expanduser("~/.libero"))

    try:
        from libero.libero.benchmark import get_benchmark  # type: ignore
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv  # type: ignore
        from libero.libero.envs.venv import SubprocVectorEnv  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import LIBERO. Install it first (see benchmarks/libero/README.md)."
        ) from e

    ckpt = torch.load(args.checkpoint, map_location=device)
    meta = ckpt.get("meta", {})
    cfg = ckpt.get("config", {})

    model_type = str(meta.get("model_type", cfg.get("model_type", "A")))
    use_language = bool(meta.get("use_language", cfg.get("use_language", False)))

    img_emb_dim = int(meta.get("img_emb_dim", 768))
    if "state_dim" not in meta:
        raise ValueError("Checkpoint missing meta['state_dim']; re-train with updated training script.")
    state_dim = int(meta["state_dim"])
    action_dim = int(meta.get("action_dim", 7))
    horizon = int(meta.get("action_horizon", 8))
    future_horizon = int(meta.get("future_offset", 1))
    world_vocab_size = int(meta.get("world_vocab_size", 1024))
    lang_dim = int(meta.get("lang_dim", 0))
    world_input_scale = float(meta.get("world_input_scale", cfg.get("world_input_scale", 1.0)))
    world_input_dropout = float(meta.get("world_input_dropout", cfg.get("world_input_dropout", 0.0)))
    world_input_layernorm = bool(meta.get("world_input_layernorm", cfg.get("world_input_layernorm", False)))
    block_world_to_action = bool(meta.get("block_world_to_action", cfg.get("block_world_to_action", False)))
    if use_language and lang_dim <= 0:
        raise ValueError(
            "use_language=True but checkpoint meta has lang_dim<=0. "
            "Train with --use_language and instruction embeddings."
        )

    # Build model.
    from world_modality.config import TransformerConfig

    transformer_cfg = TransformerConfig()
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
        lang_dim=lang_dim if use_language else 0,
        world_input_scale=world_input_scale,
        world_input_dropout=world_input_dropout,
        world_input_layernorm=world_input_layernorm,
        block_world_to_action=block_world_to_action,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    vision = VisionEncoder(args.vision_model_name, device=str(device))
    codebook = load_codebook(args.codebook_centroids)
    text_encoder = None
    if use_language:
        # Prefer checkpoint-specified text model if present.
        text_model_name = str(meta.get("text_model_name", args.text_model_name))
        text_max_length = int(meta.get("text_max_length", args.text_max_length))
        text_encoder = TextEncoder(
            TextEncoderConfig(
                model_name=text_model_name,
                max_length=text_max_length,
                device=str(device),
                dtype="float16",
            )
        )

    benchmark = get_benchmark(args.benchmark)(args.task_order_index)
    all_task_ids = list(range(benchmark.get_num_tasks()))
    if args.task_ids.strip():
        all_task_ids = [int(x) for x in args.task_ids.split(",") if x.strip()]

    results: Dict[int, float] = {}
    for task_id in all_task_ids:
        task = benchmark.get_task(task_id)
        instruction = task.language

        env_args = {
            "bddl_file_name": benchmark.get_task_bddl_file_path(task_id),
            "camera_heights": 128,
            "camera_widths": 128,
        }

        # Run rollouts in batches of env_num.
        n_success = 0
        n_total = args.n_trials
        for start in range(0, args.n_trials, args.env_num):
            this_env_num = min(args.env_num, args.n_trials - start)
            env = SubprocVectorEnv(
                [lambda: OffScreenRenderEnv(**env_args) for _ in range(this_env_num)]
            )
            env.reset()
            env.seed(args.seed)

            init_states = benchmark.get_task_init_states(task_id)
            idxs = np.arange(this_env_num) % init_states.shape[0]
            obs = env.set_init_state(init_states[idxs])

            # Sanity-check state dimensionality.
            st = extract_batch(obs, args.state_key)
            if st.ndim != 2 or st.shape[1] != state_dim:
                raise ValueError(
                    f"State dim mismatch: env '{args.state_key}' has shape {st.shape} "
                    f"but checkpoint expects state_dim={state_dim}. "
                    "Adjust --state_key or retrain with matching proprio."
                )

            # Let physics settle with null actions.
            for _ in range(args.warmup_steps):
                env.step(np.zeros((this_env_num, action_dim), dtype=np.float32))

            done = [False] * this_env_num
            for _ in range(args.max_steps):
                act = policy_step(
                    model=model,
                    model_type=model_type,
                    vision=vision,
                    codebook=codebook,
                    text_encoder=text_encoder,
                    instruction=instruction,
                    obs=obs,
                    camera_key=args.camera_key,
                    state_key=args.state_key,
                    device=device,
                )
                obs, _, d, _ = env.step(act)
                for i in range(this_env_num):
                    done[i] = done[i] or bool(d[i])
                if all(done):
                    break

            n_success += sum(int(x) for x in done)
            env.close()

        sr = n_success / max(n_total, 1)
        results[task_id] = sr
        print(f"[LIBERO] task_id={task_id:3d}  success_rate={sr:.3f}  instr='{instruction}'")

    mean_sr = float(np.mean(list(results.values()))) if results else 0.0
    print("\n=== LIBERO Summary ===")
    print(f"benchmark: {args.benchmark}")
    print(f"tasks:     {len(results)}")
    print(f"mean SR:   {mean_sr:.4f}")


if __name__ == "__main__":
    main()
