from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from world_modality.model import WorldPolicyTransformer
from world_modality.text import TextEncoder, TextEncoderConfig
from world_modality.vision import VisionEncoder
from world_modality.vq import VQCodebook


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop rollout evaluation on MetaWorld (success rate).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--codebook_centroids", type=str, required=True, help="Path to *_codebook_centroids.f32.npy")
    p.add_argument("--vision_model_name", type=str, default="facebook/dinov2-base")
    p.add_argument("--text_model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--text_max_length", type=int, default=64)

    p.add_argument("--task_indices", type=str, default="", help="Comma-separated task indices to eval (default: all).")
    p.add_argument("--task_map_json", type=str, default="", help="Optional JSON mapping task_index -> metaworld env_name.")
    p.add_argument(
        "--tasks_jsonl",
        type=str,
        default="",
        help="Optional path/URL to a tasks.jsonl file with fields like {task_index, task}.",
    )
    p.add_argument("--task_text_field", type=str, default="task")

    p.add_argument("--n_trials", type=int, default=10, help="Trials per task.")
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--env_num", type=int, default=10, help="Number of envs to run in parallel (python-level).")

    p.add_argument("--render_width", type=int, default=480)
    p.add_argument("--render_height", type=int, default=480)
    p.add_argument("--camera_name", type=str, default="corner2")

    p.add_argument(
        "--state_dim_override",
        type=int,
        default=0,
        help="Override checkpoint state_dim (useful if you want to feed a state subset).",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_codebook(centroids_path: str) -> VQCodebook:
    centroids = np.load(centroids_path).astype(np.float32)
    cb = VQCodebook(centroids=centroids)
    try:
        import faiss  # noqa: F401

        cb.faiss_index = None
        cb.faiss_index = _build_faiss_index(centroids)
    except Exception:
        cb.faiss_index = None
    return cb


def _build_faiss_index(centroids: np.ndarray):
    import faiss

    index = faiss.IndexFlatL2(centroids.shape[1])
    index.add(centroids.astype(np.float32))
    return index


def _load_tasks_jsonl(path_or_url: str, text_field: str) -> Dict[int, str]:
    if not path_or_url:
        return {}

    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import urllib.request

        with urllib.request.urlopen(path_or_url) as r:
            raw = r.read().decode("utf-8")
        lines = raw.splitlines()
    else:
        with open(path_or_url, "r") as f:
            lines = f.read().splitlines()

    mapping: Dict[int, str] = {}
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        obj = json.loads(ln)
        if "task_index" not in obj:
            continue
        ti = int(obj["task_index"])
        mapping[ti] = str(obj.get(text_field, ""))
    return mapping


def _load_task_map_json(path: str) -> Dict[int, str]:
    if not path:
        return {}
    with open(path, "r") as f:
        obj = json.load(f)
    out: Dict[int, str] = {}
    for k, v in obj.items():
        out[int(k)] = str(v)
    return out


def _render_rgb(env: Any, width: int, height: int, camera_name: str) -> np.ndarray:
    # MetaWorld envs may expose different render signatures depending on version.
    try:
        img = env.render(mode="rgb_array", width=width, height=height, camera_name=camera_name)
    except TypeError:
        try:
            img = env.render(mode="rgb_array")
        except Exception:
            img = env.render()
    img = np.asarray(img)
    # Ensure HWC uint8.
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


@torch.no_grad()
def policy_step_batch(
    model: WorldPolicyTransformer,
    model_type: str,
    vision: VisionEncoder,
    codebook: VQCodebook,
    text_encoder: Optional[TextEncoder],
    instruction: str,
    imgs: List[np.ndarray],
    states: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    # Vision embedding.
    img_emb = vision.encode(imgs).to(device)  # [B, d_e]

    current_world = None
    if model_type == "C":
        w = codebook.encode(img_emb.detach().cpu().numpy().astype(np.float32))
        current_world = torch.as_tensor(w, dtype=torch.long, device=device)

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
    return actions[:, 0, :].detach().cpu().numpy()


def _make_env(mt50: Any, env_name: str, task_idx: int, seed: int) -> Any:
    env_cls = mt50.train_classes[env_name]
    env = env_cls()

    # Best-effort camera adjustment to match the HF dataset description.
    try:
        env.model.cam_pos[2] = np.array([0.75, 0.075, 0.7], dtype=np.float32)
    except Exception:
        pass

    # Select a task variation for this env.
    tasks = [t for t in mt50.train_tasks if getattr(t, "env_name", "") == env_name]
    if tasks:
        env.set_task(tasks[task_idx % len(tasks)])

    try:
        env.seed(seed)
    except Exception:
        pass

    return env


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    try:
        import metaworld  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import metaworld. Install it first (see benchmarks/metaworld/README.md)."
        ) from e

    ckpt = torch.load(args.checkpoint, map_location=device)
    meta = ckpt.get("meta", {})
    cfg = ckpt.get("config", {})

    model_type = str(meta.get("model_type", cfg.get("model_type", "A")))
    use_language = bool(meta.get("use_language", cfg.get("use_language", False)))

    img_emb_dim = int(meta.get("img_emb_dim", 768))
    state_dim = int(meta.get("state_dim", 0))
    if args.state_dim_override > 0:
        state_dim = int(args.state_dim_override)
    action_dim = int(meta.get("action_dim", 4))
    horizon = int(meta.get("action_horizon", 8))
    future_horizon = int(meta.get("future_offset", 1))
    world_vocab_size = int(meta.get("world_vocab_size", 1024))
    lang_dim = int(meta.get("lang_dim", 0))
    world_input_scale = float(meta.get("world_input_scale", cfg.get("world_input_scale", 1.0)))
    world_input_dropout = float(meta.get("world_input_dropout", cfg.get("world_input_dropout", 0.0)))
    world_input_layernorm = bool(meta.get("world_input_layernorm", cfg.get("world_input_layernorm", False)))
    block_world_to_action = bool(meta.get("block_world_to_action", cfg.get("block_world_to_action", False)))

    if state_dim <= 0:
        raise ValueError("Checkpoint missing state_dim; retrain with updated training script.")
    if use_language and lang_dim <= 0:
        raise ValueError("Checkpoint expects language but lang_dim<=0; retrain with --use_language.")

    from world_modality.config import TransformerConfig

    model = WorldPolicyTransformer(
        model_type=model_type,  # type: ignore
        cfg=TransformerConfig(),
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

    # Prefer checkpoint-specified text model if present.
    text_model_name = str(meta.get("text_model_name", args.text_model_name))
    text_max_length = int(meta.get("text_max_length", args.text_max_length))

    text_encoder = None
    if use_language:
        text_encoder = TextEncoder(
            TextEncoderConfig(
                model_name=text_model_name,
                max_length=text_max_length,
                device=str(device),
                dtype="float16",
            )
        )

    # Task text mapping (optional) and env mapping.
    task_text = _load_tasks_jsonl(args.tasks_jsonl, args.task_text_field)
    task_map = _load_task_map_json(args.task_map_json)

    mt50 = metaworld.MT50()
    env_names = sorted(list(mt50.train_classes.keys()))

    # Task indices to evaluate.
    if args.task_indices.strip():
        task_indices = [int(x) for x in args.task_indices.split(",") if x.strip()]
    else:
        if task_text:
            task_indices = sorted(task_text.keys())
        else:
            task_indices = list(range(len(env_names)))

    # Default mapping: task_index -> env_names[task_index] (best-effort).
    def env_name_for_task(ti: int) -> str:
        if ti in task_map:
            return task_map[ti]
        if 0 <= ti < len(env_names):
            return env_names[ti]
        raise ValueError(
            f"task_index={ti} is out of range for default mapping (len(env_names)={len(env_names)}). "
            "Provide --task_map_json."
        )

    results: Dict[int, float] = {}

    for ti in task_indices:
        env_name = env_name_for_task(ti)
        instruction = task_text.get(ti, env_name)

        n_success = 0
        n_total = args.n_trials

        for start in range(0, args.n_trials, args.env_num):
            this_env_num = min(args.env_num, args.n_trials - start)

            envs = [
                _make_env(mt50, env_name=env_name, task_idx=(start + j), seed=args.seed + start + j)
                for j in range(this_env_num)
            ]
            obs = [e.reset() for e in envs]
            done = [False] * this_env_num
            success = [False] * this_env_num

            for _ in range(args.max_steps):
                imgs = [_render_rgb(e, args.render_width, args.render_height, args.camera_name) for e in envs]

                # Build state batch (best-effort: take first state_dim dims from obs).
                st = np.zeros((this_env_num, state_dim), dtype=np.float32)
                for i in range(this_env_num):
                    o = np.asarray(obs[i]).reshape(-1)
                    if o.shape[0] < state_dim:
                        raise ValueError(
                            f"Env observation dim {o.shape[0]} < state_dim {state_dim}. "
                            "Use --state_dim_override or retrain with matching state."
                        )
                    st[i] = o[:state_dim]

                act = policy_step_batch(
                    model=model,
                    model_type=model_type,
                    vision=vision,
                    codebook=codebook,
                    text_encoder=text_encoder,
                    instruction=instruction,
                    imgs=imgs,
                    states=st,
                    device=device,
                )
                # Clip actions into action space range.
                act = np.clip(act, -1.0, 1.0).astype(np.float32)

                for i, e in enumerate(envs):
                    if done[i]:
                        continue
                    o, _, d, info = e.step(act[i])
                    obs[i] = o
                    if isinstance(info, dict) and float(info.get("success", 0.0)) > 0.0:
                        success[i] = True
                        done[i] = True
                    else:
                        done[i] = done[i] or bool(d)

                if all(done):
                    break

            n_success += sum(int(x) for x in success)

            for e in envs:
                try:
                    e.close()
                except Exception:
                    pass

        sr = n_success / max(n_total, 1)
        results[ti] = sr
        print(f"[MetaWorld] task_index={ti:3d}  env='{env_name}'  success_rate={sr:.3f}  instr='{instruction}'")

    mean_sr = float(np.mean(list(results.values()))) if results else 0.0
    print("\n=== MetaWorld Summary ===")
    print(f"tasks:   {len(results)}")
    print(f"mean SR: {mean_sr:.4f}")


if __name__ == "__main__":
    main()
