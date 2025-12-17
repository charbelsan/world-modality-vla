from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .config import DataConfig
from .data_sr100 import build_cache_paths
from .text import TextEncoder, TextEncoderConfig


try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except Exception:  # pragma: no cover
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute per-frame instruction embeddings for a LeRobot dataset.")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, default="train", help="Cache prefix to write (use 'train').")
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--instruction_key", type=str, default="instruction")
    p.add_argument(
        "--tasks_jsonl",
        type=str,
        default="",
        help=(
            "Optional path/URL to a LeRobot-style meta/tasks.jsonl. "
            "If set, instruction text is derived from task_index -> task mapping instead of reading a string field."
        ),
    )
    p.add_argument("--task_index_key", type=str, default="task_index")
    p.add_argument("--task_text_field", type=str, default="task")
    p.add_argument("--episode_id_key", type=str, default="episode_id")
    p.add_argument("--text_model_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=256, help="Batch size for text encoding over unique strings.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(
        dataset_name=args.dataset_name,
        cache_dir=args.cache_dir,
        instruction_key=args.instruction_key,
        episode_id_key=args.episode_id_key,
        use_language=True,
    )
    paths = build_cache_paths(data_cfg, args.split)
    os.makedirs(os.path.dirname(paths.instruction_embeddings_path), exist_ok=True)

    ds = LeRobotDataset(args.dataset_name)

    # Optional: instruction derived from tasks.jsonl via task_index.
    task_index_to_text: Dict[int, str] = {}
    if args.tasks_jsonl:
        if args.tasks_jsonl.startswith("http://") or args.tasks_jsonl.startswith("https://"):
            import urllib.request

            with urllib.request.urlopen(args.tasks_jsonl) as r:
                raw = r.read().decode("utf-8")
            lines = raw.splitlines()
        else:
            with open(args.tasks_jsonl, "r") as f:
                lines = f.read().splitlines()

        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            if "task_index" not in obj:
                continue
            ti = int(obj["task_index"])
            task_index_to_text[ti] = str(obj.get(args.task_text_field, ""))

    # Collect instruction string per episode (first occurrence wins).
    episode_to_instruction: Dict[int, str] = {}
    has_episode_id = args.episode_id_key in ds[0]
    has_instruction = args.instruction_key in ds[0]
    has_task_index = args.task_index_key in ds[0]

    if args.tasks_jsonl:
        if not has_task_index:
            raise KeyError(
                f"--tasks_jsonl was provided, but dataset '{args.dataset_name}' lacks task index key "
                f"'{args.task_index_key}'."
            )
        if not task_index_to_text:
            raise ValueError(
                f"--tasks_jsonl was provided but no task_index->text mapping was loaded from '{args.tasks_jsonl}'."
            )
    else:
        if not has_instruction:
            raise KeyError(
                f"Dataset '{args.dataset_name}' does not contain instruction key '{args.instruction_key}'. "
                "Set --instruction_key correctly, or use --tasks_jsonl for task_index->text mapping."
            )

    for i in tqdm(range(len(ds)), desc="Scanning dataset"):
        step = ds[i]
        ep_id = int(step.get(args.episode_id_key, 0)) if has_episode_id else 0
        if ep_id in episode_to_instruction:
            continue
        if args.tasks_jsonl:
            ti = int(step.get(args.task_index_key, 0))
            if ti not in task_index_to_text:
                raise KeyError(
                    f"task_index {ti} not found in mapping from --tasks_jsonl. "
                    "Check that you passed the correct tasks.jsonl file."
                )
            episode_to_instruction[ep_id] = task_index_to_text[ti]
        else:
            episode_to_instruction[ep_id] = str(step.get(args.instruction_key, ""))

    episode_ids = sorted(episode_to_instruction.keys())
    instructions = [episode_to_instruction[eid] for eid in episode_ids]

    # Encode unique instruction strings (deduplicate exact duplicates).
    uniq: Dict[str, int] = {}
    uniq_texts: List[str] = []
    for t in instructions:
        if t not in uniq:
            uniq[t] = len(uniq_texts)
            uniq_texts.append(t)

    text_cfg = TextEncoderConfig(
        model_name=args.text_model_name,
        max_length=args.max_length,
        device=args.device,
        dtype=args.dtype,
    )
    encoder = TextEncoder(text_cfg)
    encoder.eval()

    uniq_embs: List[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(uniq_texts), args.batch_size), desc="Encoding instructions"):
            batch_texts = uniq_texts[start : start + args.batch_size]
            emb = encoder.encode(batch_texts).detach().cpu().numpy().astype(np.float16)
            uniq_embs.append(emb)
    uniq_embs_np = np.concatenate(uniq_embs, axis=0) if uniq_embs else np.zeros((0, 1), np.float16)

    # Map each episode id to embedding.
    episode_emb: Dict[int, np.ndarray] = {}
    for eid, txt in zip(episode_ids, instructions):
        episode_emb[eid] = uniq_embs_np[uniq[txt]]

    # Build per-frame embedding array aligned with dataset indices.
    d_lang = int(uniq_embs_np.shape[1]) if uniq_embs_np.ndim == 2 else 1
    out = np.zeros((len(ds), d_lang), dtype=np.float16)

    for i in tqdm(range(len(ds)), desc="Filling per-frame embeddings"):
        step = ds[i]
        ep_id = int(step.get(args.episode_id_key, 0)) if has_episode_id else 0
        out[i] = episode_emb[ep_id]

    np.save(paths.instruction_embeddings_path, out)
    print(f"Saved instruction embeddings to {paths.instruction_embeddings_path} (shape={out.shape})")


if __name__ == "__main__":
    main()