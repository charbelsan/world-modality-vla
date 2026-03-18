from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .config import DataConfig
from .llm_vla_dataset import build_latent_cache_paths


def _sharded_output_path(path: str, *, shard_index: int, num_shards: int) -> str:
    if num_shards <= 1:
        return path
    base, ext = os.path.splitext(path)
    return f"{base}.shard{shard_index:02d}-of-{num_shards:02d}{ext}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded world-latent npy files into one cache file.")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--world_latents_source", type=str, required=True)
    parser.add_argument("--latent_suffix", type=str, default="")
    parser.add_argument("--num_shards", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.num_shards) < 2:
        raise ValueError("--num_shards must be >= 2 for merging")

    cfg = DataConfig(dataset_name=args.dataset_name, cache_dir=args.cache_dir)
    cfg.latent_suffix = str(args.latent_suffix or "")
    paths = build_latent_cache_paths(cfg, args.split, args.world_latents_source)

    shard_arrays = []
    shard_metas = []
    for shard_index in range(int(args.num_shards)):
        shard_path = _sharded_output_path(paths.latents_path, shard_index=shard_index, num_shards=int(args.num_shards))
        meta_path = _sharded_output_path(
            paths.latents_path.replace(".npy", "_metadata.json"),
            shard_index=shard_index,
            num_shards=int(args.num_shards),
        )
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Missing shard file: {shard_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Missing shard metadata: {meta_path}")
        shard_arrays.append(np.load(shard_path, mmap_mode="r"))
        with open(meta_path, "r") as f:
            shard_metas.append(json.load(f))

    total_rows = int(shard_metas[0]["total_frames"])
    emb_dim = int(shard_arrays[0].shape[1])
    merged = np.lib.format.open_memmap(paths.latents_path, mode="w+", dtype=np.float16, shape=(total_rows, emb_dim))

    for shard_index, arr in enumerate(shard_arrays):
        indices = np.arange(total_rows, dtype=np.int64)[shard_index :: int(args.num_shards)]
        if int(arr.shape[0]) != int(indices.shape[0]):
            raise ValueError(
                f"Shard {shard_index} row mismatch: file has {int(arr.shape[0])}, expected {int(indices.shape[0])}"
            )
        merged[indices] = arr

    merged.flush()

    meta = dict(shard_metas[0])
    meta["merged_from_num_shards"] = int(args.num_shards)
    meta["embedding_dim"] = emb_dim
    meta["final_written_rows"] = total_rows
    meta["shard_index"] = None
    meta_path = paths.latents_path.replace(".npy", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Merged latents saved to {paths.latents_path}")
    print(f"Merged metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
