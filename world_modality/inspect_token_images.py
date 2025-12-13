from __future__ import annotations

import argparse
import os
from typing import Any, List, Optional

import numpy as np
from PIL import Image

from .config import DataConfig
from .data_sr100 import build_cache_paths


try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore
except Exception:  # pragma: no cover
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize frames that share the same world token id.")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--split", type=str, default="train", help="Cache split prefix to load (usually 'train').")
    p.add_argument("--image_key", type=str, default="rgb")
    p.add_argument("--out_dir", type=str, default="token_inspection")
    p.add_argument("--num_tokens", type=int, default=10, help="How many token ids to visualize (top by frequency).")
    p.add_argument("--samples_per_token", type=int, default=12, help="How many frames to show per token.")
    p.add_argument("--resize", type=int, default=224, help="Resize images to square size (0 = keep original).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--token_ids", type=str, default="", help="Optional comma-separated token ids to visualize.")
    return p.parse_args()


def to_pil(img: Any) -> Optional[Image.Image]:
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        if img.ndim == 3:
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            return Image.fromarray(img)
    try:
        import torch

        if isinstance(img, torch.Tensor):
            t = img.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW
                if t.dtype != torch.uint8:
                    t = (t * 255).clamp(0, 255).to(torch.uint8)
                arr = t.permute(1, 2, 0).numpy()
                return Image.fromarray(arr)
            if t.ndim == 3 and t.shape[-1] in (1, 3):  # HWC
                if t.dtype != torch.uint8:
                    t = (t * 255).clamp(0, 255).to(torch.uint8)
                return Image.fromarray(t.numpy())
    except Exception:
        pass
    return None


def make_grid(images: List[Image.Image], ncols: int, bg=(0, 0, 0)) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), bg)
    w, h = images[0].size
    nrows = (len(images) + ncols - 1) // ncols
    grid = Image.new("RGB", (ncols * w, nrows * h), bg)
    for i, im in enumerate(images):
        r, c = divmod(i, ncols)
        grid.paste(im, (c * w, r * h))
    return grid


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    data_cfg = DataConfig(dataset_name=args.dataset_name, cache_dir=args.cache_dir, image_key=args.image_key)
    paths = build_cache_paths(data_cfg, args.split)
    tokens = np.load(paths.tokens_path, mmap_mode="r").astype(np.int64)
    vocab = int(tokens.max()) + 1

    if args.token_ids.strip():
        token_ids = [int(x) for x in args.token_ids.split(",") if x.strip()]
    else:
        counts = np.bincount(tokens, minlength=vocab)
        token_ids = counts.argsort()[::-1][: args.num_tokens].tolist()

    ds = LeRobotDataset(args.dataset_name)

    for tid in token_ids:
        idxs = np.flatnonzero(tokens == tid)
        if idxs.size == 0:
            continue
        pick = rng.choice(idxs, size=min(args.samples_per_token, idxs.size), replace=False)
        ims: List[Image.Image] = []
        for gidx in pick.tolist():
            step = ds[int(gidx)]
            im = to_pil(step.get(args.image_key))
            if im is None:
                continue
            if args.resize and args.resize > 0:
                im = im.resize((args.resize, args.resize))
            ims.append(im.convert("RGB"))

        if not ims:
            continue
        ncols = int(np.ceil(np.sqrt(len(ims))))
        grid = make_grid(ims, ncols=ncols)
        out_path = os.path.join(args.out_dir, f"token_{tid:04d}_n{len(ims)}.jpg")
        grid.save(out_path, quality=90)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

