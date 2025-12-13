from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import torch

from .config import DataConfig, TransformerConfig, VisionConfig, VQConfig
from .model import WorldPolicyTransformer
from .vision import VisionEncoder
from .vq import VQCodebook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time SR100 control with world-modality policy.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--world_vocab_size", type=int, default=1024)
    parser.add_argument("--codebook_centroids", type=str, default="", help="Path to VQ centroids .npy file.")
    parser.add_argument("--vision_model_name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--hz", type=float, default=10.0, help="Control frequency.")
    return parser.parse_args()


def load_codebook(centroids_path: str) -> VQCodebook:
    centroids = np.load(centroids_path).astype(np.float32)
    return VQCodebook(centroids=centroids)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt["config"]
    meta = ckpt.get("meta", {})

    transformer_cfg = TransformerConfig()
    # Use dimensionalities stored in checkpoint metadata when available.
    img_emb_dim = int(meta.get("img_emb_dim", 768))
    state_dim = int(meta.get("state_dim", 16))
    action_dim = int(meta.get("action_dim", 8))
    horizon = int(meta.get("action_horizon", model_cfg.get("action_horizon", 8)))
    model_type = meta.get("model_type", model_cfg.get("model_type", "A"))
    world_vocab_size = int(meta.get("world_vocab_size", args.world_vocab_size))

    model = WorldPolicyTransformer(
        model_type=model_type,
        cfg=transformer_cfg,
        obs_dim=img_emb_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        world_vocab_size=world_vocab_size,
        horizon=horizon,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    vision_enc = VisionEncoder(args.vision_model_name, device=device.type)

    # Load VQ codebook if provided (required for model C).
    codebook: Optional[VQCodebook] = None
    if args.codebook_centroids:
        codebook = load_codebook(args.codebook_centroids)

    dt = 1.0 / args.hz

    while True:
        tic = time.time()

        # TODO: integrate with actual SR100 camera / state / control APIs.
        # For now, we use placeholders.
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = np.zeros((1, state_dim), dtype=np.float32)

        frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            img_emb = vision_enc.encode(frame_t)[0].unsqueeze(0).to(device)

        if codebook is not None:
            cur_token = codebook.encode(img_emb.cpu().numpy().astype(np.float32))[0]
            cur_token_t = torch.as_tensor([cur_token], dtype=torch.long, device=device)
        else:
            cur_token_t = None

        state_t = torch.from_numpy(state).to(device)

        with torch.no_grad():
            actions, world_logits = model(
                img_emb=img_emb,
                state=state_t,
                current_world_token=cur_token_t if model_type == "C" else None,
            )

        action_to_execute = actions[0, 0].cpu().numpy()

        # TODO: send `action_to_execute` to SR100 controllers.
        _ = action_to_execute

        # Sleep to maintain control frequency.
        elapsed = time.time() - tic
        if elapsed < dt:
            time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
