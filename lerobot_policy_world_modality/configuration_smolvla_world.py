from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


WorldLatentsSource = Literal["dino", "vjepa"]
WorldMemoryMode = Literal["pred", "oracle", "zero", "shuffle", "random"]


@PreTrainedConfig.register_subclass("smolvla_world")
@dataclass
class SmolVLAWorldConfig(SmolVLAConfig):
    """SmolVLA + world modality (Model-F style injection into action expert only).

    Notes:
    - Training can optionally use cached latents indexed by dataset global `index`.
    - Inference (env rollouts) uses an online world encoder + Prophet by default.
    """

    # ---- World latents / cache ----
    dataset_repo_id: str = "HuggingFaceVLA/libero"
    cache_dir: str = "cache"
    world_latents_source: WorldLatentsSource = "vjepa"
    latent_suffix: str = "m4"  # e.g. "m4" when temporal_window=4 was used for precompute

    # ---- World memory geometry ----
    context_frames: int = 4  # T_ctx (history length, z_{t-T+1..t})
    future_offset: int = 8  # K (predict z_{t+1..t+K})

    # ---- Injection ----
    enable_world_injection: bool = True
    world_inject_num_heads: int = 8
    world_gate_init: float = 0.0  # tanh(gate) starts at 0
    world_memory_mode_train: WorldMemoryMode = "pred"  # oracle only valid in offline training

    # ---- World predictor (Prophet) ----
    enable_world_predictor: bool = True
    prophet_layers: int = 2
    prophet_heads: int = 8
    prophet_dropout: float = 0.0
    delta_prediction: bool = True  # predict z_{t+k} - z_t

    # ---- Loss ----
    lambda_world: float = 0.2

    # ---- Online world encoder (rollouts) ----
    # Used when cached latents are not present (e.g., during env rollouts).
    world_vision_model_name: str = "facebook/vjepa2-vitg-fpc64-256"
    world_use_first_camera_only: bool = True  # easiest default for LIBERO

