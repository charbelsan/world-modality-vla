from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


# Type aliases for documentation (draccus can't parse Literal from CLI, so we use str)
# Valid values: WorldLatentsSource: "dino" | "vjepa"
# Valid values: WorldMemoryMode: "pred" | "oracle" | "zero" | "shuffle" | "random"
# Valid values: WorldMemoryModeRollout: "pred" | "zero" | "random"


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
    world_latents_source: str = "vjepa"  # "dino" or "vjepa"
    latent_suffix: str = "m4"  # e.g. "m4" when temporal_window=4 was used for precompute
    # NOTE: this must match the cached latents file second dimension.
    # For the default encoder `facebook/vjepa2-vitg-fpc64-256`, embedding dim is 1408.
    world_latent_dim: int = 1408

    # Optional init: load SmolVLA weights into this policy before training starts.
    # Useful for starting from `lerobot/smolvla_base` or a local exported policy dir.
    init_from_policy_path: str = ""
    init_from_strict: bool = False

    # ---- Hub pushing (disable by default for experiments) ----
    push_to_hub: bool = False

    # ---- World memory geometry ----
    context_frames: int = 4  # T_ctx (history length, z_{t-T+1..t})
    future_offset: int = 8  # K (predict z_{t+1..t+K})

    # ---- Injection ----
    enable_world_injection: bool = True
    world_inject_num_heads: int = 8
    world_gate_init: float = 0.0  # tanh(gate) starts at 0
    world_memory_mode_train: str = "pred"  # "pred" | "oracle" | "zero" | "shuffle" | "random"
    log_attn_stats: bool = True
    log_grad_stats: bool = True
    world_memory_mode_rollout: str = "pred"  # "pred" | "zero" | "random"

    # ---- Optional fusion ablations (F2/F3) ----
    # F2: inject world memory into the *suffix input embeddings* (earlier in the expert path).
    world_inject_suffix_in: bool = False

    # F3a: treat world as prefix modality by inserting projected world tokens into the prefix padding.
    # 0 disables. When >0, inserts up to this many tokens (clipped by available prefix padding).
    world_prefix_tokens: int = 0
    # Attention block for inserted prefix tokens:
    # - "state": mask_ar=1 (image/lang cannot attend to world; state/actions can)
    # - "prefix": mask_ar=0 (world tokens join image/lang block; full token competition)
    world_prefix_block: str = "state"
    world_prefix_gate_init: float = 0.0

    # F3b: cross-attend from prefix embeddings into world memory (gated), without adding tokens.
    world_prefix_cross_attn: bool = False
    world_prefix_num_heads: int = 8

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

    # Rollout temporal encoding:
    # Cached V-JEPA latents can be temporal (e.g., latent_suffix="m4"). If rollout uses single-frame
    # embeddings, Prophet sees a mismatched input distribution and predicted memory can harm SR.
    # Set to 0 to infer from `latent_suffix` ("m4" -> 4), otherwise use the provided window.
    world_rollout_temporal_window: int = 0
