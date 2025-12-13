from dataclasses import dataclass
from typing import Optional


@dataclass
class CoCDataConfig:
    dataset_name: str
    coc_jsonl: str
    image_key: str = "rgb"
    proprio_key: str = "proprio"
    action_key: str = "action"
    context_frames: int = 3
    action_horizon: int = 8
    future_offset: int = 8
    batch_size: int = 256
    num_workers: int = 8
    cache_dir: str = "cache"


@dataclass
class CoCModelConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    world_vocab_size: int = 1024
    max_coc_len: int = 128


@dataclass
class CoCTrainingConfig:
    learning_rate: float = 1e-4
    max_epochs: int = 10
    lambda_world_loss: float = 0.2
    alpha_coc_loss: float = 0.2
    warmup_steps: int = 0
    gradient_clip: float = 1.0
    seed: int = 42
    log_dir: str = "logs_coc"


@dataclass
class CoCExperimentConfig:
    data: CoCDataConfig
    model: CoCModelConfig
    train: CoCTrainingConfig

