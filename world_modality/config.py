from dataclasses import dataclass
from typing import Literal, Optional


ModelType = Literal["A", "B", "C", "C_no_world_input"]


@dataclass
class DataConfig:
    dataset_name: str
    image_key: str = "rgb"
    proprio_key: str = "proprio"
    action_key: str = "action"
    context_frames: int = 3  # T_ctx
    action_horizon: int = 8  # H
    future_offset: int = 8  # K
    train_split: str = "train"
    val_split: str = "val"
    num_workers: int = 8
    batch_size: int = 256
    cache_dir: str = "cache"
    max_train_steps: Optional[int] = None


@dataclass
class VisionConfig:
    model_name: str = "facebook/dinov2-base"
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class VQConfig:
    num_tokens: int = 1024
    sample_frames: int = 200_000
    kmeans_batch_size: int = 4096
    use_faiss: bool = True


@dataclass
class TransformerConfig:
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    model_type: ModelType = "A"
    max_epochs: int = 50
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lambda_world_loss: float = 0.2
    warmup_steps: int = 0  # Linear warmup steps (0 = no warmup)
    gradient_clip: float = 0.0  # Max gradient norm (0 = no clipping)
    log_wandb: bool = False
    wandb_project: str = "world-modality-sr100"
    seed: int = 42
    precision: str = "16-mixed"


@dataclass
class ExperimentConfig:
    data: DataConfig
    vision: VisionConfig
    vq: VQConfig
    transformer: TransformerConfig
    training: TrainingConfig
