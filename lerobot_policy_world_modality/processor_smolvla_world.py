from __future__ import annotations

from typing import Any

import torch

from lerobot.processor import (
    ComplementaryDataProcessorStep,
    DeviceProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStepRegistry,
)
from lerobot.processor.batch_processor import AddBatchDimensionProcessorStep
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

from .configuration_smolvla_world import SmolVLAWorldConfig
from .world_latents_cache import WorldLatentsCache


@ProcessorStepRegistry.register(name="world_latents_from_cache")
class WorldLatentsFromCacheStep(ComplementaryDataProcessorStep):
    """Attach cached world latents to the batch via complementary data.

    Requires `index` in complementary data (present in offline dataset batches).
    Skips silently when `index` is missing (e.g., env rollouts).
    """

    def __init__(
        self,
        *,
        dataset_repo_id: str,
        cache_dir: str,
        source: str,
        latent_suffix: str,
        context_frames: int,
        future_offset: int,
    ):
        super().__init__()
        self._cache: WorldLatentsCache | None = None
        self._cache_args = {
            "dataset_repo_id": dataset_repo_id,
            "cache_dir": cache_dir,
            "source": source,
            "latent_suffix": latent_suffix,
            "context_frames": context_frames,
            "future_offset": future_offset,
        }

    def complementary_data(self, complementary_data: dict[str, Any]) -> dict[str, Any]:
        if "index" not in complementary_data:
            return complementary_data

        if self._cache is None:
            self._cache = WorldLatentsCache(**self._cache_args)

        index = complementary_data["index"]
        batch = self._cache.get_by_index(index)
        if batch is None:
            return complementary_data

        out = dict(complementary_data)
        out["world_z_hist"] = batch.z_hist
        out["world_z_future"] = batch.z_future
        out["world_future_valid"] = batch.future_valid
        return out

    def transform_features(self, features):
        # We add extra tensors to complementary data; do not advertise as policy features.
        return features


def make_smolvla_world_pre_post_processors(
    config: SmolVLAWorldConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    pre, post = make_smolvla_pre_post_processors(config, dataset_stats=dataset_stats)

    # Insert our cache step after AddBatchDimension so `index` is batched, and before Device so it gets moved.
    step = WorldLatentsFromCacheStep(
        dataset_repo_id=config.dataset_repo_id,
        cache_dir=config.cache_dir,
        source=config.world_latents_source,
        latent_suffix=config.latent_suffix,
        context_frames=config.context_frames,
        future_offset=config.future_offset,
    )

    new_steps = []
    inserted = False
    for s in pre.steps:
        new_steps.append(s)
        if isinstance(s, AddBatchDimensionProcessorStep) and not inserted:
            new_steps.append(step)
            inserted = True
    if not inserted:
        # Fallback: insert before device step.
        out_steps = []
        for s in new_steps:
            if isinstance(s, DeviceProcessorStep) and not inserted:
                out_steps.append(step)
                inserted = True
            out_steps.append(s)
        new_steps = out_steps

    pre = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=new_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
    )
    return pre, post
