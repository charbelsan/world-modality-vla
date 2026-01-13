"""LeRobot policy plugin: world modality experiments.

LeRobot auto-discovers packages named `lerobot_policy_*` and imports them at startup.
Importing this package registers the `smolvla_world` policy config via draccus ChoiceRegistry.
"""

from __future__ import annotations

from .configuration_smolvla_world import SmolVLAWorldConfig  # noqa: F401
from .modeling_smolvla_world import SmolVLAWorldPolicy  # noqa: F401
from .processor_smolvla_world import make_smolvla_world_pre_post_processors


def _patch_lerobot_factories() -> None:
    """Patch LeRobot factories to support out-of-tree policy types.

    LeRobot 0.4.x ships a hard-coded `get_policy_class()` switch and does not know about
    our plugin policies (`smolvla_world`). Since our wrapper entrypoints (`lerobot-wm-*`)
    import this package before invoking LeRobot scripts, we can safely monkeypatch the
    factories at import time.
    """

    try:
        import lerobot.policies.factory as factory  # type: ignore
        from lerobot.processor import DeviceProcessorStep, RenameObservationsProcessorStep  # type: ignore
    except Exception:
        # LeRobot not installed; allow importing this package for docs/lint.
        return

    if getattr(factory, "_world_modality_patched", False):
        return

    orig_get_policy_class = factory.get_policy_class
    orig_make_pre_post_processors = factory.make_pre_post_processors

    def get_policy_class(name: str):  # type: ignore[no-untyped-def]
        if name == "smolvla_world":
            return SmolVLAWorldPolicy
        return orig_get_policy_class(name)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(policy_cfg, SmolVLAWorldConfig):
            pre, post = make_smolvla_world_pre_post_processors(
                config=policy_cfg, dataset_stats=kwargs.get("dataset_stats")
            )

            # Honor the two overrides used by LeRobot train/eval scripts.
            overrides = kwargs.get("preprocessor_overrides") or {}
            if "rename_observations_processor" in overrides:
                rename_map = overrides["rename_observations_processor"].get("rename_map", {})
                for step in pre.steps:
                    if isinstance(step, RenameObservationsProcessorStep):
                        step.rename_map = rename_map
                        break
            if "device_processor" in overrides:
                device = overrides["device_processor"].get("device")
                if device is not None:
                    for step in pre.steps:
                        if isinstance(step, DeviceProcessorStep):
                            step.device = device
            return pre, post

        return orig_make_pre_post_processors(
            policy_cfg=policy_cfg, pretrained_path=pretrained_path, **kwargs
        )

    factory.get_policy_class = get_policy_class  # type: ignore[assignment]
    factory.make_pre_post_processors = make_pre_post_processors  # type: ignore[assignment]
    factory._world_modality_patched = True


_patch_lerobot_factories()
