"""LeRobot policy plugin: world modality experiments.

LeRobot auto-discovers packages named `lerobot_policy_*` and imports them at startup.
Importing this package registers the `smolvla_world` policy config via draccus ChoiceRegistry.
"""

from __future__ import annotations

import json
import os

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
        import lerobot.policies.utils as policy_utils  # type: ignore
        from lerobot.processor import (  # type: ignore
            DeviceProcessorStep,
            RenameObservationsProcessorStep,
        )
    except Exception:
        # LeRobot not installed; allow importing this package for docs/lint.
        return

    if getattr(factory, "_world_modality_patched", False):
        return

    orig_get_policy_class = factory.get_policy_class
    orig_make_pre_post_processors = factory.make_pre_post_processors
    orig_validate_visual_features_consistency = policy_utils.validate_visual_features_consistency

    def _env_rename_map() -> dict[str, str] | None:
        s = os.environ.get("LEROBOT_WM_RENAME_MAP_JSON")
        if not s:
            return None
        try:
            obj = json.loads(s)
        except Exception:
            return None
        if not isinstance(obj, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()):
            return None
        return obj

    def _apply_rename_map(pre, rename_map: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
        # Best-effort: either mutate an existing rename step or insert one at the front.
        for step in getattr(pre, "steps", []):
            if isinstance(step, RenameObservationsProcessorStep):
                step.rename_map = rename_map
                return

        try:
            new_step = RenameObservationsProcessorStep(rename_map=rename_map)
        except TypeError:
            try:
                new_step = RenameObservationsProcessorStep(rename_map)
            except TypeError:
                return

        steps = list(getattr(pre, "steps", []))
        steps.insert(0, new_step)
        try:
            pre.steps = steps
        except Exception:
            return

    def get_policy_class(name: str):  # type: ignore[no-untyped-def]
        if name == "smolvla_world":
            return SmolVLAWorldPolicy
        return orig_get_policy_class(name)

    def validate_visual_features_consistency(cfg, features):  # type: ignore[no-untyped-def]
        """Patched validation that applies rename_map before checking feature consistency."""
        rename_map = _env_rename_map()
        if rename_map:
            # Transform feature keys: dataset_key → policy_key
            renamed_features = {}
            for key, value in features.items():
                new_key = rename_map.get(key, key)
                renamed_features[new_key] = value
            features = renamed_features
        return orig_validate_visual_features_consistency(cfg, features)

    def make_pre_post_processors(policy_cfg, pretrained_path=None, **kwargs):  # type: ignore[no-untyped-def]
        if isinstance(policy_cfg, SmolVLAWorldConfig):
            processors_pretrained_path = None
            init_path = str(getattr(policy_cfg, "init_from_policy_path", "") or "").strip()
            if init_path:
                processors_pretrained_path = init_path
            elif pretrained_path:
                processors_pretrained_path = pretrained_path
            pre, post = make_smolvla_world_pre_post_processors(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
                processors_pretrained_path=processors_pretrained_path,
            )
        else:
            pre, post = orig_make_pre_post_processors(
                policy_cfg=policy_cfg, pretrained_path=pretrained_path, **kwargs
            )

        # Honor the two overrides used by LeRobot train/eval scripts.
        overrides = kwargs.get("preprocessor_overrides") or {}
        if "rename_observations_processor" in overrides:
            rename_map = overrides["rename_observations_processor"].get("rename_map", {})
            if isinstance(rename_map, dict) and rename_map:
                _apply_rename_map(pre, rename_map)
        if "device_processor" in overrides:
            device = overrides["device_processor"].get("device")
            if device is not None:
                for step in getattr(pre, "steps", []):
                    if isinstance(step, DeviceProcessorStep):
                        step.device = device
                        break

        # Additionally, apply env-based rename map (set by `lerobot-wm-*` wrappers).
        env_map = _env_rename_map()
        if env_map:
            _apply_rename_map(pre, env_map)

        return pre, post

    factory.get_policy_class = get_policy_class  # type: ignore[assignment]
    factory.make_pre_post_processors = make_pre_post_processors  # type: ignore[assignment]
    # Also patch the validation function to apply rename_map before feature checking
    policy_utils.validate_visual_features_consistency = validate_visual_features_consistency  # type: ignore[assignment]
    factory.validate_visual_features_consistency = validate_visual_features_consistency  # type: ignore[assignment]
    factory._world_modality_patched = True


_patch_lerobot_factories()
