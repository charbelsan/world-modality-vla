from __future__ import annotations

import sys


def train_main() -> None:
    """Wrapper around `lerobot-train` that ensures the policy plugin is imported.

    LeRobot versions that do not auto-discover policy plugins require an explicit import
    so that `PreTrainedConfig.register_subclass("smolvla_world")` is executed.
    """

    import lerobot_policy_world_modality  # noqa: F401

    try:
        from lerobot.scripts.lerobot_train import main
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import LeRobot training entrypoint. Ensure `lerobot` is installed."
        ) from e

    sys.exit(main())


def eval_main() -> None:
    """Wrapper around `lerobot-eval` that ensures the policy plugin is imported."""

    import lerobot_policy_world_modality  # noqa: F401

    try:
        from lerobot.scripts.lerobot_eval import main
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import LeRobot eval entrypoint. Ensure `lerobot` is installed."
        ) from e

    sys.exit(main())

