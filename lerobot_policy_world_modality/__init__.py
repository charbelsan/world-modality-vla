"""LeRobot policy plugin: world modality experiments.

LeRobot auto-discovers packages named `lerobot_policy_*` and imports them at startup.
Importing this package registers the `smolvla_world` policy config via draccus ChoiceRegistry.
"""

from .configuration_smolvla_world import SmolVLAWorldConfig  # noqa: F401

