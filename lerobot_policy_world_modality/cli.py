from __future__ import annotations

import json
import os
import sys


def _extract_rename_map_from_argv() -> None:
    """Move `--rename_map` (JSON) from argv into an env var.

    Some LeRobot versions parse configs with draccus, which may not reliably parse a JSON
    string into a `dict[str, str]` field. To avoid silent fallbacks (e.g. `{}`), we
    accept `--rename_map='{\"a\":\"b\"}'`, validate it here, set an env var, and remove
    the flag from argv.

    The factory monkeypatch in `lerobot_policy_world_modality.__init__` consumes
    `LEROBOT_WM_RENAME_MAP_JSON` and applies it to the rename processor step.
    """

    argv = list(sys.argv)
    out: list[str] = [argv[0]]
    rename_value: str | None = None

    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--rename_map="):
            rename_value = a.split("=", 1)[1]
            i += 1
            continue
        if a == "--rename_map":
            if i + 1 >= len(argv):
                raise ValueError("--rename_map flag requires a value")
            rename_value = argv[i + 1]
            i += 2
            continue
        out.append(a)
        i += 1

    if rename_value is None:
        return

    try:
        rename_map = json.loads(rename_value)
    except Exception as e:
        raise ValueError(
            "Invalid `--rename_map` value. Expected a JSON object like "
            '\'{"observation.images.image":"observation.images.camera1"}\'.'
        ) from e

    if not isinstance(rename_map, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in rename_map.items()):
        raise ValueError("`--rename_map` must be a JSON object mapping strings to strings.")

    os.environ["LEROBOT_WM_RENAME_MAP_JSON"] = json.dumps(rename_map)
    sys.argv = out


def _ensure_libero_config_noninteractive() -> None:
    """Prevent LIBERO from prompting for config on first import.

    `libero.libero` creates `~/.libero/config.yaml` on first import and (unfortunately)
    prompts via `input()` if the file is missing. That breaks non-interactive runs
    (train/eval launched from scripts, tmux, SLURM, etc.).

    We pre-create a minimal config file if:
    - `libero.libero` is importable (installed), and
    - the config file does not already exist.
    """

    if os.environ.get("LEROBOT_WM_SKIP_LIBERO_CONFIG", "").lower() in {"1", "true", "yes"}:
        return

    try:
        import importlib.util
        from pathlib import Path
    except Exception:  # pragma: no cover
        return

    # If LIBERO isn't installed, do nothing.
    spec = importlib.util.find_spec("libero.libero")
    if spec is None or not spec.submodule_search_locations:
        return

    config_dir = Path(
        os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    ).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return

    try:
        benchmark_root = Path(list(spec.submodule_search_locations)[0]).resolve()
        path_dict = {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str((benchmark_root / "bddl_files").resolve()),
            "init_states": str((benchmark_root / "init_files").resolve()),
            "datasets": str((benchmark_root.parent / "datasets").resolve()),
            "assets": str((benchmark_root / "assets").resolve()),
        }
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file.write_text("\n".join(f"{k}: {v}" for k, v in path_dict.items()) + "\n")
    except Exception:
        # If anything goes wrong, fall back to LIBERO's native behavior.
        return


def train_main() -> None:
    """Wrapper around `lerobot-train` that ensures the policy plugin is imported.

    LeRobot versions that do not auto-discover policy plugins require an explicit import
    so that `PreTrainedConfig.register_subclass("smolvla_world")` is executed.
    """

    import lerobot_policy_world_modality  # noqa: F401
    _extract_rename_map_from_argv()
    _ensure_libero_config_noninteractive()

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
    _extract_rename_map_from_argv()
    _ensure_libero_config_noninteractive()

    try:
        from lerobot.scripts.lerobot_eval import main
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Could not import LeRobot eval entrypoint. Ensure `lerobot` is installed."
        ) from e

    sys.exit(main())
