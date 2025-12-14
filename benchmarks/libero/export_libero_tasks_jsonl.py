from __future__ import annotations

import argparse
import json
import os
from typing import Dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export LIBERO task language strings as tasks.jsonl.")
    p.add_argument("--benchmark", type=str, default="libero_10")
    p.add_argument("--task_order_index", type=int, default=0)
    p.add_argument("--out", type=str, default="benchmarks/libero/tasks.jsonl")
    p.add_argument(
        "--libero_config_path",
        type=str,
        default="",
        help="Set LIBERO_CONFIG_PATH for non-interactive runs.",
    )
    return p.parse_args()


def _write_libero_config_if_missing(libero_config_path: str) -> None:
    import importlib.util

    import yaml

    if not libero_config_path:
        libero_config_path = os.path.expanduser("~/.libero")
    os.makedirs(libero_config_path, exist_ok=True)
    config_file = os.path.join(libero_config_path, "config.yaml")
    if os.path.exists(config_file):
        return

    spec = importlib.util.find_spec("libero.libero")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not locate 'libero.libero'. Install LIBERO first (see benchmarks/libero/README.md)."
        )
    benchmark_root = os.path.dirname(os.path.abspath(spec.origin))

    cfg = {
        "benchmark_root": benchmark_root,
        "bddl_files": os.path.join(benchmark_root, "bddl_files"),
        "init_states": os.path.join(benchmark_root, "init_files"),
        "datasets": os.path.join(os.path.dirname(benchmark_root), "datasets"),
        "assets": os.path.join(benchmark_root, "assets"),
    }
    with open(config_file, "w") as f:
        yaml.safe_dump(cfg, f)


def main() -> None:
    args = parse_args()

    if args.libero_config_path:
        os.environ["LIBERO_CONFIG_PATH"] = args.libero_config_path
        _write_libero_config_if_missing(args.libero_config_path)
    else:
        _write_libero_config_if_missing(os.path.expanduser("~/.libero"))

    try:
        from libero.libero.benchmark import get_benchmark  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Failed to import LIBERO. Install it first (see benchmarks/libero/README.md)."
        ) from e

    benchmark = get_benchmark(args.benchmark)(args.task_order_index)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tasks: Dict[int, str] = {}
    for task_id in range(benchmark.get_num_tasks()):
        task = benchmark.get_task(task_id)
        tasks[int(task_id)] = str(task.language)

    with open(args.out, "w") as f:
        for task_id in sorted(tasks.keys()):
            f.write(json.dumps({"task_index": task_id, "task": tasks[task_id]}) + "\n")

    print(f"Wrote {len(tasks)} tasks to {args.out}")


if __name__ == "__main__":
    main()

