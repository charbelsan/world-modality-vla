# Benchmarks (Rollout Evaluation)

This folder contains **closed-loop rollout evaluation** scripts for benchmark success rates.

The core training code (`world_modality/`) is offline imitation learning. For publication-grade results, you typically need at least one **simulation success rate** benchmark in addition to offline metrics.

Current integrations:

- `benchmarks/libero/`: LIBERO task-suite rollouts (robosuite / MuJoCo).
- `benchmarks/metaworld/`: MetaWorld MT50 rollouts (MuJoCo).

These benchmark dependencies are heavier than the base repo. Prefer running them in a separate environment.