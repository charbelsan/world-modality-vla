# Research Coding Agent Prompt (copy/paste)

Use this prompt when handing the repo to an autonomous coding research agent.

---

You are an autonomous research engineering agent working in the repo `world-modality-vla`.

## Goal

Implement, run, and report experiments for the hypothesis:

> Adding a discrete **world token stream** as a **first-class modality** inside a VLA (consumed as input + predicted into the future) improves action quality and long-horizon stability vs baselines.

## Non-negotiable model matrix

Run and compare all four models (same trunk/hparams where possible):

1. **A**: action-only BC baseline.
2. **B**: action BC + future world-token prediction loss (aux head), but **do not** feed world tokens as inputs.
3. **C**: world tokens as a first-class input modality **and** predict future world tokens.
4. **C_no_world_input**: ablation: same as C but **do not** feed world tokens as inputs; keep world-token prediction loss.

## Required evaluation outputs

1. Offline action metric:
   - action MSE (or the repo’s configured action loss) on a held-out split.
2. World metrics for B/C/C_no_world_input:
   - Top‑1 and Top‑5 world-token accuracy
   - per-horizon breakdown (k=1..K)
3. Causal intervention:
   - world corruption test for C and C_no_world_input
   - report clean MSE, corrupted MSE, and ratio = corrupted/clean

## Dataset choice (start)

Start with:

- `HuggingFaceVLA/libero`

Keys usually:

- `image_key=observation.images.image`
- `proprio_key=observation.state`
- `action_key=action`

## Crash-proofing (mandatory on unstable VMs)

Do not store checkpoints only on local disk.

- Prefer `--log_dir` on a persistent mount.
- Otherwise run continuous sync (Google Drive via `rclone`):
  - read `ops/README.md`

## GPU utilization

If GPU VRAM is underused:

- Increase batch size cautiously (watch stability).
- Run multiple trainings in parallel (reduce `num_workers` per run to avoid CPU thrash).
- Prefer extra runs that answer clear research questions:
  - different `future_offset` (horizon sweep),
  - different `world_vocab_size`,
  - λ sweep,
  - 2nd seed for C.

## What to commit

Commit only code + small docs/plots.

- Do NOT commit `cache/` or `logs*/` (they are gitignored).
- Store large checkpoints in remote storage (Drive/S3/HF).

## Deliverables

1. Commands to reproduce each run.
2. A markdown summary with:
   - config table (A/B/C/C_no_world_input)
   - offline metrics table
   - corruption ratio table
   - brief interpretation (does modality help beyond aux loss?)

---

