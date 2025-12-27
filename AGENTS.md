# Agent Notes (world-modality-vla)

This file is written for future coding agents (and humans) to quickly regain context and avoid repeating known failure modes.

## Project Goal

Validate the hypothesis: **“World modality as external memory improves VLA control.”**

Backbone VLM: **Qwen3‑VL‑3B‑Instruct** (primary).

Core architecture: **Model‑F style**: predicted future latents are injected **only into `<ACT_i>` hidden states** via gated cross‑attention (gate init = 0), never as extra tokens competing in the main sequence.

## Critical Dataset Facts (LIBERO / LeRobot)

Dataset: `HuggingFaceVLA/libero` (LeRobot format).

Non‑negotiable keys:
- `image_key = observation.images.image`
- `instruction_key = task` (**NOT** `instruction`)
- `episode_id_key = episode_index`

If you train with `instruction_key=instruction`, the model gets empty strings → becomes task‑agnostic → **0% LIBERO success rate is expected**. The code now fails fast if the instruction key is missing.

## Where to Read First

- `RESEARCH_ANALYSIS.md` — current status + next steps
- `docs/L40S_RUNBOOK.md` — L40S launch guide
- `docs/LLM_VLA_FPLUS.md` — experiment matrix + principles
- `scripts/run_fplus_experiments.sh` — default launcher (E0/E2, optional E4)

## Branches

Use these intentionally:

- `master`
  - Stable MSE training pipeline + dataset-key fix.
  - `world_modality/eval_libero.py` is *not* the best choice for Qwen3 + Flow eval.

- `phaseC-flow-head`
  - Adds Flow action head (`--action_head flow`) + logging.
  - Launcher tags outputs by `{world_source}_{latent_suffix}_{delta}_{action_head}` to avoid overwrites.
  - Includes a more robust `world_modality/eval_libero.py` (Qwen3/Qwen2.5 support via `AutoModelForVision2Seq`).

Recommendation: **run training + eval from `phaseC-flow-head`** for the current experiment matrix.

## Current “Minimal, High‑Signal” Experiment Matrix

Run only:

E0 vs E2 × {MSE, Flow}. Skip E4 until SR is non‑zero and trends are clear.

- E0‑MSE: baseline (`--disable_future_injection`, `--action_head mse`)
- E2‑MSE: world memory (`--action_head mse`)
- E0‑Flow: baseline (`--disable_future_injection`, `--action_head flow`)
- E2‑Flow: world memory (`--action_head flow`)

## World Latents (V‑JEPA)

Preferred:
- `WORLD_SOURCE=vjepa`
- `TEMPORAL_WINDOW=4` → `LATENT_SUFFIX=m4`
- `DELTA_PREDICTION=1` (Prophet predicts `z_{t+k} - z_t`)

Latents are cached under:
`cache/HuggingFaceVLA/libero/train_world_latents_<source>_<suffix>.fp16.npy`

## Evaluation (LIBERO)

Closed‑loop LIBERO rollouts require:
- init state set before reset
- settle steps after reset (10 dummy steps)
- dataset camera convention: 180° flip
- success detection via `env.check_success()`

If success rate is still 0:
- First inspect checkpoint config: it must say `instruction_key=task`.
- Print the built prompt (confirm task text is present).
- Confirm eval backbone matches training backbone (Qwen3 vs Qwen2.5).

## Outputs

- Training outputs: `logs_llm/<exp_name>/llm_vla_epoch*.pt`
- Eval outputs: `eval_libero_results/*.json` (depending on flags)

## Known Good Operational Defaults

In an L40S environment:
- export HF caches to fast disk (`/mnt/fast/hf_cache`).
- set `MUJOCO_GL=egl` for headless rollouts.

