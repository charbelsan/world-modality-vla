# World Modality Research Analysis

**Last updated:** January 12, 2026
**Status:** Qwen+`<ACT_i>` pipeline still yields 0% LIBERO SR → switch to a known-working LIBERO policy (SmolVLA) and test world modality as a surgical augmentation

---

## 1. Research Hypothesis (unchanged)

> **"World model could be the modality missing from VLA control."**

We hypothesize that providing a policy with **predicted future world-state representations** as an **external modality**
(not competing as extra tokens in the main stream) improves closed-loop control.

Core ingredients:
- World representation `z_t` (e.g., V‑JEPA latents; later: Cosmos tokenizer latents/tokens)
- Future predictor `Prophet(z_{t-T+1:t}) -> ẑ_{t+1:t+K}` (action-independent)
- Fusion: **gated cross-attention into the action path only** (Model‑F do‑no‑harm)

---

## 2. Fixed Root Cause (Historical): Empty instructions

Early LIBERO runs used `instruction_key=instruction`, but `HuggingFaceVLA/libero` stores language under `task`.
This produced empty strings → task-agnostic policies → 0% SR is expected.

Fixes (already merged earlier):
- Default dataset keys now match LIBERO (`image_key`, `instruction_key=task`, `episode_id_key=episode_index`)
- Dataset loader fails fast if the instruction key is missing

---

## 3. Dec 28, 2025 Findings (Qwen F+ pipeline)

After retraining with real instructions:
- **Offline**: E2 (future memory injection) improves action loss / MSE relative to E0.
- **Closed-loop LIBERO**: still **0% SR** for E0/E2 and for MSE/Flow heads (tested on 20 episodes on `libero_spatial`).

Conclusion:
- We cannot claim “world helps control” because the baseline is not solving the benchmark.

---

## 4. Jan 2026 Direction Change: move to SmolVLA baseline

The Qwen3‑VL + `<ACT_i>` readout approach is convenient, but it is not a proven LIBERO control recipe and it has
repeatedly produced **0% success** even when offline losses improve.

To isolate the world-modality hypothesis and avoid wasting compute on an under-specified action decoder:

**Baseline:** SmolVLA (LeRobot)
- Proper state/action normalization
- Dedicated action expert (flow matching) rather than “LM hidden state → MLP”
- Prefix KV-cache design for long visual/language prefixes

**Intervention:** world modality injection into the **action expert hidden states only** (Model‑F do‑no‑harm)

This repo now provides a LeRobot policy plugin:
- `--policy.type=smolvla_world`
- Runbook: `docs/MI300X_LIBERO_SMOLVLA_WORLD.md`
- Parallel launcher (MI300X): `scripts/launch_parallel_mi300x_smolvla_world.sh`

---

## 5. Current Experiment Matrix (minimal, high-signal)

Run 2–3 seeds each; start on `libero_spatial`, then expand suites.

- **E0**: `policy.type=smolvla` (baseline)
- **E1**: `policy.type=smolvla_world` with `policy.world_memory_mode_train=zero` (capacity control)
- **E2**: `policy.type=smolvla_world` with `policy.world_memory_mode_train=pred` (main hypothesis)

Optional offline-only plumbing checks:
- `policy.world_memory_mode_train=oracle` should be an upper bound
- `policy.world_memory_mode_train=shuffle/random` should not help

### Expected outcomes (what counts as evidence)
- Oracle > E0 and `world_gate` opens → fusion wiring is correct and world reps are useful.
- E2 > E0 and corruptions hurt (rollout ablation via `policy.world_memory_mode_rollout=zero/random`) → supports the hypothesis.
- E1 improves similarly to E2 → likely confound (extra capacity/regularization), not world modality.

### Required logs (for interpretability)
`smolvla_world` logs (via LeRobot train loop):
- `world_gate`, `world_loss`, `world_cos`, `loss_total`
- `world_attn_entropy`, `world_attn_pmax`, `world_ctx_norm`, `world_act_norm` (if enabled)
- `grad_world_inject`, `grad_prophet` (previous-step grad norms; if enabled)

---

## 6. Sanity Gates (must pass before SR conclusions)

1) Demo replay in env succeeds at non-trivial rate (validates action/control semantics)
2) `lerobot-eval` on a known-good policy yields non-zero SR in the same env install

