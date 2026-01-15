# World Modality Research Analysis

**Last updated:** January 14, 2026
**Status:** SmolVLA baseline validated (~89.5% SR on `libero_spatial`). E1 (capacity control) matches baseline after processor fix. E2 (world_pred) training/eval in progress.

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

This repo provides a LeRobot policy plugin:
- `--policy.type=smolvla_world`
- Runbook: `docs/MI300X_LIBERO_SMOLVLA_WORLD.md`
- Parallel launcher (MI300X): `scripts/launch_parallel_mi300x_smolvla_world.sh`

---

## 5. Current Experiment Matrix (minimal, high-signal)

Start with 1 seed to validate plumbing; only then scale to 2–3 seeds and more suites.

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
- `world_gate`, `world_loss`, `world_cos`, `world_valid_frac`, `world_mem_norm`, `world_z_hist_norm`, `world_z_future_norm`, `loss_total`
- `world_attn_entropy`, `world_attn_pmax`, `world_ctx_norm`, `world_act_norm` (if enabled)
- `grad_world_inject`, `grad_prophet` (previous-step grad norms; if enabled)

---

## 6. Sanity Gates (must pass before SR conclusions)

1) Smoke train runs (`--steps=2`) for `smolvla_world` (validates policy/plugin + cache wiring)
2) E0 fine-tune runs end-to-end (validates dataset feature naming; see `--rename_map` note in MI300X runbook)
3) `lerobot-wm-eval` on the resulting checkpoints yields non-zero SR on at least one suite/seed before concluding anything about E1/E2

---

## 7. January 14, 2026 Results (SmolVLA + World Modality)

### 7.1 Experiment Status

As of **January 14, 2026** (MI300X VM run; `libero_spatial`, 10 tasks).

| Experiment | Config | Training | Eval SR | Notes |
|------------|--------|----------|---------|-------|
| **E0** | `smolvla` baseline | 50K steps | **89.5%** | Sanity gate passed |
| **E1** | `smolvla_world` + `world_memory_mode_train=zero` | 50K steps | **88.5%** | Capacity control, matches E0 |
| **E2** | `smolvla_world` + `world_memory_mode_train=pred` | in progress | TBD | Main hypothesis |

### 7.2 Bugs Fixed

1. **Processor mismatch** (commit `16672bc`): `smolvla_world` was building new pre/post processors from `dataset_stats` instead of loading the saved processors from the init checkpoint (`policy.init_from_policy_path`). This caused action convention / unnormalization mismatch → 0% SR despite good E0.

2. **Do-no-harm numeric guards**: skip fusion when the gate is effectively closed, and ignore non-finite world memory to avoid contaminating actions (e.g., `0 * NaN = NaN`).

### 7.3 Do-No-Harm Validation (E1 vs E0)

After fixing the processor mismatch, **E1 (world_zero) = 88.5%** matches **E0 = 89.5%** on SR (within noise).

Interpretation:
- With `world_memory_mode_train=zero`, the injected memory is uninformative by design.
- The gate stays near 0 (`world_gate ≈ -0.000002`), so `smolvla_world` behaves like baseline SmolVLA.
- The observed E1≈E0 outcome validates that the plugin wiring does not break control semantics.
- Extra parameters (Prophet, GatedCrossAttention) do not provide benefit when gate=0 (no capacity confound).

### 7.4 Interpretation

- **E0 = 89.5%**: Baseline works. Sanity gate passed.
- **E1 = 88.5% ≈ E0**: Expected. Proves extra parameters (Prophet, GatedCrossAttention) don't help when gate=0.
- **E2 > E0?**: TBD. This is the main hypothesis test.

If E2 significantly outperforms E0:
- World modality (predicted V-JEPA futures) improves control
- The improvement is due to world information, not extra capacity (since E1 ≈ E0)

### 7.5 Next Steps

1. Complete E1 eval across all 10 libero_spatial tasks
2. Complete E2 training (50K steps)
3. Run E2 eval
4. If E2 > E0: run ablations (`world_memory_mode_rollout=zero/random`) to confirm world info is used
