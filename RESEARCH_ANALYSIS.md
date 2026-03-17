# World Modality Research Analysis

**Last updated:** March 17, 2026
**Status:** SmolVLA baseline is validated. `smolvla_world` passes the do-no-harm check (`E1 ~= E0`), but the current `E2` implementation is **mixed / slightly negative overall** on matched-budget `libero_spatial`. The hypothesis survives, but the stronger implementation claim does not.

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

---

## 8. March 2026 Update: what survived and what failed

### 8.1 Current matched-budget findings

On a later matched-budget `libero_spatial` comparison, we observed:

| Experiment | Avg SR | Interpretation |
|------------|--------|----------------|
| **E0** | **83.6%** | Strong baseline |
| **E2-pred** | **80.8%** | World branch active, but not a net win |
| **E2-zero** | **74.0%** | Removing non-zero memory hurts |

Additional ablation result from the partial `random` rollout run:

- On tasks `0-4`, `pred ~= random > zero`
- This means the world branch is **not ignored**
- But it does **not** yet prove that the policy is benefiting from the **semantic correctness** of Prophet's predictions

Why the last point matters:
- `random` was not norm-matched, so `pred ~= random` could still be a **magnitude / conditioning** effect rather than a "content does not matter" result.
- We therefore added stronger rollout corruptions: `random_scaled`, `signflip`, and `shuffle`.

### 8.2 Strong claim rejected

The following stronger claim is **not** supported by the current results:

> "A small action-independent Prophet that predicts future V-JEPA latents and injects them late into the action head should reliably improve LIBERO control."

That claim is too strong for the evidence.

### 8.3 Revised hypothesis (this is the one we should test next)

The core thesis still survives in a narrower and better form:

> **World model can work as a first-class modality for control, but only if the predictive features are informative enough, view-complete enough, and fused early enough to shape action selection.**

This revised hypothesis has four practical implications:

1. **Representation quality matters**
   - The issue is not "world modality is wrong".
   - The issue may be that the current predictive signal is not the right one for contact-rich control.

2. **Action-independent futures are limited**
   - A single predicted future from observation history alone can easily be wrong for manipulation.
   - This is a first-principles limitation, not just a training bug.

3. **Fusion position matters**
   - Late side injection can turn world memory into a weak auxiliary bias.
   - If world is truly first-class, it may need to affect representation/planning earlier (`F2`, `F3b`).

4. **Observability matters**
   - Using only the front camera for world memory is likely insufficient for grasp/contact timing.
   - Wrist and possibly two-view world memory are high-value next tests.

### 8.4 Important correction: we are already using V-JEPA 2

The current codebase is **not** using an old V-JEPA-v1 encoder. The default world encoder is:

- `facebook/vjepa2-vitg-fpc64-256`

So "just upgrade to V-JEPA 2" is **not** the right explanation for the current mixed E2 result.
The more plausible bottlenecks are:

- action-independent prediction
- late fusion
- single-view world memory
- latent-space mismatch with contact-sensitive control

---

## 9. Comparison with recent video-world-model papers

Recent works support the **thesis direction**, but they do **not** support the weakest version of the current implementation.

### 9.1 What the best-performing methods have in common

Across DiT4DiT, DreamZero, Cosmos Policy, DreamDojo, mimic-video, and VPP, the common pattern is:

- predictive video features are strong and spatially grounded
- world modeling is either **action-conditioned** or tightly coupled to action learning
- fusion happens early or the world model is the backbone itself
- the policy is not asked to decide whether to trust a weak side-channel

### 9.2 Why this differs from our current branch

Our current `smolvla_world` branch does this:

- build `z_t` from a world encoder
- predict `z_hat_{t+1:t+K}` from `z_{t-T+1:t}`
- inject those futures only into the action expert through gated cross-attention

This is closer to a **late external-memory adapter** than to DiT4DiT / DreamZero / Cosmos Policy.

### 9.3 Paper-by-paper intuition

- **DiT4DiT** ([arXiv](https://arxiv.org/abs/2603.10448), [project](https://dit4dit.github.io/))
  - Extracts **intermediate hidden states** from a pretrained video diffusion transformer and feeds them into an action transformer.
  - The important lesson is not "generate video", but "use predictive video features as the main representation."

- **DreamZero** ([project](https://dreamzero0.github.io/))
  - Jointly predicts video and action in one diffusion/flow process.
  - Main lesson: action-conditioned world modeling is stronger than a separate action-free future branch.

- **Cosmos Policy** ([project](https://research.nvidia.com/labs/dir/cosmos-policy/cosmos_policy_index.html))
  - Turns a pretrained video model into the policy itself.
  - This is a strong performance reference, but it is a different thesis from "external world modality improves an existing VLA."

- **DreamDojo** ([project](https://dreamdojo-world.github.io/))
  - Uses a large video world model mainly for planning, simulation replacement, and post-training.
  - Main lesson: scale and predictive video priors are useful, but not necessarily as a drop-in late side channel.

- **VPP** ([paper](https://proceedings.mlr.press/v267/hu25g.html)) and **mimic-video** ([project](https://mimic-video.github.io/))
  - These are closer to our desired path than full video sampling.
  - They suggest that **predictive video features** can be used for action prediction without fully generating futures online.

### 9.4 First-principles synthesis

The literature does **not** say:

> "World as modality is a dead end."

It says something more precise:

> "World as modality works when the predictive representation is strong, grounded, and integrated deeply enough into the control computation."

That is still compatible with the original vision. It just rules out the weakest implementation of that vision.

---

## 10. Next best experiments (2x H100 for 24 hours)

The objective for the next 24 hours is **not** to explore everything. It is to remove the biggest remaining ambiguities with high-signal runs.

### 10.1 Success criteria

At the end of the H100 window, we want to know:

1. Does **semantic** future content matter (`pred > signflip/random_scaled`)?
2. Is the current bottleneck mainly **observability** (front-only vs wrist)?
3. Is the current bottleneck mainly **fusion position** (late F1 vs earlier F2/F3b)?

### 10.2 GPU allocation

**GPU 0: causal ablations + wrist world memory**

1. Run `E2-signflip` on the current checkpoint (`50 episodes/task`)
2. Run `E2-random_scaled` on the current checkpoint (`50 episodes/task`)
3. Precompute wrist latents:
   - `image_key=observation.images.image2`
   - `latent_suffix=m4_wrist`
4. Train `E2-wrist` from the same SmolVLA init checkpoint
5. Evaluate `E2-wrist` on `libero_spatial`

**GPU 1: earlier fusion**

1. Train `F2` (`world_inject_suffix_in=true`) with the existing front-camera latents
2. Evaluate `F2` on `libero_spatial`
3. If time remains and F2 is promising, train/eval `F3b` (`world_prefix_cross_attn=true`)

### 10.3 Why this ordering

- `signflip` and `random_scaled` tell us whether semantics matter at all.
- `E2-wrist` tests the strongest obvious missing-information hypothesis.
- `F2` tests whether late-only injection is the main architectural bottleneck.
- This yields significantly more information than spending the whole window on more baseline-like runs.

### 10.4 Stop conditions

- If `pred <= signflip` and `pred <= random_scaled`:
  - current Prophet content is not helping
  - stop investing in more F1 late-fusion variants

- If `E2-wrist > E2-front` by a meaningful margin:
  - prioritise multi-view / wrist-aware world memory

- If `F2 > F1`:
  - prioritise earlier fusion over more late-fusion tuning

- If none of `E2-wrist`, `F2`, or `F3b` beats E0:
  - stop scaling the current V-JEPA2 + action-free Prophet line
  - move to a DiT4DiT / mimic-video style video-feature branch

### 10.5 Most likely next branch if the current line stalls

The next architecture to prototype should be:

- **keep SmolVLA as the control baseline**
- **do not replace the main SmolVLA vision encoder first**
- replace the current world branch with **predictive video-model hidden features**
- prefer **feature extraction** over full video generation
- feed those predictive features as a **separate world-modality branch**
- fuse those features earlier (`F2` or `F3b`) rather than only at the end of the action expert

This is important conceptually:

- the base VLM should still process current images/language/state in the normal way
- the world model should provide an additional predictive modality to the action path

That is much closer to the original thesis than "swap the vision encoder for a video model."

### 10.6 Better formulation of the next thesis

The next thesis to test is:

> "Predictive video features can act as a first-class world modality for control when they are kept separate from the
> base VLM perception stream and fused into the action computation with the right inductive bias."

In practice, that means:

1. **Keep current-image perception intact**
   - SmolVLA still sees the real observations through its normal visual encoder

2. **Add a predictive branch**
   - frozen or lightly tuned predictive video model
   - hidden states / tokenizer latents, not full video sampling

3. **Use the world branch for action, not for replacing perception**
   - action expert attends to predictive features
   - optionally prefix cross-attn if we want world to influence planning/representation

4. **Optionally regularize the action path to stay predictive**
   - auxiliary next-feature / next-image-token loss from the action transformer
   - this is a principled way to force semantic alignment without turning the whole policy into a video generator
