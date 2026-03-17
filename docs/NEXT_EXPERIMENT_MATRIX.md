# Next Experiment Matrix: From JEPA Branch to Video-Feature Branch

**Purpose:** make the next launches obvious when GPUs free up.

This doc separates:

1. **Runnable now** with the current `smolvla_world` code
2. **Implement next** if the current JEPA-style branch stalls

The goal is not to maximize run count. The goal is to answer the next causal question with minimal ambiguity.

---

## 0. Current state

What is already true:

- `E0` baseline works on LIBERO.
- `E1 ~= E0` after the processor fix, so do-no-harm is validated.
- `E2` is **active but mixed**.
- The old `m=1` rollout bug was real and is fixed.

What is still unresolved:

- Does the policy use **semantic future content**, or just any non-zero side signal?
- Is the main issue **front-only observability**?
- Is the main issue **late fusion**?
- Is the main issue **action-independent future prediction**?

---

## 1. First-principles lessons from the recent video-world-model papers

The recent strong results suggest four principles:

| Principle | Why it matters for us | Representative works |
|---|---|---|
| Predictive features must be **useful for control**, not merely non-zero | Our current branch may be acting like a side bias rather than trusted imagination | DiT4DiT, VPP, mimic-video |
| **Observability** matters | Front-only memory is weak for grasp/contact timing | manipulation benchmarks broadly; our own task pattern |
| **Earlier fusion** can matter | If world is first-class, it should shape the action path before the final readout | DiT4DiT-style conditioning; our F2/F3b ablations |
| Purely **action-independent** future prediction is limited | One guessed future can be wrong for manipulation | DreamZero, Cosmos Policy, DreamDojo, V-JEPA2-AC style planning/action conditioning |

For this repo, that means:

- finish the current JEPA-style branch cleanly,
- then pivot to **predictive video features** rather than spending more H100 time on tiny variants of the same late-injection design.

---

## 2. Phase A: runnable now on current code

These runs require no new architecture implementation.

### A1. Semantic-content ablations on the same E2 checkpoint

**Question:** does semantic future content matter?

Run:

- `pred`
- `signflip`
- `random_scaled`

Interpretation:

- `pred > signflip` and `pred > random_scaled`
  - semantics matter
- `pred ~= signflip/random_scaled`
  - the branch is mostly using non-zero conditioning, not reliable predictive content
- `pred < signflip/random_scaled`
  - current predicted content is actively harmful

Notes:

- `signflip` is the cleanest corruption because it preserves norms exactly.
- `random_scaled` removes the obvious magnitude confound.
- Do not spend more time on `shuffle` unless you need a secondary sanity check.

### A2. Wrist-camera world memory

**Question:** is front-only observability the main bottleneck?

Run:

- precompute `m4_wrist`
- train `E2-wrist`
- evaluate on the same budget/suite as `E2-front`

Interpretation:

- `E2-wrist > E2-front`
  - the current memory branch is limited by view choice
- `E2-wrist ~= E2-front`
  - observability is not the main issue

### A3. Earlier-fusion ablations

**Question:** is late-only action-expert injection too weak?

Run:

- `F2`: `world_inject_suffix_in=true`
- `F3b`: `world_prefix_cross_attn=true`

Interpretation:

- `F2 > F1`
  - world needs to shape expert computation earlier
- `F3b > F2`
  - world should influence representation/prefix, not just the late action path
- neither beats `F1`
  - fusion placement is probably not the main bottleneck

### A4. Near-future emphasis

**Question:** is `K=8` too long / too blurry for contact-sensitive control?

Run:

- `E2-K4`: `future_offset=4`
- `E2-K2`: `future_offset=2`

Interpretation:

- shorter `K` helps
  - near-future precision is the main limit
- shorter `K` does not help
  - the issue is likely feature quality or action-independence

### A5. Generalization check on another suite

**Question:** is the effect specific to `libero_spatial`?

Run the best Phase A variant on:

- `libero_object`
- optionally `libero_goal`

Interpretation:

- if the branch helps only on one suite, the effect is brittle
- if the branch helps across suites, the direction is real

---

## 3. Phase A priority order

When GPUs free up, run in this order:

1. `signflip`
2. `random_scaled`
3. `E2-wrist`
4. `F2`
5. `F3b` only if `F2` is not clearly bad
6. `E2-K4`
7. `E2-K2`
8. best variant on `libero_object`

Why this order:

- first answer whether semantics matter at all,
- then answer whether view choice or fusion position is limiting,
- then answer whether horizon length is the real issue.

---

## 4. Stop conditions for the current JEPA-style branch

Stop spending H100 time on the current V-JEPA2 + action-free Prophet line if the following are all true:

- `pred <= signflip`
- `pred <= random_scaled`
- `E2-wrist` does not beat `E2-front`
- `F2/F3b` do not beat `F1`
- `K=2/4` does not beat `K=8`

That combination means:

- the branch is active,
- but the current predictive content is not trusted or not useful,
- and small changes in view/fusion/horizon do not rescue it.

At that point, move to Phase B.

---

## 5. Phase B: the next branch to implement

This is the video-feature branch inspired by DiT4DiT / mimic-video / VPP style results.

### B0. Design rule for the next branch

**Do not replace SmolVLA's vision encoder first.**

SmolVLA is first a VLM backbone with an added action expert. Replacing its main visual encoder immediately would mix
two changes at once:

- changing the control backbone representation
- changing the world-modality branch

That would destroy attribution.

The better first-class-modality interpretation is:

- keep the normal SmolVLA vision-language-state path intact
- add a **separate predictive video branch**
- let the action path attend to that predictive branch as world memory

So the next branch should test:

> "Can predictive video features behave like a first-class world modality for the action path, while the base VLM
> still processes normal images and language in the usual way?"

This is closer to the original thesis than "replace the visual encoder with a video model."

### B1. Frozen predictive video features as world modality

**Core idea:**

- do **not** generate videos online
- do **not** replace the policy backbone
- extract hidden features from a predictive video model
- inject them as the world modality into SmolVLA

Minimal experiment:

- frozen predictive video model
- single chosen hidden layer / timestep or tokenizer latent
- world features only go to the action path (`F1` first, then `F2/F3b`)
- compare against the current JEPA `E2`

Question answered:

- is the bottleneck **feature quality**, not the world-modality idea itself?

### B2. Video features + earlier fusion

Run:

- `video-F1`
- `video-F2`
- `video-F3b`

Question answered:

- if video features help only when fused earlier, then the right claim is:
  - world is first-class,
  - but it must shape representation/planning, not only the late action readout

### B3. Multi-view predictive video features

Run:

- front-only
- wrist-only
- front+wrist

Question answered:

- is view completeness the key enabler once the features are stronger?

### B4. Auxiliary future prediction from the action path

If we want world to be genuinely first-class, the action transformer should not only **consume** predictive video
features. It should also be regularized to preserve or predict useful future information.

Minimal version:

- keep the main policy loss unchanged
- add a small auxiliary head on top of action-expert hidden states
- predict either:
  - the next video feature block
  - or a lightweight next-image / next-token target

Question answered:

- does forcing the action path to stay predictive make the world branch more semantically useful?

This is the safer version of "predict the next image" for our setup:

- first predict **video-model features or tokens**
- only later consider pixels or full frame generation

### B5. Only after B1-B4 show signal: multi-hypothesis or action-conditioned futures

Do **not** start with MoE or imagination banks.

Only move there if:

- semantic content matters,
- but a single predicted future still hurts some tasks.

Then the right next branch is:

- multi-hypothesis futures
- or action-conditioned world prediction

because the problem becomes ambiguity, not feature absence.

---

## 6. What not to do yet

Do not spend the next H100 window on:

- full video generation / diffusion rollouts as an online planner
- replacing SmolVLA and the world branch at the same time
- MoE routing before proving semantic content matters
- large seed sweeps before the branch decision is made

Those are lower-signal than the Phase A and Phase B decisions above.

---

## 7. Recommended concrete queue

### Queue 1: current branch closure

- finish `E2-wrist`
- finish `F2`
- run `signflip`
- run `random_scaled`
- if `F2` is acceptable, run `F3b`
- run `E2-K4`
- run `E2-K2`

### Queue 2: promote the winner

Take the best of:

- `E2-front`
- `E2-wrist`
- `F2`
- `F3b`
- `E2-K4`
- `E2-K2`

Then:

- run one extra seed
- run `libero_object`

### Queue 3: new branch implementation

If Queue 1 does not produce a clear positive result:

- implement frozen predictive video-feature world memory
- run `video-F1`
- then `video-F2`

That is the cleanest pivot without abandoning the world-as-modality thesis.
