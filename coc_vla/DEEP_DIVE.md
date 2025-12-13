# Deep Dive: What to Borrow (JEPA / GR00T / Cosmos / Open VLMs)

This document is a research-engineering guide for improving the **world-token VLA + CoC** line while staying faithful to the project’s core bet:

> **World is a first-class modality inside the VLA transformer**, represented as a discrete token stream that can be consumed, rolled out, ablated, and analyzed.

We treat external projects (JEPA, GR00T, Cosmos, open VLMs) as sources of *principles and implementation patterns*, not as replacements for this bet.

---

## 1. JEPA / V-JEPA (Facebook Research)

### 1.1 Why it matters

JEPA-style models learn **predictive latent representations** that discard pixel noise but preserve dynamics-relevant state.

For us, JEPA is most valuable as:

1. A **better frozen encoder** to produce `e_t` before VQ → world tokens.
2. A source of **action-conditioned latents** (V-JEPA2-AC) that can make world tokens more dynamics-aware.

### 1.2 Concrete ways to borrow (without drifting)

**Borrow A: Replace the vision encoder**

- Current pipeline: DINOv2 pooled embedding → k-means → world token id.
- Upgrade: V-JEPA2 (or V-JEPA2-AC) embedding → k-means → world token id.

Expected benefit:

- world token clusters align more with *state* and *dynamics*, less with appearance.

**Borrow B: Use JEPA’s action-conditioned latent as a distillation target**

- Train world-token transformer as usual (predict `w_{t+1..t+K}`).
- Add an auxiliary alignment loss:
  - map predicted world token logits → expected centroid embedding,
  - align with JEPA’s predicted latent for the future frame/clip.

This keeps world tokens explicit while borrowing JEPA’s predictive structure.

### 1.3 Relevant repos

You can clone and inspect:

- `facebookresearch/jepa` (I-JEPA / core JEPA code)
- `facebookresearch/vjepa` and `facebookresearch/vjepa2` (video JEPA and V-JEPA2)

See `coc_vla/external/fetch_external_repos.sh`.

---

## 2. NVIDIA Isaac GR00T (GR00T N1.*)

### 2.1 Why it matters

GR00T’s key contributions (as a pattern) are:

- A **strong frozen VLM backbone** for perception + language grounding.
- A **policy head** designed for action distribution modeling (often diffusion / flow-matching style).
- Multi-task / multi-embodiment training infrastructure.

Even if you do not adopt GR00T’s full stack, it suggests a practical path:

> Use a strong pretrained multimodal backbone for observation grounding, while keeping *world tokens* explicit and rollable inside the policy.

### 2.2 What to borrow while keeping world-tokens central

**Borrow A: Use a stronger observation backbone**

- Instead of only pooled DINO embeddings, you can:
  - use a VLM vision tower to produce `e_t`,
  - VQ those into world tokens.

**Borrow B: Action head modernization**

Your current action head is MSE regression from ACT_QUERY states.

GR00T-style policies often use diffusion/flow for higher-quality action distributions.

You can incorporate this later as:

- same trunk (world tokens as modality),
- replace action head with a diffusion head.

**Borrow C: Cross-embodiment tokenization**

Add a `ROBOT_ID` token and normalize states/actions to a shared representation.

World tokens become a shared “world vocabulary” across embodiments (Franka, Unitree G1, etc.).

### 2.3 Relevant repo

- `NVIDIA/Isaac-GR00T` (code, configs, training recipes)

---

## 3. NVIDIA Cosmos (Reasoning / World Models)

Cosmos is useful in two different roles:

1. **Cosmos-Reason1**: an open(-ish) reasoning VLM tuned for physical AI; useful for generating high-quality CoC labels and as an LLM-as-judge for CoC consistency.
2. **Cosmos-Predict2.5**: a predictive world model; can be used as a sanity check / external critic, or to generate auxiliary synthetic rollouts.

### 3.1 How to use Cosmos without abandoning the bet

**Borrow A: CoC label generator**

Use Cosmos-Reason1 to generate chains-of-causality grounded in task and visual observations. This produces better CoC supervision than generic captioning VLMs in many physical tasks.

**Borrow B: “Physics critic”**

Use Cosmos-Reason1 (or a separate judge model) to score:

- whether a CoC explanation is physically plausible,
- whether it matches world-token rollouts.

This is compatible with world tokens as the internal world modality.

---

## 4. Which open-source VLM to use for CoC generation?

The best choice depends on two constraints:

1. **Quality of causal reasoning** (not just object recognition).
2. **Throughput** (you may need to annotate many episodes).

### 4.1 Recommended default: Qwen3-VL (Instruct / Thinking)

Pros:

- Strong multimodal reasoning; good at structured outputs.
- Well supported in `transformers`.

Suggested models:

- `Qwen/Qwen3-VL-8B-Instruct` (high throughput)
- `Qwen/Qwen3-VL-8B-Thinking` (better reasoning, slower)

### 4.2 “Physics-heavy” option: Cosmos-Reason1

Pros:

- Tuned for physical AI reasoning, often better CoC quality.

Cons:

- License may be more restrictive than Apache-style open models.
- May be more GPU/VRAM hungry depending on size.

### 4.3 Practical recommendation for the team

- Use **Qwen3-VL-8B** to generate CoC for *all* episodes quickly.
- Optionally re-run a smaller curated subset with **Cosmos-Reason1** to:
  - compare quality,
  - or act as a judge/filter for weak CoC.

---

## 5. Concrete next improvements to the repo (aligned with this doc)

1. Make `coc_vla/coc_generation.py` support real image-conditioned generation for:
   - Qwen3-VL
   - Qwen2.5-VL (Cosmos-like models)
2. Add scripts to fetch external repos for inspection:
   - JEPA, V-JEPA2, Isaac-GR00T, Cosmos
3. Add a “backbone swap” hook for world-token embedding extraction:
   - DINOv2 encoder (baseline)
   - V-JEPA2 encoder (better dynamics priors)
   - VLM vision tower (semantic grounding)

These changes preserve the research bet while pulling in the strongest principles from the ecosystem.

