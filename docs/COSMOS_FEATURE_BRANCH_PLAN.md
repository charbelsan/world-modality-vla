# Cosmos Feature Branch Plan

**Purpose:** define the first non-JEPA branch cleanly.

This branch tests the claim:

> **Predictive video features can act as a first-class world modality for control, without replacing SmolVLA's main image/language backbone.**

The goal is **not** to reproduce Cosmos Policy.
The goal is to replace the current **world branch** only.

---

## 1. Design rule

Keep the base SmolVLA perception path unchanged:

- real images -> normal SmolVLA vision encoder
- language -> normal SmolVLA language path
- state -> normal SmolVLA state path

Replace only the current world-memory source:

- **old:** V-JEPA2 pooled latent + Prophet
- **new:** Cosmos Predict hidden features / tokenizer latents

So the experiment is:

- **same policy**
- **same action expert**
- **different world modality**

This preserves attribution.

---

## 2. Why this is first priority

The current JEPA line already told us enough:

- the branch can be wired safely
- the branch can affect actions
- but the current predicted signal is not clearly useful enough

Recent results from DiT4DiT / VPP / mimic-video / Cosmos-style work suggest the main missing ingredient is likely:

- **strong predictive video features**

So the next branch should ask:

> "If we keep the same policy and only upgrade the world feature source, does the world-modality hypothesis become true?"

---

## 3. Experiment family

### C0. Cosmos features, current late action injection

**What changes**

- Replace current `world_latents_source=vjepa` branch with a new source, e.g. `world_latents_source=cosmos`
- Precompute or online-extract a compact set of predictive Cosmos features
- Feed them into the existing world-injection path

**What stays the same**

- SmolVLA backbone
- action expert
- gating mechanism
- eval protocol

**Question answered**

- Is the main bottleneck **feature quality**, not the world-modality idea itself?

**Expected outcome**

- If `C0 > E2-JEPA`, the JEPA branch was feature-limited
- If `C0 ~= E2-JEPA`, the bottleneck is probably fusion or action-independence

---

### C1. Cosmos features + earlier action fusion

Run the same Cosmos feature source with:

- `C1-F1`: current late action injection
- `C1-F2`: earlier action-side fusion (`suffix_in`)

Optional:

- `C1-F3b`: prefix cross-attn into the action-side representation path

**Question answered**

- Are Cosmos features only useful when they influence action computation earlier?

**Expected outcome**

- `C1-F2 > C1-F1` means world should shape action computation before the final readout
- `C1-F3b > C1-F2` means the world branch needs a stronger representation path

---

### C2. View ablation for Cosmos features

Run:

- `C2-front`
- `C2-wrist`
- `C2-both`

**Question answered**

- Is view completeness still a bottleneck after upgrading the world feature source?

**Expected outcome**

- wrist or both should help pickup/contact-heavy tasks more than front-only

---

## 4. Minimal implementation plan

### Step 1. Add a new world source

Extend the world-latent source enum / loader to support:

- `cosmos`

Implementation target:

- same cache interface as V-JEPA:
  - `cache/HuggingFaceVLA/libero/train_world_latents_cosmos_<suffix>.npy`
  - metadata file next to it

The rest of the training code should not care which encoder produced the features.

### Step 2. Define the feature unit

The first version should use **compact predictive features**, not full sampled video.

Use this contract first:

- per time step, extract a fixed-width tensor with the same high-level interface as the current world branch:
  - simplest v0: one pooled feature vector per predicted step, shape roughly `[B, K, D]`
  - later v1: token-level predictive features, shape roughly `[B, K, N, D]`
- start with **v0 pooled features** so the existing world-memory path can be reused with minimal surgery
- only move to token-level features after proving the source matters at all

Good first options:

1. **Tokenizer / latent tokens**
   - better if easy to extract from Cosmos Predict without denoising loops

2. **Intermediate hidden features**
   - closer to DiT4DiT
   - only if extraction is straightforward and stable

Do **not** start with:

- full video generation
- pixel reconstruction as the runtime path

### Step 2.5. First milestone: parity adapter

Before redesigning fusion, build the smallest possible comparison:

- Cosmos features -> lightweight adapter -> current world-memory interface
- same gate
- same action expert
- same training loss

This gives a clean `C0` result:

- if it beats JEPA, source quality is the main bottleneck
- if it does not, move attention to fusion or action-conditioning

### Step 3. Start with frozen features

The first Cosmos branch should be:

- frozen feature extractor
- train only the policy-side world branch / fusion

This keeps the experiment cheap and interpretable.

### Step 4. Only then add auxiliary predictive regularization

Once Cosmos features are wired:

- add a lightweight head from action hidden states to predict:
  - next Cosmos feature block, or
  - next tokenizer latent block

Do **not** start with raw next-image prediction.

### Step 5. Keep comparison clean

For the first Cosmos-vs-JEPA comparison, keep these fixed:

- same dataset and suite
- same training steps
- same eval episode budget
- same SmolVLA checkpoint initialization
- same fusion setting unless the experiment is explicitly about fusion

Otherwise "Cosmos beats JEPA" will not be attributable to the feature source itself.

---

## 5. Exact first runs

### Batch 1: prove the source matters

Run:

1. `E0` baseline
2. `E2-JEPA` current reference
3. `C0-front`

Same:

- suite: `libero_spatial`
- seed: `0`
- budget: same steps and same eval episodes

Decision:

- If `C0-front > E2-JEPA`, move immediately to C1/C2
- If not, stop and inspect feature extraction itself

Minimum success criterion for moving on:

- consistent gain on `libero_spatial` average or a clear gain on the previously weak pickup/contact tasks
- and no obvious collapse on the rest of the suite

### Batch 2: prove fusion/view matter

Run:

1. `C1-F2-front`
2. `C2-wrist`
3. `C2-both`

Decision:

- best one becomes the new main branch

### Batch 3: generalization

Run best Cosmos variant on:

- `libero_object`
- optionally `libero_goal`

### Batch 4: only if C0/C1 show signal

Run:

- auxiliary next-feature prediction from action hidden states

Question answered:

- does forcing the action path to stay predictive make world features more semantically useful?

---

## 6. What not to do first

Do **not** first:

- replace SmolVLA vision encoder with Cosmos
- reproduce Cosmos Policy end-to-end
- use full diffusion video rollout as the online modality
- combine too many changes at once

Those are all valid later, but they are poor first experiments for attribution.

---

## 7. Stop conditions

The Cosmos branch is worth continuing if **any** of these happen:

- `C0-front > E2-JEPA`
- `C1-F2 > C1-F1`
- `C2-wrist` or `C2-both` clearly beats `C2-front`

If none happen, then the problem is likely deeper:

- action-independent future as a modality may be insufficient
- we may need multi-hypothesis futures or action-conditioned world features

That would justify the next branch:

- MoE / imagination bank
- or action-conditioned video-world features

---

## 8. Summary

The correct pivot is:

- **not** "replace the whole policy with Cosmos"
- **not** "keep tuning JEPA forever"

It is:

> **Use stronger predictive video features as the world branch, while keeping SmolVLA's normal perception path intact.**

That is the cleanest next test of the original thesis.
