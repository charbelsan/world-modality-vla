# P5 24H Pivot: Move Beyond JEPA Late Fusion

**Decision:** on the new 24-hour P5 block, do **not** spend the main budget on more JEPA + late-fusion variants.

Use this block to:

1. preserve everything on durable storage
2. implement the **Cosmos feature branch**
3. run the first clean **Cosmos-vs-JEPA** comparison

See also:

- [COSMOS_FEATURE_BRANCH_PLAN.md](/home/charbel/charbel/world-modality-vla/docs/COSMOS_FEATURE_BRANCH_PLAN.md)
- [NEXT_EXPERIMENT_MATRIX.md](/home/charbel/charbel/world-modality-vla/docs/NEXT_EXPERIMENT_MATRIX.md)

---

## 1. Why pivot now

The JEPA branch already gave the important answers:

- the branch can be wired safely
- the branch can affect control
- but it is not a clean win

Recent results from DiT4DiT, Cosmos Policy, DreamZero, VPP, and mimic-video point in the same direction:

- stronger **predictive video features**
- earlier or deeper coupling to action
- less reliance on a weak late side-channel

So the new question is no longer:

> "Can we rescue JEPA with one more small tweak?"

It is:

> "If we keep SmolVLA fixed and replace only the world branch with predictive video features, does world-as-modality become useful?"

---

## 2. Non-negotiable operational rule

The previous P5 lost all NVMe-only outputs.

For this block:

- keep the repo itself on NVMe if you want
- but write **all outputs, caches, logs, checkpoints, and exported features** to `/mnt/preserved`

Suggested layout:

```bash
/mnt/preserved/world-modality-vla/
  outputs/
  cache/
  logs/
  artifacts/
```

If needed, use symlinks from the repo root into `/mnt/preserved`.

Also sync periodically:

```bash
aws s3 sync /mnt/preserved/world-modality-vla/outputs/ s3://charbel-home-backup-997291/world_model_training/
```

---

## 3. What not to do with this 24h block

Do **not** make these the main runs:

- more JEPA-only late-fusion variants
- more long JEPA training without a new source
- full Cosmos Policy reproduction
- replacing SmolVLA's main vision encoder first
- full video generation as the runtime world modality

Those either have weak attribution or the wrong thesis.

---

## 4. Main experiment family for this block

### C0. Cosmos features, same current action-side interface

Goal:

- keep SmolVLA unchanged
- keep the action expert unchanged
- replace only the world source

Implementation target:

- add `world_latents_source=cosmos`
- extract compact predictive Cosmos features
- map them through a **parity adapter** into the current world-memory interface

This is the cleanest "source-quality" test.

### C1. Cosmos features + earlier action fusion

After `C0`, run:

- `C1-F1`: same late action injection
- `C1-F2`: earlier action-side fusion (`suffix_in`)

Optional if promising:

- `C1-F3b`

### C2. View ablation

Run:

- `C2-front`
- `C2-wrist`
- `C2-both`

---

## 5. Practical 24h schedule on 8x H100

### Phase 0: bootstrap and smoke tests (first 2 hours)

Use 2-3 GPUs max.

Tasks:

- recreate the venv
- clone / pull `phaseC-flow-head`
- point outputs and caches to `/mnt/preserved`
- smoke test Cosmos model loading
- smoke test extraction of one feature tensor from:
  - front view
  - wrist view

Deliverable:

- one saved sample feature file on `/mnt/preserved`
- confirmed feature shape and dtype

### Phase 1: implement the parity adapter (hours 2-6)

Use 1 GPU for testing, leave the others free.

Tasks:

- add `cosmos` as a valid world source
- add a precompute path matching the current cache interface
- make the policy accept the new source without touching the SmolVLA backbone

Deliverable:

- `C0-front` can launch end-to-end

### Phase 2: first real runs (hours 6-16)

Run in parallel:

- GPU 0: `C0-front`
- GPU 1: `C0-wrist`
- GPU 2: JEPA `E2` reference eval only if needed for a fair same-machine comparison
- GPU 3: `C1-F2-front`
- GPU 4: feature-precompute / cache jobs
- GPU 5: backup seed or `C2-both`
- GPU 6-7: spare for failures, smoke fixes, or evals

Primary question:

- does `C0` beat the current JEPA `E2`?

### Phase 3: promote the winner (hours 16-24)

If any Cosmos branch is clearly better:

- run one extra seed
- run `libero_object`

If no Cosmos branch beats JEPA:

- stop and inspect the feature extraction itself before spending more training time

---

## 6. Minimum success criteria

Move forward with the Cosmos branch if any of these happen:

- `C0-front > E2-JEPA`
- `C1-F2 > C1-F1`
- `C2-wrist` or `C2-both` beats `C2-front`

If none happen, the problem is probably not just feature source.
Then the next branch should be:

- action-conditioned world features
- or multi-hypothesis world memory

---

## 7. Bottom line

This 24-hour block should be used to answer:

> "Does replacing JEPA with predictive video features rescue the world-modality idea, while keeping SmolVLA itself fixed?"

That is the highest-value next question.
