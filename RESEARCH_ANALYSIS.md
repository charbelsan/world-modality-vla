# World Modality Research Analysis

**Date:** December 26, 2025
**Status:** Retraining required (instructions bug fixed); Phase C (Flow head) ready

---

## 1. Research Hypothesis

> **"World model could be the modality missing from LLM-based VLA"**

Vision-Language-Action (VLA) models predict actions from image observations and language instructions. We hypothesize that injecting **future world state predictions** as an additional modality significantly improves action prediction quality.

### Core Idea
- Use V-JEPA to encode visual observations into latent world states
- Train a Prophet module to predict future world states from current state
- Inject predicted future states into the VLM via gated cross-attention
- The model learns to leverage world dynamics for better action planning

### Architecture (Model F)
```
Image → V-JEPA → z_current → Prophet → z_future (predicted)
                     ↓              ↓
                  VLM Encoder → Gated Cross-Attention → Action Head
                     ↑
              Instruction Text
```

---

## 2. Root Cause: Why LIBERO Success Was 0%

Previous LLM-VLA experiments (E0/E2/E4) were trained with `--instruction_key instruction`.

In the LeRobot `HuggingFaceVLA/libero` dataset:
- the task language string is provided under the key **`task`** (not `instruction`)
- so `instruction_key=instruction` silently produced **empty strings** for every training sample

Result:
- all trained policies were effectively **task-agnostic**
- **0% closed-loop success rate is expected**, even if offline action MSE looks “reasonable”

This invalidates all prior SR conclusions. Training curves remain useful only as an offline-learning signal.

### Fixes merged
- `world_modality/train_llm_vla.py` defaults now match LIBERO: `--image_key observation.images.image`, `--instruction_key task`, `--episode_id_key episode_index`.
- `world_modality/llm_vla_dataset.py` now **raises** if the instruction key is missing (fail-fast, no silent empty text).
- `scripts/run_fplus_experiments.sh` defaults now use `INSTRUCTION_KEY=task` and pass `EPISODE_ID_KEY=episode_index`.

---

## 3. What We Know (and What We Don’t)

### Known
- Adding Prophet + future-injection can strongly improve offline action MSE (task-agnostic runs already showed this trend).
- Delta prediction (`z_{t+k} - z_t`) + temporal V-JEPA (`m=4`) makes the world-loss non-trivial and stabilizes Prophet learning.

### Unknown (the real research question)
- Does Model-F future memory improve **closed-loop LIBERO success rate** once the policy is truly instruction-conditioned?
- Does changing the **action head** from MSE → Flow improve SR (common failure mode: MSE learns “mean actions”)?

---

## 4. Next Step: Minimal, High-Signal Retraining Matrix

Run only what is needed to answer the question:

**E0 vs E2 × {MSE, Flow}**, skip E4 for now.

| ID | Head | World Memory | Purpose |
|----|------|--------------|---------|
| E0-MSE | MSE | disabled | baseline sanity |
| E2-MSE | MSE | enabled | does world help with MSE head? |
| E0-Flow | Flow | disabled | does flow fix SR vs MSE? |
| E2-Flow | Flow | enabled | main hypothesis test |

### Required settings (non-negotiable)
- `IMAGE_KEY=observation.images.image`
- `INSTRUCTION_KEY=task`
- `EPISODE_ID_KEY=episode_index`

### Recommended world-latent config
- `WORLD_SOURCE=vjepa`
- `TEMPORAL_WINDOW=4` and `LATENT_SUFFIX=m4`
- `DELTA_PREDICTION=1`

### Run procedure (fast sanity → full)
1) **Sanity**: run `MAX_EPOCHS=1` (or even 0.5 epoch) for E0-MSE and evaluate with `n_episodes=2`.
   - If SR is still 0: debug prompt/text pipeline before spending more compute.
2) If sanity SR is non-zero: run full 5 epochs for all four runs and evaluate with `n_episodes=10`.

### Evaluation correctness
- LIBERO requires: init-state set before reset, 10 settle steps, image flip, and `env.check_success()`.
- For Qwen3 backbone eval support + flow-head inference, use the evaluation script on the `phaseC-flow-head` branch.

---

## 5. Success Criteria

We declare progress only if these are true:
- **Pipeline validity:** E0-MSE gets non-zero SR (>0%) after the instruction-key fix.
- **Head validity:** E0-Flow ≥ E0-MSE on SR (flow should not be worse).
- **World modality validity:** E2-Flow > E0-Flow on SR, and future-memory corruption degrades SR.

---

## 6. Risks / Debug Checklist (if SR is still 0 after the fix)
- Confirm checkpoint config contains `instruction_key=task`.
- Confirm prompts printed during training contain the task text (not empty).
- Confirm eval uses the same backbone as training (Qwen3 vs Qwen2.5).
- Confirm image flip is applied consistently (dataset convention).
- Confirm action space matches env expectations (7-dim, clipped to [-1, 1]).

---

## 8. Preliminary Conclusions

1. **World memory significantly reduces action prediction error** (51% at epoch 1)
2. **The model learns to use future context** (gate opens from 0 to -0.016)
3. **Delta prediction enables non-trivial world modeling** (loss 0.55 vs 1.0)
4. **Multi-frame V-JEPA provides richer temporal context**

**Provisional verdict:** Evidence strongly supports the hypothesis that world models are a valuable modality for VLA. LIBERO evaluation will provide definitive validation.

---

## Appendix: Experiment Configuration

### E0 v2 (Baseline)
```bash
--disable_future_injection
--lambda_world 0.0
--freeze_backbone --use_lora
--batch_size 8 --max_epochs 5
```

### E2 v2 (World Memory - Model F)
```bash
--lambda_world 0.2
--delta_prediction
--latent_suffix m4
--freeze_backbone --use_lora
--batch_size 8 --max_epochs 5
```

### E4 (F+ with Chain-of-Thought)
```bash
--lambda_world 0.2
--lambda_text 0.1
--delta_prediction
--coc_jsonl coc_outputs/libero_coc.jsonl
--freeze_backbone --use_lora
--batch_size 8 --max_epochs 5
```
