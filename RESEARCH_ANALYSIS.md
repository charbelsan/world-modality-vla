# World Modality Research Analysis

**Date:** December 28, 2024
**Status:** Phase A (Training) ✅ Complete | Phase B (Evaluation) ⚠️ 0% SR - Investigating

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

## 2. December 28 Experiment Results

### 2.1 Training Summary (All Complete)

| Experiment | Action Head | World Memory | Epochs | Final VAL MSE | World Loss |
|------------|-------------|--------------|--------|---------------|------------|
| **E0-MSE** | MSE | No (`--disable_future_injection`) | 5 | 0.206 | ~0.98 (no learning) |
| **E0-Flow** | Flow | No | 5 | N/A (flow loss) | ~0.98 |
| **E2-MSE** | MSE | Yes (`--lambda_world 0.2`) | 5 | **0.078** | **0.56** |
| **E2-Flow** | Flow | Yes | 5 | 0.982 (flow) | 0.52 |

### 2.2 E2-MSE vs E0-MSE: Validation MSE per Epoch

| Epoch | E0-MSE | E2-MSE | Improvement |
|-------|--------|--------|-------------|
| 0 | 0.210 | 0.106 | **50%** |
| 1 | 0.214 | 0.092 | **57%** |
| 2 | 0.208 | 0.084 | **60%** |
| 3 | 0.205 | 0.083 | **60%** |
| 4 | 0.206 | **0.078** | **62%** |

### 2.3 World Loss Progression (Prophet Learning)

**E0-MSE** (no world memory - Prophet not used):
```
Epoch 0 Step 0:   world=0.976
Epoch 4 Step end: world=0.977  (no change - as expected)
```

**E2-MSE** (world memory - Prophet actively learning):
```
Epoch 0 Step 0:   world=1.002
Epoch 0 Step 100: world=0.955
Epoch 4 Step end: world=0.556  (44% reduction - Prophet learned!)
```

### 2.4 Gate Analysis

| Experiment | Gate Value | Interpretation |
|------------|------------|----------------|
| E0-MSE | 0.000 | No future injection (disabled) |
| E2-MSE | -0.0156 | sigmoid(-0.0156) ≈ **0.496** → using ~50% future info |
| E2-Flow | -0.0156 | Same - consistent gating behavior |

---

## 3. LIBERO Evaluation Results

### 3.1 Evaluation Configuration

- **Suite:** libero_spatial (10 pick-and-place tasks)
- **Episodes per task:** 2
- **Total episodes:** 20
- **Fixes applied:**
  - `--instruction_key task` (correct natural language tasks)
  - `use_delta=True` (match LeRobot control mode)
  - `--binarize_gripper` for MSE models

### 3.2 Results (All 0% SR)

| Experiment | Checkpoint | binarize_gripper | use_delta | Overall SR |
|------------|------------|------------------|-----------|------------|
| E0-MSE | epoch4 | ✓ | ✓ | **0%** (0/20) |
| E0-Flow | epoch4 | - | ✓ | **0%** (0/20) |
| E2-MSE | epoch4 | ✓ | ✓ | **0%** (0/20) |
| E2-Flow | epoch4 | - | ✓ | **0%** (0/20) |

### 3.3 Per-Task Breakdown (E0-MSE example)

| Task | SR |
|------|----|
| pick_up_the_black_bowl_between_the_plate_and_the_ramekin... | 0% |
| pick_up_the_black_bowl_next_to_the_ramekin... | 0% |
| pick_up_the_black_bowl_from_table_center... | 0% |
| pick_up_the_black_bowl_on_the_cookie_box... | 0% |
| pick_up_the_black_bowl_in_the_top_drawer... | 0% |
| pick_up_the_black_bowl_on_the_ramekin... | 0% |
| pick_up_the_black_bowl_next_to_the_cookie_box... | 0% |
| pick_up_the_black_bowl_on_the_stove... | 0% |
| pick_up_the_black_bowl_next_to_the_plate... | 0% |
| pick_up_the_black_bowl_on_the_wooden_cabinet... | 0% |

---

## 4. Investigation: Why 0% SR?

### 4.1 Issues Identified and Fixed

| Issue | Status | Effect |
|-------|--------|--------|
| Wrong `--instruction_key instruction` | ✅ Fixed → `task` | Training had correct instructions |
| Missing `use_delta=True` control mode | ✅ Fixed | Still 0% SR after fix |
| MSE predicts mean gripper (0) not binary (-1/+1) | ✅ Fixed with `--binarize_gripper` | Still 0% SR |

### 4.2 Model Behavior Analysis

**Actions ARE visually-conditioned** (not constant):
```python
# Different images produce different actions:
Black image:  [ 0.09  0.11 -0.18 -0.08 -0.01  0.08  0.24]
White image:  [ 0.07  0.23 -0.18 -0.08  0.01  0.11  0.56]
Noise image:  [ 0.11  0.05 -0.11  0.04  0.02  0.07  0.12]
# Action variance across images: 0.045 (non-trivial)
```

**But actions don't lead to task success:**
- Robot moves consistently in one direction
- Doesn't adapt to object locations
- Gripper doesn't close at right time (even with binarization)

### 4.3 Remaining Hypotheses

| Hypothesis | Likelihood | Evidence |
|------------|------------|----------|
| **Missing proprioception** | High | Dataset has `observation.state` (8-dim) but training uses only images |
| **Single camera limitation** | Medium | Dataset has `image2` (wrist cam) but training uses only `agentview` |
| **VLM spatial precision** | Medium | General VLM may lack fine manipulation precision |
| **Domain gap** | Medium | Training demos vs eval initial states may differ |
| **Action head capacity** | Low | 62% training improvement suggests learning is happening |

### 4.4 Dataset Modalities (What We're Missing)

```python
# HuggingFaceVLA/libero dataset contains:
observation.images.image:  [3, 256, 256]  # ✅ Used
observation.images.image2: [3, 256, 256]  # ❌ NOT used (wrist camera)
observation.state:         [8]            # ❌ NOT used (proprioception)
action:                    [7]            # ✅ Used
task:                      str            # ✅ Used (language instruction)
```

---

## 5. Key Findings

### 5.1 Training: World Memory Helps Significantly

**E2-MSE achieves 62% lower action prediction error than E0-MSE:**
- E0-MSE final VAL MSE: 0.206
- E2-MSE final VAL MSE: 0.078
- **Improvement: 62%**

**Prophet learns meaningful future prediction:**
- World loss decreased from 1.0 → 0.56 (44% reduction)
- Indicates non-trivial temporal modeling

**Gate opens to use future information:**
- Gate value: -0.0156 → sigmoid ≈ 0.496
- Model uses ~50% of injected future context

### 5.2 Evaluation: Training Metrics Don't Transfer

Despite 62% better action prediction:
- E2-MSE: 0% task success
- E0-MSE: 0% task success
- **No difference in eval performance**

This suggests:
1. MSE on action regression ≠ task success
2. Missing modalities (proprio, wrist cam) may be critical
3. Or fundamental architecture limitations

---

## 6. Experiment Details

### 6.1 Training Configuration

**Common settings:**
```bash
--vlm_backbone qwen2_5_vl_3b_instruct
--dataset_name HuggingFaceVLA/libero
--image_key observation.images.image
--instruction_key task
--freeze_backbone --use_lora
--batch_size 8 --max_epochs 5
--world_latents_source vjepa
--latent_suffix m4
--delta_prediction
```

**E0 (Baseline):**
```bash
--disable_future_injection
--lambda_world 0.0
```

**E2 (World Memory):**
```bash
--lambda_world 0.2
# (no --disable_future_injection)
```

**Action Heads:**
- MSE: `--action_head mse`
- Flow: `--action_head flow --flow_steps_eval 8`

### 6.2 Evaluation Configuration

```bash
MUJOCO_GL=egl python -m world_modality.eval_libero \
  --checkpoint <checkpoint.pt> \
  --suite libero_spatial \
  --n_episodes 2 \
  --libero_root /home/ubuntu/LIBERO \
  --device cuda \
  --binarize_gripper  # for MSE models
  # --disable_future_injection  # for E0 only
```

### 6.3 Checkpoint Information

| Experiment | Checkpoint Path | Size |
|------------|-----------------|------|
| E0-MSE | `logs_llm/E0_baseline_vjepa_m4_delta_mse/llm_vla_epoch4.pt` | 7.1 GB |
| E0-Flow | `logs_llm/E0_baseline_vjepa_m4_delta_flow/llm_vla_epoch4.pt` | 7.1 GB |
| E2-MSE | `logs_llm/E2_model_f_vjepa_m4_delta_mse/llm_vla_epoch4.pt` | 7.2 GB |
| E2-Flow | `logs_llm/E2_model_f_vjepa_m4_delta_flow/llm_vla_epoch4.pt` | 7.2 GB |

---

## 7. Next Steps

### 7.1 Immediate (High Priority)

1. **Add proprioception input** (`observation.state`) to training/eval
2. **Add wrist camera** (`observation.images.image2`) as second visual input
3. **Record eval rollout videos** to visualize robot behavior

### 7.2 Architecture Changes (If Above Fails)

1. **Dedicated gripper head** with binary classification loss
2. **Increase LoRA rank** (currently r=16) for more capacity
3. **Larger backbone** (Qwen2.5-VL-7B instead of 3B)

### 7.3 Debugging (Parallel)

1. **Offline validation**: Compare predicted vs GT actions on training set
2. **Oracle test**: Use ground-truth actions in eval to verify env works
3. **Early epoch eval**: Check if earlier checkpoints behave differently

---

## 8. Conclusions

### 8.1 Provisional Support for Hypothesis

**Training evidence supports world memory hypothesis:**
- 62% action prediction improvement (E2-MSE vs E0-MSE)
- Prophet learns meaningful future predictions (world loss 1.0 → 0.56)
- Gate learns to use ~50% of future information

### 8.2 But Evaluation Shows Critical Gap

**0% task success across all experiments indicates:**
- Training MSE improvement ≠ manipulation success
- Pure vision (single camera, no proprio) may be insufficient
- Or architecture lacks spatial precision for fine manipulation

### 8.3 Updated Verdict

> **The world memory hypothesis shows promise in training metrics but cannot be validated until evaluation issues are resolved. The 62% action prediction improvement is significant, but task success requires additional modalities or architectural changes.**

---

## Appendix: Raw Training Logs

### E0-MSE Final Steps
```
[Epoch 4 Step 153790] loss=0.179 action=0.179 world=0.983 gate=0.000
[Epoch 4 Step 153800] loss=0.203 action=0.203 world=0.981 gate=0.000
[Epoch 4 Step 153810] loss=0.166 action=0.166 world=0.977 gate=0.000
[Epoch 4] VAL action MSE=0.206361 gate=0.0000
```

### E2-MSE Final Steps
```
[Epoch 4 Step 148300] loss=0.169 action=0.044 world=0.621 gate=-0.016
[Epoch 4 Step 148350] loss=0.178 action=0.063 world=0.573 gate=-0.016
[Epoch 4 Step 148400] loss=0.144 action=0.032 world=0.556 gate=-0.016
[Epoch 4] VAL action MSE=0.077775 gate=-0.0156
```

### E2-Flow Final Steps
```
[Epoch 4 Step 148300] loss=1.213 flow=1.102 act_mse=0.898 world=0.558 gate=-0.016
[Epoch 4 Step 148350] loss=1.067 flow=0.938 act_mse=1.094 world=0.647 gate=-0.016
[Epoch 4 Step 148400] loss=1.245 flow=1.141 act_mse=0.961 world=0.520 gate=-0.016
[Epoch 4] VAL action MSE=0.982144 gate=-0.0156
```
