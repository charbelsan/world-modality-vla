# World-Token VLA with Chain-of-Causality (CoC) Head

This folder contains the second line of experiments in this project:

- A **world-modality transformer** that already predicts and consumes discrete world tokens.
- A **Chain-of-Causality (CoC) head** that generates textual explanations grounded in that world stream.

The goal is to build a VLA that:

- Predicts actions and multi-step future world tokens.
- Explains its own behavior and imagined futures in natural language.
- Eventually transfers across embodiments (e.g. Franka, Unitree G1) using a shared world token vocabulary.

This is deliberately more ambitious than a “toy caption head”: CoC is intended to describe *why* actions are taken, in terms of evolving world state and task goals.

---

## 1. High-level architecture

We reuse the existing world-modality transformer as the core and add a CoC head on top.

### 1.1 Inputs and world tokens

Per training sample (episode segment), the shared trunk sees:

- `LANG`: tokenized instruction text (if the dataset is language-conditioned).
- `ROBOT_ID`: a learned embedding indicating which robot/embodiment generated the episode.
- `OBS`: a token that fuses:
  - the latest vision embedding from a frozen encoder, and
  - the latest proprio state.
- `WORLD_CUR`: embedding lookup of the current world token `w_t`.
- `FUT_QUERY × K`: learnable queries that predict future world tokens `w_{t+1..t+K}`.
- `ACT_QUERY × H`: learnable queries for action chunk prediction `[a_t..a_{t+H-1}]`.

The transformer runs self-attention over:

```text
[LANG...] [ROBOT_ID] [OBS] [WORLD_CUR] [FUT_Q1..FUT_QK] [ACT_Q1..ACT_QH]
```

This is a strict extension of the existing model C: we keep world tokens as a *first-class modality* inside the transformer.

### 1.2 Heads

We attach three heads:

1. **Action head** (existing):
   - Reads ACT_QUERY hidden states.
   - Outputs continuous actions `[B, H, D_a]`.

2. **World head** (existing):
   - Reads FUT_QUERY hidden states.
   - Outputs logits `[B, K, V]` over world tokens `w_{t+1..t+K}`.

3. **CoC head** (new):
   - Reads a summary of:
     - instruction tokens,
     - robot ID,
     - WORLD_CUR and FUT_QUERY hidden states (i.e., the world rollout),
   - Acts as a causal language decoder that generates chain-of-causality text:
     - “The robot moves towards the red block to grasp it...”.

The CoC head is not a toy classifier: it is a proper language decoder with cross-entropy training against multi-sentence causal narratives.

### 1.3 Loss

For each sample we define:

- `L_act`: behavior-cloning loss on the action chunk.
- `L_world`: world token multi-step prediction loss (cross-entropy over `[B,K,V]` vs `[B,K]`).
- `L_coc`: cross-entropy over CoC tokens (teacher forcing).

Total loss:

```text
L = L_act + λ_world * L_world + α_coc * L_coc
```

Initial coefficients:

- `λ_world = 0.2` (as in the world-modality experiments).
- `α_coc ∈ [0.1, 0.3]` (tuned).

---

## 2. Datasets and CoC generation

### 2.1 Base datasets

The CoC line is designed to run on a mix of:

- **LIBERO** (`HuggingFaceVLA/libero`):
  - Language-conditioned manipulation tasks.
  - Franka-like arm; good first testbed.
- **Open X-Embodiment (OXE) subsets** (optional but recommended):
  - A small set of robots with different kinematics (e.g. Franka, UR, Aloha).
  - For cross-embodiment training and testing.

In all cases we assume a `LeRobotDataset`-compatible interface with:

- images,
- proprio/robot state,
- actions,
- optional instruction text.

### 2.2 CoC labels (using open-source VLM/LMM)

CoC labels are generated offline by an open-source VLM or LMM. The expected pipeline is:

1. For each episode:
   - Extract instruction text (if available).
   - Sample a few representative frames:
     - start (`t0`),
     - middle (`tmid`),
     - end (`tend`).
2. Call a VLM (e.g. Qwen2-VL, LLaVA) with:
   - the instruction text,
   - the three frames (or short clips),
   - a prompt asking for a 3–6 step chain-of-causality:
     - “what the robot does”,
     - “why this moves closer to the goal”.
3. Save the CoC as a short multi-sentence text per episode.

The script `coc_vla/coc_generation.py` implements this flow generically; the specific VLM model is selectable via CLI.

We aim for **episode-level CoC** first (one CoC per episode), which is already rich enough to shape the shared representation. Later, we can move to chunk-level CoC (per action/world rollout chunk).

---

## 3. Folder structure

The `coc_vla` folder is self-contained and designed to sit alongside the existing `world_modality` code:

- `coc_vla/README.md` – this document.
- `coc_vla/__init__.py` – package marker.
- `coc_vla/config.py` – configuration dataclasses for CoC experiments.
- `coc_vla/data.py` – dataset wrappers that extend `SR100SequenceDataset` with CoC text.
- `coc_vla/coc_generation.py` – utilities to generate CoC labels using an open-source VLM.
- `coc_vla/model.py` – two-head VLA model:
  - wraps `world_modality.model.WorldPolicyTransformer`,
  - adds a CoC decoder head.
- `coc_vla/train.py` – training script for the two-head model:
  - loads world tokens, actions, and CoC,
  - optimizes `L_act + λ_world * L_world + α_coc * L_coc`.
- `coc_vla/eval_coc.py` – skeleton for evaluating CoC quality and action/world metrics.

This keeps the CoC line modular: you can develop and iterate here without disturbing the baseline world-modality experiments.

---

## 4. CoC generation utilities

The file `coc_vla/coc_generation.py` contains:

- A utility to iterate over `LeRobotDataset` episodes.
- A launcher to call a Hugging Face VLM (e.g. Qwen2-VL) via `transformers`:
  - encode a small set of images,
  - send them along with instruction text and a CoC prompt,
  - parse and save the CoC output.
- A simple JSONL output format:

```json
{
  "episode_id": 42,
  "instruction": "put the red block in the green bin",
  "frames": {
    "t0": "path/or/index/of/first_frame",
    "tmid": "path/or/index/of/mid_frame",
    "tend": "path/or/index/of/end_frame"
  },
  "coc_text": [
    "The robot moves its gripper above the red block to approach it.",
    "It lowers and closes the gripper to grasp the block.",
    "Then it lifts and moves the block over to the green bin.",
    "Finally, it opens the gripper to place the block in the bin."
  ]
}
```

Training scripts in this folder expect such a JSONL file per dataset split (`train`, `val`, etc.).

---

## 5. Two-head model implementation

The file `coc_vla/model.py` defines:

- A `WorldCocTransformer` class that:
  - takes a `WorldPolicyTransformer` instance (world + action),
  - adds a CoC decoder head.
- The CoC decoder:
  - is a small causal transformer,
  - consumes CoC tokens with teacher forcing,
  - uses pooled trunk hidden states as conditioning.

During training:

- The world and action heads behave exactly as in existing experiment C.
- The CoC head is trained in parallel on CoC text.

This reuse ensures architectural compatibility with all the world-modality ablations (A/B/C/C_no_world_input).

---

## 6. Training and evaluation (high-level)

### 6.1 Training

Once CoC JSONL files are generated for LIBERO (and later for cross-embodiment mixes), the training loop is:

1. Build world token caches using `world_modality/precompute_world_tokens.py` (once).
2. Train the two-head model:

```bash
python -m coc_vla.train \
  --dataset_name HuggingFaceVLA/libero \
  --coc_jsonl /path/to/libero_train_coc.jsonl \
  --cache_dir /mnt/worldmod/cache \
  --log_dir /mnt/worldmod/logs_coc_libero \
  --batch_size 512 \
  --learning_rate 1e-4 \
  --lambda_world_loss 0.2 \
  --alpha_coc 0.2 \
  --max_epochs 10
```

3. Later, switch `dataset_name` and `coc_jsonl` to cross-embodiment mixes.

### 6.2 Evaluation

The script `coc_vla/eval_coc.py` (skeleton) is intended to:

- Report:
  - action MSE,
  - world rollout accuracy vs horizon,
  - CoC quality metrics (e.g., LLM-as-judge scores on coherence, correctness).
- For cross-embodiment:
  - report per-robot metrics,
  - analyze how consistent CoC is across different embodiments.

---

## 7. Future extensions

Once the basic two-head CoC VLA is running, natural extensions include:

- **Chunk-level CoC**: generate and train CoC per action/world rollout segment instead of per episode.
- **Consistency constraints**: use an LLM-as-critic to enforce agreement between:
  - world rollouts,
  - CoC,
  - and success/failure labels.
- **Steering vectors on the world stream** (Introspection-style):
  - probe and steer world hidden states for properties like long-horizon foresight, safety, etc.
- **Deployment on Unitree G1 and Franka**:
  - calibrate action spaces,
  - fine-tune on a small set of G1/Franka demos using the same world token vocabulary and CoC head.

This folder provides the scaffolding to develop those ideas without entangling the core world-modality experiments.

