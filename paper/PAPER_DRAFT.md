# World Tokens as a First‑Class Modality for Vision‑Language‑Action Policies

> Paper draft (living document).  
> This repo contains the code to reproduce the core experiments (A/B/C/C_no_world_input) and the CoC data/labeling pipeline.
>
> **Status:** Methods/positioning are stable; results tables are placeholders until experiments are rerun.

---

## Abstract

Embodied policies are often **reactive**: they map recent observations to actions without an explicit internal state that is exposed, controllable, and analysable. While many systems train with auxiliary prediction losses or rely on large backbones that implicitly encode dynamics, the resulting “world model” remains **entangled** in the weights and is difficult to supervise or intervene on.

We propose **World Tokens**, a discrete world‑state stream treated as a **first‑class modality inside a Vision‑Language‑Action (VLA) transformer**. Each timestep is mapped to a discrete id via a frozen vision encoder followed by vector quantization. The policy **consumes** the current world token as part of its input sequence and **predicts** a multi‑step future world token rollout alongside action chunks. This yields an explicit, rollable internal state that can be supervised, ablated, and visualized.

We evaluate four controlled variants that disentangle “future prediction helps” from “world modality helps”: (A) standard behavior cloning, (B) auxiliary future token loss only, (C) world token input modality + future token loss, and (C_no_world_input) an ablation with future token loss but no world token input. We report offline action error, future token accuracy, and a causal **world‑corruption intervention** that scrambles the world input token at inference. Our results (to be filled) indicate that world tokens as a modality improve action quality and long‑horizon stability beyond an auxiliary loss, and that the learned policy causally relies on the world stream.

---

## 1. Introduction

Robotic control is a **partially observed** sequential decision problem: a single image is not the world, and action choices should depend on latent factors such as object contact state, grasp stability, and task progress. An effective policy therefore needs an internal **state** (a belief over the world) to avoid drift and compounding error over time.

Recent VLA systems improve robustness by scaling pretrained perception backbones, by adding predictive auxiliary objectives, or by using external world models to score rollouts. However, these approaches often leave the “world model” either:

1) **implicit**, buried in hidden states and weights (hard to supervise/diagnose), or  
2) **external**, queried as a tool or used only for data generation (not integrated into the policy’s token computation).

We take a different position:

> If “world modeling” matters for action, it should appear in the policy’s computation as a **first‑class modality**, like vision and language—explicit tokens the transformer can attend to and condition on.

This paper tests that thesis with a minimal, scalable construction of world tokens and a clean ablation matrix that isolates modality effects from auxiliary‑loss effects.

### Contributions

1. **World tokens as a modality:** a discrete world stream inserted into a VLA transformer sequence, consumed and predicted jointly with actions.
2. **Clean experimental separation:** A vs B vs C vs C_no_world_input isolates the effect of explicit modality from auxiliary prediction.
3. **Causal intervention:** a world‑corruption test shows whether the policy actually uses world tokens (beyond correlational metrics).
4. **Practical and reproducible pipeline:** frozen encoder + k‑means/VQ codebook + cached tokenization enables rapid experimentation at scale.

---

## 2. Problem Setup

We assume demonstration trajectories containing (at minimum):

- observation image(s) \(o_t\) (RGB),
- proprio/state \(s_t\) (joint positions, gripper state, etc.),
- action \(a_t\) (continuous control),
- optional instruction text \(x\) (language‑conditioned tasks).

We train a policy to predict an **action chunk** of horizon \(H\):

\[
\hat a_{t:t+H-1} = \pi(\cdot)
\]

And we additionally train it to predict a **future world token rollout** of horizon \(K\):

\[
\hat w_{t+1:t+K} = \pi(\cdot)
\]

The key question is not whether “predicting the future” helps (it often does), but whether **making the predicted world state explicit in the token sequence** provides additional gains and enables causal control/analysis.

---

## 3. World Tokens: Definition and Construction

We define a discrete world token per timestep using a frozen perceptual encoder and vector quantization.

### 3.1 Frozen perceptual embedding

We embed each observation with a frozen vision encoder:

\[
e_t = \mathrm{Enc}_v(o_t) \in \mathbb{R}^{d_e}.
\]

In the simplest instantiation, \(e_t\) is the pooled (CLS/global‑avg) embedding of a pretrained ViT (e.g. DINO‑style features). This paper treats the encoder as a modular component; it can be replaced by stronger predictive representations (Section 8).

### 3.2 Vector quantization into discrete ids

We build a codebook \(C=\{c_j\}_{j=1}^N\) by k‑means over a large sample of embeddings. World token assignment is nearest‑centroid:

\[
w_t = \arg\min_{j} \lVert e_t - c_j\rVert^2.
\]

This yields a discrete vocabulary of size \(N\) that can be predicted with next‑token‑style cross entropy, and inspected by nearest‑neighbor visualization in embedding space.

### 3.3 Why this counts as “world modeling”

This is **not** pixel prediction. The claim is narrower and more testable:

- World tokens provide a **discrete predictive state** the policy can roll forward.
- The token stream is **explicit** (supervisable, ablatable, visualizable).
- It scales to large datasets because tokenization is offline and cached.

---

## 4. Policy Architecture: World as a First‑Class Modality

We implement a single transformer trunk with different input sequences for each model variant.

### 4.1 Token layout

Let \(d\) be the transformer width. We create tokens:

- `OBS`: fused projection of the latest \((e_t, s_t)\).
- `WORLD_CUR`: an embedding lookup of the current world id \(w_t\) (only for model C).
- `FUT_QUERY × K`: learnable query tokens used to predict \(w_{t+1:t+K}\).
- `ACT_QUERY × H`: learnable query tokens used to predict action chunk \(a_{t:t+H-1}\).
- optional language tokens `LANG` if dataset provides instruction text.

The trunk is a standard self‑attention transformer encoder over the concatenated sequence.

### 4.2 Four controlled variants

**A — Behavior Cloning (BC)**  
Inputs: `[OBS] [ACT_QUERY×H]`  
Outputs: actions  
Loss: action loss only

**B — Implicit world model (auxiliary future loss only)**  
Inputs: `[OBS] [FUT_QUERY×K] [ACT_QUERY×H]`  
Outputs: actions + future world tokens  
Loss: action + \(\lambda\)·world loss  
Crucially: world tokens are **not** fed as inputs.

**C — World modality (ours)**  
Inputs: `[OBS] [WORLD_CUR] [FUT_QUERY×K] [ACT_QUERY×H]`  
Outputs: actions + future world tokens  
Loss: action + \(\lambda\)·world loss  
Here world tokens are **first‑class inputs** with their own embedding and position.

**C_no_world_input — Ablation (aux loss only, same codepath as C)**  
Inputs: `[OBS] [FUT_QUERY×K] [ACT_QUERY×H]`  
Outputs: actions + future world tokens  
Loss: action + \(\lambda\)·world loss  
This isolates “modality matters” vs “aux loss matters.”

### 4.3 Heads and losses

Action head: maps each `ACT_QUERY` hidden state to continuous actions.

World head: maps each `FUT_QUERY` hidden state to logits over \(N\) world tokens:

\[
\mathcal{L}_{world}=\sum_{k=1}^{K} \mathrm{CE}(\hat w_{t+k}, w_{t+k}).
\]

Total training objective:

\[
\mathcal{L}=\mathcal{L}_{act}+\lambda\,\mathcal{L}_{world}.
\]

---

## 5. Why “World as a Modality” Should Help (Theoretical Intuition)

This section provides first‑principles reasoning (not a formal theorem) for why model C can outperform model B even with identical world prediction loss.

### 5.1 Partial observability and belief state

In a POMDP, optimal control depends on a belief state \(b_t\) that summarizes history. In practice, transformer policies approximate this by compressing observation history into hidden states. But this belief can be:

- distributed across many dimensions and token positions,
- difficult to supervise directly,
- and fragile under distribution shift.

World tokens provide an **explicit, compressive belief proxy**: by forcing a discrete partition of perceptual embeddings that is predictive of future states, the token id acts as a low‑bandwidth state representation.

### 5.2 Modality vs auxiliary loss

Model B uses the future prediction loss to *shape* hidden states, but it does not provide an explicit world state **as an input variable**. As a result, “world information” must be:

- internally re‑derived from `OBS` each time,
- and remains entangled with the action computation.

Model C changes the computation graph:

- `WORLD_CUR` is directly visible to attention.
- The trunk can form attention patterns where action queries condition on world state **explicitly**, in the same way they condition on language or vision tokens.

This is analogous to the difference between:

- “regularize representations to contain X” (auxiliary loss), and
- “give the model X as an explicit input channel it can route computation through.”

### 5.3 Testable consequence: causal sensitivity

If `WORLD_CUR` truly functions as a modality, corrupting it should measurably degrade performance. This yields a causal test beyond metric improvements:

- Model C: corruption ratio \(>1\) expected.
- C_no_world_input: corruption ratio \(\approx 1\) expected (since it does not consume \(w_t\)).

---

## 6. Experimental Design

### 6.1 Datasets

Primary dataset (first pass, highly reproducible):

- `HuggingFaceVLA/libero` via LeRobot.

Optional extensions:

- cross‑embodiment mixtures (e.g., subsets of Open‑X‑Embodiment) to test transfer across robots (Franka → Unitree).

### 6.2 Metrics

Offline metrics (fast and stable):

- action chunk MSE (or task‑standard action loss),
- world token Top‑1 and Top‑5 accuracy per horizon \(k\),
- corruption ratio = corrupted_MSE / clean_MSE (causal dependence).

Online metrics (stronger, if available):

- success rate on a fixed task protocol (sim or real robot).

### 6.3 Implementation notes (reproducibility)

This repo saves a `config.json` next to checkpoints for each run, and provides scripts for caching embeddings/tokens.

See:

- `world_modality/precompute_world_tokens.py`
- `world_modality/train.py` (A/B/C/C_no_world_input)
- `world_modality/eval_offline.py`
- `world_modality/intervention_corrupt_world.py`

---

## 7. Results (Placeholders)

> Fill with rerun experiments after VM restart.

### 7.1 Main table

| Model | World input? | World loss? | Action MSE ↓ | World Top‑1 ↑ | World Top‑5 ↑ | Corrupt ratio ↑ |
|------:|:------------:|:-----------:|:------------:|:-------------:|:-------------:|:---------------:|
| A | ✗ | ✗ | TODO | – | – | – |
| B | ✗ | ✓ | TODO | TODO | TODO | – |
| C_no_world_input | ✗ | ✓ | TODO | TODO | TODO | ~1.0 (expected) |
| C (ours) | ✓ | ✓ | TODO | TODO | TODO | >1.0 (expected) |

### 7.2 Key takeaways to highlight (when filled)

1) A → B: future prediction helps representations (implicit dynamics).  
2) B/C_no_world_input → C: world tokens as a modality improves control beyond auxiliary loss.  
3) Corruption ratio for C supports causal reliance on the world modality.

---

## 8. Related Work (First‑Principles Comparison)

This section frames major prior directions as **principles** and explains how we borrow them without abandoning the paper’s bet.

### 8.1 Predictive representation learning: (V‑)JEPA / action‑conditioned JEPA

**Principle:** Learn latents that preserve dynamics‑relevant information while discarding pixel noise.

**Why it’s strong:** Better latents tend to yield better downstream control and generalization.

**How we borrow without drifting:** Replace the frozen encoder used to form \(e_t\) (and thus world tokens) with a JEPA/V‑JEPA‑style representation, potentially action‑conditioned. The rest of our method remains: discrete tokenization + world tokens as a modality.

### 8.2 Large pretrained robotics stacks: GR00T‑style systems

**Principle:** Scale multimodal grounding (vision+language) and action distribution modeling (often diffusion/flow) to improve robustness across tasks and embodiments.

**Why it’s strong:** Strong grounding reduces perception failures; distributional action heads reduce regression brittleness.

**How we borrow without drifting:** We keep world tokens explicit. We can:

- swap in a stronger vision/language backbone to produce \(e_t\),
- modernize the action head (diffusion/flow),
- keep the world stream as the explicit internal state modality.

### 8.3 Implicit world models and “reason‑while‑acting”: Alpamayo‑style

**Principle:** Use auxiliary predictive objectives and/or language rationales (chain‑of‑thought/causality) to shape internal computation for better action selection, without an explicit world‑state variable.

**Why it’s strong:** Joint reasoning objectives can regularize representations and improve interpretability.

**How we borrow without drifting:** Our model B corresponds to “implicit dynamics via future prediction loss.” We add the stronger claim: making the world state explicit as tokens supports **causal interventions** and future planning interfaces. We also extend toward a CoC head grounded in the world stream (Section 9).

### 8.4 Robotics systems with explicit world modeling (e.g., Unitree‑style pipelines)

**Principle:** Separate perception/state estimation, world modeling, and control; sometimes integrate planning with a learned world model.

**Why it’s strong:** Explicit state variables enable planning, debugging, and modular upgrades.

**How we borrow without drifting:** Our contribution is to re‑embed “explicit state” into a **single transformer sequence** so the policy can attend to and predict world state end‑to‑end, while remaining compatible with modular components (encoder, tokenizer, CoC generator).

---

## 9. Toward a Two‑Head “Act + Explain” Model (Chain‑of‑Causality)

We extend the world‑token VLA with a CoC head that produces a multi‑step causal narrative of behavior grounded in the world rollout.

### 9.1 CoC supervision via open VLMs (Qwen3‑VL)

Because few robotics datasets contain human CoC annotations, we generate labels offline using open VLMs:

- Sample keyframes (start/mid/end) and instruction.
- Prompt the VLM for 4–7 numbered causal steps (“what” + “why”).
- Store JSONL labels per episode.

Repo support:

- `coc_vla/coc_generation.py` (Qwen3‑VL support; resume mode)
- `coc_vla/train.py` (two‑head scaffold; requires CoC labels)

### 9.2 Why CoC is complementary to world tokens

World tokens provide a compact predictive state; CoC provides a **linguistic interface** that can:

- diagnose failure modes (“gripper missed object, then…"),
- enforce consistency between predicted world rollouts and explanations,
- support transfer by aligning behavior with task‑level causal structure.

---

## 10. Introspection and Steering (Optional Analysis Track)

Inspired by recent work on “introspection” and steering internal directions, we can treat hidden states at `WORLD_CUR` and `FUT_QUERY` positions as probes for:

- success vs failure,
- grasp/contact state,
- subgoal completion.

We can then:

- compare linear separability across A/B/C,
- and optionally apply small steering interventions at inference to test causal sensitivity beyond token corruption.

This is **analysis tooling**, not a replacement for explicit world tokens.

---

## 11. Limitations

- World tokens inherit biases of the frozen encoder and the k‑means partition.
- Discrete tokenization can collapse rare but important states without sufficient codebook capacity.
- Offline action MSE is not a perfect proxy for closed‑loop success; online evaluation is preferable when available.
- CoC labels generated by VLMs may contain hallucinations; filtering/judging is required for high‑stakes claims.

---

## 12. Conclusion

We argue that “world modeling” should not be a hidden side effect of auxiliary losses or an external tool used only for scoring. By making world state an explicit **token modality** inside the policy transformer, we obtain a controllable, rollable, and interpretable representation that improves action selection and enables causal analysis.

The key empirical question is not whether future prediction helps (it often does), but whether **world tokens as first‑class inputs** add measurable benefit beyond an auxiliary objective. Our ablation matrix and corruption tests are designed to answer exactly that.

---

## Appendix A: Reproducibility pointers (repo)

- World token precompute: `world_modality/precompute_world_tokens.py`
- Train A/B/C/C_no_world_input: `world_modality/train.py` via entrypoints `train_baseline_*.py`, `train_model_c.py`
- Offline eval: `world_modality/eval_offline.py`
- Corruption intervention: `world_modality/intervention_corrupt_world.py`
- Crash‑proof logs: `ops/README.md`
- CoC generation: `coc_vla/coc_generation.py`
- CoC design notes and external repo fetcher: `coc_vla/DEEP_DIVE.md`, `coc_vla/external/fetch_external_repos.sh`

