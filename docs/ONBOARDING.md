# Onboarding (Curtis): Qwen3‑VL + World Memory (Model F+)

This repo has legacy “small transformer” models (A/B/B_cont/C/F) **and** the new
pipeline we actually care about on L40S:

> **F+ = generalist VLM (Qwen3‑VL) + action tokens + Model‑F world memory injection + optional CoC.**

If you only run one thing on the L40S box: run `scripts/run_fplus_experiments.sh`
(E0 baseline + E2 Model‑F + optional E4 CoC).

---

## 0) What to read first
- Launch/run details: `docs/L40S_RUNBOOK.md`
- Exact F+ experiment matrix + flags: `docs/LLM_VLA_FPLUS.md`

---

## 1) First‑principles architecture (F+)

### 1.1 The action interface (VLM → actions, no decoding)
We add `H` special tokens: `<ACT_0> ... <ACT_{H-1}>` (default `H=8`).

Prompt per sample:
```
<image>
<instruction>
<ACT_0> ... <ACT_7>
```

We do **one forward pass** of the VLM (no text generation). We take the final
hidden states at the `<ACT_i>` positions and decode them with an MLP:

- `h_act`: `[B, H, d_llm]` gathered from the VLM
- `a_pred = ActionHead(h_act)`: `[B, H, action_dim]`

Key point: actions are read from hidden states, **not** from generated text.

### 1.2 What are `z_hist`, `z_future`, `z_pred`?
We precompute a per‑frame latent `z_t` from a frozen vision model:
- `world_latents_source=dino`: DINOv2 frame embedding (fast bootstrap)
- `world_latents_source=vjepa`: V‑JEPA2 video latent (preferred; more dynamics‑aware)

For each training sample at time `t` we build:
- `z_hist = [z_{t-T+1}, ..., z_t]` with `T=context_frames` → shape `[B, T, D_w]`
- `z_future = [z_{t+1}, ..., z_{t+K}]` with `K=future_offset` → shape `[B, K, D_w]`

`z_future` is **ground truth future** latents during training only (we have the
future frames in the dataset).

`z_pred = Prophet(z_hist)` predicts the same object:
- `z_pred ≈ z_future` → shape `[B, K, D_w]`

So:
- **What does `z_pred` encode?** A *sequence* of predicted future world latents
  for the next `K` steps (not “one latent for all futures”).
- **Why align to `z_future` instead of using `z_future` directly?** Because at
  inference you do not have future frames, so you must use `z_pred`. We still
  use `z_future` in training as a supervision target and (optionally) as an
  oracle memory for scheduled sampling.

### 1.3 Model‑F world injection (external memory, gated)
We do **not** insert world tokens into the main LLM context by default.
Instead, we treat predicted futures as an *external memory*:

1) Prophet predicts `z_pred` (or we use `z_future` as oracle early in training).
2) Project memory to LLM dimension: `kv = W(z_*)` → `[B, K, d_llm]`
3) Cross‑attend from `<ACT>` states into the memory:
   - `ctx = CrossAttn(query=h_act, key=kv, value=kv)` → `[B, H, d_llm]`
4) Gated residual (gate init = 0):
   - `h_act ← h_act + tanh(gate) * ctx`

This is the “do‑no‑harm” mechanism: if the future memory is useless/noisy, the
model can keep `tanh(gate)≈0`.

### 1.4 Scheduled sampling (oracle → predicted)
During training we can mix memory sources:
- early epochs: mostly oracle `z_future`
- later epochs: mostly predicted `z_pred`
- inference: predicted only

This avoids a training/inference mismatch.

---

## 2) CoC (“talk while acting”) in F+
We optionally train a language loss on CoC labels, **without letting language
tokens influence the action path**.

Default (`--text_loss_mode joint_after_act`):
- The input sequence contains CoC target tokens *after* `<ACT> ...`.
- Causality ensures tokens after `<ACT>` cannot affect `<ACT>` hidden states.
- This matches the “single autoregressive stream” style used by Alpamayo‑R1,
  but we intentionally keep the direction **act → talk** for safety.
 - If an episode has no CoC label, `L_text` is skipped for that sample (or pass
   `--require_coc` to drop unlabeled episodes).

Alternative (`--text_loss_mode separate`):
- Run a separate forward pass to compute text loss (simplest mental model, more compute).

Important: we always compute actions from `<ACT>` hidden states; we never parse
actions out of generated text.

---

## 3) Experiments you will actually run on L40S
Launcher: `scripts/run_fplus_experiments.sh`

- **E0 (baseline)**: Qwen3‑VL + `<ACT>` + ActionHead, no world injection.
- **E2 (Model‑F)**: E0 + Prophet + gated cross‑attention into `<ACT>` tokens.
- **E4 (F+)**: E2 + CoC text loss (if `COC_JSONL` is provided).

Diagnostics you might add later (optional):
- Memory corruption at validation: `--corruption_mode {zero,random,shuffle,oracle}`
- “No injection” sanity: `--disable_future_injection` (should reduce to baseline)

---

## 4) Common monitoring signals
- Action MSE (train/val)
- `gate=tanh(gate_param)` (should start ~0; may increase if future memory helps)
- Corruption gap: performance drop when future memory is corrupted (proves reliance)

---

## 5) Legacy A/B/C line (do not run by default)
The older models (`train_baseline_a.py`, `train_baseline_b.py`, `train_model_c.py`)
implement “small transformer” ablations with discrete VQ world tokens.

They’re kept for historical comparison, but the default direction is F+ on Qwen3‑VL.
