# SmolVLA + World Modality: Fusion Ablations (F1–F3)

This doc defines *optional* ablations to test whether “world modality” needs to influence **representations earlier**
than the default Model‑F style injection.

**Baseline reference**
- E0: `smolvla` fine-tune from `HuggingFaceVLA/smolvla_libero`
- E2: `smolvla_world` with predicted memory (`world_memory_mode_train=pred`)

Keep **seed, suite, sample budget, eval protocol** identical across ablations.

---

## Terminology

- `z_hist`: cached/online world history latents `[B, T_ctx, D]`
- `z_future`: cached future latents `[B, K, D]`
- `mem`: the tensor provided to fusion (depends on mode: `pred|oracle|zero|...`)
- “gate”: a learned scalar; effective strength is `tanh(gate)`.

---

## F1 (Default): Late injection in the action expert (Model‑F)

**What it is**
- Inject `mem` into the **action expert hidden states** right before `action_out_proj`.
- This happens during training and at **every denoise step** during diffusion-style inference.

**Why**
- Minimal interference with the VLM prefix stream.
- Do‑no‑harm: gate init = 0, so behavior matches baseline until the model learns to open it.

**Expected signal if world helps**
- `world_gate` increases away from 0.
- E2 SR > E0 SR, and rollout ablations (set rollout memory to `zero/random`) reduce SR.

**How to run**
- This is the current default behavior (`--policy.enable_world_injection=true`).

---

## F2: Earlier injection into the suffix *input embeddings* (expert path)

**What it is**
- Inject `mem` into the **suffix embeddings** (`suffix_embs`) *before* they go through the expert transformer.
- This lets world memory shape the expert’s internal computation, not only the final readout.

**Why**
- If world information needs to influence “planning/representation”, late-only injection may be too weak.

**Expected outcomes**
- If late injection is too weak: F2 > F1 on SR and/or faster learning (gate opens earlier).
- If the expert already has enough capacity: F2 ≈ F1.
- If fusion destabilizes: SR drops and/or gates explode (watch non-finite metrics).

**How to run**
```bash
lerobot-wm-train \
  ... \
  --policy.type=smolvla_world \
  --policy.world_inject_suffix_in=true
```

Logging:
- `world_gate_suffix_in` tracks this path.

---

## F3a: Prefix token insertion (world as a true modality)

**What it is**
- Project `mem` to the VLM prefix hidden size and **insert K “world tokens”** into the prefix padding.
- This changes the prefix stream and introduces token competition.

**Two variants**
- `world_prefix_block=prefix` (mask_ar=0): world tokens join the image/language block (max competition).
- `world_prefix_block=state` (mask_ar=1): world tokens behave like state tokens (image/lang cannot attend to them;
  state/actions can). This is safer but less “full modality”.

**Expected outcomes**
- If early fusion is necessary: F3a(prefix) can outperform F1/F2 but may be less stable.
- If token competition hurts: SR drops vs E0, gates stay near 0, or attention becomes diffuse.

**How to run**
```bash
lerobot-wm-train \
  ... \
  --policy.type=smolvla_world \
  --policy.world_prefix_tokens=8 \
  --policy.world_prefix_block=prefix \
  --policy.world_prefix_gate_init=0.0
```

Logging:
- `world_gate_prefix` tracks the prefix-token gate.

---

## F3b: Prefix cross-attention (no extra tokens)

**What it is**
- Add a gated cross-attn module that lets **prefix embeddings** attend to `mem` without increasing sequence length.

**Why**
- Tests “world influences representation” without the confound of changing token count / positions.

**Expected outcomes**
- If representation shaping is key: F3b > F1.
- If the VLM prefix is already strong / world is redundant: F3b ≈ F1.

**How to run**
```bash
lerobot-wm-train \
  ... \
  --policy.type=smolvla_world \
  --policy.world_prefix_cross_attn=true \
  --policy.world_prefix_num_heads=8 \
  --policy.world_prefix_gate_init=0.0
```

Logging:
- `world_gate_prefix_cross` tracks this path.

---

## Safety / interpretability notes

1) **Do‑no‑harm numerical guard**
   - When a gate is effectively closed, fusion code skips computing/using memory to avoid `0 * NaN = NaN` issues.

2) **Don’t mix ablations unless you intend to**
   - F2 + F3 together makes attribution harder. Run them separately first.

3) **Keep sample budget fixed**
   - Use `TOTAL_SAMPLES` in `scripts/run_mi300x_smolvla_world_matrix.sh` to compare fairly across batch sizes.

