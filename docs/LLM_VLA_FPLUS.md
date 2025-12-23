## Model F+ (Qwen3-VL + World Memory + CoC)

This document defines the F+ model and the exact experiments to run on L40S.
F+ extends Model F with optional CoC language loss and FLARE-style latent alignment.

### Core principle (do-no-harm)
Actions are always read from <ACT_i> hidden states. Language generation is a
separate pass and never feeds into action computation.

### Architecture summary
1) VLM backbone: Qwen3-VL-3B-Instruct.
2) Action interface:
   - Prompt: "<image> <instruction> <ACT_0> ... <ACT_{H-1}>"
   - Hidden states at ACT tokens -> ActionHead -> action chunk.
3) World memory (Model F):
   - Prophet predicts future latents z_pred from z_hist.
   - Cross-attend z_pred into ACT hidden states only.
   - Gated residual with gate initialized to 0.
4) CoC (Phase 2):
   - Separate forward pass for language loss on CoC text.
   - Actions are computed first and are not conditioned on CoC output.

### Losses
- L_action: MSE on action chunk.
- L_world: cosine loss between z_pred and z_future (FLARE-style alignment).
- L_text: text loss on CoC labels (optional).

Total:
  L = L_action + lambda_world * L_world + lambda_text * L_text

### Data requirements
- LIBERO dataset in LeRobot format.
- Precomputed world latents (DINO or V-JEPA).
- Optional CoC JSONL with {episode_id, coc_text}.

### Precompute world latents
```
python -m world_modality.precompute_world_latents \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image \
  --world_latents_source vjepa \
  --cache_dir cache
```

### Experiment matrix (E0-E4)
Default runs (in launcher):
- E0: VLM-BC baseline (no world loss, no injection)
- E2: Model F (world memory injection)
- E4: Full F+ (Model F + CoC + FLARE alignment)

Optional diagnostics (not run by default):
- E1: Auxiliary world loss only (no injection)
- E3: CoC loss without F+ (talk while acting)

### Corruption evaluation
Use validation corruption to verify reliance on future memory:
--corruption_mode {none, zero, random, shuffle, oracle}

Expected:
- Oracle helps early.
- Corruption hurts as gate opens.

### Notes from SmolVLA to borrow (principles)
- Keep the model small and efficient.
- Reduce token pressure and avoid bloated prompts.
- Separate policy and language outputs (avoid feedback loops).
- Prefer inference strategies that decouple prediction and execution (RTC-style).

### Quick launch
Use `scripts/run_fplus_experiments.sh` to run the default experiments (E0, E2, E4).
For a zero-ambiguity setup, see `docs/L40S_RUNBOOK.md`.

Environment variables (optional):
- DATASET, IMAGE_KEY, INSTRUCTION_KEY, CACHE_DIR
- WORLD_SOURCE (dino or vjepa)
- BACKBONE (default: qwen3_vl_3b_instruct)
- BATCH_SIZE, MAX_EPOCHS, LOG_EVERY
- COC_JSONL (required for E4)
