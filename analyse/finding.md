# Findings Log

Purpose: keep a short, append-only record of the important experimental findings, the lesson from each result, and the decision it triggered.

Rule:
- Add one new row for each result that changed the research direction, implementation, or evaluation protocol.
- Prefer concrete numbers over narrative.
- Keep the “Decision / Next Step” column actionable.

| Date | Experiment / Event | Result | Lesson | Decision / Next Step |
|---|---|---|---|---|
| 2025-12-28 | Qwen F+ closed-loop LIBERO runs (`E0/E2`, MSE + Flow) | Offline losses improved, but closed-loop SR stayed at `0%` | Better offline prediction did not translate to control; the baseline itself was not validated | Stop treating Qwen F+ as the main control instrument; pivot to a proven LIBERO policy |
| 2026-01-14 | SmolVLA baseline `E0` on `libero_spatial` | High non-zero SR (`~89.5%`) | The benchmark/control stack is valid with SmolVLA | Use SmolVLA as the control backbone for world-modality tests |
| 2026-01-14 | `E1` world-zero after processor fix | `E1 ~= E0` (`~88.5%` vs `~89.5%`) | Extra world modules do not help by themselves; the do-no-harm control passed | Keep Model-F style controlled comparisons |
| 2026-01-15 | SmolVLA processor/postprocessor mismatch | `E1` initially showed `0%` SR, then recovered after loading processors from the init checkpoint | Eval can fail because of processor/action-convention mismatch, not because the model is bad | Always reuse the saved init processors for `smolvla_world` experiments |
| 2026-01-15 | Rollout temporal mismatch (`m=1` vs cached `m=4`) | Task 0-1 performance recovered strongly after matching rollout encoding to cached latents | Train/eval world embeddings must come from the same temporal encoding distribution | Infer or force rollout temporal window from the latent suffix/config |
| 2026-03-17 | Matched-budget `E2-pred` vs `E0` on `libero_spatial` | `E0 = 83.6%`, `E2-pred = 80.8%` | The current JEPA-style implementation is active but not a net win | Reject the strong claim that JEPA+Prophet late injection reliably improves control |
| 2026-03-17 | `E2-zero` and partial `E2-random` rollout ablations | `E2-zero = 74.0%`; partial `random ~= pred > zero` on early tasks | The branch is being used, but semantic usefulness of the predicted content is still unproven | Add stronger corruption controls: `random_scaled`, `signflip`, `shuffle` |
| 2026-03-17 | Research synthesis from DiT4DiT, DreamZero, Cosmos Policy, DreamDojo, VPP, mimic-video | Strong results consistently use predictive video features and earlier/action-coupled fusion | The thesis survives, but the current JEPA implementation is probably not the right final instantiation | Keep SmolVLA fixed and replace only the world branch with predictive video features |
| 2026-03-18 | P5 capacity-block data loss | NVMe-only outputs disappeared when the instance expired | Ephemeral storage is unacceptable for checkpoints and analysis artifacts | Write all outputs, logs, caches, and checkpoints to `/mnt/preserved` and sync backups regularly |
| 2026-03-18 | Current research decision | JEPA + late fusion is no longer the main line | The next high-value question is source quality, not one more JEPA tweak | Prioritize `Cosmos features vs JEPA features` with SmolVLA kept unchanged |

## Current Working Thesis

The thesis is now:

> World model can still be a first-class modality for control, but the modality must be strong enough, view-complete enough, and fused early enough to shape action selection.

What is no longer assumed:

> A small action-independent Prophet over JEPA latents, injected late into the action head, should reliably improve control.

