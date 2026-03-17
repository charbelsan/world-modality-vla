# H100 24H Plan: World Modality Decision Runs

Goal: use **2x H100 for 24 hours** to answer the highest-value remaining research questions, not to maximize raw run count.

See also:

- [NEXT_EXPERIMENT_MATRIX.md](/home/charbel/charbel/world-modality-vla/docs/NEXT_EXPERIMENT_MATRIX.md)
- [launch_h100_followup_queue.sh](/home/charbel/charbel/world-modality-vla/scripts/launch_h100_followup_queue.sh)

Primary questions:

1. Does **semantic** future content matter?
2. Is the main bottleneck **front-only observability**?
3. Is the main bottleneck **late fusion**?

The current reference is:

- baseline `E0` works
- `E1 ~= E0` (do-no-harm validated)
- `E2` is active but mixed / slightly below baseline overall

---

## Summary

Use the two GPUs for two different purposes:

- **GPU 0**: resolve the remaining causal ambiguity in the current `E2` checkpoint
- **GPU 1**: test the most likely architectural improvement

This is the fastest path to a real decision.

---

## GPU 0: Causal ablations + wrist memory

### Stage A: content-sensitivity ablations

Run these on the **same `E2` checkpoint** with the **same eval budget**:

1. `world_memory_mode_rollout=signflip`
2. `world_memory_mode_rollout=random_scaled`

Recommended budget:

- `50 episodes/task` on `libero_spatial`

Interpretation:

- `pred > signflip` and `pred > random_scaled`
  - semantic future content matters
- `pred ~= signflip/random_scaled`
  - current branch is mostly using non-zero conditioning, not reliable predictive semantics

### Stage B: wrist-camera world memory

Precompute wrist latents:

```bash
python -m world_modality.precompute_world_latents \
  --dataset_name HuggingFaceVLA/libero \
  --image_key observation.images.image2 \
  --cache_dir cache \
  --world_latents_source vjepa \
  --temporal_window 4 \
  --latent_suffix m4_wrist \
  --device cuda
```

Then train and evaluate:

- `E2-wrist`
- same init checkpoint as `E2-front`
- same steps / batch / eval budget

Use:

- `--policy.latent_suffix=m4_wrist`
- `--policy.world_camera=wrist`

Interpretation:

- `E2-wrist > E2-front`
  - front-only observability is a real bottleneck

---

## GPU 1: Earlier fusion

### Stage A: F2

Train and eval:

- `world_inject_suffix_in=true`

This tests whether world memory needs to shape the expert earlier rather than only at the final action-expert readout.

### Stage B: F3b (only if time remains or F2 looks good)

Train and eval:

- `world_prefix_cross_attn=true`

This tests whether world memory should affect the prefix/representation stream without adding extra tokens.

Interpretation:

- `F2 > F1`
  - late-only injection is too weak
- `F3b > F2`
  - world should influence the representation path, not just the action expert

Reference details:

- [SMOLVLA_WORLD_FUSION_ABLATIONS.md](/home/charbel/charbel/world-modality-vla/docs/SMOLVLA_WORLD_FUSION_ABLATIONS.md)

---

## Recommended order

### First 2-4 hours

- GPU 0: `signflip`, then `random_scaled`
- GPU 1: start `F2`

### Next 4-12 hours

- GPU 0: precompute wrist latents, then start `E2-wrist`
- GPU 1: eval `F2`; start `F3b` only if F2 is not clearly bad

### Final 12 hours

- evaluate `E2-wrist`
- if one branch clearly wins, use remaining time for:
  - one extra seed
  - or one more suite (`libero_object` or `libero_goal`)

---

## Stop conditions

Stop the current V-JEPA2 + action-free Prophet line if:

- `pred <= signflip`
- `pred <= random_scaled`
- `E2-wrist` does not beat `E2-front`
- `F2/F3b` do not beat `F1`

If that happens, do **not** spend more H100 time on small variants of the same design.
Move to the next branch:

- keep SmolVLA as baseline
- replace the current world branch with **predictive video-model hidden features**
- prefer a **DiT4DiT / mimic-video / VPP-style feature path**

---

## What not to spend the 24h on

- more E0/E1 reruns unless needed for protocol sanity
- full video generation / diffusion sampling as an online planner
- switching both backbone and world branch at the same time

Those are lower signal than the runs above.
