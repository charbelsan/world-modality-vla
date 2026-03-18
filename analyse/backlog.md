# Research Backlog

Purpose: track the live task list with explicit status so the next runs and implementation steps do not get lost across machines or capacity-block restarts.

Conventions:
- `Done`: completed and should not be reopened unless a regression appears
- `In Progress`: currently being implemented, tested, or actively run
- `Pending`: approved next work, not started yet
- `Blocked`: cannot proceed yet; blocker must be written explicitly
- `Deprioritized`: intentionally not the main line right now

## Done

- [x] Validate a real closed-loop baseline with SmolVLA (`E0` non-zero / high SR)
- [x] Validate do-no-harm control (`E1 ~= E0`) after fixing processor mismatch
- [x] Fix rollout temporal mismatch between cached `m=4` latents and single-frame rollout encoding
- [x] Add rollout corruption modes `random_scaled`, `signflip`, and `shuffle`
- [x] Add wrist-camera option for online world encoding
- [x] Define the Cosmos feature branch and the research pivot away from JEPA-first iteration
- [x] Record the operational lesson that P5 outputs must live on `/mnt/preserved`, not NVMe

## In Progress

- [ ] Finish end-to-end `world_latents_source=cosmos` integration for SmolVLA world-modality training and rollout eval
- [ ] Add durable P5 bootstrap scripts that recreate the env and stage caches/outputs under `/mnt/preserved`
- [ ] Add Cosmos smoke-test scripts for front and wrist views
- [ ] Add an 8-GPU Cosmos precompute + merge launcher
- [ ] Add an 8-GPU Cosmos screening launcher for the first training/eval matrix

## Pending

- [ ] Run `C0-front`, seed 0: Cosmos world features, current action-side interface
- [ ] Run `C0-front`, seed 1
- [ ] Run `C0-wrist`, seed 0
- [ ] Run `C0-wrist`, seed 1
- [ ] Run `C1-F2-front`, seed 0: Cosmos world features with earlier action-side fusion
- [ ] Run `C1-F2-front`, seed 1
- [ ] Run `C1-F2-wrist`, seed 0
- [ ] Run `C1-F2-wrist`, seed 1
- [ ] Promote the best Cosmos variant to `libero_object`
- [ ] Add one extra seed for the best Cosmos variant before making a strong claim
- [ ] Decide whether token-level Cosmos features are needed after pooled-feature parity results
- [ ] Decide whether an auxiliary next-feature prediction head from action hidden states is worth adding

## Blocked

- [ ] `C2-both` (front + wrist in one world branch) is blocked because the current world branch and rollout path still assume a single selected camera per run
- [ ] Token-level Cosmos feature injection is blocked until pooled Cosmos features (`C0`) prove that source quality matters at all
- [ ] DiT4DiT-style intermediate hidden-state extraction is blocked until the simpler tokenizer-latent branch is stable and benchmarked
- [ ] Full action-conditioned world modeling is blocked until we finish the clean “better source, same policy” comparison

## Deprioritized

- [ ] More JEPA-only late-fusion sweeps
- [ ] Replacing SmolVLA’s main vision encoder before the world-branch comparison is complete
- [ ] Full Cosmos Policy reproduction as the first next step
- [ ] Full video generation as the runtime world modality path

## Exit Criteria For The Current Phase

- [ ] A Cosmos-source run beats the JEPA reference on matched-budget `libero_spatial`
- [ ] Or, if it does not, the failure mode is explicit enough to justify the next pivot:
  action-conditioned futures, multi-hypothesis memory, or deeper fusion redesign
