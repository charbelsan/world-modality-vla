# GPU Strategy (MI300X / big VRAM)

This repo’s models are small enough that you can easily underuse a 200GB GPU.

The goal is to turn extra headroom into **more scientific signal**, not “random bigger”.

## 1) First priority: finish the core ablation cleanly

Run:

- A, B, C, C_no_world_input

Same dataset, same steps, same action horizon, same future horizon, same seed(s).

## 2) Second priority: sweeps that strengthen the claim

Only change **one axis at a time** per run.

Recommended sweeps (in order):

1) **future horizon** (`--future_offset`): 4, 8, 16
2) **world vocab size** (`--world_vocab_size`): 256, 1024, 4096 (requires regenerating codebook for each)
3) **λ_world** (`--lambda_world_loss`): 0.05, 0.2, 0.5
4) **seeds**: at least 2 seeds for model C

## 3) Parallel training is often better than giant batch

If one training run uses ~10–20GB VRAM, it’s usually more stable to run 3–6 runs in parallel than to push batch size to the limit.

Watch out for CPU/DataLoader oversubscription:

- If you run N trainings in parallel, use `--num_workers 4` (not 16) for each.

## 4) Crash-proofing

If the VM is unstable:

- always sync `--log_dir` (see `ops/README.md`)
- log dirs should be separate per run (`logs_c_seed0`, `logs_c_seed1`, etc.)

