# VideoWorld (ByteDance-Seed) — Notes for World-Modality-VLA

Repo: `third_party/VideoWorld` (cloned from `ByteDance-Seed/VideoWorld`).

This note extracts the parts that are directly relevant to our hypothesis:
**“World modality as external memory improves VLA control.”**

## What VideoWorld Does (Robotics Path)

VideoWorld’s CALVIN pipeline is explicitly **two-stage**:

1) **Train an LDM (Latent Dynamics Model)** to compress a short video clip into compact “dynamics/action-like” codes.
- Entry point config: `third_party/VideoWorld/LDM/configs/calvin_ldm.py:30`
- Key knob: `act_embedding_num=9` tokens summarizing a clip (`frame_num=10`) via a QFormer.
- Encoder class: `third_party/VideoWorld/LDM/ldm/models/autoencoders/magvit_v2_ldm.py:85`
  - The encoder runs a spatiotemporal conv stack, then uses a **QFormer with learned query slots**:
    - query slots: `self.act = nn.Embedding(act_embedding_num, 512)` (`third_party/VideoWorld/LDM/ldm/models/autoencoders/magvit_v2_ldm.py:193`)
    - QFormer call: `video, _ = self.qformer(video, dense_pe, query)` (`third_party/VideoWorld/LDM/ldm/models/autoencoders/magvit_v2_ldm.py:236`)

2) **Train an autoregressive “next-token” transformer** on VQ visual tokens, augmented with the LDM codes.
- Robotics config: `third_party/VideoWorld/VideoWorld/configs/calvin_train.py:18`
- Uses `use_la_action=True` (`third_party/VideoWorld/VideoWorld/configs/calvin_train.py:23`)
- The dataset loads per-step `la_action` from `calvin_ldm_results.pth`:
  - `item['la_action'] = np.stack(la_actions)` (`third_party/VideoWorld/VideoWorld/falcon/datasets/calvin_dataset.py:639`)
- The algorithm appends `la_action` tokens into the token stream (they become “extra visual ids”):
  - `visual_ids = torch.cat([visual_ids, la_actions], dim=2)` (`third_party/VideoWorld/VideoWorld/falcon/models/algorithms/calvin_GR1_wostate_vq_idm.py:410`)

## First Principles We Should Borrow

1) **Represent changes, not states**
- Our earlier failure mode (Prophet “copies” `z_t`) happens when latents are too invariant.
- VideoWorld forces the representation to encode *what changed* by training a dedicated LDM and extracting compact change codes.

2) **Use multiple compact tokens (query slots), not a single pooled vector**
- Their `act_embedding_num` tokens are a learned bottleneck that can focus on task-relevant dynamics.
- This is a direct antidote to “global pooled embeddings barely change”.

3) **Discrete / quantized codes create a strong learning signal**
- Instead of regressing a nearly-constant continuous vector with MSE, predicting discrete indices (or codebook logits) can prevent “mean/copy collapse”.

4) **Keep “world” separate from the main language stream**
- VideoWorld concatenates tokens into the transformer’s modality stream.
- In our design, we keep the Model-F safety principle: world memory should affect **only `<ACT>`** (via gated cross-attn), to avoid interference with generalist reasoning.

## Mapping VideoWorld → Our “World Memory (Model-F)” Design

VideoWorld “`la_action` tokens” are conceptually very close to what we want `z_future` to be:
**a compact, dynamics-bearing latent** that is informative for control.

We can adopt the principle without adopting their full generative policy:

- Replace our current world source (`vjepa` pooled latents) with **dynamics tokens**:
  - simplest: **spatial tokens / patch tokens** from a strong vision encoder (instead of global pooling)
  - closer to VideoWorld: train an LDM-style encoder that outputs `K` discrete dynamics tokens

- Keep our integration mechanism:
  - `h_act = h_act + tanh(gate) * CrossAttn(h_act, kv(world_tokens))`
  - gate init = 0 (safe-by-default)

## Minimal Experiment Plan (High Signal, Low Waste)

1) **Get a non-zero SR baseline first**
- Use SmolVLA/LeRobot recipe and confirm replay + closed-loop works on the target split.

2) **Oracle dynamics tokens injection (upper bound)**
- Inject “ground-truth” future dynamics tokens to check whether perfect world information improves SR.

3) **Predicted dynamics tokens**
- Train Prophet to predict those tokens (classification / logits), then inject predicted tokens.

4) **Corruption tests**
- shuffle/zero/random tokens; SR should drop once the gate opens (proof of reliance).

## What to Be Careful About (for Our Repo)

- Don’t copy VideoWorld’s token-stream concatenation as the default; it recreates the “Model-C” interference risk.
- If we do discrete world tokens, prefer:
  - CE over code indices (or vector-quantized reconstruction loss), not pure MSE on near-invariant vectors.
  - multiple tokens (`K>1`) rather than a single global vector.

