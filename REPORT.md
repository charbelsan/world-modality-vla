# World Tokens as a New Modality on SR100 – Hackathon Report

## 1. Setup

- Robot: SR100
- Task:
- Dataset:
- Vision encoder:
- World token codebook size (N):
- Context length (T_ctx):
- Action horizon (H):
- Future offset (K):

## 2. Models

Describe the three models:

- Baseline A (standard BC):
- Baseline B (auxiliary future token loss):
- Model C (world token modality):

Include any implementation details that differ from the spec.

## 3. Training Curves

Paste or link plots:

- Training / validation action MSE
- World token accuracy (top-1 / top-5) for B/C
- Combined loss

## 4. Offline Evaluation

### 4.1 Metrics

- Action MSE on held-out set:
  - A:
  - B:
  - C:
- World token accuracy (top-1 / top-5):
  - B:
  - C:

### 4.2 Open-loop Rollouts

Describe open-loop rollout evaluation:

- How error grows with horizon for A vs C
- Any qualitative stability differences

## 5. Real Robot Evaluation

### 5.1 Protocol

Document the protocol exactly:

- Task definition:
- Number of trials per model:
- Initial condition control:
- Success criterion:

### 5.2 Results

Fill the table with actual numbers.

| Model   | Success / Trials | Success Rate |
|---------|------------------|--------------|
| A (BC)  |                  |              |
| B (Aux) |                  |              |
| C (World modality) |      |              |

Comment on:

- Absolute improvement of C vs A
- Any observed differences in stability, drift, or long-horizon behavior

## 6. World Tokens & Imagination

### 6.1 Codebook & Tokens

- Encoder backbone:
- Number of clusters (N):
- Number of frames used for k-means:

Include qualitative examples of:

- Cluster centers / nearest-neighbor frames
- How tokens correlate with distinct world states

### 6.2 Future Token Visualization

Describe how you visualized predicted future tokens, e.g.:

- Nearest-neighbor frame from dataset for predicted token id
- Any short videos or GIFs showing:
  - Current camera view
  - Predicted future frame
  - Robot behavior

## 7. Discussion

Summarize:

- Did world tokens as a modality help?
- How much did auxiliary loss alone help (B vs A)?
- How much did adding world tokens as inputs help (C vs B)?

Include any failure modes, caveats, or interesting side observations.

## 8. Future Work

Brief bullets:

- Longer world token sequences
- Better vision encoder or codebook
- Multi-task / multi-scene training
- Language-conditioned policies using world tokens

