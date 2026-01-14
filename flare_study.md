FLARE vs Our Approach: Detailed Comparison

Note (Jan 2026): This file is a study note. Some statements below are high-level comparisons and may not match the
current `smolvla_world` implementation exactly. For current behavior, prefer:
- `docs/SMOLVLA_WORLD_FUSION_ABLATIONS.md`
- `lerobot_policy_world_modality/modeling_smolvla_world.py`

  FLARE Architecture (GR00T N1.5/N1.6)

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                         FLARE Sequence Layout                                │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                              │
  │   Input Sequence (all participate in SELF-ATTENTION):                        │
  │   ┌─────────────┬─────────────────┬──────────────────┐                       │
  │   │ State Token │ Action Tokens   │ Future Tokens    │                       │
  │   │ (q_t)       │ (A_t^τ noised)  │ (M learnable)    │                       │
  │   │ [1, D]      │ [H_act, D]      │ [M, D]           │                       │
  │   └─────────────┴─────────────────┴──────────────────┘                       │
  │         │               │                 │                                  │
  │         └───────────────┴─────────────────┘                                  │
  │                         │                                                    │
  │                         ▼                                                    │
  │              ┌─────────────────────┐                                         │
  │              │   DiT Transformer   │  ← All tokens attend to ALL tokens     │
  │              │   (Self-Attention)  │    via standard self-attention         │
  │              └─────────────────────┘                                         │
  │                    │          │                                              │
  │         ┌──────────┘          └──────────┐                                   │
  │         ▼                                ▼                                   │
  │   Action Tokens Out              Future Tokens Out (from Layer L=6)          │
  │   [H_act, D]                     [M, D]                                      │
  │         │                                │                                   │
  │         ▼                                ▼                                   │
  │   action_out_proj                  future_proj (MLP)                         │
  │         │                                │                                   │
  │         ▼                                ▼                                   │
  │   v_t (velocity)                   z_pred [M, D_embed]                       │
  │         │                                │                                   │
  │         ▼                                ▼                                   │
  │   L_fm = MSE(v_t, u_t)            L_align = -cos(z_pred, z_future)           │
  │                                          ↑                                   │
  │                                   Frozen VLM embedding                       │
  │                                   of future frame ϕ_{t+H}                    │
  │                                                                              │
  │   Total Loss: L = L_fm + λ·L_align  (λ = 0.2)                               │
  └─────────────────────────────────────────────────────────────────────────────┘

  Key FLARE Properties:

  1. Future tokens are IN the sequence - they participate in self-attention with action/state tokens
  2. Bidirectional information flow - action tokens can attend TO future tokens, and future tokens can attend TO action tokens
  3. Cosine similarity loss - not MSE, aligns direction not magnitude
  4. Internal layer extraction - future token outputs extracted at Layer 6 (not final layer)
  5. Frozen target - future embeddings come from a pretrained VLM on frame t+H

  ---
  Our Approach (SmolVLA + World Modality)

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                      Our Sequence Layout                                     │
  ├─────────────────────────────────────────────────────────────────────────────┤
  │                                                                              │
  │   Input Sequence:                                                            │
  │   ┌───────────────────────────────────┬─────────────────┐                    │
  │   │ Prefix (VLM Backbone - frozen)    │ Suffix (Action Expert - trainable)  │
  │   │ [Image patches, Lang, State]      │ [Noisy action tokens]               │
  │   │ [~600 tokens, D]                  │ [50 tokens, D]                      │
  │   └───────────────────────────────────┴─────────────────┘                    │
  │                                               │                              │
  │                                               ▼                              │
  │                                    ┌─────────────────────┐                   │
  │                                    │  Action Expert      │                   │
  │                                    │  (Self-Attention)   │                   │
  │                                    └─────────────────────┘                   │
  │                                               │                              │
  │                                               ▼                              │
  │                                         suffix_out                           │
  │                                         [50, D]                              │
  │                                               │                              │
  │                          ┌────────────────────┴────────────────┐             │
  │                          ▼                                     │             │
  │               ┌─────────────────────┐                          │             │
  │               │ GatedCrossAttention │ ← LATE, EXTERNAL         │             │
  │               │ Q=suffix_out        │   injection point        │             │
  │               │ K,V=world_memory    │                          │             │
  │               └─────────────────────┘                          │             │
  │                          │                                     │             │
  │                          ▼                                     │             │
  │               suffix_out + gate * ctx                          │             │
  │                          │                                     │             │
  │                          ▼                                     │             │
  │                   action_out_proj                              │             │
  │                          │                                     │             │
  │                          ▼                                     │             │
  │                   v_t (velocity)                               │             │
  │                          │                                     │             │
  │                          ▼                                     │             │
  │                   L_fm = MSE(v_t, u_t)                         │             │
  │                                                                │             │
  │   ════════════════════════════════════════════════════════════════════════  │
  │                                                                │             │
  │   SEPARATE WORLD PREDICTION BRANCH:                            │             │
  │                                                                │             │
  │   z_hist [B, 4, 1408] ──► Prophet ──► z_pred [B, K, 1408]     │             │
  │          (V-JEPA)          (2-layer)      │                    │             │
  │                                           ▼                    │             │
  │                              L_world = MSE(z_pred, z_future)   │             │
  │                                           │                    │             │
  │                                           └────────────────────┘             │
  │                                                                              │
  │   Total Loss: L = L_fm + λ·L_world  (λ = 0.2)                               │
  └─────────────────────────────────────────────────────────────────────────────┘

  ---
  Side-by-Side Comparison
  Aspect: Future token position
  FLARE (GR00T): IN sequence (suffix)
  Ours (SmolVLA World): EXTERNAL (separate branch)
  ────────────────────────────────────────
  Aspect: Attention type
  FLARE (GR00T): Self-attention (bidirectional)
  Ours (SmolVLA World): Cross-attention (unidirectional)
  ────────────────────────────────────────
  Aspect: Information flow
  FLARE (GR00T): Actions ↔ Future (mutual)
  Ours (SmolVLA World): Actions ← Future (one-way)
  ────────────────────────────────────────
  Aspect: Injection point
  FLARE (GR00T): Throughout all layers
  Ours (SmolVLA World): Late in the expert path (pre action_out_proj), applied per denoise step
  ────────────────────────────────────────
  Aspect: Future predictor
  FLARE (GR00T): None (tokens are learnable queries)
  Ours (SmolVLA World): Prophet network (predicts z_fut)
  ────────────────────────────────────────
  Aspect: Alignment loss
  FLARE (GR00T): Cosine similarity
  Ours (SmolVLA World): cosine-style (1 - cos), masked near episode end
  ────────────────────────────────────────
  Aspect: Target embedding
  FLARE (GR00T): Frozen VLM on future frame
  Ours (SmolVLA World): Frozen V-JEPA on future frame
  ────────────────────────────────────────
  Aspect: Gate mechanism
  FLARE (GR00T): No (full participation)
  Ours (SmolVLA World): Yes (learnable gate, init=0)
  ────────────────────────────────────────
  Aspect: Layer for future output
  FLARE (GR00T): Layer 6 (middle)
  Ours (SmolVLA World): Final layer
  ---
  Why FLARE May Be More Effective

  1. Bidirectional Information Flow

  FLARE:                              Ours:
  Action ←→ Future                    Action ← Future
  (can influence each other)          (one-way only)
  In FLARE, future tokens can influence how action tokens attend to each other. In our approach, future info is added AFTER the action tokens have already processed.

  2. Early Fusion vs Late Injection

  FLARE: Future info available at EVERY transformer layer
  Ours:  Future info is injected late in the expert path (unless you enable earlier-fusion ablations)
  This means in FLARE, the model can use future predictions to guide intermediate representations. In ours, it can only refine final outputs.

  3. No Bottleneck

  FLARE: Future tokens directly in transformer = full D dimensionality
  Ours:  Prophet → project → cross-attn = potential bottleneck

  4. Implicit vs Explicit Prediction

  FLARE: Model learns WHAT future info is useful implicitly
  Ours:  We force Prophet to predict V-JEPA features explicitly
  FLARE's future tokens learn to extract whatever future info helps action prediction. Our Prophet is constrained to predict V-JEPA embeddings, which may not be optimal.

  ---
  How to Implement FLARE-style in SmolVLA

  # Current (Late Cross-Attention):
  suffix_out = transformer(suffix_embs)           # [B, 50, D]
  suffix_out = cross_attn(suffix_out, world_mem)  # Inject AFTER
  v_t = action_out_proj(suffix_out)

  # FLARE-style (In-Sequence):
  future_tokens = self.future_queries.expand(B, M, -1)  # [B, M, D]
  combined = torch.cat([suffix_embs, future_tokens], dim=1)  # [B, 50+M, D]
  out = transformer(combined)  # Future tokens participate in self-attn!
  suffix_out = out[:, :50, :]  # Action tokens
  future_out = out[:, 50:, :]  # Future tokens from layer L
  v_t = action_out_proj(suffix_out)
  z_pred = future_proj(future_out)
  L_align = -cosine_similarity(z_pred, z_future).mean()

  ---
  Recommendation

  If E2 doesn't show improvement, implement FLARE-style prefix/suffix fusion:

  1. Add M learnable future query tokens to the action expert sequence
  2. Extract future token outputs from an intermediate layer (e.g., layer 6 of 16)
  3. Switch from MSE to cosine similarity loss
     (Note: current `smolvla_world` already uses cosine-style world loss.)
  4. Remove the separate Prophet network (future tokens replace it)
  5. Remove gated cross-attention (let self-attention handle it naturally)

  This would be a cleaner, more integrated design that matches the state-of-the-art.
  
  First Principles Analysis: World Modality for Action

  The Core Question

  Why should knowing the future help action prediction?

  Action = f(current_state, goal, dynamics)

  Where:
  - current_state: what is (observation)
  - goal: what should be (language instruction)
  - dynamics: what will be (world model)

  Actions are fundamentally about state transitions. A good action policy implicitly understands:
  1. Where am I now?
  2. Where do I want to be?
  3. How does the world respond to my actions?

  The third component is the world modality - knowledge about dynamics, physics, consequences.

  ---
  First Principles Extracted from FLARE

  Principle 1: Predict Representations, Not Reconstructions

  ❌ Predict pixels:     z → Decoder → 256×256×3 image
  ✅ Predict embeddings: z → Linear  → 1408-dim vector

  Why?
  - Pixel prediction wastes capacity on irrelevant details (textures, lighting)
  - Embeddings compress task-relevant information
  - Contrastive/alignment losses scale better than reconstruction

  Your current approach already does this (V-JEPA embeddings). This principle is validated.

  ---
  Principle 2: Let the Model Decide What Future Info is Useful

  FLARE: Learnable future tokens → Model learns to extract relevant future info
  Ours:  Prophet predicts V-JEPA → We dictate what future info looks like

  The insight: We're forcing the model to predict V-JEPA features, but V-JEPA was trained for video understanding, not robot control. The optimal future representation for action might be different.

  First principle: The future representation should be action-aware, not a generic video embedding.

  Implication for your research:
  # Instead of: predict frozen V-JEPA features
  L_world = MSE(z_pred, z_future_vjepa)  # V-JEPA decides what matters

  # Consider: predict action-conditioned future
  L_world = MSE(z_pred, f(z_future, actions))  # Actions inform what matters
  # Or: let alignment emerge
  L_world = -cos(z_pred, stop_grad(z_policy))  # Policy decides what matters

  ---
  Principle 3: Bidirectional > Unidirectional Information Flow

  Unidirectional (ours):
    Actions ← Future
    "Future informs actions, but actions don't shape future understanding"

  Bidirectional (FLARE):
    Actions ↔ Future
    "Future and actions mutually inform each other"

  Why does bidirectionality help?

  Consider: "Pick up the red cup"
  - Future tokens learn: "red cup will be in gripper"
  - Action tokens learn: "move toward red cup"
  - Mutual attention: Action tokens attending to future tokens learn "which actions lead to this future"
  - Reverse: Future tokens attending to action tokens learn "what future is achievable given these actions"

  First principle: World understanding and action planning should co-evolve, not be separate modules.

  Implication for your research:
  Current: Prophet(z_hist) → z_pred → inject into actions
           (Prophet never sees what actions are being planned)

  Better:  Joint reasoning over [state, future, actions]
           (Future representation adapts to the action being considered)

  ---
  Principle 4: Integration > Injection

  Injection (ours):
    ┌─────────────────┐
    │ Action pathway  │──────► output
    └────────┬────────┘
             │
       cross-attn ← world_memory (external)

  Integration (FLARE):
    ┌─────────────────────────────────┐
    │ [State, Actions, Future] joint │──► output
    └─────────────────────────────────┘

  Why integration > injection?

  Injection treats world info as "additional context" - like a hint passed at the end.
  Integration treats world info as "fundamental to reasoning" - like a core input modality.

  Analogy:
  - Injection = reading a book, then someone whispers a hint in your ear
  - Integration = the hint is written in the book from the start

  First principle: World modality should be a first-class citizen in the representation, not an afterthought.

  ---
  Principle 5: Alignment > Reconstruction

  Reconstruction loss: MSE(z_pred, z_target)
    - Penalizes magnitude differences
    - Sensitive to scale
    - Can be dominated by easy-to-predict dimensions

  Alignment loss: -cos(z_pred, z_target)
    - Only cares about direction
    - Scale-invariant
    - Focuses on semantic similarity

  Why alignment works better for world modeling?

  The exact magnitude of future embeddings doesn't matter for action. What matters is:
  - "Is the gripper approaching the object?" (directional)
  - "Is the object moving toward the goal?" (directional)

  First principle: For auxiliary objectives, semantic alignment often beats exact reconstruction.

  ---
  Principle 6: The Future Should Influence Planning, Not Just Execution

  Late injection (ours):
    Plan actions → Refine with future info → Execute
    (Future only helps "polish" the final output)

  Early integration (FLARE):
    Consider future → Plan actions accordingly → Execute
    (Future shapes the entire planning process)

  The insight: If world knowledge only affects the final layer, it can only make small adjustments. If it affects all layers, it can fundamentally change the action strategy.

  Example: "Stack blocks A, B, C"
  - Late injection: Model plans to grab A, future info says "A is unstable" → minor adjustment
  - Early integration: Model sees "A is unstable" throughout → plans to stabilize A first

  ---
  Synthesis: What Makes World Modality Work?

  Drawing from these principles, effective world modality requires:
  ┌───────────────┬────────────────────────────────┬─────────────────────────────┐
  │   Property    │          Description           │     Your Current Design     │
  ├───────────────┼────────────────────────────────┼─────────────────────────────┤
  │ Compact       │ Predict embeddings, not pixels │ ✅ V-JEPA embeddings        │
  ├───────────────┼────────────────────────────────┼─────────────────────────────┤
  │ Action-aware  │ Future rep informed by actions │ ❌ Prophet ignores actions  │
  ├───────────────┼────────────────────────────────┼─────────────────────────────┤
  │ Bidirectional │ Mutual information flow        │ ❌ Unidirectional injection │
  ├───────────────┼────────────────────────────────┼─────────────────────────────┤
  │ Integrated    │ First-class in attention       │ ❌ External cross-attention │
  ├───────────────┼────────────────────────────────┼─────────────────────────────┤
  │ Aligned       │ Direction > magnitude          │ ❌ Using MSE                │
  ├───────────────┼────────────────────────────────┼─────────────────────────────┤
  │ Early         │ Influences planning            │ ❌ Only affects final layer │
  └───────────────┴────────────────────────────────┴─────────────────────────────┘
  ---
  Research Directions Based on First Principles

  Direction A: Action-Conditioned World Representation

  Principle applied: Future representation should be action-aware.

  # Current: Prophet predicts from history only
  z_pred = Prophet(z_hist)  # [B, K, D]

  # New: Prophet sees what actions are being considered
  z_pred = Prophet(z_hist, noisy_actions)  # Action-conditioned future

  # The future you predict should depend on what you're about to do

  Hypothesis: An action-conditioned future is more useful because it predicts "what happens IF I do this action" rather than "what happens in general."

  ---
  Direction B: Hierarchical World Modality

  Principle applied: Different layers need different temporal horizons.

  Layer 1-4:   Short-term world (next 1-2 steps) → low-level motor adjustments
  Layer 5-8:   Medium-term world (next 5-10 steps) → trajectory shaping
  Layer 9-12:  Long-term world (next 20+ steps) → goal-directed planning

  # Inject different future horizons at different depths
  for i, layer in enumerate(transformer_layers):
      h = layer(h)
      if i in [4, 8, 12]:
          horizon = [2, 10, 30][i // 4]
          h = inject_world(h, world_memory[:, :horizon])

  Hypothesis: Early layers need immediate future (physics), late layers need distant future (goals).

  ---
  Direction C: World as Constraint, Not Input

  Principle applied: World modality as regularization, not conditioning.

  # Current: World info is INPUT to action prediction
  actions = Policy(obs, world_memory)

  # Alternative: World info CONSTRAINS action prediction
  actions = Policy(obs)
  L_constraint = consistency(actions, world_model_predictions)

  # "Your predicted actions should lead to predicted world states"

  Hypothesis: Instead of injecting world features, use world predictions to regularize the action manifold.

  ---
  Direction D: Contrastive World Alignment

  Principle applied: Alignment > Reconstruction.

  # Current: MSE between predicted and actual future
  L_world = MSE(z_pred, z_future)

  # Contrastive: Pull predicted future close to actual, push away from others
  L_world = -log(exp(cos(z_pred, z_future)) /
                 sum(exp(cos(z_pred, z_negative))))

  # Or simple cosine alignment (FLARE-style)
  L_world = -cos(z_pred, z_future).mean()

  Hypothesis: Contrastive/alignment losses are more robust to embedding scale issues and focus on semantic similarity.

  ---
  Direction E: World Modality as Memory

  Principle applied: Integration as first-class modality.

  Current modalities in SmolVLA:
    [Vision] [Language] [State] [Actions]

  Add world as explicit modality:
    [Vision] [Language] [State] [World] [Actions]
                                 ↑
                          Future predictions as tokens

  # Instead of cross-attention injection, add world tokens to sequence
  world_tokens = self.world_encoder(z_pred)  # [B, K, D]
  sequence = torch.cat([
      vision_tokens,   # What I see
      lang_tokens,     # What I should do
      state_tokens,    # Where I am
      world_tokens,    # What will happen  ← NEW
      action_tokens,   # What I'm planning
  ], dim=1)

  Hypothesis: Treating world as a modality (like vision/language) rather than an injection gives it equal standing in attention.

  ---
  Concrete Experiment Proposals

  Based on first principles, here are experiments ordered by expected impact:

  Experiment 1: Cosine Alignment (Low Effort, Tests Principle 5)

  # Change one line
  L_world = -F.cosine_similarity(z_pred, z_future, dim=-1).mean()
  # Instead of MSE

  Experiment 2: Multi-Layer Injection (Medium Effort, Tests Principle 6)

  # Inject at layers 4, 8, 12 instead of only final
  for i, layer in enumerate(action_expert):
      h = layer(h)
      if i % 4 == 0:
          h = cross_attn(h, world_memory)

  Experiment 3: World Tokens in Sequence (Higher Effort, Tests Principle 4)

  # Add K world tokens to action expert input
  world_query = self.world_queries.expand(B, K, -1)
  suffix = torch.cat([action_tokens, world_query], dim=1)
  # Extract world token outputs for alignment loss

  Experiment 4: Action-Conditioned Prophet (Higher Effort, Tests Principle 2)

  # Prophet sees actions being denoised
  z_pred = Prophet(z_hist, x_t)  # x_t = noisy actions
  # Now future prediction is "what happens if I do x_t"

  ---
  Summary: First Principles for World Modality Research
  ┌───────────────┬─────────────────────────────────────────┬────────────────────────────────┐
  │   Principle   │                 Insight                 │       Research Direction       │
  ├───────────────┼─────────────────────────────────────────┼────────────────────────────────┤
  │ Compact       │ Embeddings > pixels                     │ Already doing this ✅          │
  ├───────────────┼─────────────────────────────────────────┼────────────────────────────────┤
  │ Action-aware  │ Future should depend on actions         │ Action-conditioned Prophet     │
  ├───────────────┼─────────────────────────────────────────┼────────────────────────────────┤
  │ Bidirectional │ Mutual info flow                        │ World tokens in self-attention │
  ├───────────────┼─────────────────────────────────────────┼────────────────────────────────┤
  │ Integrated    │ First-class modality                    │ Add world to token sequence    │
  ├───────────────┼─────────────────────────────────────────┼────────────────────────────────┤
  │ Aligned       │ Semantic similarity > exact match       │ Cosine loss instead of MSE     │
  ├───────────────┼─────────────────────────────────────────┼────────────────────────────────┤
  │ Hierarchical  │ Different horizons for different depths │ Multi-layer injection          │
  └───────────────┴─────────────────────────────────────────┴────────────────────────────────┘
  The key insight: World modality should be a co-evolving representation with actions, not a separate module that injects late.

  Your research contribution could be showing which of these principles matter most for robot manipulation, independent of the specific architecture (FLARE vs your design).
