# Hybrid RL + MCTS Draft Bot — Implementation TODO

## Goal
Transform the current pure-PPO approach into a hybrid system where RL learns a strong policy prior and value estimator, and MCTS handles online adversarial reasoning at test time.

---

## Phase 1: Reframe RL Training (Policy as Prior, Value as Evaluator)

### 1.1 Split Policy & Value Heads into Distinct Objectives
- [ ] Add separate optimizers or param groups for `policy_head` vs `value_head` + shared backbone.
- [ ] Use a 10:1 or 5:1 LR ratio: `lr_policy = 2e-4`, `lr_value = 1e-4` (value converges faster, should not dominate).
- [ ] Consider freezing the shared backbone for the first N steps of value warm-up.

### 1.2 Add Auxiliary Losses for Stronger Priors
- [ ] **Action prediction loss (supervised):** On completed rollouts from self-play / historical data, add a cross-entropy loss predicting the action taken by the final oracle-argmax policy.
  - This teaches the policy head what "good" moves look like before PPO ratio clipping kicks in.
- [ ] **Hero embedding contrastive loss:** Pull embeddings of heroes that co-occur in winning drafts; push embeddings of heroes that counter each other.
  - Improves generalization across the 123-hero action space.

### 1.3 Improve Value Target Quality
- [ ] During rollouts, store the **final oracle win probability** as the terminal reward.
- [ ] For intermediate value targets, experiment with **n-step returns** (n=5 or 10) instead of full GAE over 20 steps.
  - Reduces variance in value targets for early-pick states.
- [ ] Normalize returns to zero mean / unit variance *per batch* in `compute_gae`.

---

## Phase 2: Enable MCTS at Inference Time

### 2.1 Integrate MCTS into Evaluation Battles
- [ ] Modify `eval/trueskill_rating.py:BPBattleSimulator.run_bp_battle()` to use MCTS for the current agent instead of greedy argmax.
  - Keep opponent as greedy argmax or historical checkpoint policy.
- [ ] Run MCTS with `num_simulations=256` per move initially; tune up to `1024` if latency allows.
- [ ] Use the trained `agent` inside MCTS as:
  - **Policy prior:** `P(a|s) = softmax(agent.logits(s) / temp)` where `temp` narrows to top-k.
  - **Value estimator:** `V(s) = agent.value(s)` to truncate search depth instead of random rollouts.
  - **Oracle:** only called at terminal leaf nodes.

### 2.2 Prune MCTS Action Space with Policy Prior
- [ ] In `search/mcts_draft.py`, add `top_k_pruning=20`.
  - Only expand the top-20 actions by policy prior probability.
  - This reduces the effective branching factor from 123 → 20, making search tractable.
- [ ] Add `dirichlet_noise` to the root node prior (AlphaZero-style) to ensure some exploration.

### 2.3 Add MCTS Self-Play Data Generation (Optional but Powerful)
- [ ] Use MCTS (not the raw policy) to generate a fraction of training rollouts.
  - The actions become higher-quality supervision for the policy head.
  - This is the core of AlphaZero-style training.
- [ ] Trade-off: MCTS rollout collection is ~10-50× slower. Start with 10% MCTS rollouts, 90% policy rollouts.

---

## Phase 3: Fix the Data Bottleneck

### 3.1 Increase Effective Batch Size
- [ ] Increase `batch_size` from `64` to `128` or `256`.
- [ ] Keep `gradient_accumulation_steps = 8`, giving effective batch sizes of `1024-2048`.
- [ ] Decrease `ppo_epochs` to `2` to prevent over-optimizing on small data slices.

### 3.2 Increase Samples Per Epoch
- [ ] Raise `samples_per_epoch` from `4096` to `8192` or `16384`.
  - With 123 heroes, the policy needs to see far more diversity to escape the uniform-random basin.

### 3.3 Improve Historical Opponent Sampling
- [ ] Ensure `historical_opponent_prob` is actually using *all* saved checkpoints, not just the most recent.
- [ ] Add a "best opponent" buffer: always include the top-3 TrueSkill-rated historical checkpoints in the opponent pool.
  - Training against strong opponents provides stronger gradients than training against random past selves.

---

## Phase 4: Supervised Value Warm-Up & Stable Pre-Training

### 4.1 Extended Value Warm-Up
- [ ] Increase `value_warmup_epochs` from `1` to `3`.
- [ ] During warm-up, log value prediction MSE against terminal oracle rewards.
  - Target: MSE < 0.05 (i.e., average error < 0.22 in win probability) before unlocking policy training.

### 4.2 Behavioral Cloning (BC) Warm-Up
- [ ] Before PPO, run 1-2 epochs of Behavioral Cloning:
  - Generate 1000-2000 rollouts using a simple heuristic (e.g., greedy argmax on oracle win rate delta).
  - Train the policy head via cross-entropy to predict these heuristic actions.
  - This gives the policy a non-random initialization, making PPO ratio updates meaningful from the start.

---

## Phase 5: Evaluation & Iteration Loop

### 5.1 Add A/B Evaluation Metrics
- [ ] Track three evaluation variants in TensorBoard:
  1. `TrueSkill/greedy` — agent plays argmax.
  2. `TrueSkill/mcts_256` — agent plays with 256 MCTS simulations.
  3. `TrueSkill/mcts_1024` — agent plays with 1024 MCTS simulations.
- [ ] Expect `mcts_1024 > mcts_256 > greedy` even with the same RL checkpoint.

### 5.2 Stabilize TrueSkill Convergence
- [ ] Increase `num_opponents` from `8` to `16` during evaluation.
  - More opponents → lower variance in TrueSkill updates.
- [ ] Decrease `eval_interval` from `1` to `2` or `4` epochs.
  - Less frequent but more stable rating updates.

### 5.3 Debugging Checkpoints
- [ ] Save checkpoints every epoch regardless of eval interval.
- [ ] Add a script `eval/compare_checkpoints.py` that pits any two checkpoints against each other with MCTS to verify progress.

---

## Quick Wins (Do First)

1. **Integrate MCTS into eval battles** (Phase 2.1) — this is the highest-leverage change. Even a weak policy + strong search beats a weak policy playing greedily.
2. **Add BC warm-up** (Phase 4.2) — cheap to implement and will immediately fix the vanishing-policy-gradient problem.
3. **Split policy/value LR** (Phase 1.1) — one-line config change that prevents value gradients from suppressing policy learning.

---

## Success Criteria

- [ ] `Loss/entropy` drops from `-4.76` to below `-4.0` within 5 epochs.
- [ ] `Loss/value` stabilizes below `0.1` MSE after warm-up.
- [ ] `TrueSkill/mcts_256` is at least `5` rating points above `TrueSkill/greedy` for the same checkpoint.
- [ ] The top checkpoint achieves a `TrueSkill rating > 30` (i.e., `mu - 3*sigma > 30`).
