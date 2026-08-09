# Project Notes

> Companion docs: **`AGENTS.md`** (behavioral guidelines, docs navigation, load-bearing
> conclusions, verdict rules, common commands) and **`docs/experiments-*.md`** (experiment
> records). This file covers architecture and training details.

## Environment

- Use the `torch` conda environment for all Python execution in this project.
  - Example: `conda run -n torch python train_bp_agent.py ...`
  - PyTorch version: 2.11.0+cu130

## Project Structure

```
├── configs/
│   ├── bp_agent_config.yaml         # Main training config
│   └── bp_agent_config_debug.yaml   # Debug config
├── data/
│   ├── hero_features.xlsx           # Hero attributes (21-dim per hero)
│   ├── hero_semantic_embeddings.pt  # 1024-dim text embeddings from ability descriptions
│   ├── hero_static_features.pt      # Precomputed static hero features
│   ├── hero_ability_descriptions.json
│   ├── hero_positions.json
│   ├── hero_winrates.json
│   └── high_mmr_with_stats*.json    # Training match data
├── model/
│   ├── bp_agent.py                  # BPTransformerAgent (policy + value network)
│   ├── win_rate_oracle.py           # WinRateOracle + OracleTrainingDataset
│   └── hero_encoder.py              # MultiModalHeroEncoder with attention fusion
├── utils/
│   ├── bp_env.py                    # BPState, collect_rollout, PPO loss helpers
│   ├── raw_data.py                  # Lazy hero feature loading, valid hero IDs
│   ├── device.py                    # CUDA/CPU device singleton
│   ├── player_preference_sampler_optimized.py  # Batch player preference generation
│   ├── batched_rollout.py           # Fully batched rollout collection
│   └── get_data_*.py                # Data fetching scripts
├── search/
│   ├── mcts_draft.py                # Lightweight single-threaded MCTS
│   └── mcts_batched.py              # Batched MCTS with shared model evaluations
├── eval/
│   ├── __init__.py                  # EvalMethod enum, evaluator factory
│   ├── rating_base.py               # Abstract base classes
│   ├── elo_rating.py                # ELO rating system
│   └── trueskill_rating.py          # TrueSkill rating system
├── trainer/
│   ├── config.py                    # TrainingConfig (YAML loader)
│   ├── bp_agent_trainer.py          # Main trainer orchestrator
│   ├── epoch_runner.py              # PPO epoch loop with minibatch shuffle
│   ├── rollout_collector.py         # Rollout collection (sequential/parallel/batched MCTS)
│   ├── loss_computer.py             # PPO loss + MCTS policy loss
│   ├── evaluator.py                 # EvaluatorManager, save_checkpoint
│   ├── data_generator.py            # Training sample generation
│   ├── checkpoint_manager.py        # Historical checkpoint discovery/caching
│   ├── model_initializer.py         # Oracle/agent/optimizer/scheduler init
│   └── tensorboard_logger.py        # SummaryWriter + TensorBoard server
├── train_bp_agent.py                # Main training entry point
├── train_winrate_oracle.py          # Oracle training script
├── eval_bp_agent.py                 # Tournament evaluation script
└── dash_app/                        # TODO: Dashboard has known bugs
    ├── app.py
    ├── layout.py
    ├── callbacks.py
    └── components/
```

## Model Architecture

### BPTransformerAgent (`model/bp_agent.py`)

Dual-backbone transformer with completely separate actor and value networks to prevent gradient interference:

**Input encoding:**
- `HeroEncoder`: `MultiModalHeroEncoder` (id embedding + attributes + semantic text) with modality dropout
- `ActionEncoder`: encodes `(actor_team, action_type, target_hero)` tuples
  - `team_embed`: `nn.Embedding(3, ACTOR_DIM=8)`
  - `action_embed`: `nn.Embedding(3, ACTION_DIM=8)`
  - fusion: `Linear(8+8+256, 256) -> LayerNorm -> SiLU`
- `PlayerEncoder`: `Linear(NUM_HEROES, 256) -> Linear(256, 256) -> LayerNorm -> SiLU`
- `cls_tokens`: learnable `[radiant, dire]` tokens, shape `[1, 2, EMBED_DIM]`

**Sequence construction:**
```
[CLS_r, CLS_d, player_r[0..4], player_d[0..4], action_history[0..T], current_query]
```
where `current_query` encodes `(current_actor, current_action_type, dummy_hero)`.

**Actor backbone** (policy):
- `nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True, dropout=0.1)`
- 4 layers
- Policy head: `Linear(256, NUM_HEROES=160)` with orthogonal init (gain=0.1)

**Value backbone** (value-first design, significantly larger):
- `nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=2048, batch_first=True, dropout=0.1)`
- 6 layers (actor_layers + 2)
- Value head MLP: `Linear(256, 1024) -> SiLU -> Dropout(0.1) -> Linear(1024, 512) -> SiLU -> Dropout(0.1) -> Linear(512, 256) -> SiLU -> Dropout(0.1) -> Linear(256, 1)`

**Value computation:**
Value transformer output yields two CLS features. The value is computed from the perspective of the `current_actor` (Radiant or Dire):
```python
radiant_cls_feat = value_out[:, 0, :]
dire_cls_feat = value_out[:, 1, :]
cls_feat = torch.where(current_actor == 0, radiant_cls_feat, dire_cls_feat)
value = value_head(cls_feat)
```

**Temperature:**
- Optional learnable temperature (`nn.Parameter(torch.ones(1))` if `learnable_temperature=True`)
- `get_temperature()` clamps to `min=0.1`
- Behavior policy samples with `softmax(logits / temp)`; target policy (for PPO ratio) always uses `temp=1.0`

**Checkpoint compatibility:**
`load_state_dict` handles old checkpoints:
- Missing `temperature` -> initialize to `ones_like`
- Old single `cls_token` -> duplicate to `cls_tokens`
- Missing `hero_encoder` -> load with `strict=False`, random init for hero encoder

### WinRateOracle (`model/win_rate_oracle.py`)

Transformer-based predictor for final team composition win probability:

**Architecture:**
- `embed_dim=128, nhead=8, num_layers=6`
- Hero encoder: same `MultiModalHeroEncoder` as agent
- Team indicator: `nn.Embedding(2, 16)` -> fusion `Linear(128+16, 128)`
- `PlayerHeroEncoder`: per-player win-rate vector encoder with transformer interaction (5 players -> team vector)
- Predict token: `nn.Parameter(torch.randn(1, 1, 128))`
- Transformer: `nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, batch_first=True, dropout=0.1)`, 6 layers
- Head: `Linear(head_input_dim, 64) -> LayerNorm -> SiLU -> Dropout(0.1) -> Linear(64, 1) -> Sigmoid`

**Current performance:** ~90.42% accuracy on held-out high-MMR matches.

**Default checkpoint path:**
```
./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth
```

### MultiModalHeroEncoder (`model/hero_encoder.py`)

Encodes each hero from three modalities:
1. **ID branch**: `nn.Embedding(NUM_HEROES, id_hidden_dim=128) -> LayerNorm`
2. **Attribute branch**: `Linear(NUM_HERO_FEATURES, 64) -> LayerNorm -> SiLU -> Linear(64, 64) -> LayerNorm`
3. **Semantic branch** (optional, `use_text=True`): `Linear(1024, 128) -> LayerNorm -> SiLU -> Linear(128, 128) -> LayerNorm`

**AttentionFusion:**
- Projects each modality to `embed_dim`
- Stacks as `[B, L*M, E]` with modality type embeddings
- Self-attention across modalities + FFN
- Gating: `Softmax` over concatenated raw features to weight modalities
- Output: weighted sum of attended modality features

**DeepResBlock:**
- `num_res_layers=3` residual sub-blocks
- Each: `LayerNorm -> Linear(E, 2E) -> SiLU -> Linear(2E, E)`

**Modality dropout:** During training, each modality has independent probability `modality_dropout=0.1` of being zeroed out.

## Environment (BPState)

Standard Dota 2 Captain Mode sequence, 20 steps total (`utils/bp_env.py`):

| Phase | Steps | Sequence |
|-------|-------|----------|
| Ban Phase 1 | 4 | R, D, R, D |
| Pick Phase 1 | 4 | R, D, D, R |
| Ban Phase 2 | 4 | D, R, D, R |
| Pick Phase 2 | 4 | D, R, R, D |
| Ban Phase 3 | 2 | R, D |
| Pick Phase 3 | 2 | R, D |

**State machine:**
- `step_idx` advances through `CM_SEQUENCE` list
- `get_current_action_type()` returns `"ban"` or `"pick"` based on `step_idx`
- Terminal when `pick_count["radiant"] + pick_count["dire"] >= 10` or `step_idx >= len(CM_SEQUENCE)`
- Action history stored as dict: `{"teams": [], "actions": [], "heroes": []}` where heroes are **0-based** indices

**State serialization:**
- `to_dict()`: single state -> batched dict with `unsqueeze(0)`
- `batch_to_dict()`: requires **all states have the same action history length**. Groups by history length internally.

**Reward:**
- Oracle predicts Radiant win probability `p`
- Mapped to `[-1, 1]`: `mapped = 2p - 1`
- Radiant steps receive `+mapped`, Dire steps receive `-mapped` (zero-sum)

**Action masking:**
- `STATIC_HERO_MASK` precomputed in `utils/raw_data.py` (invalid hero IDs = `-1e9`)
- Used heroes additionally masked at runtime

## Training & Resume

### Checkpoint format

Full checkpoints (`bp_agent_epoch{N}.pth`, `bp_agent_final.pth`) contain:
- `agent_state`: model weights
- `optimizer_state`: AdamW optimizer state
- `scheduler_state`: LR scheduler state
- `entropy_step`: entropy annealer step count
- `epoch`, `global_step`, `grad_accum_step`

### Resume behavior

```bash
conda run -n torch python train_bp_agent.py --resume ./ckpts/bp_agent-<timestamp>/bp_agent_epoch<N>.pth
```

- **New directories always created** for both checkpoints and TensorBoard logs when resuming. Old runs are never modified.
- Previous checkpoint directory is automatically added to `checkpoint_dirs` scanned for historical opponents.
- `EvaluatorManager` receives `additional_dirs` so the combined leaderboard shows ratings from both previous and current runs.
- Optimizer, scheduler, and entropy annealer states are restored from the checkpoint.
- Epoch parsing: if filename matches `bp_agent_epoch{N}.pth`, extracts `N` as start epoch.

## MCTS

GITCGRL-compatible persistent tree with cross-step backprop and fast rollout.

### Tree Structure (`search/mcts_batched.py`)

Alternating node types (GITCGRL: BattleNode → ActionNode → BattleNode → ...):

| Level | Type | Expanded? | Role |
|-------|------|-----------|------|
| 0, 2, 4, ... | BattleNode | Yes | Decision point: evaluate → expand → ActionNode children |
| 1, 3, 5, ... | ActionNode | **No** | Action candidate: evaluate for value only, no children |

- `BatchedMCTSNode`: Uses `__slots__`, tracks `_is_action_node`, `_depth_from_root`, `_eval_queued`.
- ActionNodes get one BattleNode child (keyed `"_battle"`) created on-demand during playout traversal.
- `_creation_state` stored on game root for correct state reconstruction across steps.

### Persistent Tree (GITCGRL `GameRoots` + `CurrentNodes`)

- `game_roots[i]` = GameRoot (`root_0`), created at step 0, persists entire draft.
- `pivots[i]` = CurrentNodes, the BattleNode at the current game state, fresh each step.
- Playouts start from `pivot` (not root), backprop through parent chain to `root_0`.
- Cross-step backprop: early decisions' ActionNodes accumulate visit statistics from ALL subsequent steps' playouts.

### Fast Rollout

- `max_search_depth`: cap on playout depth from search root (default 4).
- When a playout hits this depth: winrate-greedy heuristic simulates remaining picks, oracle evaluates final lineup, value backpropped as terminal.
- Eliminates model forwards for deep leaves, caps per-step cost regardless of game progress.

### Batched Evaluation

- `BatchMCTSEngine.search_batch(root_states, prev_roots, pivots)` runs MCTS for multiple rollouts.
- States grouped by action history length for efficient `batch_to_dict()` collation.
- All leaf evaluations in one round are batched into a single model forward.
- Dirichlet root noise (`alpha=0.3`, `epsilon=0.25`) applied only at game root (first step).

### MCTS Config (`configs/bp_agent_config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mcts.num_simulations` | 32 | Leaves per round |
| `mcts.c_puct` | 2 | UCB exploration |
| `mcts.top_k` | 12 | Action pruning |
| `mcts.max_search_depth` | 4 | Fast rollout depth cap |
| `mcts.dirichlet_alpha` | 0.3 | Root noise concentration |
| `mcts.dirichlet_epsilon` | 0.25 | Root noise mixing ratio |
| `mcts.use_mcts_policy_loss` | true | Use MCTS visit distribution as CE target |
| `mcts.policy_loss_weight` | 1.0 | Weight for MCTS policy loss (PPO weight = 0) |

## PPO Training

### Rollout collection (`utils/batched_rollout.py::collect_batched_rollouts` — ACTIVE path)

- Behavior policy samples with temperature (exploration)
- `old_log_prob` always computed from target policy (`temperature=1.0`) for PPO ratio consistency
- MCTS actions: when `use_mcts=True` and `is_current`, runs `BatchMCTSEngine.search_batch`
  (persistent tree, batched evaluation); log_prob still from target policy
- Legacy fallback (`use_batched=false`): `utils/bp_env.py::collect_rollout` with
  `search/mcts_draft.py::DraftMCTS` — kept for compatibility, not the default
- Final bootstrap value: uses value network's prediction on terminal state, not raw oracle reward
- `step_teams` recorded for per-team reward assignment

### Return computation (`trainer/loss_computer.py::prepare_rollout`)

**Uses Monte Carlo returns, not GAE with discounting.** For fixed-horizon deterministic draft (20 steps), all steps for a team share the same terminal reward target with `gamma=1.0`:
```python
mapped_reward = 2.0 * final_reward - 1.0  # [-1, 1]
returns = torch.full_like(team_step_values, team_reward)  # Same for all team steps
advantages = returns - team_step_values  # MC advantage
```

Radiant steps get `+mapped_reward`, Dire steps get `-mapped_reward`.

### PPO epoch loop (`trainer/epoch_runner.py`)

Inspired by standard PPO / GITCGRL:
1. Flatten all valid steps across rollouts
2. Shuffle into minibatches (`minibatch_size=64`)
3. Multiple PPO epochs (default `ppo_epochs=3`) over the same data
4. KL early stopping: if average KL > `kl_threshold=0.1`, break epoch loop
5. Advantage normalization per minibatch (`normalize_advantages`)

### Gradient handling

- Gradient accumulation: `gradient_accumulation_steps=2` (effective batch size = 128)
- **Separate gradient clipping:**
  - Non-value params: `max_grad_norm=0.5`
  - Value head params: `max_grad_norm * 4 = 2.0`
- Value-only warmup: first `value_warmup_epochs=1` epochs train only value head (`loss = value_loss_coeff * value_loss`)

### Loss (`trainer/loss_computer.py::compute_minibatch`)

Primary training signal: **MCTS policy loss** (AlphaZero-style cross-entropy against visit distribution):
```python
mcts_policy_loss = -(mcts_policies * log_probs_all).sum(dim=-1).mean()
total_loss = policy_loss_weight * mcts_policy_loss + value_loss_coeff * value_loss + entropy_coeff * entropy_loss
```

PPO clip loss is computed but weighted to zero when `use_mcts_policy_loss=true` and `policy_loss_weight=1.0`.

## Training Configuration (`configs/bp_agent_config.yaml`)

| Parameter | Value | Location |
|-----------|-------|----------|
| `actor_lr` | 1e-4 | config top-level |
| `agent_temperature` | 1.0 | behavior policy temp |
| `agent_learnable_temperature` | false | |
| `value_loss_coeff` | **2.0** | loss weights |
| `entropy_loss_coeff` | 0.03 | |
| `training.epochs` | **64** | |
| `training.batch_size` | **64** | |
| `training.gradient_accumulation_steps` | 2 | effective batch = 128 |
| `training.samples_per_epoch` | 128 | |
| `training.value_warmup_epochs` | 2 | |
| `training.historical_opponent_prob` | **0.3** | 30% vs historical |
| `training.policy_staleness_tolerance` | 2 | stale opponent steps also train |
| `training.num_strata` | 3 | TrueSkill stratified sampling |
| `ppo.clip_ratio` | 0.2 | PPO clip epsilon |
| `ppo.value_clip_ratio` | **0.5** | value clip (effectively disabled for [-1,1]) |
| `ppo.ppo_epochs` | **3** | PPO updates per batch |
| `ppo.minibatch_size` | 64 | |
| `ppo.kl_threshold` | 0.1 | KL early stop |
| `ppo.kl_early_stop` | true | |
| `ppo.max_grad_norm` | 0.5 | gradient clipping |
| `mcts.enabled` | true | |
| `mcts.use_batched` | true | batched vs threaded |
| `mcts.num_simulations` | **32** | leaves per round |
| `mcts.c_puct` | 2 | UCB exploration |
| `mcts.top_k` | **12** | action pruning |
| `mcts.max_search_depth` | **4** | fast rollout depth cap |
| `mcts.dirichlet_alpha` | 0.3 | root exploration noise |
| `mcts.dirichlet_epsilon` | 0.25 | noise mixing ratio |
| `mcts.use_mcts_policy_loss` | **true** | MCTS visits as CE target |
| `mcts.policy_loss_weight` | **1.0** | weight for MCTS CE loss |

### Entropy annealing

```yaml
entropy_annealing:
  enabled: true
  initial_coeff: 0.03
  final_coeff: 0.01
  type: "linear"          # linear, exponential, cosine
  total_epochs: 16
  warmup_steps: 500
  annealing_steps: 8000
```

## Rating Systems

### ELO (`eval/elo_rating.py`)

- Initial rating: 1500
- K-factor: 32
- Scale: 400
- Opponent sampling: Gaussian weighting around current ELO with `opponent_sample_std=200`

### TrueSkill (`eval/trueskill_rating.py`)

Uses official `trueskill` library:

| Parameter | Default | Config override |
|-----------|---------|-----------------|
| `initial_mu` | 25.0 | 25.0 |
| `initial_sigma` | 25.0/3 ≈ 8.333 | 8.33 |
| `beta` | 25.0/6 ≈ 4.167 | **2.5** |
| `tau` | 25.0/300 ≈ 0.083 | **0.167** |
| `draw_probability` | 0.0 | 0.0 |
| `opponent_sample_std` | 2.0 | **5.0** |
| `staleness_threshold` | 5 | 4 |
| `num_active_models` | 5 | 16 |

**Staleness mechanism:**
- Models not evaluated for `staleness_threshold` epochs are "stale"
- Before each evaluation, stale models refresh by battling active models
- Staleness resets after evaluation

**Rating display:** `mu - 3 * sigma` (conservative estimate, 99.7% confidence bound)

## Evaluation (`eval_bp_agent.py`)

Tournament mode for watching trained models battle:

```bash
# Top 3 models by TrueSkill, 3 matches per pair
python eval_bp_agent.py --top_n 3 --matches 3

# Specific models
python eval_bp_agent.py --models ./ckpts/model1.pth ./ckpts/model2.pth --matches 5

# ELO-based selection
python eval_bp_agent.py --top_n 3 --rating elo
```

- Models play round-robin with random side assignment
- Winner determined by oracle win probability threshold (>0.5)
- Verbose output shows full BP sequence with hero names

## Code Conventions

- **Device**: Always use `utils.device.DEVICE` for model/data placement
- **Environment variable**: All Python files set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the top
- **Hero IDs**: Data files use **1-based** (1-160); model internals use **0-based** (0-159)
- **Valid heroes**: Use `get_valid_hero_ids()` from `utils.raw_data` for the actual existing hero subset
- **Naming**: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants

## Data Files

### `hero_features.xlsx`
- Columns: `id`, `name`, `attr_str`, `attr_agi`, `attr_int`, `attr_all`, plus role tags (`role_Carry`, `role_Support`, etc.)
- 21-dimensional feature vector per hero

### `hero_semantic_embeddings.pt`
- 1024-dimensional vectors from processing hero ability description text
- Loaded lazily via `_LazySemanticMap`

### `high_mmr_with_stats.json`
- Training data for Oracle and agent evaluation
- Fields: `match_id`, `picks_bans`, `players` (with `hero_history`), `radiant_win`
- `hero_history` format: `{hero_id_str: {games: int, wins: int}}`

## Dashboard (TODO)

The Dash-based visualization dashboard (`dash_app/`, `dash_server.py`) currently has known bugs and should not be used until fixed.

## Known Code Issues

1. **Old checkpoints lack `hero_encoder`**: Checkpoints from before the hero_encoder was added
   load with `strict=False`, leaving the hero encoder randomly initialized. This doesn't affect
   MCTS search correctness within a single game (the weights don't change between steps), but
   does affect model quality in training.
2. **Oracle data-leak suspicion**: `hero_history` in the training data may include the match
   itself (see `TECH_ANALYSIS.md`) — the deployed 90.42% oracle predates the suspicion.
3. **`search/mcts_draft.py` is legacy**: only used when `use_batched=false`; keep it working
   but prefer the batched path.

*(The former `TS_ENV` undefined bug and the `embed_dim=128` eval mismatch are fixed in the
working tree — do not reintroduce.)*
