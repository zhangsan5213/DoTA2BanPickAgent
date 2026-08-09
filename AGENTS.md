# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## Behavioral Guidelines

1. **Think before coding.** State assumptions explicitly. If something is unclear, ask. If a
   simpler approach exists, say so.
2. **Simplicity first.** No features beyond what was asked, no abstractions for single-use code.
   If 200 lines could be 50, rewrite it.
3. **Surgical changes.** Touch only what you must. Match existing style. Remove orphans your
   change creates, but don't clean up pre-existing code unless asked.
4. **Goal-driven execution.** Define success criteria before starting. For multi-step tasks,
   state a brief plan with verification checkpoints.

## Project Overview

DoTA2BanPickAgent is an RL project for Dota 2 Captain Mode drafting (Ban/Pick): a transformer
policy agent (BPTransformerAgent) plays the full 20-step CM sequence against historical
checkpoints and itself, with a supervised WinRateOracle providing terminal rewards.

**Current main line (2026-07)**: MCTS + AlphaZero-style CE training. Rollouts are collected with
a batched persistent-tree MCTS (`search/mcts_batched.py`); the only per-action training signal is
cross-entropy against the MCTS visit distribution (`mcts.policy_loss_weight: 1.0`). Plain PPO
(the pre-July era) never produced policy learning — see `docs/experiments-2026-02-04-ppoa-era.md`.

**Language note**: Code comments are mixed Chinese/English (core training files primarily
Chinese); documentation is English. Write new comments in the language of the surrounding code.

## Docs Navigation (read first)

| Doc | Content |
|-----|---------|
| `docs/experiments-2026-07-mcts-ce-line.md` | **MCTS+CE 主线实验记录（2026-07）**：resume 链、5 个 run 数据表、entropy 信号（−4.47→−1.82）、结论与 verdict 规则 |
| `docs/experiments-2026-02-04-ppoa-era.md` | **旧 PPO 时代（2026-02~04）**：策略冻结证据（entropy −4.76 不动）、oracle 泄漏嫌疑、为何弃用 plain PPO |
| `CLAUDE.md` | **架构细节**：模型（BPTransformerAgent / WinRateOracle / MultiModalHeroEncoder）、BPState 环境、MCTS 树结构、PPO 训练管线、评分系统、恢复机制 |
| `TECH_ANALYSIS.md` | 2026-04 技术评审：6 个已修复 bug、7 个设计缺口（其中多数已由 MCTS 线解决） |

## Package Structure

```
├── configs/
│   ├── bp_agent_config.yaml         # Main training config (source of truth for hyperparams)
│   └── bp_agent_config_debug.yaml   # Debug config
├── data/                            # Hero features, semantic embeddings, winrates, match data
├── model/
│   ├── bp_agent.py                  # BPTransformerAgent (dual-backbone policy + value)
│   ├── win_rate_oracle.py           # WinRateOracle (terminal reward) + OracleTrainingDataset
│   └── hero_encoder.py              # MultiModalHeroEncoder (id + attributes + text, attention fusion)
├── utils/
│   ├── bp_env.py                    # BPState (20-step CM env), collect_rollout (legacy path), PPO loss helpers
│   ├── batched_rollout.py           # Fully batched rollout collection (ACTIVE training path)
│   ├── raw_data.py                  # Lazy hero feature loading, valid hero IDs, STATIC_HERO_MASK
│   ├── device.py                    # CUDA/CPU device singleton
│   └── player_preference_sampler_optimized.py / get_data_*.py
├── search/
│   ├── mcts_batched.py              # Batched persistent-tree MCTS (ACTIVE; GITCGRL-style BattleNode/ActionNode)
│   └── mcts_draft.py                # Legacy single-threaded MCTS (fallback only, keep in sync cautiously)
├── eval/
│   ├── smoke_test_persistent_mcts.py    # Pre-training gate: persistent-tree backprop (DummyAgent, CPU)
│   ├── smoke_comprehensive.py           # Pre-training gate: real ckpt + full pipeline
│   ├── smoke_rollout_persistent.py      # Pre-training gate: batched rollout collection
│   ├── profile_mcts.py                  # MCTS per-decision timing profile (--ckpt)
│   ├── elo_rating.py / trueskill_rating.py / rating_base.py
├── trainer/
│   ├── bp_agent_trainer.py          # Main trainer orchestrator
│   ├── rollout_collector.py         # Rollout collection dispatch (batched MCTS / threaded / sequential)
│   ├── loss_computer.py             # MCTS CE loss + value loss + entropy; MC returns
│   ├── epoch_runner.py              # PPO epoch loop with minibatch shuffle + KL early stop
│   ├── evaluator.py                 # EvaluatorManager, save_checkpoint
│   └── checkpoint_manager.py / model_initializer.py / config.py / data_generator.py / tensorboard_logger.py
├── train_bp_agent.py                # Main training entry point
├── train_winrate_oracle.py          # Oracle training script
├── eval_bp_agent.py                 # Tournament evaluation script
└── dash_app/                        # TODO: has known bugs, do not use
```

## Setup

```bash
conda run -n torch python ...   # ALL Python execution uses the `torch` conda env (torch 2.11.0+cu130)
```

Dependencies in `requirements.txt` (torch, numpy, scipy, trueskill, pyyaml, tensorboard, tqdm, pandas, openpyxl, dash…).

## Common Commands

### Pre-training gate — run BEFORE any MCTS training experiment

```bash
conda run -n torch python eval/smoke_test_persistent_mcts.py    # CPU, DummyAgent, ~seconds
conda run -n torch python eval/smoke_comprehensive.py           # real ckpt, full pipeline (uses e35 by default)
conda run -n torch python eval/smoke_rollout_persistent.py      # batched rollout collection w/ real ckpt
```

Status: `smoke_test_persistent_mcts.py` was repaired on 2026-08-02 (it crashed at step 1 —
see Load-Bearing Conclusions #11). Full re-verification of all three is pending; run them when
compute is free, before the next training run.

### Training

```bash
# Fresh start
conda run -n torch python train_bp_agent.py

# Resume (new ckpt/tb dirs always created; previous dir added to historical opponent pool)
conda run -n torch python train_bp_agent.py --resume ./ckpts/bp_agent-<ts>/bp_agent_epoch<N>.pth

# Debug config
conda run -n torch python train_bp_agent.py --config configs/bp_agent_config_debug.yaml
```

### Evaluation & benchmarking

```bash
# Tournament between trained models
conda run -n torch python eval_bp_agent.py --top_n 3 --matches 3

# MCTS search timing profile (A/B comparable — MUST fix PYTHONHASHSEED)
PYTHONHASHSEED=0 conda run -n torch python eval/profile_mcts.py --ckpt ./ckpts/bp_agent-20260725-085756/bp_agent_epoch35.pth

# TensorBoard
tensorboard --logdir runs --port 6006
```

### A/B methodology

- Fix `PYTHONHASHSEED=0` for any comparative bench.
- Judge training by: (1) batch entropy moving down (the only signal that has ever separated
  learning from frozen), (2) CE loss, (3) TrueSkill vs the historical pool — never by a single
  run's eval winrate.
- Record results in `docs/experiments-*.md` and update the Docs Navigation table.

## Code Conventions & Critical Patterns

- **Device**: always `utils.device.DEVICE` for model/data placement.
- **KMP**: every Python file sets `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` at the top.
- **Hero IDs**: data files are **1-based (1–160)**; model internals are **0-based (0–159)**.
  Conversion happens at the boundary (`BPState.step` stores `hero_id - 1`; masks index `h - 1`).
- **Valid heroes**: use `get_valid_hero_ids()` from `utils.raw_data`; `STATIC_HERO_MASK` is the
  precomputed static mask (invalid heroes = −1e9).
- **Naming**: PascalCase classes, snake_case functions, UPPER_SNAKE_CASE constants.
- **MCTS state reconstruction**: nodes store no state; `_reconstruct_state` replays actions from
  the search root's `_creation_state` or the current state, using cached parent states as a fast
  path. `BPState.copy()` is a deep copy of history/hero lists (player features are shared by
  reference but never mutated).

## Architecture (summary — full detail in `CLAUDE.md`)

- **Environment** (`utils/bp_env.py`): standard CM sequence, 20 steps (ban/pick alternating),
  action mask = static mask + used heroes. Terminal reward = oracle's Radiant win prob, mapped
  to [−1, 1]; Radiant steps +mapped, Dire steps −mapped.
- **MCTS** (`search/mcts_batched.py`): persistent tree per draft (`game_roots` anchor +
  `pivots` current node, GITCGRL style), alternating BattleNode/ActionNode levels, batched
  model evaluation grouped by history length, root Dirichlet noise (game root only),
  `max_search_depth` fast rollout (winrate-greedy heuristic → oracle terminal eval), top-k
  pruning by prior.
- **Training** (`trainer/loss_computer.py`): MC returns (γ=1, same terminal target for all steps
  of a team), MCTS CE loss `−(π · log p)` with `policy_loss_weight=1.0`, value loss coeff 2.0,
  entropy annealed 0.03→0.01, KL early stop, value-only warmup 2 epochs, grad accumulation 2.
- **Opponents**: 30% of rollouts vs historical checkpoints (TrueSkill-stratified sampling,
  `num_strata=3`); stale opponents within tolerance 2 also contribute training data.

## Important Configuration (`configs/bp_agent_config.yaml` — the source of truth)

| Param | Value | | Param | Value |
|-------|-------|-|-------|-------|
| `actor_lr` | 1e-4 | | `mcts.num_simulations` | 32 |
| `value_loss_coeff` | 2.0 | | `mcts.c_puct` | 2 |
| `entropy_loss_coeff` (init/final) | 0.03 / 0.01 | | `mcts.top_k` | 12 |
| `training.epochs` | 64 | | `mcts.max_search_depth` | 4 |
| `training.samples_per_epoch` | 128 | | `mcts.dirichlet_alpha/epsilon` | 0.3 / 0.25 |
| `training.batch_size` / `grad_accum` | 64 / 2 | | `mcts.policy_loss_weight` | 1.0 |
| `training.value_warmup_epochs` | 2 | | `mcts.use_mcts_policy_loss` | true |
| `training.historical_opponent_prob` | 0.3 | | `rating.trueskill.beta / tau` | 2.5 / 0.167 |
| `training.num_strata` | 3 | | `rating.num_player_sets` | 48 (→384 battles/eval) |
| `ppo.clip_ratio` / `ppo_epochs` / `minibatch_size` | 0.2 / 3 / 64 | | `ppo.kl_threshold` | 0.1 |

## Load-Bearing Conclusions (do not re-derive)

1. **Plain PPO never produced policy learning** (all Feb–Apr runs): entropy pinned at −4.76 nats
   (≈ uniform), actor loss ≈ 0, only value trained. Do not re-tune plain-PPO hyperparameters.
   See `docs/experiments-2026-02-04-ppoa-era.md`.
2. **MCTS CE is the only line that moved the policy**: entropy −4.47 → −1.82 in the first 5
   epochs of the July line. It is the main line; the CE gradient is real (verified by the entropy
   signal, not just code reading).
3. **July models (~6/64 epochs) do NOT yet beat April baselines**: final leaderboard top =
   151105/e11 (23.79) and e29 (23.76); July peak e33 = 23.52. This is not a verdict — verdict
   rules below.
4. **TrueSkill deltas < ~1σ (σ≈0.87) are ties**; 384-battle evals swing ±2.7 mu epoch-to-epoch.
   Eval winrate (~0.47–0.54) is NOT a strength metric (self-play + oracle reward).
5. **Interrupted runs are untrusted**: `bp_agent_final.pth` from a run killed mid-epoch has 2–8
   eval games and σ≈2–4; its rating is noise. (3 of 5 July resume runs died mid-batch.)
6. **Oracle**: deployed default is `win_rate_oracle-20260309033516-000-0.9042.pth` (90.42%).
   The 95.15% checkpoint is the data-leak suspect — never swap it in without re-verifying the
   leakage question (player `hero_history` may include the match itself).
7. **Do NOT regress the persistent-tree design**: cross-step backprop through `root_0` and
   playouts from `pivots` are load-bearing; the smoke test verifies visit accumulation.
8. **Checkpoint compatibility**: pre-hero_encoder checkpoints load with `strict=False` (random
   hero encoder) — fine within a game, degrades training quality. The old eval scripts' bugs
   (TS_ENV undefined, embed_dim mismatch) are FIXED in the working tree — do not reintroduce.
9. **Everything July 2026 is uncommitted** (last commit 2026-04-16). Results are not reproducible
   from HEAD — commit the working tree before starting the next experiment.
10. **MCTS root Dirichlet noise is applied only at the TRUE game root** (first step of a draft),
    matching GITCGRL — do not apply it per-step.
11. **An ActionNode must never be a search root** (found 2026-08-02 via the smoke test, which
    crashed passing an ActionNode as `prev_root`): an ActionNode's only child key is the string
    `"_battle"`, which would leak into finalized action selection. Contract: `prev_roots[i]` =
    top BattleNode (trace `next_root` to its parent chain end, as `collect_batched_rollouts`
    does), `pivots[i]` = the `_battle` BattleNode. `_finalize_searches` filters non-int keys as
    a guard — do not remove.
12. **Pivots must be reset before reuse — the July KL-spike root cause** (found 2026-08-03):
    the `_battle` pivot is created on-demand inside selection, so by the end of a step it is
    already EXPANDED. `collect_batched_rollouts` reuses it as the next step's pivot; re-expanding
    only adds children (never removes stale ones), so heroes that were legal last step but are
    already banned/picked now stay in the tree, get selected, and land in the visit policy π.
    The training CE mask (−1e9) then overlaps π → `-π·(-1e9)` explodes → `Loss/total` ~1e7–1e8
    and `KL = inf` on the first batch. This explains the July line's batch-KL 0.17–0.46 spikes,
    the run that died after 1 batch (KL 0.46), and 3/5 resume runs dying mid-epoch. FIXED:
    `search_batch` clears the pivot's children/stats before reuse. Do NOT remove — a regression
    here re-poisons every training run.

## Verdict Rules

- **No kill before ~50 epochs** of the MCTS-CE line unless a true collapse signature appears:
  entropy → 0, CE ≈ random on all decisions, wins concentrated on 1–2 player sets.
- A rating dip alone (even 3+ mu) is NOT a verdict.
- Before any new MCTS experiment: run the three smoke tests (see Common Commands).

## Known Issues

1. Old checkpoints (pre-hero_encoder, e.g. `bp_agent-20260418-*`) load with `strict=False`
   leaving a randomly-initialized hero encoder. Weights don't change between steps within a game
   (MCTS correctness unaffected) but model quality degrades in training.
2. Oracle data-leak suspicion: `hero_history` in training data may include the match itself
   (see `TECH_ANALYSIS.md`); the deployed 0.9042 model predates the suspicion.
3. `search/mcts_draft.py` is a legacy single-threaded implementation used only by the
   sequential/threaded fallback paths (`use_batched: false`); it is NOT used in the default
   config. Keep it working but prefer the batched path.
4. `dash_app/` has known bugs — do not use until fixed.
