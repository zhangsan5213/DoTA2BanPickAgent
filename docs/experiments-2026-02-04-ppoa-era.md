# Experiments: Plain PPO Era (2026-02 → 04) — the Policy Freeze

All training before the July MCTS-CE line. Historical record — the conclusions here are
**load-bearing**: they explain why the current line exists and why old TrueSkill ratings
must not be read as policy-strength evidence.

## The core finding: the policy never learned

Across every readable run (Mar 16 – Apr 19), the same signature appears:

- `Loss/entropy` pinned at **−4.76 nats** (ln 160 = 5.075 → ≈ uniform over all heroes); never
  moved by more than 0.05 nats over an entire run.
- `Loss/actor` / `Epoch/policy_loss` ≈ **0** — the policy head received no effective gradient.
- Value loss did improve (e.g. 2.21 → 0.008 in 20260330), and value-only warmup epochs behaved
  normally.

TrueSkill display ratings of 20–24 during this era therefore measured a **near-random policy
with a learned value head**, not draft strength. The leaderboard spread (mu 23.2–26.3) was
noise-level.

## Run table (selected)

| Run | Epochs | Rollouts/ep | lr | Final eval WR | TR final display | Notes |
|-----|--------|-------------|-----|---------------|------------------|-------|
| 20260316-021828 | 386 batches | — | — | — | 15.1 | entropy frozen |
| 20260321-115441 | 348 batches | — | — | — | 13.4 | entropy frozen |
| 20260321-145332 | 828 batches | — | — | — | 17.0 | entropy frozen |
| 20260330-185327 | 32 | 416 | 3e-4 | 0.52 | 23.05 | value 2.21→0.008; total loss went negative (−0.13) — entropy-term dominance, not a bug |
| 20260413-211808 | 28 | 512 | 1e-4 | 0.53 | 22.49 | 8 battles/eval |
| 20260414-131158 | 9 | **6144** | 5e-5 | 0.52 | 23.10 | value never improved (0.78–1.26); highest mu ever logged (28.16 at e5) — noise |
| 20260418-011530 | 16 | 128 | 5e-5 | 0.544 | 20.83 | e1 value-only warmup; policy loss ≈ 0 all run |
| 20260418-151105 | 29 | 256 | 1e-4 | 0.518 | 21.88 | same frozen pattern; **ancestor of the July line** |

Notes:

- `20260415-134443` / `20260416-003455` runs exist in git history but their event files contain
  only the header — results unrecoverable.
- No NaN/inf anywhere; the recurring pathology was exclusively the frozen policy.
- Eval winrate ~0.47–0.58 everywhere and is NOT a strength metric (self-play + oracle reward).

## Oracle history (supervised)

| Checkpoint | Acc | Notes |
|------------|-----|-------|
| win_rate_oracle-num_heroes_160 / 20260226-211124-040-0.9515.pth | **0.9515** | highest ever; simplest (no text/player-attention); **data-leak suspect** (player `hero_history` may include the match itself — see TECH_ANALYSIS.md) |
| …-text-embd_dim_128 / …-038-0.9048.pth | 0.9048 | |
| …-text-embd_dim_128-player_attention / …-083-0.9055.pth | 0.9055 | |
| …-player_attention / 20260309033516-000-0.9042.pth | 0.9042 | **the deployed default** feeding agent rewards |

Do NOT swap in the 0.9515 checkpoint as a "better oracle" without first verifying the leakage
question — a leaky oracle corrupts the reward for every downstream experiment.

## Conclusions (load-bearing)

1. **Plain PPO with MC returns never produced policy learning in this repo.** Root cause never
   fully diagnosed; the July switch to MCTS visit-distribution CE (AlphaZero-style) was the first
   change that moved entropy. Do not spend time re-tuning plain-PPO hyperparameters.
2. Old TrueSkill ratings (Feb–Apr) are not comparable to July ratings as policy evidence.
3. Highest-mu readings (27–28) across ALL eras are within eval noise (~±0.9 σ per model,
   ±2.7 mu epoch-to-epoch at 384 battles).
4. The strongest April checkpoint set is `bp_agent-20260418-151105` (e11/e29 top of the July
   combined leaderboard). It is the reference baseline for any "did we improve" comparison.
