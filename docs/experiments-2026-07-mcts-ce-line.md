# Experiments: MCTS + CE Line (2026-07)

The current main line: AlphaZero-style MCTS with cross-entropy policy loss on visit distributions,
running inside PPO training (PPO term weighted to 0). All runs resumed from the April baseline
`bp_agent-20260418-151105/bp_agent_epoch29.pth`.

## Resume chain

```
151105/epoch29 (Apr baseline, ~6984 global steps)
  → bp_agent-20260724-204330  (1 batch of epoch 30, then killed)  ← interrupted
  → bp_agent-20260724-205735  (epochs 30–34, 5 full epochs)       ← the only substantive run
  → bp_agent-20260725-064641  (1 batch of epoch 35, then killed)  ← interrupted
  → bp_agent-20260725-072854  (1 batch, KL 0.46 → early stop)     ← dead, no checkpoint
  → bp_agent-20260725-085756  (epoch 35 complete)                 ← latest model
```

3 of 5 July runs died mid-batch. The resume chain works but is fragile — treat any run that
stops in < 1 epoch as untrusted (its `bp_agent_final.pth` rating is noise: e.g. 2–8 games only).

## Run table

| Run | Epochs | Rollouts/ep | Eval battles | Final mu / display | Final eval WR | Notes |
|-----|--------|-------------|--------------|--------------------|---------------|-------|
| 20260724-204332 | ~0 (1 batch of e30) | — | 8 | 25.62 / 19.74 (σ=1.96) | — | Interrupted ~12 min; KL 0.103 early-stop |
| 20260724-205739 | e30–e34 | 256 | 384 | 24.58 / 21.96 | 0.471 | Only full run. Entropy −4.47→−1.82 |
| 20260725-064645 | ~0 (1 batch of e35) | — | 2 | 26.21 / 14.85 (σ=3.78) | — | Interrupted ~4 min |
| 20260725-072854 | ~0 (1 batch) | — | — | — | — | KL 0.46 → PPO early stop; no ckpt saved |
| 20260725-085756 | e35 | 128 | 384 | 25.15 / 22.54 | 0.505 | Batch KL 0.17–0.37, epoch-avg 0.0043 |

## Key signals

- **Entropy finally moved** (the headline): batch entropy went −4.47 → −1.82 nats across
  205739's 5 epochs. Every pre-July run had entropy pinned at −4.76 (≈ uniform over 160 heroes,
  ln 160 = 5.075). The MCTS CE loss has real gradient and is the only per-action training signal
  that has ever worked here.
- **No improvement over April baselines yet**: final combined leaderboard (085756) top models:
  151105/e11 display **23.79**, 151105/e29 **23.76**, 205735/e33 **23.52** (July peak), new e35
  **22.54**. July models sit mid-pack. Only ~6 of the configured 64 epochs were trained.
- **Eval noise is large**: 384-battle evals swing ~2.7 mu between adjacent epochs
  (e32 display 23.46 / WR 0.466 → e33 26.20 / WR 0.539). Single-epoch rating deltas are not evidence.
- **KL spikes**: batch KL 0.17–0.46 triggered PPO early-stop frequently; epoch-averaged KL stayed
  small (0.0043). No NaN/inf in any run.
- The single highest mu ever logged anywhere is 28.16 (e5 of the 6144-rollouts/epoch April run) —
  within noise; do not cite it as a ceiling.

## Conclusions (load-bearing)

1. MCTS+CE is the only line that has ever trained a real policy in this repo. Do NOT revert to
   plain PPO (the pre-July policy-freeze era) — see `experiments-2026-02-04-ppoa-era.md`.
2. After ~6 epochs of MCTS-CE, new models do NOT measurably beat April checkpoints. This is
   NOT a verdict against the line — config targets 64 epochs and the old checkpoints were trained
   with a far larger value-head-only budget.
3. Never judge a model on a run that was interrupted mid-epoch: `bp_agent_final.pth` from such a
   run has 2–8 evaluation games and σ ≈ 2–4.
4. TrueSkill display rating (mu − 3σ) against the historical pool is the only cross-model signal;
   treat < 1σ (≈ 0.87) differences as ties.

## Verdict rules

- **No kill before ~50 epochs** unless a true collapse signature appears: entropy → 0,
  CE ≈ random on all decisions, or wins concentrated on 1–2 player sets.
- A single run's rating dip (even 3+ mu) is NOT a verdict — it's eval noise.
- Before ANY new MCTS experiment: run the smoke tests (see AGENTS.md → Common Commands) and
  confirm the persistent-tree visit-accumulation check passes.

## Post-mortem: 2026-08-03 first-run failure (load-bearing)

The first resumed run (from 151105-era code, epoch 35 → 36) died after one batch with
`Loss/total ≈ 2.6e7` and `KL = inf`. Root cause: **stale pivot children** (see
`AGENTS.md` conclusion #12). The on-demand `_battle` pivot is already expanded when reused as
the next step's search root; its children hold heroes that are no longer legal, they get
selected, and π overlaps the training CE mask (−1e9) → `-π·(-1e9)` explodes.

This also retroactively explains the July run anomalies: batch-KL 0.17–0.46 spikes,
`20260725-072854` dying after 1 batch (KL 0.46), and 3/5 resume runs dying mid-epoch.
`search_batch` now resets pivots before reuse; verified: π always legal, CE ≈ 4.6, KL ≈ 0 on
the e35 checkpoint. **Any run that shows batch-KL > 0.1 with huge Loss/total should re-check
this invariant before anything else.**
