# Project Notes

## Environment
- Use the `torch` conda environment for all Python execution in this project.
  - Example: `conda run -n torch python train_bp_agent.py ...`
  - PyTorch version: 2.11.0+cu130

## Training & Resume
- `train_bp_agent.py` supports `--resume <checkpoint.pth>` to resume training.
- Checkpoints (both periodic `bp_agent_epoch{N}.pth` and final `bp_agent_final.pth`) save full training state:
  - model weights (`agent_state`)
  - optimizer state (`optimizer_state`)
  - scheduler state (`scheduler_state`)
  - entropy annealer step (`entropy_step`)
  - epoch, global_step, grad_accum_step
- **Important**: When resuming, NEW directories are always created for both checkpoints and TensorBoard logs to avoid contamination. Old runs are never modified.
- **Previous Checkpoint Inclusion**: When resuming, the directory of the checkpoint you're resuming from is automatically added to the list of directories scanned for historical checkpoints. This means checkpoints from the previous run are available for evaluation and as historical opponents.
- **Combined Leaderboard**: When resuming, the TrueSkill/ELO leaderboard automatically loads and displays rating records from both the previous run(s) and the current resuming run, showing all historical models together.
- Resume command:
  ```bash
  conda run -n torch python train_bp_agent.py --resume ./ckpts/bp_agent-<timestamp>/bp_agent_epoch<N>.pth
  ```

## MCTS Optimization
- The MCTS implementation has been optimized to avoid deep copying `BPState` for every node.
- Key optimizations:
  - `MCTSNode` no longer stores `BPState` - only stores the action taken to reach it
  - Tree selection phase uses only UCB scores, no state reconstruction
  - States are reconstructed lazily only when needed for evaluation/expansion
  - Terminal state information is cached after first check
  - `__slots__` used for memory efficiency and faster attribute access
- Performance impact: From O(N * D) state copies to O(K) state reconstructions per search (N=simulations, D=depth, K=unique leaves)
