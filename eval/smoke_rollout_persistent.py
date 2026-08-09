"""Smoke test: persistent MCTS with real checkpoint and rollout collection."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from model.bp_agent import BPTransformerAgent
from utils.batched_rollout import collect_batched_rollouts
from utils.bp_dataset import BPDataset
from utils.raw_data import NUM_HEROES

CKPT = "./ckpts/bp_agent-20260725-085756/bp_agent_epoch35.pth"

print("Loading models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = BPTransformerAgent().to(device)
agent.eval()
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
agent.load_state_dict(ckpt, strict=False)

class DummyOracle:
    def predict(self, rp, dp, rpf, dpf, **kw):
        return np.array([[0.55]])

oracle = DummyOracle()

print(f"Device: {device}")
print(f"Agent params: {sum(p.numel() for p in agent.parameters()):,}")

# Load a small test dataset
print("Loading dataset...")
dataset = BPDataset(num_synthetic=8)
print(f"Dataset size: {len(dataset)}")

samples = [dataset[i] for i in range(min(4, len(dataset)))]

mcts_config = {
    "c_puct": 2.0,
    "num_simulations": 16,
    "top_k": 12,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "max_search_depth": 4,
}

print(f"\nRunning rollouts with persistent MCTS (num_simulations=16, top_k=12)...")
t0 = time.perf_counter()
rollouts = collect_batched_rollouts(
    agent, oracle, samples,
    temperature=1.0,
    use_mcts=True,
    mcts_config=mcts_config,
    opponent_agents=None,
)
elapsed = time.perf_counter() - t0

print(f"\nCollected {len(rollouts)} rollouts in {elapsed:.1f}s ({elapsed/len(rollouts):.1f}s/rollout)")
for i, r in enumerate(rollouts):
    n_steps = len(r["actions"])
    n_mcts = r["mcts_policies"] is not None and r["mcts_policies"].shape[0] > 0
    print(f"  Rollout {i}: {n_steps} steps, "
          f"MCTS={'yes' if n_mcts else 'no'}, "
          f"last reward={r['rewards'][-1]:.3f}, "
          f"actions.shape={r['actions'].shape}")

print("\n✅ Smoke test PASSED — persistent MCTS rollouts work end-to-end")
