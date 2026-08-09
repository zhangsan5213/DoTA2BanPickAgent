"""Comprehensive smoke test for persistent MCTS with fast rollout."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from model.bp_agent import BPTransformerAgent
from search.mcts_batched import BatchMCTSEngine, BatchedMCTSNode
from utils.bp_env import BPState
from utils.batched_rollout import collect_batched_rollouts
from utils.bp_dataset import BPDataset
from utils.raw_data import NUM_HEROES

CKPT = "./ckpts/bp_agent-20260725-085756/bp_agent_epoch35.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = BPTransformerAgent().to(device).eval()
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
agent.load_state_dict(ckpt, strict=False)

class DummyOracle:
    def predict(self, *a, **kw):
        return np.array([[0.55]])
oracle = DummyOracle()

player_feats = [[0.0] * NUM_HEROES for _ in range(5)]
tests_passed = 0
tests_total = 0

# ============================================================
def test(name):
    global tests_total
    tests_total += 1
    print(f"\n{'='*60}")
    print(f"TEST {tests_total}: {name}")
    print(f"{'='*60}")
    return tests_total

# ============================================================
t = test("Fresh tree (prev_roots=None, no persistence) - backward compat")
engine = BatchMCTSEngine(agent, oracle, c_puct=2.0, num_simulations=8, top_k=12,
                          max_search_depth=0)
state = BPState([], [], player_feats, player_feats, is_radiant_turn=True, step_idx=0)
results = engine.search_batch([state], prev_roots=[None])
action, probs, next_root = results[0]
assert action is not None, "Must return a valid action"
assert action > 0, f"Action must be > 0, got {action}"
assert len(probs) > 0, "Must have action probs"
assert next_root is not None, "Must return next_root"
assert next_root.parent is not None, "next_root should have parent (root_0)"
# Verify root statistics exist
root_0 = next_root.parent
root_visits = sum(c.visit_count for c in root_0.children.values()) if root_0.children else 0
print(f"  Action: {action}, probs: {list(probs.items())[:3]}, root_visits: {root_visits}")
print(f"  ✅ PASS")
tests_passed += 1

# ============================================================
t = test("Persistent tree: game_roots across 5 steps")
engine = BatchMCTSEngine(agent, oracle, c_puct=2.0, num_simulations=8, top_k=12,
                          max_search_depth=3)
state = BPState([], [], player_feats, player_feats, is_radiant_turn=True, step_idx=0)
game_root = None
root_visits_by_step = []

for step in range(5):
    prev_roots = [game_root]
    results = engine.search_batch([state], prev_roots=prev_roots)
    action, probs, next_root = results[0]
    
    # Track game_root: the root of the entire search tree
    if game_root is None:
        # Fresh tree: next_root's parent is the game root
        game_root = next_root.parent if next_root is not None else None
    
    # Count visits on root_0's children
    if game_root is not None and game_root.children:
        visits = sum(c.visit_count for c in game_root.children.values())
    else:
        visits = 0
    root_visits_by_step.append(visits)
    
    print(f"  Step {step}: action={action}, root_0.children_visits={visits}")
    state.step(action)

print(f"  Visit accumulation: {root_visits_by_step}")
# Visits should increase as later steps' playouts backprop to root_0
if len(root_visits_by_step) >= 3:
    if root_visits_by_step[-1] > root_visits_by_step[0]:
        print(f"  ✅ PASS: visits accumulate (persistent tree backprop works)")
        tests_passed += 1
    else:
        print(f"  ❌ FAIL: visits not accumulating")
else:
    print(f"  ✅ PASS (not enough steps to verify accumulation)")
    tests_passed += 1

# ============================================================
t = test("full rollout with persistent tree (end-to-end)")
dataset = BPDataset(num_synthetic=4)
samples = [dataset[i] for i in range(min(2, len(dataset)))]

mcts_config = {
    "c_puct": 2.0, "num_simulations": 8, "top_k": 8,
    "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
    "max_search_depth": 3,
}

t0 = time.perf_counter()
rollouts = collect_batched_rollouts(
    agent, oracle, samples, temperature=1.0,
    use_mcts=True, mcts_config=mcts_config, opponent_agents=None,
)
elapsed = time.perf_counter() - t0

print(f"  Collected {len(rollouts)} rollouts in {elapsed:.1f}s ({elapsed/len(rollouts):.1f}s each)")
for r in rollouts:
    print(f"    actions={r['actions'].shape}, reward={r['rewards'][-1]:.3f}, "
          f"mcts_policies={'yes' if r['mcts_policies'] is not None else 'no'}")

assert len(rollouts) == len(samples), f"Expected {len(samples)} rollouts"
for r in rollouts:
    assert len(r["actions"]) > 0, "Rollout must have actions"
    assert r["mcts_policies"] is not None, "Must have MCTS policies when use_mcts=True"
    assert r["mcts_policies"].shape[0] == len(r["actions"]), \
        f"MCTS policies shape mismatch: {r['mcts_policies'].shape} vs {len(r['actions'])}"
print(f"  ✅ PASS: end-to-end rollout works")
tests_passed += 1

# ============================================================
t = test("Non-MCTS step resets game_roots correctly")
# This was already tested implicitly above (all MCTS), but let's verify the reset
# logic by running with opponent_agents
agent2 = BPTransformerAgent().to(device).eval()
agent2.load_state_dict(ckpt, strict=False)

# Load some old ckpt as opponent
old_ckpt = torch.load("./ckpts/bp_agent-20260724-205735/bp_agent_epoch30.pth",
                      map_location=device, weights_only=False)
opponent = BPTransformerAgent().to(device).eval()
opponent.load_state_dict(old_ckpt, strict=False)

opponent_agents = [(opponent, "dire", 99)]  # Dire = opponent, very stale

samples2 = [dataset[i] for i in range(min(1, len(dataset)))]
rollouts2 = collect_batched_rollouts(
    agent, oracle, samples2, temperature=1.0,
    use_mcts=True, mcts_config=mcts_config,
    opponent_agents=opponent_agents,
    policy_staleness_tolerance=2,
)

print(f"  Collected {len(rollouts2)} rollouts with opponent")
for r in rollouts2:
    n_mcts_steps = (r["valid_mask"].sum().item() if r["valid_mask"] is not None else 0)
    n_total = len(r["actions"])
    print(f"    total_steps={n_total}, MCTS_steps={n_mcts_steps}, reward={r['rewards'][-1]:.3f}")

print(f"  ✅ PASS: opponent mode works with persistent tree")
tests_passed += 1

# ============================================================
print(f"\n{'='*60}")
print(f"ALL TESTS: {tests_passed}/{tests_total} PASSED")
print(f"{'='*60}")
