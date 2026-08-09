"""Profile persistent MCTS timing to find the true bottleneck."""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch, numpy as np
from model.bp_agent import BPTransformerAgent
from search.mcts_batched import BatchMCTSEngine, BatchedMCTSNode
from utils.bp_env import BPState
from utils.raw_data import NUM_HEROES

parser = argparse.ArgumentParser(description="Profile persistent MCTS per-decision timing")
parser.add_argument(
    "--ckpt",
    default="./ckpts/bp_agent-20260725-085756/bp_agent_epoch35.pth",
    help="Agent checkpoint path",
)
parser.add_argument(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device (cuda/cpu)",
)
args = parser.parse_args()

CKPT = args.ckpt
device = torch.device(args.device)

agent = BPTransformerAgent().to(device).eval()
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
agent.load_state_dict(ckpt, strict=False)

class DummyOracle:
    def predict(self, *a, **kw):
        return np.array([[0.55]])
oracle = DummyOracle()

player_feats = [[0.0] * NUM_HEROES for _ in range(5)]

engine = BatchMCTSEngine(
    agent, oracle,
    c_puct=2.0, num_simulations=16, top_k=12,
    dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
)

# --- Bench: single step, deep persistent tree ---
# First, build a deep tree by running 10 steps
print("Building persistent tree (10 steps)...")
state = BPState([], [], player_feats, player_feats, is_radiant_turn=True, step_idx=0)
prev_root = None
for step in range(10):
    results = engine.search_batch([state], prev_roots=[prev_root])
    action, _, next_root = results[0]
    state.step(action)
    prev_root = next_root
print(f"Tree built. Next search root depth: {len(state.history['actions'])}")

# Now profile a single search step with the deep tree
print(f"\nProfiling step 11 at depth {len(state.history['actions'])}...")

torch.cuda.synchronize()
t_start = time.perf_counter()

# Use wrapper to time sub-parts
results = engine.search_batch([state], prev_roots=[prev_root])

torch.cuda.synchronize()
t_total = time.perf_counter() - t_start

action, probs, _ = results[0]
print(f"\nTotal time: {t_total*1000:.0f}ms")
print(f"Action: {action}, top probs: {list(probs.items())[:3]}")

# --- Now check: what's taking time? ---
# Re-run with manual breakdown
print("\n=== Detailed profile ===")
torch.cuda.synchronize()
t1 = time.perf_counter()

# Create a single MCTS instance and manually profile the stages
from search.mcts_batched import BatchedDraftMCTS
mcts = BatchedDraftMCTS(agent, oracle, c_puct=2.0, num_simulations=16, top_k=12)

torch.cuda.synchronize()
t_setup = time.perf_counter()

# Stage 1: Collect pending (selection only)
t_sel_total = 0
n_selections = 0
for round_i in range(8):  # first 8 rounds
    t_sel_start = time.perf_counter()
    pending = mcts._collect_pending_evaluations(prev_root, state)
    t_sel = time.perf_counter() - t_sel_start
    t_sel_total += t_sel
    n_selections += len(pending)
    
    if not pending:
        break
    
    # Stage 2: Reconstruct states
    t_rec_start = time.perf_counter()
    eval_states = []
    for p in pending:
        s = mcts._reconstruct_state(p.root_state, p.node, search_root=p.search_root)
        eval_states.append(s)
    t_rec = time.perf_counter() - t_rec_start
    
    # Stage 3: Model forward (batch)
    torch.cuda.synchronize()
    t_fwd_start = time.perf_counter()
    batch_dict = BPState.batch_to_dict(eval_states, device=device)
    logits, values = agent(batch_dict)
    torch.cuda.synchronize()
    t_fwd = time.perf_counter() - t_fwd_start
    
    print(f"  Round {round_i}: selection={t_sel*1000:.1f}ms, "
          f"reconstruct={t_rec*1000:.1f}ms, forward={t_fwd*1000:.1f}ms, "
          f"leaves={len(pending)}")

torch.cuda.synchronize()
t_total2 = time.perf_counter() - t1
print(f"\nTotal rounds profiled: {n_selections} selections in {t_sel_total*1000:.0f}ms")
print(f"Full search_batch: {t_total*1000:.0f}ms")
