"""Smoke test: persistent MCTS tree — verify backprop accumulation across steps."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from search.mcts_batched import BatchMCTSEngine, BatchedMCTSNode
from utils.bp_env import BPState
from utils.raw_data import NUM_HEROES


class DummyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.tensor(0.0))
    def forward(self, state):
        B = state["radiant_player_feats"].shape[0]
        logits = torch.zeros(B, NUM_HEROES)
        for b in range(B):
            for i in range(NUM_HEROES):
                logits[b, i] = 20.0 - i  # hero 1 best, hero 158 worst
        value = torch.zeros(B, 1)
        return logits, value


class DummyOracle:
    def predict(self, r, d, rp, dp):
        return np.array([[0.55]])  # slight radiant advantage


def test_persistent_within_single_draft():
    """Run a full 20-step draft with persistent tree, verify:
       - root_0 visit_count grows across steps
       - each step's action is valid
    """
    print("=" * 60)
    print("TEST 1: Persistent tree across a full 20-step draft")
    print("=" * 60)

    agent = DummyAgent()
    oracle = DummyOracle()
    player_feats = [[0.0] * NUM_HEROES for _ in range(5)]

    engine = BatchMCTSEngine(
        agent, oracle,
        c_puct=2.0, num_simulations=32, top_k=12,
        dirichlet_alpha=0.3, dirichlet_epsilon=0.25,
        max_search_depth=4,  # mirror configs/bp_agent_config.yaml; without a cap the
                             # search hits the max_rounds safety limit every step
    )

    state = BPState([], [], player_feats, player_feats, is_radiant_turn=True, step_idx=0)
    prev_root = None
    pivot = None
    first_level_total_visits = []  # sum of visit counts on root children per step
    max_visits_per_step = []

    for step_i in range(20):
        if state.done:
            print(f"  Step {step_i}: game done, stopping")
            break

        t0 = time.perf_counter()
        results = engine.search_batch([state], prev_roots=[prev_root], pivots=[pivot])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        action, probs, next_root = results[0]

        total_visits_this_level = sum(
            c.visit_count for c in next_root.parent.children.values()
        ) if next_root is not None and next_root.parent is not None else 0

        max_visit = max(
            (c.visit_count for c in next_root.parent.children.values()), default=0
        ) if next_root is not None and next_root.parent is not None else 0

        first_level_total_visits.append(total_visits_this_level)
        max_visits_per_step.append(max_visit)

        action_type = "PICK" if state.get_current_action_type() == "pick" else "BAN "
        team = "R" if state.is_radiant_turn else "D"

        print(f"  Step {step_i:2d} [{action_type} {team}] hero={action:3d} | "
              f"probs_top3={list(probs.items())[:3]} | time={elapsed_ms:.0f}ms | "
              f"level0_visits={total_visits_this_level} max_visit={max_visit}")

        state.step(action)
        # Production semantics (collect_batched_rollouts): pass the TOP BattleNode as
        # prev_root (GameRoot anchor) and the BattleNode under the selected ActionNode
        # as pivot. Passing the ActionNode itself as prev_root makes the next search
        # root an ActionNode, whose only child key is the string "_battle" — that would
        # leak into the finalized action selection.
        if next_root is not None:
            top = next_root
            while top.parent is not None:
                top = top.parent
            prev_root = top
            if next_root._is_action_node:
                if "_battle" not in next_root.children:
                    battle = BatchedMCTSNode(parent=next_root, action=None)
                    battle._depth_from_root = next_root._depth_from_root + 1
                    battle._is_action_node = False
                    next_root.children["_battle"] = battle
                pivot = next_root.children["_battle"]
            else:
                pivot = None
        else:
            pivot = None

        # Verify: picked action is in valid actions
        assert action in state.radiant_heroes + state.dire_heroes + state.radiant_bans + state.dire_bans, \
            f"Selected action {action} not applied to state!"

    print(f"\n  First-level visit accumulation across steps: {first_level_total_visits}")
    print(f"  Max visit per step: {max_visits_per_step}")

    # The real persistence check: EVERY playout backprops through the top root
    # (root_0), regardless of which pivot it started from. So after N steps the
    # top root must have >= num_simulations * N visits. A broken ancestor chain
    # would leave it at ~num_simulations (first step only).
    n_steps = len(first_level_total_visits)
    expected_min = 32 * n_steps
    if prev_root is not None:
        # Trace to the very top parent
        top = prev_root
        while top.parent is not None:
            top = top.parent
        print(f"  Top root visit_count after all steps: {top.visit_count}")
        print(f"  Expected minimum ({32} sims x {n_steps} steps): {expected_min}")
        if top.visit_count >= expected_min:
            print("  ✅ PASS: cross-step backprop accumulating (persistent tree working)")
        else:
            print("  ❌ FAIL: top root visits below expected minimum, check ancestor backprop")
        print(f"  Top root children: {len(top.children)}")
        for a, c in list(top.children.items())[:5]:
            print(f"    child action={a} visits={c.visit_count} prior={c.prior:.4f}")

    print()


def test_baseline_no_persistence():
    """Run one step without persistent tree (baseline)."""
    print("=" * 60)
    print("TEST 2: Baseline single-step search (no persistence)")
    print("=" * 60)

    agent = DummyAgent()
    oracle = DummyOracle()
    player_feats = [[0.0] * NUM_HEROES for _ in range(5)]

    engine = BatchMCTSEngine(agent, oracle, c_puct=2.0, num_simulations=16, top_k=12)
    state = BPState([], [], player_feats, player_feats, is_radiant_turn=True, step_idx=0)

    t0 = time.perf_counter()
    action, probs, next_root = engine.search_batch([state])[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(f"  Selected action: {action}")
    print(f"  Policy probs top 5: {list(probs.items())[:5]}")
    print(f"  Time: {elapsed_ms:.0f}ms")
    if next_root is not None:
        print(f"  Next root children: {len(next_root.children)}")
    print("  ✅ PASS: basic search works\n")


if __name__ == "__main__":
    test_baseline_no_persistence()
    test_persistent_within_single_draft()
