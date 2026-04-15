"""Lightweight MCTS for DOTA 2 draft (training-stage rollout enhancement)."""

import math
from typing import List

import torch
import torch.nn.functional as F

from utils.bp_env import BPState
from utils.raw_data import NUM_HEROES


class MCTSNode:
    """MCTS tree node - stores only tree structure, no deep state copies."""

    __slots__ = ['parent', 'action', 'prior', 'children', 'visit_count',
                 'value_sum', 'eval_value', 'action_priors', 'is_terminal',
                 'terminal_value']

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action  # hero_id that led to this node (None for root)
        self.prior = prior
        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        # Cached evaluation results (only set after expansion/evaluation)
        self.eval_value = None
        self.action_priors = None
        self.is_terminal = None  # None = unknown, True/False = known
        self.terminal_value = None  # Only valid if is_terminal is True

    def is_expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct, parent_visits):
        if self.visit_count == 0:
            return float('inf')
        q = self.value()
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u


class DraftMCTS:
    """Lightweight MCTS for BP using policy priors + value/oracle evaluation."""

    def __init__(
        self,
        agent,
        oracle,
        c_puct=1.5,
        num_simulations=64,
        top_k=20,
    ):
        self.agent = agent
        self.oracle = oracle
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.top_k = top_k

    def _reconstruct_state(self, root_state: BPState, node: MCTSNode) -> BPState:
        """Reconstruct the state for a node by replaying actions from root."""
        # Collect path from node back to root
        path = []
        current = node
        while current.parent is not None:
            path.append(current.action)
            current = current.parent

        # Now replay from root - ONE copy per evaluation batch, not per node!
        state = root_state.copy()
        for action in reversed(path):
            state.step(action)

        return state

    @torch.no_grad()
    def _get_policy_and_value(self, state: BPState):
        """Get policy logits and value estimate from agent.

        Returns:
            (masked_logits, value_for_current_player, [(action, prior), ...])
        """
        state_dict = state.to_dict(device=next(self.agent.parameters()).device)
        logits, value = self.agent(state_dict)
        logits = logits.squeeze(0)  # [NUM_HEROES]

        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return None, 0.0, []

        # Build mask: only valid actions allowed
        mask = torch.full((NUM_HEROES,), -1e9, device=logits.device)
        for h in valid_actions:
            if 1 <= h <= NUM_HEROES:
                mask[h - 1] = 0.0
        masked_logits = logits + mask

        # Compute probabilities over valid actions
        probs = F.softmax(masked_logits, dim=-1)
        valid_hero_indices = [h - 1 for h in valid_actions]
        valid_probs = probs[valid_hero_indices]

        # Top-k pruning
        if len(valid_actions) > self.top_k:
            topk_vals, topk_idx = torch.topk(valid_probs, self.top_k)
            pruned_actions = [valid_actions[i] for i in topk_idx.tolist()]
            pruned_probs = topk_vals / topk_vals.sum()
        else:
            pruned_actions = valid_actions
            pruned_probs = valid_probs / valid_probs.sum()

        action_priors = list(zip(pruned_actions, pruned_probs.tolist()))
        return masked_logits, value.item(), action_priors

    def _expand_node(self, root_state: BPState, node: MCTSNode):
        """Expand a leaf node using policy priors."""
        if node.action_priors is None:
            # Need to evaluate this node
            state = self._reconstruct_state(root_state, node)

            if state.done:
                node.is_terminal = True
                reward = state.get_reward(self.oracle)
                if reward is None:
                    reward = 0.5
                mapped = 2.0 * reward - 1.0  # [-1, 1]
                if state.is_radiant_turn:
                    node.terminal_value = mapped
                else:
                    node.terminal_value = -mapped
                node.eval_value = node.terminal_value
                return

            _, value, action_priors = self._get_policy_and_value(state)
            if not action_priors:
                node.eval_value = 0.0
                return
            node.is_terminal = False
            node.eval_value = value
            node.action_priors = action_priors

        for action, prior in node.action_priors:
            child = MCTSNode(parent=node, action=action, prior=prior)
            node.children[action] = child

    def _select_child(self, node: MCTSNode):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            score = child.ucb_score(self.c_puct, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backpropagate(self, path: List[MCTSNode], value: float):
        """Backpropagate value up the tree."""
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def search(self, root_state: BPState):
        """Run MCTS and return (selected_action, action_probs_dict).

        selected_action: hero_id with highest visit count.
        action_probs_dict: {hero_id: probability} from normalized visit counts.
        """
        root = MCTSNode()  # Root has no parent or action

        # Expand root first - we need its children for selection
        if root_state.done:
            # Already done - just return valid actions with uniform prob
            valid_actions = root_state.get_valid_actions()
            if valid_actions:
                return valid_actions[0], {a: 1.0/len(valid_actions) for a in valid_actions}
            return None, {}

        _, value, action_priors = self._get_policy_and_value(root_state)
        root.is_terminal = False
        root.eval_value = value
        root.action_priors = action_priors
        self._expand_node(root_state, root)

        for _ in range(self.num_simulations):
            node = root
            path = [node]

            # Selection - FAST: NO state reconstruction needed!
            # We just traverse the tree using UCB scores
            while node.is_expanded() and not node.is_terminal:
                node = self._select_child(node)
                path.append(node)

            # Now we've reached a leaf - check if terminal (once, lazily)
            if node.is_terminal is None:
                # Lazily determine if terminal by reconstructing state once
                state = self._reconstruct_state(root_state, node)
                if state.done:
                    node.is_terminal = True
                    reward = state.get_reward(self.oracle)
                    if reward is None:
                        reward = 0.5
                    mapped = 2.0 * reward - 1.0  # [-1, 1]
                    if state.is_radiant_turn:
                        node.terminal_value = mapped
                    else:
                        node.terminal_value = -mapped
                    node.eval_value = node.terminal_value
                else:
                    node.is_terminal = False

            if node.is_terminal:
                # Terminal - backpropagate immediately
                self._backpropagate(path, node.terminal_value)
                continue

            # Non-terminal, unexpanded leaf: expand and backpropagate
            if not node.is_expanded():
                self._expand_node(root_state, node)
                value = node.eval_value if node.eval_value is not None else 0.0
                self._backpropagate(path, value)

        # Build visit-count policy
        visits = {action: child.visit_count for action, child in root.children.items()}
        total_visits = sum(visits.values())
        if total_visits == 0:
            valid_actions = root_state.get_valid_actions()
            action_probs = {a: 1.0 / len(valid_actions) for a in valid_actions}
            selected_action = valid_actions[0] if valid_actions else None
        else:
            action_probs = {a: v / total_visits for a, v in visits.items()}
            selected_action = max(visits, key=visits.get)

        return selected_action, action_probs


if __name__ == "__main__":
    print("=" * 60)
    print("Testing DraftMCTS")
    print("=" * 60)

    # Dummy agent/oracle for shape verification
    class DummyAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Add a dummy parameter so next(self.parameters()) works
            self._dummy = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, state):
            B = state["radiant_player_feats"].shape[0]
            return torch.randn(B, NUM_HEROES), torch.randn(B, 1)

    class DummyOracle:
        def predict(self, r, d, rp, dp):
            import numpy as np
            return np.array([[0.6]])

    from utils.raw_data import get_valid_hero_ids

    agent = DummyAgent()
    oracle = DummyOracle()
    mcts = DraftMCTS(agent, oracle, num_simulations=8, top_k=10)

    player_feats = [[0.0] * NUM_HEROES for _ in range(5)]
    state = BPState([], [], player_feats, player_feats, is_radiant_turn=True, step_idx=0)

    action, probs = mcts.search(state)
    print(f"Selected action: {action}")
    print(f"Action probs sum: {sum(probs.values()):.4f}")
    print(f"Num actions in policy: {len(probs)}")
    print("\n[OK] MCTS search successful!")
