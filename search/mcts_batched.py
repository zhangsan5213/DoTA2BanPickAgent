"""Batched MCTS for DOTA 2 draft - collects all state evaluations and runs in single batch."""

import math
import json
import random
import time
from pathlib import Path
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from utils.bp_env import BPState
from utils.raw_data import NUM_HEROES, STATIC_HERO_MASK


# --- Cached hero winrate data for fast rollout ---
@lru_cache(maxsize=1)
def _load_hero_winrates():
    path = Path(__file__).parent.parent / "data" / "hero_winrates.json"
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v.get("winrate", 0.5) or 0.5 for k, v in raw.items()}


@dataclass
class PendingEvaluation:
    """Represents a node that needs model evaluation."""
    node: 'BatchedMCTSNode'
    root_state: BPState
    path: List['BatchedMCTSNode']
    search_root: 'BatchedMCTSNode' = None  # Persistent tree: root of THIS search


class BatchedMCTSNode:
    """MCTS tree node - stores only tree structure, no deep state copies."""

    __slots__ = ['parent', 'action', 'prior', 'children', 'visit_count',
                 'value_sum', 'eval_value', 'action_priors', 'is_terminal',
                 'terminal_value', '_cached_state', '_creation_state',
                 '_depth_from_root', '_is_action_node', '_eval_queued']

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action  # hero_id that led to this node (None for BattleNodes)
        self.prior = prior
        self.children = {}  # action -> BatchedMCTSNode
        self.visit_count = 0
        self.value_sum = 0.0
        self.eval_value = None
        self.action_priors = None
        self.is_terminal = None  # None = unknown, True/False = known
        self.terminal_value = None  # Only valid if is_terminal is True
        self._cached_state = None
        self._creation_state = None
        self._depth_from_root = 0  # 0=root, 1=ActionNode, 2=BattleNode, 3=ActionNode, ...
        self._is_action_node = False  # True if ActionNode (odd depth): never expanded
        self._eval_queued = False  # Avoid duplicate pending entries for BattleNodes

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

    def collect_paths(self, max_trajectories=4, temperature=0.5) -> List[List[int]]:
        """Collect multiple trajectories from this MCTS tree by sampling visit distributions.
        
        GITCGRL CollectPath equivalent: walks the tree, at each level samples an
        action from the visit distribution (with temperature), then descends.
        Produces diverse trajectories for multi-trajectory training data.
        
        Args:
            max_trajectories: Maximum number of trajectories to sample
            temperature: Softmax temperature for visit distribution sampling
                         (lower = greedier, 1.0 = proportional to visits)
        
        Returns:
            List of trajectories, each a list of hero_ids (actions)
        """
        trajectories = []
        for _ in range(max_trajectories):
            node = self
            traj = []
            while node.is_expanded() and len(node.children) > 0:
                visits = {a: c.visit_count for a, c in node.children.items()}
                total = sum(visits.values())
                if total == 0:
                    break
                actions = list(visits.keys())
                probs = np.array([visits[a] / total for a in actions], dtype=np.float64)
                # Apply temperature
                if temperature != 1.0 and temperature > 0:
                    probs = probs ** (1.0 / temperature)
                    probs /= probs.sum()
                # Sample action
                action = actions[np.random.choice(len(actions), p=probs)]
                traj.append(action)
                node = node.children[action]
            
            if len(traj) > 0:
                trajectories.append(traj)
        return trajectories


class CachedStatePool:
    """Pool to share cached states across multiple MCTS trees to avoid memory explosion."""

    def __init__(self, max_size: int = 1000):
        self._cache = {}  # (id(root), path_tuple), node) -> state
        self._max_size = max_size

    def get(self, key):
        return self._cache.get(key)

    def put(self, key, state):
        if len(self._cache) >= self._max_size:
            # Simple cache eviction - clear oldest half
            keys = list(self._cache.keys())
            for k in keys[:len(keys)//2]:
                del self._cache[k]
        self._cache[key] = state


class BatchedDraftMCTS:
    """Batched MCTS for BP - multiple trees share a single batch evaluation."""

    def __init__(
        self,
        agent,
        oracle,
        c_puct=1.5,
        num_simulations=64,
        top_k=20,
        max_search_depth=0,  # 0 = no cap; >0 = treat leaves at this depth as terminal
    ):
        self.agent = agent
        self.oracle = oracle
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.top_k = top_k
        self.max_search_depth = max_search_depth

    def _reconstruct_state(self, root_state: BPState, node: BatchedMCTSNode,
                           search_root: Optional['BatchedMCTSNode'] = None) -> BPState:
        """Reconstruct the state for a node by replaying actions from root.

        Optimized: Uses cached parent state when available to minimize replay depth.

        When search_root is provided (persistent tree), parent traversal stops at
        the search root — only actions below the search root are replayed on top
        of root_state (which already represents the state at search_root).
        """
        # First check if this node already has a cached state
        if node._cached_state is not None:
            return node._cached_state

        # Check if parent has a cached state (common case)
        if node.parent is not None and node.parent._cached_state is not None:
            # Fast path: step from parent's cached state
            state = node.parent._cached_state.copy()
            if node.action is not None:  # BattleNodes skip (actionless state nodes)
                state.step(node.action)
            node._cached_state = state
            return state

        # Fallback: collect path from node up to (but not including) search_root
        path = []
        current = node
        while current.parent is not None and current.parent is not search_root:
            if current.action is not None:  # BattleNodes have action=None, skip
                path.append(current.action)
            current = current.parent

        # Replay from root_state
        state = root_state.copy()
        for action in reversed(path):
            state.step(action)

        # Cache the result
        node._cached_state = state
        return state

    def _select_child(self, node: BatchedMCTSNode):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            score = child.ucb_score(self.c_puct, node.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def _backpropagate(self, path: List[BatchedMCTSNode], value: float):
        """Backpropagate value up the tree."""
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def _fast_rollout(self, state: BPState) -> float:
        """Fast rollout to game end using winrate-greedy heuristic, then oracle eval.
        
        Returns value from the CURRENT player's perspective (in [-1, 1]).
        """
        hero_winrates = _load_hero_winrates()
        sim = state.copy()
        while not sim.done:
            valid = sim.get_valid_actions()
            if not valid:
                break
            # Greedy: pick highest winrate hero
            # (with small random noise to avoid deterministic behavior)
            best = max(valid, key=lambda h: hero_winrates.get(h, 0.5) + random.uniform(-0.02, 0.02))
            sim.step(best)
        
        # Oracle evaluation at terminal state
        reward = sim.get_reward(self.oracle)
        if reward is None:
            return 0.0
        mapped = 2.0 * reward - 1.0  # [-1, 1]
        # Convert to CURRENT (original state) player's perspective
        if state.is_radiant_turn:
            return mapped
        else:
            return -mapped

    def _collect_pending_evaluations(
        self,
        root: BatchedMCTSNode,
        root_state: BPState,
    ) -> List[PendingEvaluation]:
        """Run selection phase and collect nodes that need evaluation.
        
        For persistent trees, the path includes ancestor nodes (above root)
        so backprop accumulates visit statistics across multiple decisions.
        
        When max_search_depth > 0, nodes at exactly that depth (relative to the
        search root) get a fast oracle rollout: no model forward, no expansion,
        just an oracle value estimate and immediate backprop. This caps tree
        growth and eliminates model forwards for deep leaves.
        """
        # Collect ancestor chain (above current search root) for cross-step backprop
        ancestor_chain = []
        node = root.parent
        while node is not None:
            ancestor_chain.append(node)
            node = node.parent
        
        pending = []

        for _ in range(self.num_simulations):
            node = root
            path = list(ancestor_chain)  # [root_0, ..., parent_of_search_root]
            path.append(node)            # [root_0, ..., search_root]
            depth_from_root = 0  # levels expanded below search_root
            
            # Selection - FAST: NO state reconstruction needed!
            while node.is_expanded() and not node.is_terminal:
                node = self._select_child(node)
                path.append(node)
                depth_from_root += 1
                # Fast rollout: at depth cap, break out and use oracle
                if self.max_search_depth > 0 and depth_from_root >= self.max_search_depth:
                    break

            # --- Fast rollout at depth cap ---
            if self.max_search_depth > 0 and depth_from_root >= self.max_search_depth:
                if node.is_terminal is None:
                    # Fast rollout: simulate to end with heuristic, no model forward
                    state = self._reconstruct_state(root_state, node, search_root=root)
                    value = self._fast_rollout(state)
                    node.terminal_value = value
                    node.is_terminal = True
                    node.eval_value = value
                    self._backpropagate(path, value)
                elif node.is_terminal:
                    self._backpropagate(path, node.terminal_value)
                continue

            # Check if this node already has evaluation results
            if node.is_terminal is not None:
                if node.is_terminal:
                    self._backpropagate(path, node.terminal_value)
                else:
                    if not node.is_expanded() and node.action_priors is not None:
                        if node._is_action_node:
                            # GITCGRL: ActionNode → create BattleNode child on-demand
                            if "_battle" not in node.children:
                                battle = BatchedMCTSNode(parent=node, action=None)
                                battle._depth_from_root = node._depth_from_root + 1
                                battle._is_action_node = False
                                node.children["_battle"] = battle
                            else:
                                battle = node.children["_battle"]
                            # Only queue once per round (avoid duplicate pending)
                            if not battle._eval_queued and battle.is_terminal is None:
                                battle._eval_queued = True
                                pending.append(PendingEvaluation(battle, root_state, path + [battle], search_root=root))
                        else:
                            self._expand_node_from_cache(node)
                            self._backpropagate(path, node.eval_value)
                continue

            # Need to evaluate this node (model forward)
            pending.append(PendingEvaluation(node, root_state, path, search_root=root))

        return pending

    def _expand_node_from_cache(self, node: BatchedMCTSNode):
        """Expand a node using cached action priors.
        ActionNodes (odd-depth candidates) are NEVER expanded — their BattleNode
        children are created on-demand during playout traversal (GITCGRL style)."""
        if node._is_action_node:
            return  # ActionNodes don't expand
        for action, prior in node.action_priors:
            child = BatchedMCTSNode(parent=node, action=action, prior=prior)
            child._depth_from_root = node._depth_from_root + 1
            child._is_action_node = (child._depth_from_root % 2 == 1)
            node.children[action] = child

    def _finalize_searches(
        self,
        roots: List[BatchedMCTSNode],
    ) -> List[Tuple[int, Dict[int, float]]]:
        """Finalize all searches and return selected actions and policies."""
        results = []
        for root in roots:
            # Build visit-count policy. Only integer action keys are actions:
            # ActionNodes also hold a "_battle" navigation key, which must never be
            # selectable as an action (silent-bug guard, GITCGRL lesson).
            visits = {
                action: child.visit_count
                for action, child in root.children.items()
                if isinstance(action, int)
            }
            total_visits = sum(visits.values())
            if total_visits == 0:
                # Fallback - should not happen
                selected_action = None
                action_probs = {}
            else:
                action_probs = {a: v / total_visits for a, v in visits.items()}
                selected_action = max(visits, key=visits.get)
            results.append((selected_action, action_probs))
        return results


class BatchMCTSEngine:
    """Engine that runs multiple MCTS searches with batched model evaluations."""

    def __init__(
        self,
        agent,
        oracle,
        c_puct=1.5,
        num_simulations=64,
        top_k=20,
        device=None,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        max_search_depth=0,
    ):
        self.agent = agent
        self.oracle = oracle
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.top_k = top_k
        self.device = device if device is not None else next(agent.parameters()).device
        # AlphaZero-style root exploration noise: prior = (1-eps)*prior + eps*Dir(alpha)
        # 0.0 disables it (backward compatible)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        # Fast rollout: oracle evaluation beyond this depth from search root
        self.max_search_depth = max_search_depth

    def _apply_root_noise(self, action_priors):
        """Mix Dirichlet noise into root action priors (in-place on a new list)."""
        n = len(action_priors)
        if n == 0:
            return action_priors
        noise = np.random.dirichlet([self.dirichlet_alpha] * n)
        eps = self.dirichlet_epsilon
        return [
            (action, (1 - eps) * prior + eps * noise[i])
            for i, (action, prior) in enumerate(action_priors)
        ]

    @torch.no_grad()
    def search_batch(
        self,
        root_states: List[BPState],
        prev_roots: Optional[List[Optional['BatchedMCTSNode']]] = None,
        pivots: Optional[List[Optional['BatchedMCTSNode']]] = None,
    ) -> List[Tuple[int, Dict[int, float], 'BatchedMCTSNode']]:
        """Run MCTS search for multiple root states with batched evaluation.
        
        GITCGRL-compatible persistent trees:
        - prev_roots[i]: GameRoot anchor (for _creation_state and parent chain)
        - pivots[i]: current BattleNode to start playouts from.
          If None, falls back to prev_roots[i] (first step: fresh tree).
          If a pivot is provided, it should be a fresh BattleNode (no children).

        Args:
            root_states: List of current BP states for each search
            prev_roots: GameRoot anchor per search (for cross-step backprop)
            pivots: Current BattleNode to start playouts from per search

        Returns:
            List of (selected_action, action_probs_dict, pivot_for_next_step)
            pivot_for_next_step is the BattleNode child under the selected action.
        """
        total_start = time.perf_counter()
        n_searches = len(root_states)
        prev_roots = prev_roots or [None] * n_searches
        pivots = pivots or [None] * n_searches

        # Create MCTS instances and determine root nodes
        mcts_instances = []
        roots = []
        is_fresh_root = []  # True if root needs expansion
        root_reconstruction_states = []  # Original game state for correct replay

        for i in range(n_searches):
            mcts = BatchedDraftMCTS(
                self.agent, self.oracle,
                c_puct=self.c_puct,
                num_simulations=self.num_simulations,
                top_k=self.top_k,
                max_search_depth=self.max_search_depth,
            )
            mcts_instances.append(mcts)

            game_root = prev_roots[i]
            pivot = pivots[i]
            
            if pivot is not None:
                # Pivot must be a FRESH BattleNode (no children, no stats).
                # A pivot created on-demand in a previous step's selection
                # (ActionNode -> "_battle") may already be EXPANDED: its children
                # hold heroes that were legal then but are already banned/picked
                # now. Re-expanding such a pivot only adds new children (never
                # removes stale ones), so the stale heroes leak into the visit
                # policy -> training CE mask (-1e9) overlaps pi -> loss explodes
                # (observed 1e8-scale Loss/total + KL=inf on 2026-08-03). Reset
                # the pivot's children and statistics before reuse.
                if len(pivot.children) > 0:
                    pivot.children.clear()
                pivot.visit_count = 0
                pivot.value_sum = 0.0
                pivot.action_priors = None
                pivot.eval_value = None
                pivot._eval_queued = False
                # Descend: playouts start from the pivot (GITCGRL pivot_node)
                roots.append(pivot)
                is_fresh_root.append(True)  # Pivot is always fresh (BattleNode)
                recon_state = root_states[i]  # Current state IS the state at pivot
                root_reconstruction_states.append(recon_state)
            elif game_root is not None:
                # Always from game root (GITCGRL shared_root)
                roots.append(game_root)
                is_fresh_root.append(not game_root.is_expanded())
                recon_state = getattr(game_root, '_creation_state', root_states[i])
                root_reconstruction_states.append(recon_state)
            else:
                # Fresh tree
                new_root = BatchedMCTSNode()
                new_root._creation_state = root_states[i].copy()
                roots.append(new_root)
                is_fresh_root.append(True)
                root_reconstruction_states.append(new_root._creation_state)

        # First: expand any UNEXPANDED roots and collect their initial evaluations
        pending_roots = []
        for idx, (root, root_state, mcts, fresh) in enumerate(
            zip(roots, root_states, mcts_instances, is_fresh_root)
        ):
            if root_state.done:
                continue
            if not fresh:
                # Already expanded from previous step — skip root evaluation
                continue
            pending_roots.append((root, root_state, mcts, idx))

        if pending_roots:
            # Batch evaluate roots
            root_states_list = [s for _, s, _, _ in pending_roots]
            root_eval = self._batch_evaluate_states(root_states_list)

            # Process root evaluations (only for fresh roots)
            for (root, root_state, mcts, idx), action_priors, value in zip(
                pending_roots, root_eval['action_priors'], root_eval['values']
            ):
                if not action_priors:
                    continue

                # Cache results in root node
                root.is_terminal = False
                root.eval_value = value
                # AlphaZero root noise: only on TRUE game roots (first step of a draft)
                is_game_root = prev_roots[idx] is None
                if self.dirichlet_epsilon > 0 and is_game_root:
                    action_priors = self._apply_root_noise(action_priors)
                root.action_priors = action_priors

                # Expand root
                mcts._expand_node_from_cache(root)

        # Now run the main simulation loop with batch evaluations
        round_count = 0
        max_rounds = self.num_simulations * 2  # Safety limit
        for _ in range(max_rounds):
            round_count += 1

            # Collect all pending evaluations from all trees
            all_pending = []
            for mcts, root, root_state, recon_state in zip(
                mcts_instances, roots, root_states, root_reconstruction_states
            ):
                pending = mcts._collect_pending_evaluations(root, recon_state)
                all_pending.extend(pending)

            if not all_pending:
                break  # All simulations complete

            # Collect states for batch evaluation
            eval_states = []
            valid_pending = []

            for pending in all_pending:
                # Reconstruct state for this node.
                # search_root: the root of this particular simulation's tree.
                # With persistent trees, this ensures we only replay actions
                # below the current search root (not the ancestor chain).
                state = mcts._reconstruct_state(pending.root_state, pending.node,
                                                search_root=pending.search_root)

                if state.done:
                    # Terminal state - handle immediately
                    pending.node.is_terminal = True
                    reward = state.get_reward(self.oracle)
                    if reward is None:
                        reward = 0.5
                    mapped = 2.0 * reward - 1.0  # [-1, 1]
                    if state.is_radiant_turn:
                        pending.node.terminal_value = mapped
                    else:
                        pending.node.terminal_value = -mapped
                    pending.node.eval_value = pending.node.terminal_value
                    # Backpropagate immediately
                    mcts._backpropagate(pending.path, pending.node.terminal_value)
                else:
                    # Need model evaluation
                    eval_states.append(state)
                    valid_pending.append((pending, state, mcts))

            if not eval_states:
                continue

            # Batch evaluate all non-terminal states (includes logits -> action_priors)
            eval_result = self._batch_evaluate_states(eval_states)

            # Process evaluations
            for (pending, state, mcts), action_priors, value in zip(valid_pending, eval_result['action_priors'], eval_result['values']):
                # Cache results
                pending.node.is_terminal = False
                pending.node.eval_value = value
                pending.node.action_priors = action_priors
                pending.node._eval_queued = False  # Allow re-queuing in future rounds

                # Expand and backpropagate
                if action_priors:
                    mcts._expand_node_from_cache(pending.node)
                mcts._backpropagate(pending.path, value)

        total_elapsed = time.perf_counter() - total_start
        if hasattr(self, '_last_timing'):
            self._last_timing['total_ms'] = total_elapsed * 1000
            self._last_timing['rounds'] = round_count
            self._last_timing['num_states'] = len(root_states)
            self._last_timing['num_simulations'] = self.num_simulations

        # Finalize all searches: extract decisions + next_root for persistence
        raw_results = mcts_instances[0]._finalize_searches(roots)
        results = []
        for i, (action, probs) in enumerate(raw_results):
            if action is not None and action in roots[i].children:
                next_root = roots[i].children[action]
            else:
                next_root = None
            results.append((action, probs, next_root))
        return results

    def _batch_evaluate_states(
        self,
        states: List[BPState],
    ) -> Dict[str, Any]:
        """Batch evaluate multiple BP states: model forward + batched logits -> action_priors.

        Returns:
            {'action_priors': [...], 'values': [...]}
        """
        if not states:
            return {'action_priors': [], 'values': []}

        eval_start = time.perf_counter()

        # Group states by their action history length FIRST
        # This allows us to use batch_to_dict() efficiently
        groups = {}  # history_length -> (list_of_indices, list_of_states)
        for idx, state in enumerate(states):
            hist_len = len(state.history["teams"])
            if hist_len not in groups:
                groups[hist_len] = ([], [])
            groups[hist_len][0].append(idx)
            groups[hist_len][1].append(state)

        batch_size = len(states)
        all_logits = [None] * batch_size
        all_values = [None] * batch_size

        # Evaluate each group separately using batch_to_dict()
        for hist_len, (indices, group_states) in groups.items():
            if len(group_states) == 1:
                # Single state: use regular to_dict()
                state_dict = group_states[0].to_dict(device=self.device)
                logits, value = self.agent(state_dict)
                all_logits[indices[0]] = logits[0]
                all_values[indices[0]] = value[0, 0].item()
            else:
                # Multiple states with same history length: use batch_to_dict()
                # This eliminates 200k+ torch.tensor() calls!
                batch_dict = BPState.batch_to_dict(group_states, device=self.device)
                logits_batch, value_batch = self.agent(batch_dict)
                for i, orig_idx in enumerate(indices):
                    all_logits[orig_idx] = logits_batch[i]
                    all_values[orig_idx] = value_batch[i, 0].item()

        # === BATCHED logits -> action_priors (the big win) ===
        # Stack all logits into one tensor: [B, NUM_HEROES]
        logits_stacked = torch.stack(all_logits, dim=0)  # [B, NUM_HEROES]

        # --- OPTIMIZED mask building using precomputed static mask ---
        # Start with precomputed static mask, then only mask out used heroes
        state_valid_actions = []
        # Base static mask: [NUM_HEROES+1], index 0 = hero_id 0
        # We need [batch_size, NUM_HEROES], index 0 = hero_id 1
        static_mask_slice = STATIC_HERO_MASK[1:NUM_HEROES+1].to(self.device)  # [NUM_HEROES]
        mask = static_mask_slice.unsqueeze(0).repeat(batch_size, 1).clone()  # [batch_size, NUM_HEROES]
        for i, state in enumerate(states):
            valid_actions = state.get_valid_actions()
            state_valid_actions.append(valid_actions)
            # Only negate USED heroes (few iterations)
            used = state.radiant_heroes + state.dire_heroes + state.radiant_bans + state.dire_bans
            for h in used:
                if 1 <= h <= NUM_HEROES:
                    mask[i, h - 1] = -1e9

        # Batched masked softmax: each row normalizes over valid actions
        batch_masked_logits = logits_stacked + mask
        batch_probs = F.softmax(batch_masked_logits, dim=-1)  # [B, NUM_HEROES]

        # --- OPTIMIZATION: Move to CPU ONCE before per-state extraction ---
        # This avoids GPU sync on every .tolist() call below
        batch_probs_cpu = batch_probs.cpu()

        # Extract per-state action priors (reuse cached valid_actions)
        action_priors_list = []
        for i, valid_actions in enumerate(state_valid_actions):
            valid_hero_indices = [h - 1 for h in valid_actions]
            valid_probs = batch_probs_cpu[i, valid_hero_indices]

            if len(valid_actions) > self.top_k:
                topk_vals, topk_idx = torch.topk(valid_probs, self.top_k)
                pruned_actions = [valid_actions[j] for j in topk_idx.tolist()]
                pruned_probs = topk_vals / topk_vals.sum()
            else:
                pruned_actions = valid_actions
                pruned_probs = valid_probs / valid_probs.sum()

            action_priors_list.append(list(zip(pruned_actions, pruned_probs.tolist())))

        # Detailed logging
        eval_elapsed = time.perf_counter() - eval_start
        if hasattr(self, '_last_timing'):
            calls = self._last_timing.get('batch_eval_calls', 0) + 1
            self._last_timing['batch_eval_calls'] = calls
            key = f'batch_eval_call_{calls}'
            self._last_timing[key] = {
                'states': batch_size,
                'groups': len(groups),
                'eval_ms': eval_elapsed * 1000,
            }
            self._last_timing['total_eval_states'] = self._last_timing.get('total_eval_states', 0) + batch_size
            self._last_timing['total_eval_groups'] = self._last_timing.get('total_eval_groups', 0) + len(groups)

        return {'action_priors': action_priors_list, 'values': all_values}

    def _collate_states_same_length(
        self,
        state_dicts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Collate multiple state dicts that have the SAME action history length.

        All states in the list must have identical history tensor shapes.
        """
        def collate_recursive(dicts_list):
            if not dicts_list:
                return {}

            first = dicts_list[0]
            if isinstance(first, dict):
                batch = {}
                for key in first.keys():
                    values = [d[key] for d in dicts_list]
                    batch[key] = collate_recursive(values)
                return batch
            elif isinstance(first, torch.Tensor):
                return torch.cat(dicts_list, dim=0)
            else:
                return dicts_list

        return collate_recursive(state_dicts)

    def _logits_to_action_priors(
        self,
        logits: torch.Tensor,
        valid_actions: List[int],
        state: BPState,
    ) -> List[Tuple[int, float]]:
        """Convert logits to pruned action priors list."""
        if not valid_actions:
            return []

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

        return list(zip(pruned_actions, pruned_probs.tolist()))


# Convenience function for single search (backward compatible)
@torch.no_grad()
def search_single(
    agent,
    oracle,
    root_state: BPState,
    c_puct=1.5,
    num_simulations=64,
    top_k=20,
) -> Tuple[int, Dict[int, float]]:
    """Run batched MCTS for a single state (backward compatible interface)."""
    engine = BatchMCTSEngine(
        agent, oracle,
        c_puct=c_puct,
        num_simulations=num_simulations,
        top_k=top_k,
    )
    results = engine.search_batch([root_state])
    action, probs, _ = results[0]
    return action, probs


if __name__ == "__main__":
    print("=" * 70)
    print("Batched MCTS - INTERNAL TIMING & CORRECTNESS TEST")
    print("=" * 70)

    import time as time_module

    # Deterministic agent for reproducible comparison
    class DummyAgent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._dummy = torch.nn.Parameter(torch.tensor(0.0))
            self.forward_count = 0

        def forward(self, state):
            self.forward_count += 1
            B = state["radiant_player_feats"].shape[0]
            # Deterministic but varied logits
            logits = torch.zeros(B, NUM_HEROES)
            for b in range(B):
                for i in range(NUM_HEROES):
                    logits[b, i] = 10.0 - (i % 20) * 0.3
            value = torch.full((B, 1), 0.5)
            return logits, value

    class DummyOracle:
        def predict(self, r, d, rp, dp):
            import numpy as np
            return np.array([[0.6]])

    from utils.raw_data import get_valid_hero_ids

    NUM_SIMULATIONS = 16
    TOP_K = 10
    BATCH_SIZES = [1, 2, 4, 8]

    player_feats = [[0.0] * NUM_HEROES for _ in range(5)]

    print(f"\nConfiguration: num_simulations={NUM_SIMULATIONS}, top_k={TOP_K}")
    print("\n" + "-" * 70)

    for batch_size in BATCH_SIZES:
        print(f"\n>>> Testing batch_size={batch_size}")

        # Create states
        states = []
        for _ in range(batch_size):
            s = BPState([], [], player_feats, player_feats, is_radiant_turn=True, step_idx=0)
            states.append(s)

        # --- Individual MCTS (baseline) ---
        agent = DummyAgent()
        oracle = DummyOracle()
        agent.forward_count = 0

        ind_start = time_module.perf_counter()
        ind_results = []
        for s in states:
            action, probs = search_single(agent, oracle, s, num_simulations=NUM_SIMULATIONS, top_k=TOP_K)
            ind_results.append((action, probs))
        ind_time_ms = (time_module.perf_counter() - ind_start) * 1000
        ind_forwards = agent.forward_count

        # --- Batched MCTS ---
        agent = DummyAgent()
        oracle = DummyOracle()
        agent.forward_count = 0

        engine = BatchMCTSEngine(agent, oracle, num_simulations=NUM_SIMULATIONS, top_k=TOP_K)
        engine._last_timing = {}

        batch_start = time_module.perf_counter()
        batch_results_full = engine.search_batch(states)
        batch_results = [(a, p) for a, p, _ in batch_results_full]
        batch_time_ms = (time_module.perf_counter() - batch_start) * 1000
        batch_forwards = agent.forward_count

        # Print timing details
        timing = engine._last_timing
        print(f"  [INDIVIDUAL] time={ind_time_ms:8.2f}ms | forward_passes={ind_forwards}")
        print(f"  [BATCHED]    time={batch_time_ms:8.2f}ms | forward_passes={batch_forwards}")
        if ind_time_ms > 0:
            print(f"  Speedup: {ind_time_ms / batch_time_ms:.2f}x")
        if ind_forwards > 0:
            reduction = 100 - (batch_forwards / ind_forwards * 100)
            print(f"  Forward pass reduction: {reduction:.1f}%")
        print(f"  MCTS rounds: {timing.get('rounds', 0)}")
        print(f"  Total batch_eval calls: {timing.get('batch_eval_calls', 0)}")
        print(f"  Total eval states: {timing.get('total_eval_states', 0)}")
        print(f"  Total eval groups: {timing.get('total_eval_groups', 0)}")
        print(f"  Total forward time: {timing.get('total_forward_ms', 0):.2f}ms")

        # Print first few batch eval calls for detail
        for call_idx in range(1, min(6, timing.get('batch_eval_calls', 0) + 1)):
            key = f'batch_eval_call_{call_idx}'
            if key in timing:
                call = timing[key]
                print(f"    Call {call_idx}: {call['states']} states -> {call['groups']} groups | eval={call['eval_ms']:.2f}ms")

        # Correctness check
        actions_match = all(
            ind_results[i][0] == batch_results[i][0]
            for i in range(batch_size)
        )
        print(f"  Actions match: {actions_match}")

    print("\n" + "=" * 70)
    print("INTERNAL TEST COMPLETE")
    print("=" * 70)
