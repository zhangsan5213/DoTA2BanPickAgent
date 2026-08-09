"""BP Simulator for Dash app - wraps the actual BP agent and MCTS."""

import os
import sys
import torch
import random
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from utils.bp_env import BPState, NUM_HEROES
from utils.raw_data import get_valid_hero_ids


class BPSimulator:
    """Simulator for BP visualization."""

    def __init__(self, agent=None, oracle=None):
        self.agent = agent
        self.oracle = oracle
        self.state = None
        self.trajectory = []
        self.mcts = None
        self.current_mcts_tree = None
        self.hero_id_to_name = {}
        self._load_hero_names()

    def _load_hero_names(self):
        """Load hero ID to name mapping."""
        try:
            import pandas as pd
            df = pd.read_excel("data/hero_features.xlsx")

            def clean_name(name):
                if name.startswith("npc_dota_hero_"):
                    return name[len("npc_dota_hero_"):]
                return name

            self.hero_id_to_name = {
                int(row["id"]): clean_name(row["name"])
                for _, row in df.iterrows()
            }
        except Exception as e:
            print(f"Warning: Could not load hero names: {e}")

    def generate_sample(self):
        """Generate a random player preference sample."""
        # Create random player preferences: [5, NUM_HEROES] with values between 0.45 and 0.7
        r_players = np.random.uniform(0.45, 0.7, (5, NUM_HEROES)).tolist()
        d_players = np.random.uniform(0.45, 0.7, (5, NUM_HEROES)).tolist()
        return {
            'r_players': r_players,
            'd_players': d_players
        }

    def initialize(self, sample=None):
        """Initialize with player preferences."""
        if sample is None:
            sample = self.generate_sample()

        self.state = BPState(
            [],
            [],
            sample["r_players"],
            sample["d_players"],
            radiant_bans=[],
            dire_bans=[],
            is_radiant_turn=True,
            step_idx=0,
        )
        self.trajectory = [self._capture_state()]
        return self.trajectory[0]

    def _get_phase_name(self, step_idx):
        """Get phase name for step index."""
        phases = [
            ("Ban 1", 0, 4),
            ("Pick 1", 4, 8),
            ("Ban 2", 8, 12),
            ("Pick 2", 12, 16),
            ("Ban 3", 16, 18),
            ("Pick 3", 18, 20),
        ]
        for phase_name, start, end in phases:
            if start <= step_idx < end:
                return phase_name
        return "Complete"

    def _format_history(self):
        """Format history for visualization."""
        from dash_app.components.bp_stage import CM_SEQUENCE

        history = []
        for i in range(len(self.state.history["heroes"])):
            team = self.state.history["teams"][i]
            action = "pick" if self.state.history["actions"][i] == 1 else "ban"
            hero_id = self.state.history["heroes"][i] + 1  # convert to 1-based
            history.append({
                "team": team,
                "action": action,
                "hero_id": hero_id,
                "step_idx": i
            })
        return history

    def _capture_state(self):
        """Return state dict for visualization."""
        return {
            "step_idx": self.state.step_idx,
            "phase": self._get_phase_name(self.state.step_idx),
            "radiant_heroes": list(self.state.radiant_heroes),
            "dire_heroes": list(self.state.dire_heroes),
            "radiant_bans": list(self.state.radiant_bans),
            "dire_bans": list(self.state.dire_bans),
            "radiant_players": [list(row) for row in self.state.radiant_players],
            "dire_players": [list(row) for row in self.state.dire_players],
            "history": self._format_history(),
            "current_actor": 0 if self.state.is_radiant_turn else 1,
            "current_action_type": self.state.get_current_action_type(),
            "done": self.state.done
        }

    def _agent_select(self, use_mcts=False, mcts_callback=None):
        """Let agent select an action."""
        if self.agent is None:
            # Random selection if no agent
            valid = self.state.get_valid_actions()
            return random.choice(valid) if valid else None

        device = next(self.agent.parameters()).device
        state_dict = self.state.to_dict(device=device)

        with torch.no_grad():
            action_logits, _ = self.agent(state_dict)

        # Mask valid actions
        valid_actions = self.state.get_valid_actions()
        mask = torch.full((NUM_HEROES,), -1e9, device=action_logits.device)
        all_valid_ids = get_valid_hero_ids()
        for h in all_valid_ids:
            if h <= NUM_HEROES:
                mask[h - 1] = 0.0
        used = set(
            self.state.radiant_heroes + self.state.dire_heroes +
            self.state.radiant_bans + self.state.dire_bans
        )
        for h in used:
            if h <= NUM_HEROES:
                mask[h - 1] = -1e9
        action_logits = action_logits + mask

        hero_id = torch.argmax(action_logits, dim=-1).item() + 1
        return hero_id

    def step(self, use_mcts=False, mcts_callback=None):
        """Execute one step, capture state."""
        if self.state.done:
            return True

        hero_id = self._agent_select(use_mcts=use_mcts, mcts_callback=mcts_callback)

        if hero_id is None:
            return True

        self.state.step(hero_id)
        self.trajectory.append(self._capture_state())
        return self.state.done

    def get_trajectory(self):
        """Get full trajectory."""
        return self.trajectory

    def get_state_at_step(self, step_idx):
        """Get state at specific step."""
        if 0 <= step_idx < len(self.trajectory):
            return self.trajectory[step_idx]
        return None

    def get_mcts_tree_cytoscape(self):
        """Get MCTS tree in Cytoscape format.

        Note: This is a placeholder. Actual implementation would
        traverse the real MCTS tree.
        """
        from dash_app.components.mcts_tree import create_empty_mcts_tree
        return create_empty_mcts_tree()

    def run_full_simulation(self, use_mcts=False, mcts_callback=None):
        """Run full BP simulation."""
        done = False
        while not done:
            done = self.step(use_mcts=use_mcts, mcts_callback=mcts_callback)
        return self.trajectory
