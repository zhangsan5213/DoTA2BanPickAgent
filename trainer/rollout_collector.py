"""Rollout collection for training."""

import random
from typing import List, Dict, Any, Optional
import torch

from model.bp_agent import BPTransformerAgent
from model.win_rate_oracle import WinRateOracle
from utils.bp_env import collect_rollout
from utils.device import DEVICE


class RolloutCollector:
    """Collects rollouts for training batches."""

    def __init__(self, agent: BPTransformerAgent, oracle: WinRateOracle,
                 historical_prob: float = 0.6, embed_dim: int = 256,
                 nhead: int = 8, num_layers: int = 4):
        """
        Args:
            agent: Current training agent
            oracle: Win rate oracle model
            historical_prob: Probability of using historical opponent
            embed_dim: Agent embedding dimension
            nhead: Agent number of attention heads
            num_layers: Agent number of transformer layers
        """
        self.agent = agent
        self.oracle = oracle
        self.historical_prob = historical_prob
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers

    def collect_batch(
        self,
        batch_samples: List[Dict[str, Any]],
        checkpoints: List,
        checkpoint_manager,
        batch_idx: int
    ) -> List[Dict[str, Any]]:
        """Collect rollouts for a batch.

        Args:
            batch_samples: List of samples for this batch
            checkpoints: List of available checkpoints
            checkpoint_manager: CheckpointManager instance
            batch_idx: Current batch index

        Returns:
            List of rollouts
        """
        batch_size = len(batch_samples)
        num_hist = int(batch_size * self.historical_prob)

        rollouts = []

        # Historical opponent rollouts
        if num_hist > 0 and checkpoints:
            hist_assignments = self._assign_historical_opponents(
                num_hist, batch_idx, batch_size, len(checkpoints)
            )
            rollouts.extend(self._collect_historical_rollouts(
                hist_assignments, batch_samples, checkpoints, checkpoint_manager
            ))

        # Self-play rollouts
        for i in range(num_hist, batch_size):
            sample = batch_samples[i]
            rollouts.append(collect_rollout(self.agent, self.oracle, sample))

        return rollouts

    def _assign_historical_opponents(
        self, num_hist: int, batch_idx: int, batch_size: int,
        num_checkpoints: int
    ) -> List[tuple]:
        """Assign historical opponents to samples.

        Returns:
            List of (sample_idx, ckpt_idx) tuples
        """
        assignments = []
        for i in range(num_hist):
            sample_idx = i
            ckpt_idx = (batch_idx * batch_size + i) % num_checkpoints
            assignments.append((sample_idx, ckpt_idx))
        return assignments

    def _collect_historical_rollouts(
        self, assignments: List[tuple], batch_samples: List[Dict[str, Any]],
        checkpoints: List, checkpoint_manager
    ) -> List[Dict[str, Any]]:
        """Collect rollouts against historical opponents."""
        rollouts = []

        # Group by checkpoint index
        ckpt_idx_to_samples = {}
        for sample_idx, ckpt_idx in assignments:
            ckpt_path = checkpoints[ckpt_idx][0]
            if ckpt_idx not in ckpt_idx_to_samples:
                ckpt_idx_to_samples[ckpt_idx] = []
            ckpt_idx_to_samples[ckpt_idx].append((sample_idx, ckpt_path))

        # Load each model once and collect all assigned rollouts
        for ckpt_idx, sample_list in ckpt_idx_to_samples.items():
            ckpt_path = sample_list[0][1]

            opponent = checkpoint_manager.load_opponent(ckpt_path)
            if opponent is None:
                # Fallback to self-play if loading fails
                for sample_idx, _ in sample_list:
                    rollouts.append(collect_rollout(self.agent, self.oracle, batch_samples[sample_idx]))
                continue

            for sample_idx, _ in sample_list:
                sample = batch_samples[sample_idx]
                current_side = random.choice(["radiant", "dire"])
                rollout = collect_rollout(
                    self.agent, self.oracle, sample,
                    opponent_agent=opponent, current_side=current_side
                )
                rollouts.append(rollout)

            # Note: opponent is NOT deleted here - it stays in LRU cache

        return rollouts
