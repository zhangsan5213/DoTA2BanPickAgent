"""Loss computation for PPO training."""

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from functools import lru_cache

from utils.bp_env import compute_gae, ppo_loss, normalize_advantages, compute_value_loss
from utils.raw_data import NUM_HEROES, get_valid_hero_ids
from utils.device import DEVICE


def compute_entropy(logits: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Compute policy entropy.

    Args:
        logits: Raw logits [num_actions]
        mask: Optional mask, used heroes are -inf

    Returns:
        entropy: Policy entropy (scalar)
    """
    if mask is not None:
        logits = logits + mask
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    return entropy


def compute_kl_divergence(new_log_probs: torch.Tensor, old_log_probs: torch.Tensor) -> float:
    """Compute KL divergence (true KL from old to new).

    KL(old || new) = sum_{a} old_policy(a) * (log old_policy(a) - log new_policy(a))

    Args:
        new_log_probs: New policy log probabilities
        old_log_probs: Old policy log probabilities

    Returns:
        KL divergence
    """
    # Compute in probability space for numerical stability
    old_probs = torch.exp(old_log_probs)
    kl = (old_probs * (old_log_probs - new_log_probs)).sum()
    return kl.item()


class LossComputer:
    """Computes PPO losses for a rollout."""

    def __init__(self, agent, value_loss_coeff: float = 2.0,
                 entropy_loss_coeff: float = 0.03, clip_eps: float = 0.2):
        """
        Args:
            agent: The agent model
            value_loss_coeff: Coefficient for value loss
            entropy_loss_coeff: Coefficient for entropy loss
            clip_eps: Clipping epsilon for PPO
        """
        self.agent = agent
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.clip_eps = clip_eps
        self.valid_hero_ids = get_valid_hero_ids()

        # Precompute valid hero mask (cached)
        self._base_mask = self._create_base_mask()

    def _create_base_mask(self) -> torch.Tensor:
        """Create base mask with only valid heroes allowed."""
        mask = torch.full((NUM_HEROES,), -1e9, device=DEVICE)
        for h in self.valid_hero_ids:
            if h <= NUM_HEROES:
                mask[h - 1] = 0.0
        return mask

    def compute(self, rollout: Dict) -> tuple:
        """Compute losses for a single rollout.

        Args:
            rollout: Dictionary containing rollout data

        Returns:
            Tuple of (loss, policy_loss, value_loss, entropy_loss, kl_div)
        """
        valid_mask = rollout['valid_mask'].to(DEVICE)
        actions = rollout['actions'].to(DEVICE)
        old_log_probs = rollout['log_probs'].to(DEVICE)
        values = rollout['values'].to(DEVICE)
        rewards = rollout['rewards'].to(DEVICE)

        actions = actions[valid_mask]
        old_log_probs = old_log_probs[valid_mask]

        # Compute GAE
        T = len(rewards)
        dones = torch.zeros(T, device=DEVICE)
        advantages, returns = compute_gae(
            rewards.unsqueeze(-1),
            values.unsqueeze(-1),
            dones.unsqueeze(-1),
            normalize_returns=True
        )
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)

        # Normalize advantages
        advantages = normalize_advantages(advantages)

        # Compute all policy outputs in a single pass (no duplicate forward passes)
        new_log_probs, new_values, entropies = self._compute_policy_outputs_single_pass(
            rollout['states'], actions, valid_mask
        )

        # Compute losses
        policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages)

        old_values_filtered = values[:-1][valid_mask]
        returns_filtered = returns[valid_mask]
        value_loss = compute_value_loss(
            new_values, old_values_filtered, returns_filtered,
            clip_eps=self.clip_eps, use_clipping=True
        )

        kl_div = compute_kl_divergence(new_log_probs, old_log_probs)

        # Use precomputed entropies
        entropy_loss = -entropies.mean()

        # Combined loss
        loss = policy_loss + self.value_loss_coeff * value_loss + self.entropy_loss_coeff * entropy_loss

        return loss, policy_loss, value_loss, entropy_loss, kl_div

    def _compute_policy_outputs_single_pass(
        self, states, actions, valid_mask
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute new log probabilities, values, and entropies in a single forward pass.

        This avoids calling agent(state) twice for the same state.

        Returns:
            Tuple of (new_log_probs, new_values, entropies)
        """
        new_log_probs_list = []
        new_values = []
        entropies_list = []

        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        action_idx = 0

        for idx in valid_indices:
            state = states[idx]
            logits, v = self.agent(state)

            # Create action mask
            mask = self._base_mask.clone()
            heroes = state['action_history']['heroes']
            used = set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
            for h in used:
                if h < NUM_HEROES:
                    mask[h] = -1e9

            logits = logits + mask

            # Compute log prob for the action
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs_list.append(dist.log_prob(actions[action_idx]))
            new_values.append(v.squeeze(-1))

            # Compute entropy (used for entropy loss)
            entropy = -(probs * torch.log_softmax(logits, dim=-1)).sum()
            entropies_list.append(entropy)

            action_idx += 1

        new_log_probs = torch.stack(new_log_probs_list)
        new_values = torch.cat(new_values)
        entropies = torch.stack(entropies_list)

        return new_log_probs, new_values, entropies

    def _create_action_mask(self, state) -> torch.Tensor:
        """Create action mask for valid and used heroes."""
        mask = self._base_mask.clone()

        # Block used heroes
        heroes = state['action_history']['heroes']
        used = set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
        for h in used:
            if h < NUM_HEROES:
                mask[h] = -1e9

        return mask
