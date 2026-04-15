"""Loss computation for PPO training (GITCGRL-inspired design)."""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from utils.bp_env import ppo_loss, normalize_advantages, compute_value_loss
from utils.raw_data import NUM_HEROES, get_valid_hero_ids
from utils.device import DEVICE


def compute_entropy(
    logits: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
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


def compute_kl_divergence(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor
) -> float:
    """Compute approximate KL divergence (PPO standard).

    approx_kl = mean((ratio - 1) - log(ratio))
    This is more stable than true KL when using sampled actions.

    Args:
        new_log_probs: New policy log probabilities
        old_log_probs: Old policy log probabilities

    Returns:
        Approximate KL divergence
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    approx_kl = ((ratio - 1) - ratio.log()).mean()
    return approx_kl.item()


class LossComputer:
    """Computes PPO losses for rollouts (GITCGRL-style)."""

    def __init__(
        self,
        agent,
        value_loss_coeff: float = 0.5,  # GITCGRL default
        entropy_loss_coeff: float = 0.01,  # GITCGRL default
        clip_eps: float = 0.2,
        value_clip_eps: float = 0.2,
    ):
        """
        Args:
            agent: The agent model
            value_loss_coeff: Coefficient for value loss (GITCGRL: 0.5)
            entropy_loss_coeff: Coefficient for entropy loss (GITCGRL: 0.01)
            clip_eps: Clipping epsilon for PPO
            value_clip_eps: Value function clipping epsilon
        """
        self.agent = agent
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self.clip_eps = clip_eps
        self.value_clip_eps = value_clip_eps
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

    def prepare_rollout(self, rollout_data: Dict) -> Optional[Dict]:
        """Prepare a single rollout: compute GAE per team and return flattened valid data.

        Args:
            rollout_data: Dictionary containing prepared rollout tensors

        Returns:
            Dictionary with flattened valid tensors, or None if empty.
        """
        valid_mask = rollout_data["valid_mask"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["old_log_probs"]
        values = rollout_data["values"]
        rewards = rollout_data["rewards"]
        states = rollout_data["states"]
        step_teams = rollout_data.get("step_teams")

        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        if len(valid_indices) == 0:
            return None

        actions_valid = actions[valid_mask]
        old_log_probs_valid = old_log_probs[valid_mask]

        # Map final reward to [-1, 1]
        final_reward = rewards[-1].item() if rewards.numel() > 0 else 0.0
        mapped_reward = 2.0 * final_reward - 1.0

        # values shape: [T+1], last is bootstrap value (not needed for MC returns)
        step_values = values[:-1]

        T = len(states)
        all_advantages = torch.empty(T, device=DEVICE)
        all_returns = torch.empty(T, device=DEVICE)
        all_old_values = torch.empty(T, device=DEVICE)

        # Use Monte Carlo returns for fixed-horizon deterministic draft (20 steps).
        # All steps for a team share the same terminal reward target.
        # No discounting (gamma=1.0) since every step is equally consequential.
        for team_id, team_reward in [(0, mapped_reward), (1, -mapped_reward)]:
            if step_teams is not None and len(step_teams) > 0:
                team_mask = step_teams == team_id
            else:
                # Fallback: treat all steps as Radiant
                team_mask = torch.ones(T, dtype=torch.bool, device=DEVICE) if team_id == 0 else torch.zeros(T, dtype=torch.bool, device=DEVICE)

            team_indices = team_mask.nonzero(as_tuple=True)[0]
            if len(team_indices) == 0:
                continue

            team_step_values = step_values[team_mask]
            # MC return = terminal reward for every step of this team
            returns = torch.full_like(team_step_values, team_reward)
            # Advantage = return - baseline (old value estimate)
            advantages = returns - team_step_values

            all_advantages[team_indices] = advantages
            all_returns[team_indices] = returns
            all_old_values[team_indices] = team_step_values

        # Filter valid data
        advantages_valid = all_advantages[valid_mask]
        returns_valid = all_returns[valid_mask]
        old_values_valid = all_old_values[valid_mask]
        states_valid = [states[i] for i in valid_indices.tolist()]

        result = {
            "states": states_valid,
            "actions": actions_valid,
            "old_log_probs": old_log_probs_valid,
            "advantages": advantages_valid,
            "returns": returns_valid,
            "old_values": old_values_valid,
        }

        if "mcts_policies" in rollout_data and rollout_data["mcts_policies"] is not None:
            result["mcts_policies"] = rollout_data["mcts_policies"][valid_mask]

        return result

    def compute_minibatch(
        self,
        states: List[Dict],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        entropy_coeff: Optional[float] = None,
        mcts_policies: Optional[torch.Tensor] = None,
        mcts_policy_weight: float = 0.0,
    ) -> Optional[tuple]:
        """Compute losses for a minibatch of flattened data.

        Args:
            states: List of state dicts (flattened valid steps)
            actions: [N] action indices
            old_log_probs: [N] old log probabilities
            advantages: [N] advantages
            returns: [N] returns
            old_values: [N] old value estimates
            entropy_coeff: Dynamic entropy coefficient
            mcts_policies: [N, NUM_HEROES] MCTS visit-count policy (optional)
            mcts_policy_weight: Weight for MCTS policy loss vs PPO policy loss

        Returns:
            Tuple of (loss, policy_loss, value_loss, entropy_loss, kl_div, mcts_policy_loss)
        """
        if len(actions) == 0:
            return None

        # Normalize advantages over the minibatch (standard PPO)
        advantages = normalize_advantages(advantages)

        new_log_probs, new_values, entropies, full_logits = self._compute_policy_outputs_optimized(
            states, actions
        )

        # Compute losses
        policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages, clip_eps=self.clip_eps)

        value_loss = compute_value_loss(
            new_values,
            old_values,
            returns,
            clip_eps=self.value_clip_eps,
            use_clipping=True,
        )

        # MCTS policy loss (AlphaZero-style cross-entropy)
        mcts_policy_loss = torch.tensor(0.0, device=DEVICE)
        if mcts_policies is not None and mcts_policy_weight > 0:
            log_probs_all = torch.log_softmax(full_logits, dim=-1)
            mcts_policy_loss = -(mcts_policies * log_probs_all).sum(dim=-1).mean()
            effective_policy_loss = (
                (1 - mcts_policy_weight) * policy_loss
                + mcts_policy_weight * mcts_policy_loss
            )
        else:
            effective_policy_loss = policy_loss

        kl_div = compute_kl_divergence(new_log_probs, old_log_probs)
        entropy_loss = -entropies.mean()

        coeff = entropy_coeff if entropy_coeff is not None else self.entropy_loss_coeff
        loss = effective_policy_loss + self.value_loss_coeff * value_loss + coeff * entropy_loss

        return loss, policy_loss, value_loss, entropy_loss, kl_div, mcts_policy_loss

    def _compute_policy_outputs_optimized(
        self, states: List[Dict], actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """优化版本：按history长度分组，组内batch处理。

        核心优化：相同history长度的states可以安全地组成batch进行单次前向传播，
        避免了为不同长度进行复杂padding的问题。

        Args:
            states: List of state dicts (already filtered to valid steps)
            actions: [N] action indices corresponding to states

        Returns:
            Tuple of (new_log_probs, new_values, entropies, full_logits)
        """
        num_steps = len(states)
        if num_steps == 0:
            return (
                torch.tensor([], device=DEVICE),
                torch.tensor([], device=DEVICE),
                torch.tensor([], device=DEVICE),
                torch.tensor([], device=DEVICE),
            )

        # 按history长度分组
        groups = {}
        for action_idx, state in enumerate(states):
            hist_len = state["action_history"]["heroes"].shape[1] if state["action_history"]["heroes"].dim() > 1 else state["action_history"]["heroes"].shape[0]
            if hist_len not in groups:
                groups[hist_len] = []
            groups[hist_len].append(action_idx)

        # 收集结果
        all_new_log_probs = [None] * num_steps
        all_new_values = [None] * num_steps
        all_entropies = [None] * num_steps
        all_logits = [None] * num_steps

        # 对每个组进行batch处理
        for hist_len, action_indices in groups.items():
            group_states = [states[i] for i in action_indices]

            # 打包成batch
            batch_state = self._pack_states(group_states)

            # 单次前向传播（需要梯度，用于反向传播）
            batch_logits, batch_values = self.agent(batch_state)

            # 批量构建mask并处理
            batch_mask = self._build_batch_action_mask(group_states)
            batch_logits = batch_logits + batch_mask

            # PPO 的 log_prob 计算应基于目标策略（temperature=1.0），
            # 保证 old_log_prob 和 new_log_prob 在同一分布下计算 ratio
            batch_probs = torch.softmax(batch_logits, dim=-1)
            batch_log_probs = torch.log_softmax(batch_logits, dim=-1)

            # 获取对应动作的log_prob
            group_actions = actions[action_indices]
            group_new_log_probs = batch_log_probs.gather(1, group_actions.unsqueeze(1)).squeeze(1)
            group_entropies = -(batch_probs * batch_log_probs).sum(dim=-1)

            # 保存结果
            for i, action_idx in enumerate(action_indices):
                all_new_log_probs[action_idx] = group_new_log_probs[i]
                all_new_values[action_idx] = batch_values[i].squeeze(-1)
                all_entropies[action_idx] = group_entropies[i]
                all_logits[action_idx] = batch_logits[i]

        # 堆叠成tensor
        new_log_probs = torch.stack(all_new_log_probs)
        new_values = torch.stack(all_new_values)
        entropies = torch.stack(all_entropies)
        full_logits = torch.stack(all_logits)

        return new_log_probs, new_values, entropies, full_logits

    def _pack_states(self, states: List[Dict]) -> Dict:
        """将相同history长度的states列表打包成batch。
        
        由于传入的states都有相同的history长度，不需要padding处理。
        """
        batch_size = len(states)
        
        # 提取玩家特征
        r_player_feats = torch.cat([s["radiant_player_feats"] for s in states], dim=0)
        d_player_feats = torch.cat([s["dire_player_feats"] for s in states], dim=0)
        
        # 提取current_actor和current_action
        current_actor = torch.cat([s["current_actor"] for s in states], dim=0)
        current_action = torch.cat([s["current_action"] for s in states], dim=0)
        
        # 提取action_history（所有state长度相同）
        if states[0]["action_history"]["heroes"].shape[1] > 0:
            teams = torch.cat([s["action_history"]["teams"] for s in states], dim=0)
            actions = torch.cat([s["action_history"]["actions"] for s in states], dim=0)
            heroes = torch.cat([s["action_history"]["heroes"] for s in states], dim=0)
        else:
            # 空history
            teams = torch.zeros(batch_size, 0, dtype=torch.long, device=DEVICE)
            actions = torch.zeros(batch_size, 0, dtype=torch.long, device=DEVICE)
            heroes = torch.zeros(batch_size, 0, dtype=torch.long, device=DEVICE)
        
        return {
            "radiant_player_feats": r_player_feats,
            "dire_player_feats": d_player_feats,
            "action_history": {
                "teams": teams,
                "actions": actions,
                "heroes": heroes,
            },
            "current_actor": current_actor,
            "current_action": current_action,
        }

    def _build_batch_action_mask(self, states: List[Dict]) -> torch.Tensor:
        """批量构建action mask。
        
        Args:
            states: 状态列表（相同history长度）
        
        Returns:
            batch_mask: [B, NUM_HEROES] mask张量
        """
        batch_size = len(states)
        batch_mask = self._base_mask.unsqueeze(0).expand(batch_size, -1).clone()
        
        for i, state in enumerate(states):
            # 从state中获取已使用的英雄
            if state["action_history"]["heroes"].numel() > 0:
                # 获取这个state中所有已使用的英雄（包括pick和ban）
                heroes = state["action_history"]["heroes"].flatten()  # [T]
                for h in heroes:
                    h_id = h.item()
                    if h_id < NUM_HEROES:
                        batch_mask[i, h_id] = -1e9
        
        return batch_mask
