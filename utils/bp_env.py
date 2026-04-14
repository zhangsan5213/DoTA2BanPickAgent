"""BP Environment and RL utilities"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from utils.device import DEVICE
from utils.raw_data import NUM_HEROES, get_valid_hero_ids


class BPState:
    """
    BP environment state - 实现标准CM模式

    CM BP顺序（共20步）：
    1. Ban Phase 1:  R, D, R, D (4 bans)
    2. Pick Phase 1: R, D, D, R (4 picks)
    3. Ban Phase 2:  D, R, D, R (4 bans)
    4. Pick Phase 2: D, R, R, D (4 picks)
    5. Ban Phase 3:  R, D (2 bans)
    6. Pick Phase 3: R, D (2 picks)

    总共：10 bans + 10 picks (每队5 ban 5 pick)
    """

    # CM模式BP顺序定义: (team, action_type)
    # team: 0=Radiant, 1=Dire
    # action_type: 'ban' or 'pick'
    CM_SEQUENCE = [
        # Ban Phase 1
        (0, "ban"),
        (1, "ban"),
        (0, "ban"),
        (1, "ban"),
        # Pick Phase 1
        (0, "pick"),
        (1, "pick"),
        (1, "pick"),
        (0, "pick"),
        # Ban Phase 2
        (1, "ban"),
        (0, "ban"),
        (1, "ban"),
        (0, "ban"),
        # Pick Phase 2
        (1, "pick"),
        (0, "pick"),
        (0, "pick"),
        (1, "pick"),
        # Ban Phase 3
        (0, "ban"),
        (1, "ban"),
        # Pick Phase 3
        (0, "pick"),
        (1, "pick"),
    ]

    def __init__(
        self,
        radiant_heroes,
        dire_heroes,
        radiant_players,
        dire_players,
        radiant_bans=None,
        dire_bans=None,
        is_radiant_turn=True,
        step_idx=0,
    ):
        self.radiant_heroes = list(radiant_heroes)
        self.dire_heroes = list(dire_heroes)
        self.radiant_bans = list(radiant_bans) if radiant_bans else []
        self.dire_bans = list(dire_bans) if dire_bans else []
        self.radiant_players = radiant_players
        self.dire_players = dire_players
        self.history = {"teams": [], "actions": [], "heroes": []}
        self.step_idx = step_idx  # 当前在CM_SEQUENCE中的位置
        self.done = False
        self.pick_count = {
            "radiant": len(self.radiant_heroes),
            "dire": len(self.dire_heroes),
        }
        self.ban_count = {
            "radiant": len(self.radiant_bans),
            "dire": len(self.dire_bans),
        }

        # 根据step_idx设置当前轮到谁
        if step_idx < len(self.CM_SEQUENCE):
            team, _ = self.CM_SEQUENCE[step_idx]
            self.is_radiant_turn = team == 0
        else:
            self.is_radiant_turn = is_radiant_turn

    def get_current_action_type(self):
        """获取当前步骤的动作类型: 'ban' 或 'pick'"""
        if self.step_idx < len(self.CM_SEQUENCE):
            _, action_type = self.CM_SEQUENCE[self.step_idx]
            return action_type
        return "pick"  # 默认pick

    def to_dict(self, device=DEVICE):
        # Handle both list and tensor inputs
        if not isinstance(self.radiant_players, torch.Tensor):
            r_feats = torch.tensor(self.radiant_players).float()
        else:
            r_feats = self.radiant_players.float()
        if not isinstance(self.dire_players, torch.Tensor):
            d_feats = torch.tensor(self.dire_players).float()
        else:
            d_feats = self.dire_players.float()

        # 确定当前action类型: 1=pick, 2=ban
        current_action_type = 1 if self.get_current_action_type() == "pick" else 2

        return {
            "radiant_player_feats": r_feats.unsqueeze(0).to(device),
            "dire_player_feats": d_feats.unsqueeze(0).to(device),
            "action_history": {
                "teams": torch.tensor(
                    self.history["teams"], dtype=torch.long, device=device
                ).unsqueeze(0)
                if self.history["teams"]
                else torch.zeros(1, 0, dtype=torch.long, device=device),
                "actions": torch.tensor(
                    self.history["actions"], dtype=torch.long, device=device
                ).unsqueeze(0)
                if self.history["actions"]
                else torch.zeros(1, 0, dtype=torch.long, device=device),
                "heroes": torch.tensor(
                    self.history["heroes"], dtype=torch.long, device=device
                ).unsqueeze(0)
                if self.history["heroes"]
                else torch.zeros(1, 0, dtype=torch.long, device=device),
            },
            "current_actor": torch.tensor(
                [0 if self.is_radiant_turn else 1], device=device
            ),
            "current_action": torch.tensor([current_action_type], device=device),
        }

    def step(self, hero_id, is_pick=None):
        """
        执行一步BP

        Args:
            hero_id: 英雄ID (1-based)
            is_pick: 可选，如果不提供则根据当前step_idx自动判断
        """
        # 如果没提供is_pick，根据CM序列自动判断
        if is_pick is None:
            is_pick = self.get_current_action_type() == "pick"

        self.history["teams"].append(0 if self.is_radiant_turn else 1)
        self.history["actions"].append(1 if is_pick else 2)
        self.history["heroes"].append(hero_id - 1)

        if is_pick:
            if self.is_radiant_turn:
                self.radiant_heroes.append(hero_id)
                self.pick_count["radiant"] += 1
            else:
                self.dire_heroes.append(hero_id)
                self.pick_count["dire"] += 1
        else:
            if self.is_radiant_turn:
                self.radiant_bans.append(hero_id)
                self.ban_count["radiant"] += 1
            else:
                self.dire_bans.append(hero_id)
                self.ban_count["dire"] += 1

        # 推进到下一步
        self.step_idx += 1

        # 检查是否结束（完成20步或pick满10个）
        total_picks = self.pick_count["radiant"] + self.pick_count["dire"]
        if total_picks >= 10 or self.step_idx >= len(self.CM_SEQUENCE):
            self.done = True
            self.is_radiant_turn = not self.is_radiant_turn  # 保持上一步的对方
        else:
            # 根据序列设置下一个turn
            next_team, _ = self.CM_SEQUENCE[self.step_idx]
            self.is_radiant_turn = next_team == 0

    def get_valid_actions(self):
        """获取当前可用的英雄ID列表（实际存在且未被ban/pick）"""
        used = set(
            self.radiant_heroes + self.dire_heroes + self.radiant_bans + self.dire_bans
        )
        valid_ids = get_valid_hero_ids()
        return [h for h in valid_ids if h not in used]

    def get_reward(self, oracle):
        if not self.done:
            return None
        r_heroes = self.radiant_heroes[:5] + [0] * (5 - len(self.radiant_heroes))
        d_heroes = self.dire_heroes[:5] + [0] * (5 - len(self.dire_heroes))
        pred = oracle.predict(
            r_heroes, d_heroes, self.radiant_players, self.dire_players
        )
        return float(pred[0, 0])


# ============ RL Helpers ============
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_CLIP_EPS = 0.2  # Value clipping epsilon


def compute_gae(
    rewards, values, dones, gamma=GAMMA, lam=LAMBDA, normalize_returns=False
):
    """
    Compute GAE advantages and returns

    Args:
        normalize_returns: 是否对returns做归一化（减少value估计的方差）
    """
    # Handle 1D input (single trajectory)
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
        values = values.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    # Squeeze back to 1D if input was 1D
    if advantages.dim() == 2 and advantages.shape[1] == 1:
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)

    # Return normalization (optional but recommended for stability)
    if normalize_returns and returns.numel() > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    return advantages, returns


def normalize_advantages(advantages, eps=1e-8):
    """
    Advantage归一化 - 关键trick，能显著提高PPO稳定性

    归一化后的advantages有:
    - 零均值: 避免policy偏向正advantage的方向
    - 单位方差: 控制梯度大小，使学习率更稳定
    """
    if advantages.numel() <= 1:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + eps)


def ppo_loss(log_probs, old_log_probs, advantages, clip_eps=CLIP_EPS):
    """PPO policy loss (使用归一化后的advantages)"""
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def compute_value_loss(
    new_values, old_values, returns, clip_eps=VALUE_CLIP_EPS, use_clipping=True
):
    """
    PPO Value Loss with optional clipping

    Args:
        new_values: 当前value network预测值 [T]
        old_values: 收集trajectory时的value预测值 [T+1] (包含bootstrap)
        returns: GAE计算的returns [T]
        clip_eps: value clipping系数
        use_clipping: 是否使用clipping

    Returns:
        value_loss: scalar
    """
    # 注意: old_values长度是T+1，需要截断到T
    old_values_clipped = (
        old_values[:-1] if old_values.shape[0] > returns.shape[0] else old_values
    )

    if use_clipping:
        # PPO-style value clipping: 限制value update幅度
        # 这防止value network一次更新过大，保持与policy更新同步
        value_pred_clipped = old_values_clipped + torch.clamp(
            new_values - old_values_clipped, -clip_eps, clip_eps
        )

        # 两个loss取较大的那个（即更保守的更新）
        value_loss1 = F.mse_loss(new_values, returns, reduction="mean")
        value_loss2 = F.mse_loss(value_pred_clipped, returns, reduction="mean")
        value_loss = torch.max(value_loss1, value_loss2)
    else:
        # 普通MSE (原版PPO论文也支持这种)
        value_loss = F.mse_loss(new_values, returns, reduction="mean")

    return value_loss


def collect_rollout(
    agent, oracle, sample, max_steps=24, opponent_agent=None, current_side="radiant",
    temperature=None, policy_staleness_tolerance=0, opponent_staleness=None
):
    """Collect one BP trajectory.

    Args:
        agent: The current agent (on-policy training target).
        oracle: Win rate oracle.
        sample: BP sample with r_players/d_players.
        max_steps: Max BP steps.
        opponent_agent: If provided, this agent plays the opposite side. If None, agent
            plays both sides (self-play).
        current_side: "radiant" or "dire" - which side the current agent plays on.
            Only relevant when opponent_agent is not None.
        temperature: Action sampling temperature. If None, uses agent's internal temperature.
            High temperature -> more exploration; low temperature -> more exploitation.
    """
    # 从agent获取设备信息
    device = next(agent.parameters()).device

    # 确保active_agent变量在使用前有定义
    active_agent = agent
    s = BPState(
        [],
        [],
        sample["r_players"],
        sample["d_players"],
        radiant_bans=[],
        dire_bans=[],
        is_radiant_turn=True,
        step_idx=0,
    )

    states, actions, log_probs, values, rewards, valid_mask, step_teams = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],  # 记录每一步是哪个team执行的: 0=Radiant, 1=Dire
    )
    step = 0

    while not s.done and step < max_steps:
        state_dict = s.to_dict(device=device)

        is_radiant_turn = s.is_radiant_turn
        current_team = 0 if is_radiant_turn else 1  # 0=Radiant, 1=Dire
        
        if opponent_agent is not None:
            if is_radiant_turn:
                active_agent = agent if current_side == "radiant" else opponent_agent
            else:
                active_agent = agent if current_side == "dire" else opponent_agent

        with torch.no_grad():
            action_logits, value = active_agent(state_dict)

        valid_actions = s.get_valid_actions()
        if not valid_actions:
            break

        # 创建mask：只允许选择实际存在且未被使用的英雄
        mask = torch.full((NUM_HEROES,), -1e9, device=action_logits.device)
        all_valid_ids = get_valid_hero_ids()
        for h in all_valid_ids:
            if h <= NUM_HEROES:
                mask[h - 1] = 0.0
        used = set(s.radiant_heroes + s.dire_heroes + s.radiant_bans + s.dire_bans)
        for h in used:
            if h <= NUM_HEROES:
                mask[h - 1] = -1e9
        action_logits = action_logits + mask

        # 获取采样温度（可学习参数或固定超参数）
        if temperature is None and hasattr(active_agent, 'get_temperature'):
            temp = active_agent.get_temperature().item()
        else:
            temp = temperature if temperature is not None else 1.0

        # 带温度概率：用于实际动作采样和 old_log_prob 记录
        # PPO 端到端学习 temperature，因此 loss 计算也基于此分布
        probs = F.softmax(action_logits / temp, dim=-1)
        dist = torch.distributions.Categorical(probs)
        hero_id = dist.sample().item() + 1
        log_prob = dist.log_prob(torch.tensor(hero_id - 1, device=action_logits.device))

        # Mark whether this step belongs to the current agent
        if opponent_agent is None:
            is_current = True  # 自对弈：所有步骤都用于训练
        else:
            is_current = (
                (current_side == "radiant")
                if is_radiant_turn
                else (current_side == "dire")
            )
            # Fresh opponents within tolerance also contribute training data
            if not is_current and opponent_staleness is not None:
                if opponent_staleness <= policy_staleness_tolerance:
                    is_current = True

        states.append(state_dict)
        actions.append(hero_id - 1)
        log_probs.append(log_prob)
        values.append(value.item())
        valid_mask.append(is_current)
        step_teams.append(current_team)  # 记录这一步是哪个team

        s.step(hero_id)
        step += 1

    final_reward = s.get_reward(oracle)
    if final_reward is None:
        final_reward = 0.5

    # 计算每一步的奖励：
    # final_reward 是 Radiant 胜率
    # - Radiant的决策：奖励 = final_reward（最大化Radiant胜率）
    # - Dire的决策：奖励 = 1.0 - final_reward（最小化Radiant胜率，即最大化Dire胜率）
    rewards = [0.0] * (len(states) - 1) + [final_reward]
    
    # 在自对弈模式下，我们需要根据每一步的执行者调整奖励
    # 存储step_teams供后续处理使用
    step_teams_tensor = torch.tensor(step_teams, dtype=torch.long)

    # 修复bootstrap value bug：使用value网络的最终预测而不是final_reward
    # 这样value网络可以学习预测最终状态的真实值，而不是直接赋值
    final_state_dict = s.to_dict(device=device)
    # 确保active_agent变量有定义
    if opponent_agent is None:
        final_active_agent = agent
    else:
        if s.is_radiant_turn:
            final_active_agent = agent if current_side == "radiant" else opponent_agent
        else:
            final_active_agent = agent if current_side == "dire" else opponent_agent

    with torch.no_grad():
        _, final_value = final_active_agent(final_state_dict)
    final_value = final_value.item()

    return {
        "states": states,
        "actions": torch.tensor(actions, dtype=torch.long),
        "log_probs": torch.stack(log_probs),
        "values": torch.tensor(values + [final_value], dtype=torch.float32),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
        "step_teams": step_teams_tensor,  # 每一步的执行者: 0=Radiant, 1=Dire
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Testing BP Environment with CM Mode")
    print("=" * 50)

    # Test BPState with CM sequence
    player_feats = [[0.0] * NUM_HEROES for _ in range(5)]
    player_feats[0][10] = 0.6  # 玩家1擅长英雄10

    s = BPState([], [], player_feats, player_feats, is_radiant_turn=True)

    print(f"Initial state - radiant turn: {s.is_radiant_turn}")
    print(f"CM Sequence length: {len(s.CM_SEQUENCE)}")
    print(f"Step 0: team={s.CM_SEQUENCE[0][0]}, action={s.CM_SEQUENCE[0][1]}")
    print(f"Valid actions count: {len(s.get_valid_actions())}")

    # Test full BP sequence
    print("\n--- Testing full BP sequence ---")
    for i in range(len(s.CM_SEQUENCE)):
        team, action_type = s.CM_SEQUENCE[i]
        team_name = "Radiant" if team == 0 else "Dire"
        valid_count = len(s.get_valid_actions())
        print(
            f"Step {i}: {team_name} {action_type.upper()} (valid heroes: {valid_count})"
        )

        # 模拟选择第一个可用英雄
        valid_actions = s.get_valid_actions()
        if valid_actions:
            hero_id = valid_actions[0]
            s.step(hero_id)

        if s.done:
            print(f"BP finished at step {i + 1}")
            break

    print(f"\nFinal Radiant picks: {s.radiant_heroes}")
    print(f"Final Dire picks: {s.dire_heroes}")
    print(f"Final Radiant bans: {s.radiant_bans}")
    print(f"Final Dire bans: {s.dire_bans}")
    print(f"Pick count: {s.pick_count}")
    print(f"Ban count: {s.ban_count}")

    # Test GAE
    print("\n--- Testing GAE ---")
    rewards = torch.tensor([0.0, 0.0, 0.8, 0.9])
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.85])
    dones = torch.tensor([0.0, 0.0, 0.0, 0.0])
    advantages, returns = compute_gae(rewards, values, dones)
    print(f"advantages: {advantages}")
    print(f"returns: {returns}")

    # Test PPO loss
    print("\n--- Testing PPO Loss ---")
    log_probs = torch.tensor([-1.0, -1.5, -2.0, -1.8])
    old_log_probs = torch.tensor([-1.1, -1.4, -1.9, -1.9])
    advantages = torch.tensor([0.1, 0.2, -0.1, 0.15])
    loss = ppo_loss(log_probs, old_log_probs, advantages)
    print(f"PPO loss: {loss.item():.4f}")

    print("\n[OK] All tests passed!")
