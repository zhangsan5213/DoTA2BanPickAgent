"""
BP Rollout收集器 - 收集trajectory并计算returns和advantages
"""
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from bp_framework.environment import BPState, BPAction, ActionType


@dataclass
class BPTransition:
    """单个BP转移（一步）"""
    # 观测
    hero_ids: torch.Tensor           # [seq_len]
    action_types: torch.Tensor       # [seq_len]
    teams: torch.Tensor              # [seq_len]
    positions: torch.Tensor          # [seq_len]
    seq_mask: torch.Tensor           # [seq_len]
    action_mask: torch.Tensor        # [num_heroes]
    
    # 动作
    action: int                      # 选择的英雄ID (1-based)
    action_idx: int                  # 选择的索引 (0-based)
    
    # 策略信息
    log_prob: float                  # 动作的对数概率
    value: float                     # 状态价值估计
    
    # 执行动作的阵营（用于多智能体场景）
    acting_team: int                 # 0=radiant, 1=dire
    action_type: int                 # 0=ban, 1=pick
    
    # 奖励（在episode结束后计算）
    reward: float = 0.0


@dataclass
class BPRollout:
    """一次完整的BP轨迹"""
    transitions: List[BPTransition] = field(default_factory=list)
    
    # 最终状态
    final_radiant_picks: List[int] = field(default_factory=list)
    final_dire_picks: List[int] = field(default_factory=list)
    
    # 计算结果（由compute_returns_and_advantages填充）
    returns: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    
    def add_transition(self, transition: BPTransition):
        """添加一个转移"""
        self.transitions.append(transition)
    
    def set_final_picks(self, radiant: List[int], dire: List[int]):
        """设置最终的pick结果"""
        self.final_radiant_picks = radiant
        self.final_dire_picks = dire
    
    def compute_returns_and_advantages(
        self,
        final_value: float = 0.0,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        use_gae: bool = True,
    ) -> Tuple[List[float], List[float]]:
        """
        计算returns和advantages
        
        Args:
            final_value: 最终状态的价值（通常BP结束时为0，或用oracle评估）
            gamma: 折扣因子
            gae_lambda: GAE lambda参数
            use_gae: 是否使用GAE，False时使用简单TD
        
        Returns:
            returns: 每个状态的折扣回报
            advantages: 每个状态的优势估计
        """
        num_steps = len(self.transitions)
        if num_steps == 0:
            return [], []
        
        # 提取values和rewards
        values = np.array([t.value for t in self.transitions] + [final_value])
        rewards = np.array([t.reward for t in self.transitions])
        
        returns = np.zeros(num_steps)
        advantages = np.zeros(num_steps)
        
        if use_gae:
            # GAE (Generalized Advantage Estimation)
            gae = 0
            for t in reversed(range(num_steps)):
                # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
                delta = rewards[t] + gamma * values[t + 1] - values[t]
                gae = delta + gamma * gae_lambda * gae
                advantages[t] = gae
                returns[t] = advantages[t] + values[t]
        else:
            # 简单TD: return = r + gamma * V(s')
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    returns[t] = rewards[t] + gamma * final_value
                else:
                    returns[t] = rewards[t] + gamma * returns[t + 1]
                advantages[t] = returns[t] - values[t]
        
        self.returns = returns.tolist()
        self.advantages = advantages.tolist()
        
        # 更新transitions
        for i, trans in enumerate(self.transitions):
            trans.reward = rewards[i]  # 确保reward是最新的
        
        return self.returns, self.advantages
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        获取batch格式的数据（用于训练）
        
        Returns:
            包含所有transition的batch tensor
        """
        if len(self.transitions) == 0:
            return {}
        
        # 找到最大序列长度（用于padding）
        max_seq_len = max(len(t.hero_ids) for t in self.transitions)
        num_heroes = len(self.transitions[0].action_mask)
        
        batch_size = len(self.transitions)
        
        # 初始化tensors
        hero_ids_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        action_types_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        teams_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        positions_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_mask_batch = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        action_mask_batch = torch.zeros((batch_size, num_heroes), dtype=torch.float32)
        
        actions = torch.zeros(batch_size, dtype=torch.long)
        log_probs = torch.zeros(batch_size, dtype=torch.float32)
        values = torch.zeros(batch_size, dtype=torch.float32)
        rewards = torch.zeros(batch_size, dtype=torch.float32)
        returns = torch.zeros(batch_size, dtype=torch.float32)
        advantages = torch.zeros(batch_size, dtype=torch.float32)
        
        # 填充数据
        for i, trans in enumerate(self.transitions):
            seq_len = len(trans.hero_ids)
            if seq_len > 0:
                hero_ids_batch[i, :seq_len] = trans.hero_ids
                action_types_batch[i, :seq_len] = trans.action_types
                teams_batch[i, :seq_len] = trans.teams
                positions_batch[i, :seq_len] = trans.positions
                seq_mask_batch[i, :seq_len] = trans.seq_mask
            
            action_mask_batch[i] = trans.action_mask
            actions[i] = trans.action_idx  # 0-based index for cross entropy
            log_probs[i] = trans.log_prob
            values[i] = trans.value
            rewards[i] = trans.reward
            
            if len(self.returns) > 0:
                returns[i] = self.returns[i]
                advantages[i] = self.advantages[i]
        
        return {
            'hero_ids': hero_ids_batch,
            'action_types': action_types_batch,
            'teams': teams_batch,
            'positions': positions_batch,
            'seq_mask': seq_mask_batch,
            'action_mask': action_mask_batch,
            'actions': actions,
            'old_log_probs': log_probs,
            'old_values': values,
            'rewards': rewards,
            'returns': returns,
            'advantages': advantages,
        }
    
    def __len__(self):
        return len(self.transitions)


class RolloutBuffer:
    """多个rollout的缓冲区"""
    
    def __init__(self):
        self.rollouts: List[BPRollout] = []
    
    def add_rollout(self, rollout: BPRollout):
        """添加一个完整的rollout"""
        self.rollouts.append(rollout)
    
    def clear(self):
        """清空缓冲区"""
        self.rollouts.clear()
    
    def get_all_batches(self) -> List[Dict[str, torch.Tensor]]:
        """获取所有rollout的batch数据"""
        return [r.get_batch() for r in self.rollouts if len(r) > 0]
    
    def compute_all_advantages(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        use_gae: bool = True,
    ):
        """为所有rollout计算advantages"""
        for rollout in self.rollouts:
            rollout.compute_returns_and_advantages(
                final_value=0.0,  # BP结束时没有后续价值
                gamma=gamma,
                gae_lambda=gae_lambda,
                use_gae=use_gae,
            )
    
    def __len__(self):
        return len(self.rollouts)
    
    def total_transitions(self) -> int:
        """总转移数"""
        return sum(len(r) for r in self.rollouts)
