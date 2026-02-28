"""
Reward计算器 - 使用WinRateOracle评估阵容并分配奖励
"""
import torch
import numpy as np
from typing import List, Optional, Dict, Tuple

from model.win_rate_oracle import WinRateOracle
from bp_framework.environment import BPState, Team, ActionType


class RewardCalculator:
    """
    奖励计算器
    
    策略:
    1. 只有最终的pick阶段会产生奖励（由Oracle评估阵容胜率）
    2. ban阶段的奖励为0（或者可以设计成sparse reward）
    3. 支持为每个pick动作分配shaped reward
    """
    
    def __init__(
        self,
        oracle: WinRateOracle,
        device: str = 'cpu',
        reward_type: str = 'final',  # 'final', 'intermediate', 'shaped'
        radiant_bias: float = 0.0,   # 天辉胜率偏置（用于平衡数据集偏差）
    ):
        """
        Args:
            oracle: 胜率预言模型
            device: 计算设备
            reward_type: 奖励类型
                - 'final': 只有最后有奖励
                - 'intermediate': 每个pick动作都有奖励（评估当前阵容）
                - 'shaped': 基于阵容变化分配shaped reward
            radiant_bias: 天辉胜率偏置修正
        """
        self.oracle = oracle
        self.device = device
        self.reward_type = reward_type
        self.radiant_bias = radiant_bias
        
        self.oracle.eval()
        for param in self.oracle.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def evaluate_picks(
        self,
        radiant_picks: List[int],
        dire_picks: List[int],
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ) -> float:
        """
        使用Oracle评估阵容胜率
        
        Returns:
            win_prob: 天辉胜率 (0-1)
        """
        if len(radiant_picks) != 5 or len(dire_picks) != 5:
            raise ValueError("Need 5 picks for each team")
        
        # 转为tensor
        radiant_tensor = torch.tensor([radiant_picks], dtype=torch.long, device=self.device)
        dire_tensor = torch.tensor([dire_picks], dtype=torch.long, device=self.device)
        
        # 使用oracle的predict接口
        win_prob = self.oracle.predict(
            radiant_picks=radiant_tensor,
            dire_picks=dire_tensor,
            radiant_player_feats=radiant_player_feats,
            dire_player_feats=dire_player_feats,
            return_tensor=True,
        )
        
        return win_prob.item() - self.radiant_bias
    
    def calculate_final_rewards(
        self,
        state: BPState,
        acting_team: Team = Team.RADIANT,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        计算最终奖励
        
        Args:
            state: 最终BP状态
            acting_team: 我们控制的阵营（从谁的视角计算奖励）
        
        Returns:
            rewards: 包含'radiant_reward', 'dire_reward', 'win_prob'的字典
        """
        radiant_picks, dire_picks = state.get_final_picks()
        
        if len(radiant_picks) != 5 or len(dire_picks) != 5:
            # BP未完成，奖励为0
            return {
                'radiant_reward': 0.0,
                'dire_reward': 0.0,
                'win_prob': 0.5,
            }
        
        win_prob = self.evaluate_picks(
            radiant_picks, dire_picks,
            radiant_player_feats, dire_player_feats
        )
        
        # 从acting_team视角的奖励
        # 如果控制天辉，reward = win_prob (越高越好)
        # 如果控制夜魇，reward = 1 - win_prob (天辉胜率越低越好)
        radiant_reward = win_prob
        dire_reward = 1.0 - win_prob
        
        return {
            'radiant_reward': radiant_reward,
            'dire_reward': dire_reward,
            'win_prob': win_prob,
        }
    
    def assign_rewards_to_rollout(
        self,
        rollout,
        final_state: BPState,
        acting_team: Team = Team.RADIANT,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ):
        """
        为rollout中的每个transition分配奖励
        
        Args:
            rollout: BPRollout对象
            final_state: 最终BP状态
            acting_team: 我们控制的阵营
        """
        # 计算最终奖励
        rewards_info = self.calculate_final_rewards(
            final_state, acting_team,
            radiant_player_feats, dire_player_feats
        )
        
        final_reward = rewards_info['radiant_reward'] if acting_team == Team.RADIANT else rewards_info['dire_reward']
        
        # 根据reward_type分配奖励
        if self.reward_type == 'final':
            # 只有最后一步有奖励
            for i, trans in enumerate(rollout.transitions):
                if i == len(rollout.transitions) - 1:
                    trans.reward = final_reward
                else:
                    trans.reward = 0.0
        
        elif self.reward_type == 'intermediate':
            # 每个pick动作都评估当前阵容
            # 注意：这会多次调用oracle，比较慢
            self._assign_intermediate_rewards(
                rollout, final_state, acting_team,
                radiant_player_feats, dire_player_feats
            )
        
        elif self.reward_type == 'shaped':
            # 基于阵容变化的shaped reward
            self._assign_shaped_rewards(
                rollout, final_reward, acting_team,
                radiant_player_feats, dire_player_feats
            )
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")
        
        return rewards_info
    
    @torch.no_grad()
    def _assign_intermediate_rewards(
        self,
        rollout,
        final_state: BPState,
        acting_team: Team,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ):
        """
        为每个pick动作分配中间奖励
        每个pick后都评估当前（不完整的）阵容
        """
        radiant_picks, dire_picks = [], []
        
        for trans in rollout.transitions:
            if trans.action_type == int(ActionType.PICK):
                # 更新阵容
                if trans.acting_team == int(Team.RADIANT):
                    radiant_picks.append(trans.action)
                else:
                    dire_picks.append(trans.action)
                
                # 评估当前阵容（用占位符补齐到5个）
                r_padded = self._pad_picks(radiant_picks)
                d_padded = self._pad_picks(dire_picks)
                
                win_prob = self.evaluate_picks(
                    r_padded, d_padded,
                    radiant_player_feats, dire_player_feats
                )
                
                reward = win_prob if acting_team == Team.RADIANT else (1.0 - win_prob)
                trans.reward = reward
            else:
                trans.reward = 0.0
    
    @torch.no_grad()
    def _assign_shaped_rewards(
        self,
        rollout,
        final_reward: float,
        acting_team: Team,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ):
        """
        分配shaped reward：基于阵容胜率的变化
        reward_t = V(s_{t+1}) - V(s_t)
        最后一步加上terminal reward保证无偏
        """
        values = []
        radiant_picks, dire_picks = [], []
        
        # 计算每个pick后的阵容价值
        for trans in rollout.transitions:
            if trans.action_type == int(ActionType.PICK):
                if trans.acting_team == int(Team.RADIANT):
                    radiant_picks.append(trans.action)
                else:
                    dire_picks.append(trans.action)
                
                r_padded = self._pad_picks(radiant_picks)
                d_padded = self._pad_picks(dire_picks)
                
                win_prob = self.evaluate_picks(
                    r_padded, d_padded,
                    radiant_player_feats, dire_player_feats
                )
                
                value = win_prob if acting_team == Team.RADIANT else (1.0 - win_prob)
                values.append((trans, value))
            else:
                trans.reward = 0.0
        
        # 计算shaped reward: delta V
        prev_value = 0.5  # 初始价值（随机阵容）
        for i, (trans, value) in enumerate(values):
            if i < len(values) - 1:
                trans.reward = value - prev_value
                prev_value = value
            else:
                # 最后一步加上terminal reward保证无偏
                trans.reward = final_reward - prev_value
    
    def _pad_picks(self, picks: List[int], target_len: int = 5) -> List[int]:
        """用默认英雄补齐阵容（用于中间评估）"""
        # 用英雄1作为占位符（或者可以用一个特殊的"未知"英雄）
        padded = picks + [1] * (target_len - len(picks))
        return padded[:target_len]


class DualTeamRewardCalculator:
    """
    双边奖励计算器 - 同时计算两个阵营的奖励
    用于self-play训练
    """
    
    def __init__(
        self,
        oracle: WinRateOracle,
        device: str = 'cpu',
    ):
        self.oracle = oracle
        self.device = device
        self.calculator = RewardCalculator(oracle, device, reward_type='final')
    
    def calculate_both_rewards(
        self,
        final_state: BPState,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[List[float], List[float]]:
        """
        计算两个阵营的奖励序列
        
        Returns:
            radiant_rewards: 天辉每个动作的奖励
            dire_rewards: 夜魇每个动作的奖励
        """
        rewards_info = self.calculator.calculate_final_rewards(
            final_state, Team.RADIANT,
            radiant_player_feats, dire_player_feats
        )
        
        win_prob = rewards_info['win_prob']
        
        # 天辉希望win_prob高，夜魇希望win_prob低
        radiant_final = win_prob
        dire_final = 1.0 - win_prob
        
        # 分配到各个动作（final reward方式）
        radiant_rewards = []
        dire_rewards = []
        
        # 统计每个阵营的动作数
        radiant_count = sum(1 for a in final_state.action_history if a.team == Team.RADIANT)
        dire_count = sum(1 for a in final_state.action_history if a.team == Team.DIRE)
        
        # 为每个阵营的动作分配奖励（只有最后一步有非零奖励）
        for i, action in enumerate(final_state.action_history):
            is_last = (i == len(final_state.action_history) - 1)
            
            if action.team == Team.RADIANT:
                reward = radiant_final if is_last else 0.0
                radiant_rewards.append(reward)
            else:
                reward = dire_final if is_last else 0.0
                dire_rewards.append(reward)
        
        return radiant_rewards, dire_rewards
