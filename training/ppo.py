"""
PPO (Proximal Policy Optimization) Algorithm

包含：
- GAE (Generalized Advantage Estimation)
- PPO更新
- 支持多epoch训练
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 GAE (Generalized Advantage Estimation)
    
    Args:
        rewards: [T] 奖励序列
        values: [T+1] 价值估计（包含最终状态）
        dones: [T] 结束标记
        gamma: 折扣因子
        lam: GAE参数
    
    Returns:
        advantages: [T] 优势估计
        returns: [T] 回报估计
    """
    advantages = []
    gae = 0
    values_np = values.detach().cpu().numpy()
    rewards_np = rewards.detach().cpu().numpy()
    dones_np = dones.detach().cpu().numpy()

    for t in reversed(range(len(rewards))):
        delta = rewards_np[t] + gamma * values_np[t + 1] * (1 - dones_np[t]) - values_np[t]
        gae = delta + gamma * lam * (1 - dones_np[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
    returns = advantages + torch.tensor(values_np[:-1], dtype=torch.float32)

    return advantages, returns


class PPOTrainer:
    """
    PPO训练器
    
    支持：
    - 多epoch更新
    - Mini-batch训练
    - 梯度裁剪
    - 自适应优势估计（根据team调整符号）
    """
    
    def __init__(self, agent, optimizer, config, device):
        """
        Args:
            agent: 策略模型（需实现encode_state, evaluate_actions）
            optimizer: 优化器
            config: 配置对象（需包含PPO相关参数）
            device: torch设备
        """
        self.agent = agent
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
    def update(self, trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        PPO更新
        
        Args:
            trajectories: 轨迹列表（每个轨迹是一个dict）
        
        Returns:
            损失统计 dict
        """
        self.agent.train()
        
        # 合并所有轨迹数据
        all_states = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_old_log_probs = []
        
        # 为每个轨迹计算GAE
        for traj in trajectories:
            states = traj['states']
            rewards = traj['rewards'].to(self.device)
            dones = traj['dones'].to(self.device)
            old_log_probs = traj['log_probs'].to(self.device)
            old_values = traj['values'].to(self.device)
            teams = traj.get('teams', [0] * len(states))
            
            # 计算GAE
            advantages, returns = compute_gae(
                rewards, old_values, dones,
                self.config.GAMMA, self.config.LAMBDA
            )
            
            # 根据team调整advantage符号
            # team=0（天辉）：advantage不变
            # team=1（夜魇）：advantage取反（要降低天辉胜率）
            for i in range(len(states)):
                if teams[i] == 1:
                    advantages[i] = -advantages[i]
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # 存储
            for i in range(len(states)):
                all_states.append(states[i])
                all_actions.append(traj['actions'][i])
                all_advantages.append(advantages[i])
                all_returns.append(returns[i])
                all_old_log_probs.append(old_log_probs[i])
        
        # 转换为tensor
        all_actions = torch.stack(all_actions).to(self.device)
        all_advantages = torch.stack(all_advantages).to(self.device)
        all_returns = torch.stack(all_returns).to(self.device)
        all_old_log_probs = torch.stack(all_old_log_probs).to(self.device)
        
        # 多epoch训练
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        dataset_size = len(all_states)
        
        for epoch in range(self.config.PPO_EPOCHS):
            # 随机打乱
            indices = torch.randperm(dataset_size)
            
            # Mini-batch训练
            batch_size = self.config.BATCH_SIZE
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # 收集batch数据
                batch_states = [all_states[i] for i in batch_indices]
                batch_actions = all_actions[batch_indices]
                batch_advantages = all_advantages[batch_indices]
                batch_returns = all_returns[batch_indices]
                batch_old_log_probs = all_old_log_probs[batch_indices]
                
                # 重新计算log_prob和value
                new_log_probs = []
                new_values = []
                entropies = []
                
                for i, state in enumerate(batch_states):
                    state_feat = self.agent.encode_state(
                        hero_ids=state['hero_ids'].to(self.device),
                        team_flags=state['team_flags'].to(self.device),
                        action_types=state['action_types'].to(self.device),
                        valid_mask=state['valid_mask'].to(self.device),
                        radiant_player_feats=state['radiant_player_feats'].to(self.device) if state['radiant_player_feats'] is not None else None,
                        dire_player_feats=state['dire_player_feats'].to(self.device) if state['dire_player_feats'] is not None else None,
                    )
                    
                    candidate_ids = state['candidate_ids'].to(self.device)
                    action_idx = state['action_idx'].to(self.device)
                    
                    log_prob, value, entropy = self.agent.evaluate_actions(
                        state_feat=state_feat,
                        candidate_hero_ids=candidate_ids,
                        actions=action_idx.unsqueeze(0),
                    )
                    
                    new_log_probs.append(log_prob)
                    new_values.append(value)
                    entropies.append(entropy)
                
                new_log_probs = torch.cat(new_log_probs)
                new_values = torch.cat(new_values).squeeze(-1)
                entropies = torch.cat(entropies)
                
                # PPO损失
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.CLIP_RATIO, 1 + self.config.CLIP_RATIO) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 确保batch_returns维度一致
                batch_returns = batch_returns.squeeze(-1)
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                entropy_loss = -entropies.mean()
                
                loss = policy_loss + self.config.VALUE_COEF * value_loss + self.config.ENTROPY_COEF * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.GRAD_CLIP)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1
        
        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0,
        }
