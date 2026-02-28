"""
PPO训练脚本 - 用于训练BP Agent

设计目标：
1. 高速批量生成trajectory（支持batch推理）
2. 使用PPO-clip更新策略
3. 支持Self-Play（双方共享模型）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json
from pathlib import Path

from model.bp_agent import BPActorCritic
from model.win_rate_oracle import WinRateOracle
from bp_framework.engine import BPEngine
from bp_framework.environment import Team
from bp_framework.rollout import BPRollout, RolloutBuffer
from utils.player_preference_sampler import sample_player_preference


@dataclass
class PPOConfig:
    """PPO训练配置"""
    # 模型配置
    embed_dim: int = 128
    nhead: int = 4
    num_layers: int = 2
    num_heroes: int = 160
    use_hero_encoder: bool = True
    use_player_heroes: bool = True
    
    # Oracle配置（用于奖励计算）
    oracle_path: str = './ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260228235818-083-0.9055.pth'
    oracle_embed_dim: int = 128
    oracle_num_layers: int = 6
    
    # 训练配置
    total_iterations: int = 1000
    episodes_per_iter: int = 32  # 每次迭代收集32个episode
    batch_size: int = 128
    mini_batch_size: int = 32
    ppo_epochs: int = 4
    
    # PPO超参数
    gamma: float = 1.0  # BP是有限horizon，通常gamma=1
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # 优化器
    lr: float = 3e-4
    weight_decay: float = 1e-5
    
    # 设备
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 日志和保存
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 50
    output_dir: str = './ckpts/ppo_bp'
    
    # 奖励类型
    reward_type: str = 'final'


def generate_team_player_features(num_heroes=160):
    """生成一个队伍（5个位置）的玩家特征"""
    player_feats = torch.zeros(5, num_heroes)
    
    for position in range(1, 6):
        heroes = sample_player_preference(
            position=position,
            m=3,
            n=5,
            random_seed=None,
        )
        
        for hero in heroes:
            hero_id = hero['id']
            win_rate = hero['win_rate']
            if 1 <= hero_id <= num_heroes:
                player_feats[position - 1, hero_id - 1] = win_rate
    
    return player_feats


def collate_batch_for_ppo(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    将多个rollout的transitions合并成一个batch
    
    由于每个transition的seq_len不同，需要padding
    """
    if len(batch) == 0:
        return {}
    
    # 找到最大序列长度
    max_seq_len = max(
        max(len(t['hero_ids']) for t in b['transitions']) 
        if b['transitions'] else 0
        for b in batch
    )
    
    # 收集所有数据
    all_data = {
        'hero_ids': [],
        'action_types': [],
        'teams': [],
        'positions': [],
        'seq_mask': [],
        'action_mask': [],
        'actions': [],
        'old_log_probs': [],
        'old_values': [],
        'returns': [],
        'advantages': [],
    }
    
    for rollout_data in batch:
        for t in rollout_data['transitions']:
            seq_len = len(t['hero_ids'])
            
            # Padding
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                hero_ids = torch.cat([t['hero_ids'], torch.zeros(pad_len, dtype=torch.long)])
                action_types = torch.cat([t['action_types'], torch.zeros(pad_len, dtype=torch.long)])
                teams = torch.cat([t['teams'], torch.zeros(pad_len, dtype=torch.long)])
                positions = torch.cat([t['positions'], torch.zeros(pad_len, dtype=torch.long)])
                seq_mask = torch.cat([t['seq_mask'], torch.zeros(pad_len, dtype=torch.long)])
            else:
                hero_ids = t['hero_ids']
                action_types = t['action_types']
                teams = t['teams']
                positions = t['positions']
                seq_mask = t['seq_mask']
            
            all_data['hero_ids'].append(hero_ids)
            all_data['action_types'].append(action_types)
            all_data['teams'].append(teams)
            all_data['positions'].append(positions)
            all_data['seq_mask'].append(seq_mask)
            all_data['action_mask'].append(t['action_mask'])
            all_data['actions'].append(t['action_idx'])
            all_data['old_log_probs'].append(t['log_prob'])
            all_data['old_values'].append(t['value'])
            all_data['returns'].append(t['return'])
            all_data['advantages'].append(t['advantage'])
    
    # Stack成tensor
    result = {}
    for key in all_data:
        if all_data[key]:
            result[key] = torch.stack(all_data[key])
        else:
            result[key] = torch.tensor([])
    
    return result


class PPOTrainer:
    """PPO训练器"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建模型
        self.actor_critic = BPActorCritic(
            embed_dim=config.embed_dim,
            nhead=config.nhead,
            num_layers=config.num_layers,
            num_heroes=config.num_heroes,
            use_hero_encoder=config.use_hero_encoder,
            use_player_heroes=config.use_player_heroes,
        ).to(self.device)
        
        # 加载Oracle（用于奖励计算，固定不训练）
        self.oracle = self._load_oracle()
        
        # 创建优化器
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # 创建BPEngine
        self.engine = BPEngine(
            actor_critic=self.actor_critic,
            oracle=self.oracle,
            device=config.device,
            first_team=Team.RADIANT,
            reward_type=config.reward_type,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            use_gae=True,
        )
        
        # 日志
        self.train_logs = []
        self.iteration = 0
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def _load_oracle(self) -> WinRateOracle:
        """加载训练好的Oracle"""
        oracle = WinRateOracle(
            embed_dim=self.config.oracle_embed_dim,
            nhead=4,
            num_layers=self.config.oracle_num_layers,
            use_text=True,
            use_player_heroes=True,
        ).to(self.device)
        
        if os.path.exists(self.config.oracle_path):
            checkpoint = torch.load(self.config.oracle_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                oracle.load_state_dict(checkpoint['model_state_dict'])
            else:
                oracle.load_state_dict(checkpoint)
            print(f"Oracle loaded from {self.config.oracle_path}")
        else:
            print(f"Warning: Oracle checkpoint not found at {self.config.oracle_path}")
        
        oracle.eval()
        for param in oracle.parameters():
            param.requires_grad = False
        
        return oracle
    
    def collect_rollouts(self, num_episodes: int) -> Tuple[RolloutBuffer, RolloutBuffer]:
        """收集trajectory（批量）"""
        r_buffer = RolloutBuffer()
        d_buffer = RolloutBuffer()
        
        for _ in range(num_episodes):
            # 生成随机玩家特征
            r_feats = generate_team_player_features(self.config.num_heroes).to(self.device)
            d_feats = generate_team_player_features(self.config.num_heroes).to(self.device)
            
            # 运行episode
            r_rollout, d_rollout = self.engine.run_episode(
                deterministic=False,
                radiant_player_feats=r_feats,
                dire_player_feats=d_feats,
                verbose=False,
            )
            
            r_buffer.add_rollout(r_rollout)
            d_buffer.add_rollout(d_rollout)
        
        return r_buffer, d_buffer
    
    def prepare_batch_data(self, r_buffer: RolloutBuffer, d_buffer: RolloutBuffer) -> List[Dict]:
        """准备batch数据（合并双方数据）"""
        batch_data = []
        
        for rollout in r_buffer.rollouts + d_buffer.rollouts:
            if len(rollout) == 0:
                continue
            
            transitions_data = []
            for i, t in enumerate(rollout.transitions):
                transitions_data.append({
                    'hero_ids': t.hero_ids,
                    'action_types': t.action_types,
                    'teams': t.teams,
                    'positions': t.positions,
                    'seq_mask': t.seq_mask,
                    'action_mask': t.action_mask,
                    'action_idx': t.action_idx,
                    'log_prob': t.log_prob,
                    'value': t.value,
                    'return': rollout.returns[i] if i < len(rollout.returns) else 0.0,
                    'advantage': rollout.advantages[i] if i < len(rollout.advantages) else 0.0,
                })
            
            batch_data.append({
                'transitions': transitions_data,
            })
        
        return batch_data
    
    def ppo_update(self, batch_data: List[Dict]) -> Dict[str, float]:
        """PPO更新"""
        if len(batch_data) == 0:
            return {}
        
        # Collate batch
        batch = collate_batch_for_ppo(batch_data)
        if len(batch) == 0:
            return {}
        
        # 移动到device
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # 标准化advantages
        advantages = batch['advantages']
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # PPO多次epoch
        dataset_size = len(batch['actions'])
        indices = torch.randperm(dataset_size)
        
        for epoch in range(self.config.ppo_epochs):
            # Mini-batch训练
            for start in range(0, dataset_size, self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                mb_indices = indices[start:end]
                
                # 获取mini-batch数据
                mb_hero_ids = batch['hero_ids'][mb_indices]
                mb_action_types = batch['action_types'][mb_indices]
                mb_teams = batch['teams'][mb_indices]
                mb_positions = batch['positions'][mb_indices]
                mb_seq_mask = batch['seq_mask'][mb_indices]
                mb_action_mask = batch['action_mask'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['old_log_probs'][mb_indices]
                mb_old_values = batch['old_values'][mb_indices]
                mb_returns = batch['returns'][mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Forward pass
                action_probs, values = self.actor_critic(
                    hero_ids=mb_hero_ids,
                    action_types=mb_action_types,
                    teams=mb_teams,
                    positions=mb_positions,
                    action_mask=mb_action_mask,
                    seq_mask=mb_seq_mask,
                )
                
                # 计算新log probs
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                
                # 计算ratio
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_pred_clipped = mb_old_values + torch.clamp(
                    values.squeeze() - mb_old_values,
                    -self.config.clip_epsilon,
                    self.config.clip_epsilon
                )
                value_loss1 = (values.squeeze() - mb_returns) ** 2
                value_loss2 = (value_pred_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                
                # 总loss
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                
                # 记录
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        return {
            'loss': total_loss / num_updates if num_updates > 0 else 0,
            'policy_loss': total_policy_loss / num_updates if num_updates > 0 else 0,
            'value_loss': total_value_loss / num_updates if num_updates > 0 else 0,
            'entropy': total_entropy / num_updates if num_updates > 0 else 0,
        }
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """评估当前策略"""
        return self.engine.evaluate(num_episodes=num_episodes, verbose=False)
    
    def save_checkpoint(self, iteration: int):
        """保存checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        
        save_path = Path(self.config.output_dir) / f'ppo_bp_iter_{iteration:04d}.pth'
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def train(self):
        """主训练循环"""
        print("=" * 60)
        print("Starting PPO Training")
        print(f"Device: {self.device}")
        print(f"Total iterations: {self.config.total_iterations}")
        print(f"Episodes per iteration: {self.config.episodes_per_iter}")
        print("=" * 60)
        
        for iteration in range(self.config.total_iterations):
            self.iteration = iteration
            
            # 1. 收集trajectory
            print(f"\n[Iter {iteration}] Collecting rollouts...")
            r_buffer, d_buffer = self.collect_rollouts(self.config.episodes_per_iter)
            
            total_transitions = r_buffer.total_transitions() + d_buffer.total_transitions()
            print(f"Collected {len(r_buffer)} + {len(d_buffer)} = {len(r_buffer) + len(d_buffer)} episodes")
            print(f"Total transitions: {total_transitions}")
            
            # 2. 准备batch数据
            batch_data = self.prepare_batch_data(r_buffer, d_buffer)
            
            # 3. PPO更新
            print(f"[Iter {iteration}] Updating policy...")
            update_info = self.ppo_update(batch_data)
            
            # 4. 记录日志
            log_entry = {
                'iteration': iteration,
                'episodes': len(r_buffer) + len(d_buffer),
                'transitions': total_transitions,
                **update_info,
            }
            self.train_logs.append(log_entry)
            
            # 5. 打印日志
            if iteration % self.config.log_interval == 0:
                print(f"\n[Iter {iteration}] Loss: {update_info.get('loss', 0):.4f}, "
                      f"Policy: {update_info.get('policy_loss', 0):.4f}, "
                      f"Value: {update_info.get('value_loss', 0):.4f}, "
                      f"Entropy: {update_info.get('entropy', 0):.4f}")
            
            # 6. 评估
            if iteration % self.config.eval_interval == 0 and iteration > 0:
                print(f"\n[Iter {iteration}] Evaluating...")
                eval_stats = self.evaluate(num_episodes=10)
                print(f"Mean win prob: {eval_stats['mean_win_prob']:.4f}")
            
            # 7. 保存
            if iteration % self.config.save_interval == 0 and iteration > 0:
                self.save_checkpoint(iteration)
        
        # 保存最终模型
        self.save_checkpoint(self.config.total_iterations)
        
        # 保存训练日志
        log_path = Path(self.config.output_dir) / 'train_logs.json'
        with open(log_path, 'w') as f:
            json.dump(self.train_logs, f, indent=2)
        print(f"\nTraining logs saved to {log_path}")


def main():
    config = PPOConfig(
        total_iterations=200,
        episodes_per_iter=32,
        batch_size=256,
        mini_batch_size=64,
        ppo_epochs=4,
        lr=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
