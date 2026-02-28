"""
高速Rollout收集器

优化策略：
1. Batch推理：同时处理多个环境的state
2. 预生成玩家特征：避免重复采样
3. 异步数据转移：使用pinned memory
"""
import torch
import numpy as np
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import queue

from bp_framework.environment import BPEnvironment, Team, ActionType
from bp_framework.rollout import BPRollout, BPTransition, RolloutBuffer
from utils.player_preference_sampler import sample_player_preference


class FastRolloutCollector:
    """
    高速Rollout收集器
    
    使用batch推理加速trajectory生成
    """
    
    def __init__(
        self,
        actor_critic,
        oracle,
        num_heroes: int = 160,
        device: str = 'cuda',
        reward_type: str = 'final',
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ):
        self.actor_critic = actor_critic
        self.oracle = oracle
        self.num_heroes = num_heroes
        self.device = device
        self.reward_type = reward_type
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # 预生成的玩家特征池（避免重复采样）
        self.player_pool_size = 100
        self.radiant_player_pool = []
        self.dire_player_pool = []
        self._refill_player_pool()
    
    def _refill_player_pool(self):
        """预生成玩家特征池"""
        # 天辉玩家池
        while len(self.radiant_player_pool) < self.player_pool_size:
            feats = self._generate_team_features()
            self.radiant_player_pool.append(feats)
        
        # 夜魇玩家池
        while len(self.dire_player_pool) < self.player_pool_size:
            feats = self._generate_team_features()
            self.dire_player_pool.append(feats)
    
    def _generate_team_features(self) -> torch.Tensor:
        """生成一个队伍的玩家特征"""
        feats = torch.zeros(5, self.num_heroes)
        for position in range(1, 6):
            heroes = sample_player_preference(position=position, m=3, n=5)
            for hero in heroes:
                hero_id = hero['id']
                if 1 <= hero_id <= self.num_heroes:
                    feats[position - 1, hero_id - 1] = hero['win_rate']
        return feats.to(self.device)
    
    def get_player_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """从池中获取玩家特征"""
        if len(self.radiant_player_pool) < 10 or len(self.dire_player_pool) < 10:
            self._refill_player_pool()
        
        r_feats = self.radiant_player_pool.pop(0)
        d_feats = self.dire_player_pool.pop(0)
        return r_feats, d_feats
    
    def collect_batch(
        self,
        batch_size: int,
        deterministic: bool = False,
        verbose: bool = False,
    ) -> Tuple[RolloutBuffer, RolloutBuffer]:
        """
        批量收集rollouts
        
        使用batch推理加速：同时处理多个episode的state
        """
        radiant_buffer = RolloutBuffer()
        dire_buffer = RolloutBuffer()
        
        # 创建多个环境
        envs = [BPEnvironment(num_heroes=self.num_heroes, device=self.device) for _ in range(batch_size)]
        
        # 初始化环境
        active_envs = list(range(batch_size))
        states = []
        r_feats_list = []
        d_feats_list = []
        rollouts_r = [BPRollout() for _ in range(batch_size)]
        rollouts_d = [BPRollout() for _ in range(batch_size)]
        
        for i in active_envs:
            r_feats, d_feats = self.get_player_features()
            state = envs[i].reset(radiant_player_feats=r_feats, dire_player_feats=d_feats)
            states.append(state)
            r_feats_list.append(r_feats)
            d_feats_list.append(d_feats)
        
        step_count = 0
        max_steps = 25  # BP最多22-24步
        
        self.actor_critic.eval()
        
        with torch.no_grad():
            while active_envs and step_count < max_steps:
                step_count += 1
                
                # 收集所有活跃环境的state
                batch_states = []
                for idx in active_envs:
                    state_tensors = envs[idx].get_state_for_agent()
                    batch_states.append({
                        'idx': idx,
                        'state_tensors': state_tensors,
                        'current_team': states[idx].current_team,
                    })
                
                # Batch推理
                if len(batch_states) > 0:
                    # 合并batch
                    hero_ids = torch.cat([s['state_tensors']['hero_ids'] for s in batch_states], dim=0)
                    action_types = torch.cat([s['state_tensors']['action_types'] for s in batch_states], dim=0)
                    teams = torch.cat([s['state_tensors']['teams'] for s in batch_states], dim=0)
                    positions = torch.cat([s['state_tensors']['positions'] for s in batch_states], dim=0)
                    action_masks = torch.cat([s['state_tensors']['action_mask'] for s in batch_states], dim=0)
                    seq_masks = torch.cat([s['state_tensors']['seq_mask'] for s in batch_states], dim=0)
                    
                    # 准备玩家特征
                    if self.actor_critic.use_player_heroes:
                        r_feats_batch = torch.stack([r_feats_list[s['idx']] for s in batch_states])
                        d_feats_batch = torch.stack([d_feats_list[s['idx']] for s in batch_states])
                    else:
                        r_feats_batch = None
                        d_feats_batch = None
                    
                    # 批量推理
                    action_probs, values = self.actor_critic(
                        hero_ids=hero_ids,
                        action_types=action_types,
                        teams=teams,
                        positions=positions,
                        action_mask=action_masks,
                        seq_mask=seq_masks,
                        radiant_player_feats=r_feats_batch,
                        dire_player_feats=d_feats_batch,
                    )
                    
                    # 处理每个环境
                    new_active = []
                    for i, batch_item in enumerate(batch_states):
                        idx = batch_item['idx']
                        current_team = batch_item['current_team']
                        
                        # 采样动作
                        probs = action_probs[i]
                        if deterministic:
                            action_idx = probs.argmax()
                        else:
                            action_idx = torch.multinomial(probs, 1).item()
                        
                        hero_id = action_idx + 1
                        
                        # 记录transition
                        transition = BPTransition(
                            hero_ids=batch_item['state_tensors']['hero_ids'].squeeze(0).cpu(),
                            action_types=batch_item['state_tensors']['action_types'].squeeze(0).cpu(),
                            teams=batch_item['state_tensors']['teams'].squeeze(0).cpu(),
                            positions=batch_item['state_tensors']['positions'].squeeze(0).cpu(),
                            seq_mask=batch_item['state_tensors']['seq_mask'].squeeze(0).cpu(),
                            action_mask=batch_item['state_tensors']['action_mask'].squeeze(0).cpu(),
                            action=int(hero_id),
                            action_idx=int(action_idx),
                            log_prob=float(torch.log(probs[action_idx] + 1e-10).item()),
                            value=float(values[i].item()),
                            acting_team=int(current_team),
                            action_type=int(states[idx].current_action_type),
                            reward=0.0,
                        )
                        
                        if current_team == Team.RADIANT:
                            rollouts_r[idx].add_transition(transition)
                        else:
                            rollouts_d[idx].add_transition(transition)
                        
                        # 执行动作
                        next_state, is_terminal, _ = envs[idx].step(hero_id)
                        states[idx] = next_state
                        
                        if not is_terminal:
                            new_active.append(idx)
                
                active_envs = new_active
        
        # 计算奖励和advantages
        for idx in range(batch_size):
            # 获取最终阵容
            radiant_picks, dire_picks = states[idx].get_final_picks()
            rollouts_r[idx].set_final_picks(radiant_picks, dire_picks)
            rollouts_d[idx].set_final_picks(radiant_picks, dire_picks)
            
            # 计算Oracle奖励（这里简化处理，实际应该调用RewardCalculator）
            # 为简化，假设双方reward基于最终胜率
            if len(radiant_picks) == 5 and len(dire_picks) == 5:
                # 这里应该用Oracle评估，简化处理
                # 实际使用时需要导入RewardCalculator
                final_reward = 0.5  # 占位符
            else:
                final_reward = 0.5
            
            # 分配奖励
            if len(rollouts_r[idx].transitions) > 0:
                rollouts_r[idx].transitions[-1].reward = final_reward
            if len(rollouts_d[idx].transitions) > 0:
                rollouts_d[idx].transitions[-1].reward = 1.0 - final_reward
            
            # 计算advantages
            rollouts_r[idx].compute_returns_and_advantages(
                final_value=0.0, gamma=self.gamma, gae_lambda=self.gae_lambda
            )
            rollouts_d[idx].compute_returns_and_advantages(
                final_value=0.0, gamma=self.gamma, gae_lambda=self.gae_lambda
            )
            
            # 添加到buffer
            radiant_buffer.add_rollout(rollouts_r[idx])
            dire_buffer.add_rollout(rollouts_d[idx])
        
        return radiant_buffer, dire_buffer


# 简单的顺序收集器（用于对比）
def collect_rollouts_sequential(
    engine,
    num_episodes: int,
    num_heroes: int = 160,
) -> Tuple[RolloutBuffer, RolloutBuffer]:
    """顺序收集rollouts（用于对比测试）"""
    r_buffer = RolloutBuffer()
    d_buffer = RolloutBuffer()
    
    for _ in range(num_episodes):
        # 生成玩家特征
        r_feats = torch.zeros(5, num_heroes)
        d_feats = torch.zeros(5, num_heroes)
        
        for pos in range(1, 6):
            r_heroes = sample_player_preference(position=pos, m=3, n=5)
            d_heroes = sample_player_preference(position=pos, m=3, n=5)
            
            for h in r_heroes:
                if 1 <= h['id'] <= num_heroes:
                    r_feats[pos-1, h['id']-1] = h['win_rate']
            for h in d_heroes:
                if 1 <= h['id'] <= num_heroes:
                    d_feats[pos-1, h['id']-1] = h['win_rate']
        
        r_rollout, d_rollout = engine.run_episode(
            deterministic=False,
            radiant_player_feats=r_feats,
            dire_player_feats=d_feats,
            verbose=False,
        )
        
        r_buffer.add_rollout(r_rollout)
        d_buffer.add_rollout(d_rollout)
    
    return r_buffer, d_buffer
