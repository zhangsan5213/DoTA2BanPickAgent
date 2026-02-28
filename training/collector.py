"""
Trajectory Collector for RL training

算法无关的轨迹收集器，可以被PPO、MCTS等任意算法使用
"""
import random
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from env.bp_env import BPEnvironment


class Trajectory:
    """单条轨迹"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.teams = []  # 记录每一步的team
        
    def add_step(self, state: Dict, action: int, reward: float, done: bool,
                 log_prob: torch.Tensor, value: torch.Tensor, team: int):
        """添加一步"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.teams.append(team)
        
    def finalize(self, final_value: torch.Tensor):
        """添加最终value（用于GAE）"""
        self.values.append(final_value)
        
    def to_tensors(self, device) -> Dict[str, torch.Tensor]:
        """转换为tensor格式"""
        return {
            'states': self.states,
            'actions': torch.tensor(self.actions, dtype=torch.long, device=device),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32, device=device),
            'dones': torch.tensor(self.dones, dtype=torch.float32, device=device),
            'log_probs': torch.cat(self.log_probs),
            'values': torch.cat(self.values),
            'teams': self.teams,
        }


class TrajectoryCollector:
    """
    轨迹收集器
    
    支持：
    - 多环境并行收集
    - 批量推理
    - 支持不同算法（PPO、MCTS等）
    """
    
    def __init__(self, envs: List[BPEnvironment], device: torch.device):
        """
        Args:
            envs: 环境列表
            device: torch设备
        """
        self.envs = envs
        self.device = device
        self.n_envs = len(envs)
        
    def collect(self, agent, oracle_reward_fn=None, temperature: float = 1.0) -> List[Trajectory]:
        """
        收集轨迹
        
        Args:
            agent: 策略模型（需实现encode_state和get_action）
            oracle_reward_fn: 可选的Oracle奖励函数（终局评估）
                              签名: fn(env, agent, device) -> float
            temperature: 温度参数，>1 增加随机性，<1 减少随机性
        
        Returns:
            轨迹列表
        """
        trajectories = [Trajectory() for _ in range(self.n_envs)]
        env_states = []
        
        # 初始化环境
        for env in self.envs:
            state = env.reset()
            env_states.append({
                'state': state,
                'env': env,
                'active': True
            })
        
        # 批量收集直到所有环境完成
        while any(env_states[i]['active'] for i in range(self.n_envs)):
            # 收集活跃环境
            active_indices = [i for i in range(self.n_envs) if env_states[i]['active']]
            if not active_indices:
                break
            
            # 构建batch
            batch_data = self._build_batch(env_states, active_indices)
            
            # 批量推理
            with torch.no_grad():
                actions, log_probs, values = self._inference(agent, batch_data, temperature)
            
            # 执行步骤
            for batch_idx, env_idx in enumerate(active_indices):
                env_data = env_states[env_idx]
                env = env_data['env']
                state = env_data['state']
                trajectory = trajectories[env_idx]
                
                # 获取当前team
                current_step = env.current_step
                current_team = env.action_sequence[current_step][0] if current_step < len(env.action_sequence) else 0
                
                # 执行动作
                action_idx = actions[batch_idx].item()
                actual_action = batch_data['candidate_ids'][batch_idx, action_idx].item()
                
                # 存储转换
                trajectory.add_step(
                    state={
                        'hero_ids': state['hero_ids'].clone(),
                        'team_flags': state['team_flags'].clone(),
                        'action_types': state['action_types'].clone(),
                        'valid_mask': state['valid_mask'].clone(),
                        'radiant_player_feats': state['radiant_player_feats'].clone() if state['radiant_player_feats'] is not None else None,
                        'dire_player_feats': state['dire_player_feats'].clone() if state['dire_player_feats'] is not None else None,
                        'candidate_ids': batch_data['candidate_ids'][batch_idx:batch_idx+1].clone(),
                        'action_idx': actions[batch_idx].clone(),
                        'team': current_team,
                    },
                    action=action_idx,
                    reward=0.0,  # 中间步骤为0
                    done=False,
                    log_prob=log_probs[batch_idx],
                    value=values[batch_idx],
                    team=current_team
                )
                
                # 执行步骤
                next_state, reward, done = env.step(actual_action)
                
                if done:
                    # 终局奖励
                    if oracle_reward_fn is not None:
                        final_reward = oracle_reward_fn(env, agent, self.device)
                        trajectory.rewards[-1] = final_reward
                    
                    trajectory.finalize(torch.zeros(1, 1).to(self.device))
                    env_data['active'] = False
                else:
                    env_data['state'] = next_state
        
        return trajectories
    
    def _build_batch(self, env_states: List[Dict], active_indices: List[int]) -> Dict:
        """构建batch输入"""
        batch_hero_ids = []
        batch_team_flags = []
        batch_action_types = []
        batch_valid_mask = []
        batch_radiant_player_feats = []
        batch_dire_player_feats = []
        batch_candidate_ids = []
        
        for idx in active_indices:
            state = env_states[idx]['state']
            env = env_states[idx]['env']
            
            batch_hero_ids.append(state['hero_ids'])
            batch_team_flags.append(state['team_flags'])
            batch_action_types.append(state['action_types'])
            batch_valid_mask.append(state['valid_mask'])
            batch_radiant_player_feats.append(state['radiant_player_feats'])
            batch_dire_player_feats.append(state['dire_player_feats'])
            
            # 采样候选英雄
            valid_heroes = env.get_valid_actions()
            K = min(32, len(valid_heroes)) if len(valid_heroes) > 0 else 0
            if K > 0:
                candidate_ids = random.sample(valid_heroes, K)
            else:
                candidate_ids = [0] * 32
            while len(candidate_ids) < 32:
                candidate_ids.append(0)
            batch_candidate_ids.append(candidate_ids)
        
        result = {
            'hero_ids': torch.cat(batch_hero_ids, dim=0).to(self.device),
            'team_flags': torch.cat(batch_team_flags, dim=0).to(self.device),
            'action_types': torch.cat(batch_action_types, dim=0).to(self.device),
            'valid_mask': torch.cat(batch_valid_mask, dim=0).to(self.device),
            'candidate_ids': torch.tensor(batch_candidate_ids, dtype=torch.long).to(self.device),
        }
        
        # 处理player feats
        if batch_radiant_player_feats[0] is not None:
            result['radiant_player_feats'] = torch.cat(batch_radiant_player_feats, dim=0).to(self.device)
            result['dire_player_feats'] = torch.cat(batch_dire_player_feats, dim=0).to(self.device)
        else:
            result['radiant_player_feats'] = None
            result['dire_player_feats'] = None
        
        return result
    
    def _inference(self, agent, batch_data: Dict, temperature: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """批量推理"""
        # 编码状态
        state_feat = agent.encode_state(
            hero_ids=batch_data['hero_ids'],
            team_flags=batch_data['team_flags'],
            action_types=batch_data['action_types'],
            valid_mask=batch_data['valid_mask'],
            radiant_player_feats=batch_data['radiant_player_feats'],
            dire_player_feats=batch_data['dire_player_feats'],
        )
        
        # 逐个获取动作（因为每个环境可能有不同的候选集）
        batch_size = state_feat.shape[0]
        actions = []
        log_probs = []
        values = []
        
        for i in range(batch_size):
            action, log_prob, value = agent.get_action(
                state_feat=state_feat[i:i+1],
                candidate_hero_ids=batch_data['candidate_ids'][i:i+1],
                deterministic=False,
                temperature=temperature,
            )
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        
        return torch.cat(actions), log_probs, values
