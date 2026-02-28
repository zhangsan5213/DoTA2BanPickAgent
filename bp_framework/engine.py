"""
BP引擎 - Self-Play训练框架

核心设计：Radiant和Dire使用同一个模型（通过team embedding区分阵营）
"""
import torch
import numpy as np
from typing import Optional, Dict, List, Tuple

from model.bp_agent import BPActorCritic
from model.win_rate_oracle import WinRateOracle
from bp_framework.environment import BPEnvironment, BPState, Team, ActionType
from bp_framework.rollout import BPRollout, BPTransition, RolloutBuffer
from bp_framework.reward import RewardCalculator


class BPEngine:
    """
    BP引擎：执行Self-Play，收集双方的trajectory
    
    Self-Play设计：
    - 同一个actor_critic模型服务双方
    - 通过teams embedding区分"我是天辉"还是"我是夜魇"
    - 分别收集双方的transition，计算各自的advantage
    """
    
    def __init__(
        self,
        actor_critic: BPActorCritic,
        oracle: WinRateOracle,
        device: str = 'cpu',
        first_team: Team = Team.RADIANT,
        reward_type: str = 'final',
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        use_gae: bool = True,
    ):
        """
        Args:
            actor_critic: Actor-Critic模型（双方共享）
            oracle: WinRateOracle用于计算奖励
            device: 计算设备
            first_team: BP先手阵营（0=天辉先ban，1=夜魇先ban）
            reward_type: 奖励类型 ('final', 'intermediate', 'shaped')
            gamma: 折扣因子
            gae_lambda: GAE lambda
            use_gae: 是否使用GAE
        """
        self.actor_critic = actor_critic.to(device)
        self.oracle = oracle.to(device)
        self.device = device
        self.first_team = first_team
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.use_gae = use_gae
        
        # 创建环境
        self.env = BPEnvironment(
            num_heroes=actor_critic.num_heroes,
            first_team=first_team,
            device=device,
        )
        
        # 创建奖励计算器
        self.reward_calculator = RewardCalculator(
            oracle=oracle,
            device=device,
            reward_type=reward_type,
        )
        
        # Rollout缓冲区
        self.rollout_buffer = RolloutBuffer()
    
    def run_episode(
        self,
        deterministic: bool = False,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
        verbose: bool = False,
        hero_name_fn: Optional[callable] = None,
    ) -> Tuple[BPRollout, BPRollout]:
        """
        运行一个完整的Self-Play BP episode
        
        Args:
            deterministic: 是否使用确定性策略（greedy）
            radiant_player_feats: 天辉玩家特征（可选）
            dire_player_feats: 夜魇玩家特征（可选）
            verbose: 是否打印详细信息
        
        Returns:
            radiant_rollout: 天辉的trajectory
            dire_rollout: 夜魇的trajectory
        """
        self.actor_critic.eval()
        
        # 重置环境（传入玩家特征）
        state = self.env.reset(
            radiant_player_feats=radiant_player_feats,
            dire_player_feats=dire_player_feats,
        )
        radiant_rollout = BPRollout()
        dire_rollout = BPRollout()
        
        if verbose:
            print("=" * 60)
            print(f"Self-Play BP - First team: {self.first_team.name}")
            print("=" * 60)
        
        step_count = 0
        while not state.is_terminal:
            # 获取当前状态输入（包含完整的历史序列）
            state_tensors = self.env.get_state_for_agent()
            current_team = state.current_team
            
            # 准备玩家特征（如果模型支持且提供了特征）
            r_feats = state.radiant_player_feats
            d_feats = state.dire_player_feats
            if r_feats is not None:
                r_feats = r_feats.unsqueeze(0).to(self.device)  # [1, 5, NUM_HEROES]
            if d_feats is not None:
                d_feats = d_feats.unsqueeze(0).to(self.device)  # [1, 5, NUM_HEROES]
            
            # 双方使用同一个模型决策（team embedding会自动处理阵营信息）
            with torch.no_grad():
                action_idx, log_prob, value = self.actor_critic.select_action(
                    hero_ids=state_tensors['hero_ids'],
                    action_types=state_tensors['action_types'],
                    teams=state_tensors['teams'],
                    positions=state_tensors['positions'],
                    action_mask=state_tensors['action_mask'],
                    seq_mask=state_tensors['seq_mask'],
                    deterministic=deterministic,
                    radiant_player_feats=r_feats,
                    dire_player_feats=d_feats,
                )
            
            # action_idx是0-based，转为1-based hero_id
            hero_id = action_idx.item() + 1
            
            if verbose:
                action_name = "PICK" if state.current_action_type == ActionType.PICK else "BAN"
                team_name = "RAD" if current_team == Team.RADIANT else "DIRE"
                hero_display = hero_name_fn(hero_id) if hero_name_fn else f"Hero_{hero_id}"
                print(f"Step {step_count:2d} | {team_name} {action_name:4s} {hero_display}")
            
            # 创建transition
            transition = BPTransition(
                hero_ids=state_tensors['hero_ids'].squeeze(0).cpu(),
                action_types=state_tensors['action_types'].squeeze(0).cpu(),
                teams=state_tensors['teams'].squeeze(0).cpu(),
                positions=state_tensors['positions'].squeeze(0).cpu(),
                seq_mask=state_tensors['seq_mask'].squeeze(0).cpu(),
                action_mask=state_tensors['action_mask'].squeeze(0).cpu(),
                action=int(hero_id),
                action_idx=int(action_idx.item()),
                log_prob=float(log_prob.item()),
                value=float(value.item()),
                acting_team=int(current_team),
                action_type=int(state.current_action_type),
                reward=0.0,
            )
            
            # 根据阵营添加到对应的rollout
            if current_team == Team.RADIANT:
                radiant_rollout.add_transition(transition)
            else:
                dire_rollout.add_transition(transition)
            
            # 执行动作
            state, is_terminal, info = self.env.step(hero_id)
            step_count += 1
        
        # 记录最终阵容
        radiant_picks, dire_picks = state.get_final_picks()
        radiant_rollout.set_final_picks(radiant_picks, dire_picks)
        dire_rollout.set_final_picks(radiant_picks, dire_picks)
        
        if verbose:
            print("-" * 60)
            if hero_name_fn:
                r_display = [hero_name_fn(hid) for hid in radiant_picks]
                d_display = [hero_name_fn(hid) for hid in dire_picks]
                print(f"Radiant picks: {r_display}")
                print(f"Dire picks:    {d_display}")
            else:
                print(f"Radiant picks: {radiant_picks}")
                print(f"Dire picks:    {dire_picks}")
        
        # 计算Oracle评估
        rewards_info = self.reward_calculator.calculate_final_rewards(
            state, Team.RADIANT,
            radiant_player_feats, dire_player_feats
        )
        
        win_prob = rewards_info['win_prob']
        radiant_reward = rewards_info['radiant_reward']  # = win_prob
        dire_reward = rewards_info['dire_reward']        # = 1 - win_prob
        
        # 分配奖励（final reward：只有最后一步有非零奖励）
        if len(radiant_rollout.transitions) > 0:
            radiant_rollout.transitions[-1].reward = radiant_reward
        if len(dire_rollout.transitions) > 0:
            dire_rollout.transitions[-1].reward = dire_reward
        
        if verbose:
            print(f"Oracle: Win prob = {win_prob:.4f}")
            print(f"Reward: R={radiant_reward:+.4f}, D={dire_reward:+.4f}")
            print("=" * 60)
        
        # 计算returns和advantages（双方独立计算）
        radiant_rollout.compute_returns_and_advantages(
            final_value=0.0,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            use_gae=self.use_gae,
        )
        dire_rollout.compute_returns_and_advantages(
            final_value=0.0,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            use_gae=self.use_gae,
        )
        
        return radiant_rollout, dire_rollout
    
    def collect_rollouts(
        self,
        num_episodes: int,
        deterministic: bool = False,
        verbose_interval: int = 10,
    ) -> Tuple[RolloutBuffer, RolloutBuffer]:
        """
        收集多个episode的rollouts（双方）
        
        Args:
            num_episodes: episode数量
            deterministic: 是否确定性策略
            verbose_interval: 打印间隔（0表示不打印）
        
        Returns:
            radiant_buffer: 天辉的rollout缓冲区
            dire_buffer: 夜魇的rollout缓冲区
        """
        radiant_buffer = RolloutBuffer()
        dire_buffer = RolloutBuffer()
        
        for episode in range(num_episodes):
            verbose = (verbose_interval > 0) and ((episode + 1) % verbose_interval == 0)
            
            r_rollout, d_rollout = self.run_episode(
                deterministic=deterministic,
                verbose=verbose,
            )
            
            radiant_buffer.add_rollout(r_rollout)
            dire_buffer.add_rollout(d_rollout)
            
            if verbose:
                r_return = sum(t.reward for t in r_rollout.transitions)
                d_return = sum(t.reward for t in d_rollout.transitions)
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"R_steps={len(r_rollout)}, D_steps={len(d_rollout)}, "
                      f"R_return={r_return:+.4f}, D_return={d_return:+.4f}")
        
        return radiant_buffer, dire_buffer
    
    def evaluate(
        self,
        num_episodes: int = 10,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        评估当前policy（计算平均胜率）
        
        Returns:
            stats: 统计信息
        """
        win_probs = []
        radiant_returns = []
        dire_returns = []
        
        for i in range(num_episodes):
            r_rollout, d_rollout = self.run_episode(
                deterministic=True,
                verbose=verbose and (i == 0),  # 只打印第一个episode
            )
            
            r_return = sum(t.reward for t in r_rollout.transitions)
            d_return = sum(t.reward for t in d_rollout.transitions)
            
            win_probs.append(r_return)  # 天辉胜率
            radiant_returns.append(r_return)
            dire_returns.append(d_return)
        
        return {
            'mean_win_prob': np.mean(win_probs),
            'std_win_prob': np.std(win_probs),
            'mean_radiant_return': np.mean(radiant_returns),
            'mean_dire_return': np.mean(dire_returns),
            'mean_episode_length': 22,  # CM模式固定22步
        }
