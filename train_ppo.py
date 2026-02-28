"""
PPO训练脚本 - 用于训练BP Agent

设计目标：
1. 高速批量生成trajectory（支持batch推理）
2. 使用PPO-clip更新策略
3. 支持Self-Play（双方共享模型）
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

import json

import torch
import torch.nn as nn
import torch.optim as optim

from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path

from model.bp_agent import BPActorCritic
from model.win_rate_oracle import WinRateOracle
from bp_framework.engine import BPEngine
from bp_framework.environment import Team
from bp_framework.rollout import BPRollout, RolloutBuffer
from bp_framework.elo_manager import ELOManager
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
    total_iterations: int = 2
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
    
    # ELO配置
    elo_num_candidates: int = 8  # 候选模型数量
    elo_games_per_match: int = 10  # 每对模型之间的比赛场次
    elo_initial_rating: float = 1500.0
    elo_k_factor: float = 32.0


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


def _collect_single_episode(
    model_config: Dict,
    actor_critic_state: Dict,
    oracle_path: str,
    device: str,
    num_heroes: int,
) -> Tuple['BPRollout', 'BPRollout']:
    """收集单个episode（模块级别函数用于多进程）"""
    import os
    from bp_framework.engine import BPEngine
    from model.win_rate_oracle import WinRateOracle
    
    # 创建模型
    actor_critic = BPActorCritic(**model_config).to(device)
    actor_critic.load_state_dict(actor_critic_state)
    actor_critic.eval()
    
    # 加载Oracle（复制 _load_oracle 的逻辑）
    oracle = WinRateOracle(
        embed_dim=128,  # oracle_embed_dim
        nhead=4,
        num_layers=6,   # oracle_num_layers
        use_text=True,
        use_player_heroes=True,
    ).to(device)
    
    if os.path.exists(oracle_path):
        checkpoint = torch.load(oracle_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            oracle.load_state_dict(checkpoint['model_state_dict'])
        else:
            oracle.load_state_dict(checkpoint)
    
    oracle.eval()
    for param in oracle.parameters():
        param.requires_grad = False
    
    # 创建engine
    engine = BPEngine(
        actor_critic=actor_critic,
        oracle=oracle,
        device=device,
    )
    
    # 生成随机玩家特征
    r_feats = generate_team_player_features(num_heroes).to(device)
    d_feats = generate_team_player_features(num_heroes).to(device)
    
    # 运行episode
    r_rollout, d_rollout = engine.run_episode(
        deterministic=False,
        radiant_player_feats=r_feats,
        dire_player_feats=d_feats,
        verbose=False,
    )
    
    return r_rollout, d_rollout


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
        'current_picks': [],
        'radiant_player_feats': [],
        'dire_player_feats': [],
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
            all_data['current_picks'].append(t['current_picks'])
            all_data['radiant_player_feats'].append(t['radiant_player_feats'])
            all_data['dire_player_feats'].append(t['dire_player_feats'])
            all_data['actions'].append(torch.tensor(t['action_idx'], dtype=torch.long))
            all_data['old_log_probs'].append(torch.tensor(t['log_prob'], dtype=torch.float32))
            all_data['old_values'].append(torch.tensor(t['value'], dtype=torch.float32))
            all_data['returns'].append(torch.tensor(t['return'], dtype=torch.float32))
            all_data['advantages'].append(torch.tensor(t['advantage'], dtype=torch.float32))
    
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
        
        # 创建输出目录
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # 加载Oracle（用于奖励计算，固定不训练）
        self.oracle = self._load_oracle()
        
        # 模型配置
        self.model_config = {
            'embed_dim': config.embed_dim,
            'nhead': config.nhead,
            'num_layers': config.num_layers,
            'num_heroes': config.num_heroes,
            'use_hero_encoder': config.use_hero_encoder,
            'use_player_heroes': config.use_player_heroes,
            'hero_encoder_dim': 128,
            'use_pick_state': True,
        }
        
        # 初始化ELO管理器
        self.elo_manager = ELOManager(
            models_dir=os.path.join(config.output_dir, 'elo_models'),
            model_config=self.model_config,
            oracle_path=config.oracle_path,
            device=config.device,
            num_opponents=config.elo_num_candidates,  # 8个对手
            games_per_match=config.elo_games_per_match,
            initial_elo=config.elo_initial_rating,
            k_factor=config.elo_k_factor,
        )
        
        # 检查是否需要初始化8个模型
        self._maybe_initialize_elo_models()
        
        # 创建主模型（使用ELO最高的模型，或新建）
        self.actor_critic = BPActorCritic(**self.model_config).to(self.device)
        best_model = self.elo_manager.get_best_model()
        if best_model is not None:
            best_id, best_path = best_model
            checkpoint = torch.load(best_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.actor_critic.load_state_dict(state_dict)
            print(f"\nLoaded best ELO model: {best_id}")
        
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
    
    def _maybe_initialize_elo_models(self):
        """如果需要，初始化8个ELO模型"""
        # 检查是否已有模型
        existing_models = list(Path(self.elo_manager.models_dir).glob('*.pth'))
        
        if len(existing_models) > 0:
            print(f"\nFound {len(existing_models)} existing ELO models")
            return
        
        print(f"\n{'='*60}")
        print(f"Initializing 8 ELO models...")
        print(f"{'='*60}")
        
        # 创建临时模型用于初始化
        temp_model = BPActorCritic(**self.model_config).to(self.device)
        
        # 初始化8个模型
        self.elo_manager.initialize_models(temp_model, iteration=0)
        
        # 运行初始ELO定分赛（让所有初始模型相互对战）
        print(f"\nRunning initial ELO tournament...")
        self._run_elo_for_all_initial_models()
    
    def _run_elo_for_all_initial_models(self):
        """为所有初始模型两两对战定分"""
        import itertools
        
        model_files = self.elo_manager._get_all_model_files()
        
        if len(model_files) < 2:
            return
        
        print(f"Running full tournament for {len(model_files)} models...")
        print(f"Each pair plays 8 matches to reduce variance")
        
        # 生成所有对战组合（每对模型出现8次）
        base_pairs = list(itertools.combinations([str(p) for p in model_files], 2))
        matches = base_pairs * 8  # 每对模型比8场
        
        games_per_side = (self.config.elo_games_per_match + 1) // 2
        
        print(f"Total matches: {len(matches)} ({len(base_pairs)} pairs × 8 rounds)")
        
        # 使用多进程并行（ELO对战统一用CPU）
        results = self.elo_manager.run_matches(matches, games_per_side, use_multiprocessing=True)
        
        # 汇总结果（同一对模型的多场比赛合并）
        pair_results = {}
        for model_a_id, model_b_id, a_wins, b_wins in results:
            pair_key = tuple(sorted([model_a_id, model_b_id]))
            if pair_key not in pair_results:
                pair_results[pair_key] = [0.0, 0.0]
            
            if pair_key[0] == model_a_id:
                pair_results[pair_key][0] += a_wins
                pair_results[pair_key][1] += b_wins
            else:
                pair_results[pair_key][0] += b_wins
                pair_results[pair_key][1] += a_wins
        
        # 更新ELO（基于汇总后的结果）
        for (model_a, model_b), (a_wins, b_wins) in pair_results.items():
            if a_wins > b_wins:
                self.elo_manager.elo_system.update_match(model_a, model_b)
            elif b_wins > a_wins:
                self.elo_manager.elo_system.update_match(model_b, model_a)
            else:
                self.elo_manager.elo_system.update_match(model_a, model_b, draw=True)
        
        # 保存ELO记录
        self.elo_manager._save_elo_ratings()
        
        # 打印结果
        print(f"\nInitial ELO Rankings:")
        rankings = self.elo_manager.elo_system.get_rankings()
        for rank, info in enumerate(rankings, 1):
            print(f"  #{rank}: {info.model_id} - ELO {info.elo:.1f}")
    
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
        """收集trajectory（批量）- 多进程并行版本"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        
        # 确定并行worker数量
        num_workers = min(mp.cpu_count(), num_episodes, 8)
        print(f"  Using {num_workers} workers to collect {num_episodes} episodes...")
        
        # 获取模型配置和状态字典
        model_config = {
            'embed_dim': self.config.embed_dim,
            'nhead': self.config.nhead,
            'num_layers': self.config.num_layers,
            'num_heroes': self.config.num_heroes,
            'use_hero_encoder': self.config.use_hero_encoder,
            'hero_encoder_dim': 128,
            'use_player_heroes': self.config.use_player_heroes,
            'use_pick_state': True,
        }
        actor_critic_state = self.actor_critic.state_dict()
        oracle_path = self.config.oracle_path
        device = self.config.device
        num_heroes = self.config.num_heroes
        
        # 并行收集
        r_buffer = RolloutBuffer()
        d_buffer = RolloutBuffer()
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = [
                executor.submit(
                    _collect_single_episode,
                    model_config,
                    actor_critic_state,
                    oracle_path,
                    device,
                    num_heroes,
                ) for _ in range(num_episodes)
            ]
            
            # 收集结果
            for future in tqdm(as_completed(futures), total=num_episodes, desc="Collecting", ncols=90):
                try:
                    r_rollout, d_rollout = future.result()
                    r_buffer.add_rollout(r_rollout)
                    d_buffer.add_rollout(d_rollout)
                except Exception as e:
                    print(f"Error in episode: {e}")
        
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
                    'current_picks': t.current_picks,
                    'radiant_player_feats': t.radiant_player_feats,
                    'dire_player_feats': t.dire_player_feats,
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
            print(f"  PPO epoch {epoch + 1}/{self.config.ppo_epochs}...")
            # Mini-batch训练
            num_batches = (dataset_size + self.config.mini_batch_size - 1) // self.config.mini_batch_size
            batch_idx = 0
            for start in range(0, dataset_size, self.config.mini_batch_size):
                batch_idx += 1
                if batch_idx % 5 == 0:
                    print(f"    Batch {batch_idx}/{num_batches}...")
                end = start + self.config.mini_batch_size
                mb_indices = indices[start:end]
                
                # 获取mini-batch数据
                mb_hero_ids = batch['hero_ids'][mb_indices]
                mb_action_types = batch['action_types'][mb_indices]
                mb_teams = batch['teams'][mb_indices]
                mb_positions = batch['positions'][mb_indices]
                mb_seq_mask = batch['seq_mask'][mb_indices]
                mb_action_mask = batch['action_mask'][mb_indices]
                mb_current_picks = batch['current_picks'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['old_log_probs'][mb_indices]
                mb_old_values = batch['old_values'][mb_indices]
                mb_returns = batch['returns'][mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # 获取player features（如果可用）
                mb_r_player_feats = batch.get('radiant_player_feats')
                mb_d_player_feats = batch.get('dire_player_feats')
                if mb_r_player_feats is not None:
                    mb_r_player_feats = mb_r_player_feats[mb_indices]
                    mb_d_player_feats = mb_d_player_feats[mb_indices]
                
                # Forward pass
                action_probs, values = self.actor_critic(
                    hero_ids=mb_hero_ids,
                    action_types=mb_action_types,
                    teams=mb_teams,
                    positions=mb_positions,
                    action_mask=mb_action_mask,
                    seq_mask=mb_seq_mask,
                    current_picks=mb_current_picks,
                    radiant_player_feats=mb_r_player_feats,
                    dire_player_feats=mb_d_player_feats,
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
        """
        评估当前策略（显示ELO排名）
        """
        rankings = self.elo_manager.elo_system.get_rankings()
        if len(rankings) == 0:
            return {'elo_rating': self.config.elo_initial_rating}
        
        best = rankings[0]
        return {
            'elo_rating': best.elo,
            'best_model_id': best.model_id,
            'num_models': len(rankings),
        }
    
    def save_checkpoint(self, iteration: int):
        """
        保存checkpoint并进行ELO定分
        
        流程：
        1. 保存当前模型到ELO模型目录
        2. 随机抽取8个历史模型对战
        3. 更新所有参与模型的ELO分数
        """
        print(f"\n{'='*60}")
        print(f"Saving checkpoint at iteration {iteration} with ELO rating...")
        print(f"{'='*60}")
        
        # 1. 保存当前模型到ELO目录
        model_id = f"iter_{iteration:06d}"
        model_path = os.path.join(self.elo_manager.models_dir, f"{model_id}.pth")
        
        checkpoint = {
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': iteration,
            'model_id': model_id,
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")
        
        # 2. 运行ELO定分赛
        new_info = self.elo_manager.run_tournament_for_new_model(
            new_model_path=model_path,
            iteration=iteration,
            use_multiprocessing=True,  # 强制并行
        )
        
        # 3. 显示当前ELO排名
        print(f"\nCurrent ELO Rankings (Top 10):")
        rankings = self.elo_manager.elo_system.get_rankings()[:10]
        for rank, info in enumerate(rankings, 1):
            marker = " <-- NEW" if info.model_id == model_id else ""
            print(f"  #{rank}: {info.model_id} - ELO {info.elo:.1f} ({info.wins}W/{info.losses}L){marker}")
    
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
            
            # 6. 评估（显示ELO信息）
            if iteration % self.config.eval_interval == 0 and iteration > 0:
                print(f"\n[Iter {iteration}] ELO Status...")
                eval_stats = self.evaluate()
                print(f"Best ELO: {eval_stats['elo_rating']:.1f} ({eval_stats['best_model_id']})")
                rankings = self.elo_manager.elo_system.get_rankings()[:8]
                print(f"Top 8: " + ", ".join([f"{r.model_id.split('_')[-1]}:{r.elo:.0f}" for r in rankings]))
            
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
        total_iterations=64000,
        episodes_per_iter=64,
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
