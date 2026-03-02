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
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from typing import List, Dict, Tuple, Optional
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
    log_interval: int = 1
    save_interval: int = 2
    eval_interval: int = 2
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
        
        # 进程池（延迟初始化，避免过早创建）
        self._executor = None
        
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
        
        # 创建主模型（使用最新的模型，或新建）
        self.actor_critic = BPActorCritic(**self.model_config).to(self.device)
        latest_model = self.get_latest_model()
        # latest_model = self.elo_manager.get_best_model()  # 优先使用ELO最高的模型
        if latest_model is not None:
            latest_id, latest_path = latest_model
            checkpoint = torch.load(latest_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.actor_critic.load_state_dict(state_dict)
            print(f"\nLoaded latest model: {latest_id}")
            # print(f"\nLoaded best model: {latest_id}")
        
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
        
        # 进程池最大worker数
        self._max_workers = min(os.cpu_count(), config.episodes_per_iter, 8)
        
        # TensorBoard writer
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = f'runs/bp_agent_exp_{timestamp}'
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    
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
    
    def _get_executor(self):
        """获取进程池（延迟初始化）"""
        if self._executor is None:
            from concurrent.futures import ProcessPoolExecutor
            self._executor = ProcessPoolExecutor(max_workers=self._max_workers)
        return self._executor
    
    def shutdown_executor(self):
        """关闭进程池"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def collect_rollouts(self, num_episodes: int) -> Tuple[RolloutBuffer, RolloutBuffer]:
        """收集trajectory（批量）- 多进程并行版本（使用持久化进程池）"""
        from concurrent.futures import as_completed
        
        print(f"  Using {self._max_workers} workers to collect {num_episodes} episodes...")
        
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
        device = 'cpu'  # 强制使用CPU进行数据收集，避免CUDA内存碎片
        num_heroes = self.config.num_heroes
        
        # 并行收集
        r_buffer = RolloutBuffer()
        d_buffer = RolloutBuffer()
        
        executor = self._get_executor()
        
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
        
        total_kl = 0  # 用于累积KL散度
        
        for epoch in tqdm(range(self.config.ppo_epochs), desc="PPO Epochs", ncols=90):
            # print(f"  PPO epoch {epoch + 1}/{self.config.ppo_epochs}...")
            # Mini-batch训练
            # num_batches = (dataset_size + self.config.mini_batch_size - 1) // self.config.mini_batch_size
            batch_idx = 0
            for start in tqdm(range(0, dataset_size, self.config.mini_batch_size), desc=f"PPO Epoch {epoch + 1}", ncols=90):
                batch_idx += 1
                # if batch_idx % 5 == 0:
                #     print(f"    Batch {batch_idx}/{num_batches}...")
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
                
                # 计算近似KL散度: KL(old||new) ≈ E[log π_old - log π_new]
                with torch.no_grad():
                    kl_div = (mb_old_log_probs - new_log_probs).mean().item()
                    total_kl += kl_div
                
                num_updates += 1
        
        avg_policy_loss = total_policy_loss / num_updates if num_updates > 0 else 0
        avg_value_loss = total_value_loss / num_updates if num_updates > 0 else 0
        avg_kl = total_kl / num_updates if num_updates > 0 else 0
        
        return {
            'loss': total_loss / num_updates if num_updates > 0 else 0,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': total_entropy / num_updates if num_updates > 0 else 0,
            'kl': avg_kl,
        }
    
    def get_latest_model(self) -> Optional[Tuple[str, str]]:
        """
        获取最新的模型（按文件修改时间）
        
        Returns:
            (model_id, model_path) 或 None
        """
        model_files = list(Path(self.elo_manager.models_dir).glob('*.pth'))
        if len(model_files) == 0:
            return None
        
        # 按文件修改时间排序，最新的排在前面
        model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_path = model_files[0]
        latest_id = latest_path.stem
        
        return latest_id, str(latest_path)
    
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
    
    def _run_single_match_with_details(
        self,
        new_model_path: str,
        opponent_path: str,
    ) -> Tuple[bool, List[int], List[int], float, bool]:
        """
        运行单场比赛并返回详细信息
        
        Returns:
            (new_model_won, radiant_picks, dire_picks, win_prob, new_model_as_radiant)
        """
        from bp_framework.environment import Team
        
        # 加载新模型
        new_model = BPActorCritic(**self.model_config).to(self.device)
        ckpt = torch.load(new_model_path, map_location=self.device)
        new_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        new_model.eval()
        
        # 加载对手模型
        opponent = BPActorCritic(**self.model_config).to(self.device)
        ckpt_opp = torch.load(opponent_path, map_location=self.device)
        opponent.load_state_dict(ckpt_opp.get('model_state_dict', ckpt_opp))
        opponent.eval()
        
        # 随机决定阵营（50%概率新模型作为Radiant）
        new_model_as_radiant = random.random() < 0.5
        
        # 生成随机玩家特征
        r_feats = generate_team_player_features(self.config.num_heroes).to(self.device)
        d_feats = generate_team_player_features(self.config.num_heroes).to(self.device)
        
        # 创建完整的对战环境
        from bp_framework.environment import BPEnvironment
        
        env = BPEnvironment(
            num_heroes=self.config.num_heroes,
            first_team=Team.RADIANT,
            device=self.device,
        )
        
        state = env.reset(
            radiant_player_feats=r_feats,
            dire_player_feats=d_feats,
        )
        
        # 运行对战
        while not state.is_terminal:
            state_tensors = env.get_state_for_agent()
            current_team = state.current_team
            
            r_feats_batch = r_feats.unsqueeze(0).to(self.device) if r_feats is not None else None
            d_feats_batch = d_feats.unsqueeze(0).to(self.device) if d_feats is not None else None
            
            # 构建current_picks
            picks = [0] * 10
            for i, hero_id in enumerate(state.radiant_picks):
                if i < 5:
                    picks[i] = hero_id
            for i, hero_id in enumerate(state.dire_picks):
                if i < 5:
                    picks[5 + i] = hero_id
            current_picks = torch.tensor([picks], dtype=torch.long, device=self.device)
            
            # 根据当前阵营选择模型
            if current_team == Team.RADIANT:
                model = new_model if new_model_as_radiant else opponent
            else:
                model = opponent if new_model_as_radiant else new_model
            
            with torch.no_grad():
                action_idx, _, _ = model.select_action(
                    hero_ids=state_tensors['hero_ids'],
                    action_types=state_tensors['action_types'],
                    teams=state_tensors['teams'],
                    positions=state_tensors['positions'],
                    action_mask=state_tensors['action_mask'],
                    seq_mask=state_tensors['seq_mask'],
                    deterministic=True,
                    radiant_player_feats=r_feats_batch,
                    dire_player_feats=d_feats_batch,
                    current_picks=current_picks,
                )
            
            hero_id = action_idx.item() + 1
            state, _, _ = env.step(hero_id)
        
        # 获取最终阵容
        radiant_picks, dire_picks = state.get_final_picks()
        
        # 计算胜负
        win_prob = self._calculate_win_prob(radiant_picks, dire_picks, r_feats, d_feats)
        
        # 判断新模型是否获胜
        if new_model_as_radiant:
            new_model_won = win_prob > 0.5
        else:
            new_model_won = win_prob < 0.5
        
        return new_model_won, radiant_picks, dire_picks, win_prob, new_model_as_radiant
    
    def _calculate_win_prob(
        self,
        radiant_picks: List[int],
        dire_picks: List[int],
        r_feats: torch.Tensor,
        d_feats: torch.Tensor,
    ) -> float:
        """计算Radiant的胜率"""
        with torch.no_grad():
            # 转换pick为0-based索引
            radiant_hero_ids = torch.tensor([radiant_picks], dtype=torch.long, device=self.device) - 1
            dire_hero_ids = torch.tensor([dire_picks], dtype=torch.long, device=self.device) - 1
            
            # 获取英雄属性 (Oracle中有预计算的all_hero_attrs和all_hero_sem)
            radiant_hero_attrs = self.oracle.all_hero_attrs[radiant_hero_ids[0]]
            dire_hero_attrs = self.oracle.all_hero_attrs[dire_hero_ids[0]]
            
            # 添加batch维度 [5] -> [1, 5, features]
            radiant_hero_attrs = radiant_hero_attrs.unsqueeze(0)
            dire_hero_attrs = dire_hero_attrs.unsqueeze(0)
            
            # 获取语义特征
            if self.oracle.all_hero_sem is not None:
                radiant_hero_semantics = self.oracle.all_hero_sem[radiant_hero_ids[0]].unsqueeze(0)
                dire_hero_semantics = self.oracle.all_hero_sem[dire_hero_ids[0]].unsqueeze(0)
            else:
                # 如果不使用text，用零填充
                text_dim = 1024
                radiant_hero_semantics = torch.zeros(1, 5, text_dim, device=self.device)
                dire_hero_semantics = torch.zeros(1, 5, text_dim, device=self.device)
            
            # 处理player_feats
            r_feats_batch = r_feats.unsqueeze(0) if r_feats.dim() == 2 else r_feats
            d_feats_batch = d_feats.unsqueeze(0) if d_feats.dim() == 2 else d_feats
            
            win_prob = self.oracle(
                radiant_hero_ids=radiant_hero_ids,
                radiant_hero_attrs=radiant_hero_attrs,
                radiant_hero_semantics=radiant_hero_semantics,
                dire_hero_ids=dire_hero_ids,
                dire_hero_attrs=dire_hero_attrs,
                dire_hero_semantics=dire_hero_semantics,
                radiant_player_feats=r_feats_batch,
                dire_player_feats=d_feats_batch,
            )
        return win_prob.item()
    
    def _visualize_matches(self, model_id: str, model_path: str, iteration: int):
        """
        可视化对战：与ELO较高的4个对手各打1场，随机选取2胜2负展示最终阵容
        """
        import random
        from pathlib import Path
        
        print(f"\n{'='*60}")
        print(f"Visualizing matches for {model_id}")
        print(f"{'='*60}")
        
        # 获取所有对手（排除自己）
        all_models = self.elo_manager.elo_system.models
        opponents = [
            (mid, info) for mid, info in all_models.items()
            if mid != model_id and info.elo > 0
        ]
        
        if len(opponents) < 4:
            print(f"Not enough opponents ({len(opponents)} < 4), skipping visualization")
            return
        
        # 按ELO排序，取前4
        opponents.sort(key=lambda x: x[1].elo, reverse=True)
        top4_opponents = opponents[:4]
        
        print(f"Selected top 4 opponents by ELO:")
        for opp_id, opp_info in top4_opponents:
            print(f"  {opp_id}: ELO {opp_info.elo:.1f}")
        
        # 与每个对手打1场
        match_results = []
        for opp_id, opp_info in top4_opponents:
            opp_path = os.path.join(self.elo_manager.models_dir, f"{opp_id}.pth")
            if not os.path.exists(opp_path):
                continue
            
            try:
                won, r_picks, d_picks, win_prob, new_as_r = self._run_single_match_with_details(
                    model_path, opp_path
                )
                match_results.append({
                    'opponent_id': opp_id,
                    'opponent_elo': opp_info.elo,
                    'won': won,
                    'radiant_picks': r_picks,
                    'dire_picks': d_picks,
                    'win_prob': win_prob,
                    'new_model_as_radiant': new_as_r,
                })
            except Exception as e:
                print(f"  Error playing against {opp_id}: {e}")
        
        if len(match_results) < 4:
            print(f"Only played {len(match_results)} matches, need at least 4")
            return
        
        # 分离胜负
        wins = [m for m in match_results if m['won']]
        losses = [m for m in match_results if not m['won']]
        
        # 随机选取：优先2胜2负，不够就全展示
        selected = []
        # 尝试各选2个
        win_target = min(2, len(wins))
        loss_target = min(2, len(losses))
        
        # 如果一边不足2个，从另一边补足
        if win_target < 2 and len(losses) > loss_target:
            loss_target = min(4 - win_target, len(losses))
        elif loss_target < 2 and len(wins) > win_target:
            win_target = min(4 - loss_target, len(wins))
        
        if win_target > 0:
            selected.extend(random.sample(wins, win_target))
        if loss_target > 0:
            selected.extend(random.sample(losses, loss_target))
        
        if len(selected) == 0:
            print(f"No matches to show")
            return
        
        # 打印结果
        print(f"\n{'='*60}")
        print(f"Match Visualization (2 Wins + 2 Losses)")
        print(f"{'='*60}")
        
        # 加载英雄名称映射
        hero_names = self._load_hero_names()
        
        for i, match in enumerate(selected, 1):
            result_str = "WIN" if match['won'] else "LOSS"
            opp_id = match['opponent_id']
            opp_elo = match['opponent_elo']
            new_as_r = match['new_model_as_radiant']
            
            print(f"\n--- Match {i}: {result_str} vs {opp_id} (ELO: {opp_elo:.1f}) ---")
            
            # 确定新模型和对手的英雄
            if new_as_r:
                new_picks = match['radiant_picks']
                opp_picks = match['dire_picks']
                new_team = "RADIANT"
                opp_team = "DIRE"
            else:
                new_picks = match['dire_picks']
                opp_picks = match['radiant_picks']
                new_team = "DIRE"
                opp_team = "RADIANT"
            
            # 打印阵容
            new_names = [hero_names.get(hid, f"Hero_{hid}") for hid in new_picks]
            opp_names = [hero_names.get(hid, f"Hero_{hid}") for hid in opp_picks]
            
            print(f"  [{new_team}] New Model: {new_names}")
            print(f"  [{opp_team}] {opp_id}:    {opp_names}")
            print(f"  Oracle Win Prob: {match['win_prob']:.4f}")
        
        print(f"{'='*60}\n")
    
    def _load_hero_names(self) -> Dict[int, str]:
        """加载英雄名称映射"""
        # 优先从 hero_winrates.json 加载（包含ID到名称的映射）
        hero_winrates_path = Path('./data/hero_winrates.json')
        if hero_winrates_path.exists():
            with open(hero_winrates_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 格式: {hero_id: {'name': hero_name, 'winrate': ...}, ...}
                if isinstance(data, dict):
                    return {
                        int(k): v['name'] 
                        for k, v in data.items() 
                        if isinstance(v, dict) and 'name' in v
                    }
        
        # 尝试 hero_names.json
        hero_names_path = Path('./data/hero_names.json')
        if hero_names_path.exists():
            with open(hero_names_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {int(k): v for k, v in data.items()}
                elif isinstance(data, list):
                    return {int(item['id']): item['name'] for item in data if 'id' in item and 'name' in item}
        
        # 尝试 heroes.json
        hero_data_path = Path('./data/heroes.json')
        if hero_data_path.exists():
            with open(hero_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return {int(item.get('id', i+1)): item.get('name', f"Hero_{i+1}") 
                            for i, item in enumerate(data)}
        
        # 如果都没有，返回空字典，使用默认命名
        return {}
    
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
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        model_id = f"model_{timestamp}"
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
        
        # 记录ELO到TensorBoard
        new_model_elo = self.elo_manager.elo_system.get_or_create(model_id).elo
        self.writer.add_scalar('elo/rating', new_model_elo, iteration)
        self.writer.add_scalar('elo/wins', new_info.wins, iteration)
        self.writer.add_scalar('elo/losses', new_info.losses, iteration)
        
        # 4. 可视化对战：与ELO较高的4个对手各打1场，随机选取2胜2负展示
        self._visualize_matches(model_id, model_path, iteration)
    
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
            
            # 记录到TensorBoard
            self.writer.add_scalar('metrics/actor_loss', update_info.get('policy_loss', 0), iteration)
            self.writer.add_scalar('metrics/value_loss', update_info.get('value_loss', 0), iteration)
            self.writer.add_scalar('metrics/kl_divergence', update_info.get('kl', 0), iteration)
            self.writer.add_scalar('metrics/entropy', update_info.get('entropy', 0), iteration)
            
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
        
        # 关闭TensorBoard writer
        self.writer.close()
        
        # 关闭进程池
        self.shutdown_executor()
        
        print(f"TensorBoard logs closed")


def main():
    config = PPOConfig(
        total_iterations=64000,
        episodes_per_iter=64,
        batch_size=256,
        mini_batch_size=64,
        ppo_epochs=4,
        save_interval=4,
        eval_interval=4,
        lr=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
