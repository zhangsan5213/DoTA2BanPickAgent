"""
ELO评分管理器 - 基于历史模型库的ELO评分系统

设计：
1. 所有模型保存到本地目录
2. 使用JSON记录所有模型的ELO分数
3. 新模型保存时，随机抽取8个历史模型对战
4. 更新参与对战的所有模型分数
"""
import os
import json
import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

from model.bp_agent import BPActorCritic
from model.win_rate_oracle import WinRateOracle
from bp_framework.environment import Team


@dataclass
class ModelELOInfo:
    """模型ELO信息"""
    model_id: str  # 模型唯一标识（如文件名）
    elo: float
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_updated: int = 0  # 最后更新的iteration


class ELORatingSystem:
    """ELO评分系统"""
    
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.models: Dict[str, ModelELOInfo] = {}
    
    def get_or_create(self, model_id: str) -> ModelELOInfo:
        """获取或创建模型ELO信息"""
        if model_id not in self.models:
            self.models[model_id] = ModelELOInfo(
                model_id=model_id,
                elo=self.initial_rating,
            )
        return self.models[model_id]
    
    def update_match(self, winner_id: str, loser_id: str, draw: bool = False):
        """更新单场比赛结果"""
        winner = self.get_or_create(winner_id)
        loser = self.get_or_create(loser_id)
        
        # 计算期望胜率
        expected_w = 1.0 / (1.0 + 10.0 ** ((loser.elo - winner.elo) / 400.0))
        expected_l = 1.0 - expected_w
        
        # 实际得分
        if draw:
            score_w, score_l = 0.5, 0.5
        else:
            score_w, score_l = 1.0, 0.0
        
        # 更新ELO
        winner.elo += self.k_factor * (score_w - expected_w)
        loser.elo += self.k_factor * (score_l - expected_l)
        
        # 更新统计
        winner.games_played += 1
        loser.games_played += 1
        
        if draw:
            winner.draws += 1
            loser.draws += 1
        else:
            winner.wins += 1
            loser.losses += 1
    
    def get_rankings(self) -> List[ModelELOInfo]:
        """获取排名列表（按ELO降序）"""
        return sorted(self.models.values(), key=lambda x: x.elo, reverse=True)
    
    def to_dict(self) -> dict:
        """转为字典用于JSON序列化"""
        return {
            'initial_rating': self.initial_rating,
            'k_factor': self.k_factor,
            'models': {k: asdict(v) for k, v in self.models.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ELORatingSystem':
        """从字典加载"""
        system = cls(
            initial_rating=data.get('initial_rating', 1500.0),
            k_factor=data.get('k_factor', 32.0)
        )
        for model_id, model_data in data.get('models', {}).items():
            system.models[model_id] = ModelELOInfo(**model_data)
        return system


def _run_elo_match_worker(
    model_a_path: str,
    model_b_path: str,
    model_config: Dict,
    oracle_path: str,
    device: str,
    games_per_side: int,
) -> Tuple[str, str, float, float]:
    """
    ELO对战worker（模块级别函数用于多进程）
    
    Args:
        model_a_path: 模型A路径
        model_b_path: 模型B路径
        model_config: 模型配置
        oracle_path: Oracle路径
        device: 计算设备
        games_per_side: 每方作为Radiant的场次
    
    Returns:
        (model_a_id, model_b_id, a_wins, b_wins)
    """
    import os
    from bp_framework.engine import BPEngine
    from bp_framework.environment import Team
    from model.win_rate_oracle import WinRateOracle
    from model.bp_agent import BPActorCritic
    from utils.player_preference_sampler import sample_player_preference
    
    model_a_id = Path(model_a_path).stem
    model_b_id = Path(model_b_path).stem
    
    # 创建模型A
    model_a = BPActorCritic(**model_config).to(device)
    ckpt_a = torch.load(model_a_path, map_location=device)
    state_dict = ckpt_a.get('model_state_dict', ckpt_a)
    model_a.load_state_dict(state_dict)
    model_a.eval()
    
    # 创建模型B
    model_b = BPActorCritic(**model_config).to(device)
    ckpt_b = torch.load(model_b_path, map_location=device)
    state_dict = ckpt_b.get('model_state_dict', ckpt_b)
    model_b.load_state_dict(state_dict)
    model_b.eval()
    
    # 加载Oracle
    oracle = WinRateOracle(
        embed_dim=128,
        nhead=4,
        num_layers=6,
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
    
    def generate_player_feats(num_heroes=160):
        """生成随机玩家特征"""
        feats = torch.zeros(5, num_heroes)
        for pos in range(1, 6):
            heroes = sample_player_preference(position=pos, m=3, n=5, random_seed=None)
            for h in heroes:
                if 1 <= h['id'] <= num_heroes:
                    feats[pos - 1, h['id'] - 1] = h['win_rate']
        return feats
    
    a_wins, b_wins = 0.0, 0.0
    
    # 进行多场比赛，交替阵营
    for game_idx in range(games_per_side * 2):
        # 生成玩家特征
        r_feats = generate_player_feats().to(device)
        d_feats = generate_player_feats().to(device)
        
        if game_idx % 2 == 0:
            # A作为Radiant，B作为Dire
            # 使用A的模型作为engine（A是Radiant）
            engine = BPEngine(
                actor_critic=model_a,
                oracle=oracle,
                device=device,
                first_team=Team.RADIANT,
            )
            r_rollout, d_rollout = engine.run_episode(
                deterministic=True,
                radiant_player_feats=r_feats,
                dire_player_feats=d_feats,
                verbose=False,
            )
            # A是Radiant
            reward = sum(t.reward for t in r_rollout.transitions)
        else:
            # B作为Radiant，A作为Dire
            # 使用B的模型作为engine（B是Radiant）
            engine = BPEngine(
                actor_critic=model_b,
                oracle=oracle,
                device=device,
                first_team=Team.RADIANT,
            )
            r_rollout, d_rollout = engine.run_episode(
                deterministic=True,
                radiant_player_feats=r_feats,
                dire_player_feats=d_feats,
                verbose=False,
            )
            # A是Dire，看Dire的reward
            reward = sum(t.reward for t in d_rollout.transitions)
        
        # 判断胜负
        if reward > 0.01:  # A赢
            a_wins += 1.0
        elif reward < -0.01:  # B赢
            b_wins += 1.0
        else:  # 平局
            a_wins += 0.5
            b_wins += 0.5
    
    return model_a_id, model_b_id, a_wins, b_wins


# 保留原函数名作为别名，用于兼容
_run_single_match = _run_elo_match_worker


class ELOManager:
    """
    ELO管理器 - 管理历史模型库和ELO评分
    """
    
    def __init__(
        self,
        models_dir: str,
        model_config: Optional[Dict] = None,
        oracle_path: str = '',
        device: str = 'cpu',
        num_opponents: int = 8,
        games_per_match: int = 10,
        initial_elo: float = 1500.0,
        k_factor: float = 32.0,
    ):
        """
        Args:
            models_dir: 模型保存目录
            model_config: 模型配置
            oracle_path: Oracle路径
            device: 计算设备
            num_opponents: 每次对战的对手数量
            games_per_match: 每对模型之间的比赛场次
            initial_elo: 初始ELO
            k_factor: ELO更新系数
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_config = model_config or {}
        self.oracle_path = oracle_path
        self.device = device
        self.num_opponents = num_opponents
        self.games_per_match = games_per_match
        
        self.elo_system = ELORatingSystem(initial_elo, k_factor)
        self.elo_json_path = self.models_dir / 'elo_ratings.json'
        
        # 加载已有ELO记录
        self._load_elo_ratings()
    
    def _load_elo_ratings(self):
        """从JSON加载ELO评分"""
        if self.elo_json_path.exists():
            with open(self.elo_json_path, 'r') as f:
                data = json.load(f)
            self.elo_system = ELORatingSystem.from_dict(data)
            print(f"Loaded ELO ratings for {len(self.elo_system.models)} models")
    
    def _save_elo_ratings(self):
        """保存ELO评分到JSON"""
        with open(self.elo_json_path, 'w') as f:
            json.dump(self.elo_system.to_dict(), f, indent=2)
    
    def _get_all_model_files(self) -> List[Path]:
        """获取所有模型文件列表"""
        return list(self.models_dir.glob('*.pth'))
    
    def _sample_opponents(self, exclude_id: str) -> List[Path]:
        """
        随机采样对手
        
        Args:
            exclude_id: 需要排除的模型ID（新模型本身）
        
        Returns:
            对手模型文件路径列表
        """
        all_models = self._get_all_model_files()
        
        # 排除当前模型
        available = [p for p in all_models if p.stem != exclude_id]
        
        if len(available) == 0:
            return []
        
        # 随机采样
        num_to_sample = min(self.num_opponents, len(available))
        sampled = random.sample(available, num_to_sample)
        
        return sampled
    
    def initialize_models(self, model: BPActorCritic, iteration: int = 0) -> List[str]:
        """
        初始化8个模型并保存
        
        Args:
            model: 基础模型（用于复制或参考）
            iteration: 当前迭代
        
        Returns:
            初始化的模型ID列表
        """
        print(f"\n{'='*60}")
        print(f"Initializing 8 models with random weights...")
        print(f"{'='*60}")
        
        model_ids = []
        
        for i in range(8):
            new_model = BPActorCritic(**self.model_config).to(self.device)
            
            # 复制基础模型并添加噪声
            with torch.no_grad():
                for param, base_param in zip(new_model.parameters(), model.parameters()):
                    noise = torch.randn_like(param) * 0.01
                    param.copy_(base_param + noise)
            
            # 保存模型
            model_id = f"init_{i:02d}_iter_{iteration:04d}"
            save_path = self.models_dir / f"{model_id}.pth"
            
            checkpoint = {
                'model_state_dict': new_model.state_dict(),
                'iteration': iteration,
                'model_id': model_id,
            }
            torch.save(checkpoint, save_path)
            
            # 初始化ELO
            info = self.elo_system.get_or_create(model_id)
            info.last_updated = iteration
            
            model_ids.append(model_id)
            print(f"  Saved {model_id}")
        
        # 保存ELO记录
        self._save_elo_ratings()
        
        return model_ids
    
    def run_tournament_for_new_model(
        self,
        new_model_path: str,
        iteration: int,
        use_multiprocessing: bool = True,
    ) -> ModelELOInfo:
        """
        为新模型运行ELO定分赛
        
        流程：
        1. 随机抽取8个历史对手
        2. 与新模型两两对战
        3. 更新所有参与模型的ELO分数
        4. 保存更新后的ELO记录
        
        Args:
            new_model_path: 新模型文件路径
            iteration: 当前迭代
            use_multiprocessing: 是否并行
        
        Returns:
            新模型的ELO信息
        """
        new_model_id = Path(new_model_path).stem
        
        print(f"\n{'='*60}")
        print(f"ELO Tournament for {new_model_id}")
        print(f"{'='*60}")
        
        # 采样对手
        opponents = self._sample_opponents(new_model_id)
        
        if len(opponents) == 0:
            print("No opponents available, initializing with base ELO")
            new_info = self.elo_system.get_or_create(new_model_id)
            new_info.last_updated = iteration
            self._save_elo_ratings()
            return new_info
        
        print(f"Selected {len(opponents)} opponents: {[p.stem for p in opponents]}")
        
        # 准备对战列表（每个对手比8场）
        matches = []
        for opp_path in opponents:
            for _ in range(8):  # 比8场
                matches.append((new_model_path, str(opp_path)))
        
        games_per_side = (self.games_per_match + 1) // 2
        
        print(f"Running {len(matches)} matches ({len(opponents)} opponents × 8 rounds), {self.games_per_match} games each...")
        
        results = self.run_matches(matches, games_per_side, use_multiprocessing)
        
        # 汇总结果（同一对手的多场比赛合并）
        opponent_results = {}
        for model_a_id, model_b_id, a_wins, b_wins in results:
            # model_a_id 是新模型，model_b_id 是对手
            opp_id = model_b_id
            if opp_id not in opponent_results:
                opponent_results[opp_id] = [0.0, 0.0]  # [新模型总胜场, 对手总胜场]
            opponent_results[opp_id][0] += a_wins
            opponent_results[opp_id][1] += b_wins
        
        # 更新ELO分数（基于汇总结果）
        print(f"\nUpdating ELO ratings...")
        for opp_id, (new_wins, opp_wins) in opponent_results.items():
            if new_wins > opp_wins:
                self.elo_system.update_match(new_model_id, opp_id, draw=False)
            elif opp_wins > new_wins:
                self.elo_system.update_match(opp_id, new_model_id, draw=False)
            else:
                self.elo_system.update_match(new_model_id, opp_id, draw=True)
        
        # 更新新模型的时间戳
        new_info = self.elo_system.get_or_create(new_model_id)
        new_info.last_updated = iteration
        
        # 保存ELO记录
        self._save_elo_ratings()
        
        # 打印结果
        print(f"\nELO Results for {new_model_id}:")
        print(f"  ELO: {new_info.elo:.1f}")
        print(f"  Record: {new_info.wins}W/{new_info.losses}L/{new_info.draws}D")
        
        # 打印所有参与模型的分数
        print(f"\nAll participants:")
        all_participants = [new_model_id] + [p.stem for p in opponents]
        for pid in sorted(all_participants):
            info = self.elo_system.get_or_create(pid)
            print(f"  {pid}: ELO={info.elo:.1f} ({info.wins}W/{info.losses}L)")
        
        return new_info
    
    def run_matches(
        self,
        matches: List[Tuple[str, str]],
        games_per_side: int,
        use_multiprocessing: bool = True,
    ) -> List[Tuple[str, str, float, float]]:
        """
        运行多场比赛（多进程并行）
        
        参考 _collect_single_episode 的实现，支持CPU和GPU多进程
        
        Args:
            matches: 对战列表 [(model_a_path, model_b_path), ...]
            games_per_side: 每方作为Radiant的场次
            use_multiprocessing: 是否使用多进程
        
        Returns:
            对战结果列表
        """
        if len(matches) == 0:
            return []
        
        if use_multiprocessing:
            # 多进程并行（参考 _collect_single_episode）
            num_workers = min(mp.cpu_count(), len(matches), 8)
            print(f"  Using {num_workers} workers for {len(matches)} matches (device: {self.device})...")
            
            results = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        _run_elo_match_worker,
                        model_a,
                        model_b,
                        self.model_config,
                        self.oracle_path,
                        self.device,  # 使用配置的device（CPU或GPU）
                        games_per_side,
                    ): (model_a, model_b) for model_a, model_b in matches
                }
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="ELO Matches", ncols=90):
                    model_a, model_b = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        model_a_id, model_b_id, a_wins, b_wins = result
                        # print(f"    {model_a_id} vs {model_b_id}: {a_wins:.0f}-{b_wins:.0f}")
                    except Exception as e:
                        print(f"    Error in {Path(model_a).stem} vs {Path(model_b).stem}: {e}")
            
            return results
        else:
            # 串行模式（调试用）
            results = []
            for model_a, model_b in tqdm(matches, desc="ELO Matches", ncols=90, total=len(matches)):
                result = _run_elo_match_worker(
                    model_a, model_b,
                    self.model_config,
                    self.oracle_path,
                    self.device,
                    games_per_side,
                )
                results.append(result)
                model_a_id, model_b_id, a_wins, b_wins = result
                # print(f"    {model_a_id} vs {model_b_id}: {a_wins:.0f}-{b_wins:.0f}")
            return results
    
    def get_best_model(self) -> Optional[Tuple[str, str]]:
        """
        获取当前ELO最高的模型
        
        Returns:
            (model_id, model_path) 或 None
        """
        rankings = self.elo_system.get_rankings()
        if len(rankings) == 0:
            return None
        
        best = rankings[0]
        model_path = str(self.models_dir / f"{best.model_id}.pth")
        
        if not os.path.exists(model_path):
            return None
        
        return best.model_id, model_path
    
    def get_elo_info(self, model_id: str) -> Optional[ModelELOInfo]:
        """获取模型的ELO信息"""
        return self.elo_system.models.get(model_id)
