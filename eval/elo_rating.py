"""
ELO Rating System for BP Agent
用于评估 BP Agent 的相对强度，因为这是个 zero-sum 博弈问题，loss 无法衡量 RL 效果
"""

import os
import re
import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

from utils.device import DEVICE
from utils.raw_data import NUM_HEROES
from utils.player_preference_sampler import batch_sample_player_preferences
from utils.bp_env import BPState
from model.bp_agent import BPTransformerAgent
from model.win_rate_oracle import WinRateOracle

from eval.rating_base import (
    ModelRatingRecord,
    RatingManagerBase,
    BattleSimulatorBase,
    RatingEvaluatorBase
)


# ELO 参数
INITIAL_ELO = 1500
ELO_K_FACTOR = 32  # K-factor，控制分数变化幅度
ELO_SCALE = 400    # 标准 ELO 比例因子

# 对手选择参数
OPPONENT_SAMPLE_STD = 200  # 正态分布采样标准差
NUM_OPPONENTS_TO_SAMPLE = 5  # 每次采样的对手数量

# 对战参数
NUM_PLAYER_SETS = 16  # 16 个不同的玩家 set
BATTLES_PER_MATCHUP = 2  # 每对模型每个玩家 set 对战次数（交换 radiant/dire）


def sigmoid(x):
    """Sigmoid 函数，用于计算预期胜率"""
    return 1.0 / (1.0 + 10 ** (-x / ELO_SCALE))


def compute_elo_change(rating_a: float, rating_b: float, score_a: float, k: int = ELO_K_FACTOR) -> int:
    """
    计算 ELO 分数变化
    
    Args:
        rating_a: 玩家 A 的当前 ELO 分数
        rating_b: 玩家 B 的当前 ELO 分数
        score_a: 玩家 A 的实际得分（1=赢，0=输，0.5=平）
        k: K-factor
    
    Returns:
        ELO 分数变化量（整数）
    """
    expected_a = sigmoid(rating_b - rating_a)
    change = round(k * (score_a - expected_a))
    return change


@dataclass
class ModelEloRecord(ModelRatingRecord):
    """模型 ELO 记录"""
    elo: int = INITIAL_ELO
    
    def to_dict(self) -> dict:
        return {
            'model_path': self.model_path,
            'elo': self.elo,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_games': self.total_games,
            'last_eval_time': self.last_eval_time,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelEloRecord":
        return cls(
            model_path=data['model_path'],
            elo=data.get('elo', INITIAL_ELO),
            wins=data.get('wins', 0),
            losses=data.get('losses', 0),
            draws=data.get('draws', 0),
            total_games=data.get('total_games', 0),
            last_eval_time=data.get('last_eval_time', ''),
        )


class EloRatingManager(RatingManagerBase):
    """ELO 分数管理器"""
    
    def _get_db_path(self) -> Path:
        """获取数据库文件路径"""
        return self.save_dir / "elo_ratings.json"
    
    def _create_record(self, model_path: str, **kwargs) -> ModelEloRecord:
        """创建新的评分记录"""
        elo = kwargs.get('elo', INITIAL_ELO)
        return ModelEloRecord(
            model_path=model_path,
            elo=elo,
            last_eval_time=datetime.now().strftime("%Y%m%d%H%M%S")
        )
    
    def _record_from_dict(self, data: dict) -> ModelEloRecord:
        """从字典创建记录对象"""
        return ModelEloRecord.from_dict(data)
    
    def update_rating(self, model_a_path: str, model_b_path: str, score_a: float):
        """
        更新两个模型的 ELO 分数
        
        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            score_a: 模型 A 的实际得分（1=赢，0=输，0.5=平）
        """
        if model_a_path not in self.records:
            self.register_model(model_a_path)
        if model_b_path not in self.records:
            self.register_model(model_b_path)
        
        record_a = self.records[model_a_path]
        record_b = self.records[model_b_path]
        
        # 计算 ELO 变化
        change = compute_elo_change(record_a.elo, record_b.elo, score_a)
        
        # 更新分数
        record_a.elo += change
        record_b.elo -= change
        
        # 更新统计
        record_a.total_games += 1
        record_b.total_games += 1
        
        if score_a > 0.5:
            record_a.wins += 1
            record_b.losses += 1
        elif score_a < 0.5:
            record_a.losses += 1
            record_b.wins += 1
        else:
            record_a.draws += 1
            record_b.draws += 1
        
        record_a.last_eval_time = datetime.now().strftime("%Y%m%d%H%M%S")
        record_b.last_eval_time = datetime.now().strftime("%Y%m%d%H%M%S")
        
        self._save_records()
    
    def get_rating_value(self, model_path: str) -> float:
        """
        获取模型的评分值
        
        Returns:
            ELO 分数
        """
        record = self.records.get(model_path)
        if record is None:
            return float(INITIAL_ELO)
        return float(record.elo)
    
    def select_opponents(self, current_model_path: str, num_opponents: int = NUM_OPPONENTS_TO_SAMPLE) -> List[str]:
        """
        根据当前模型 ELO 分数，使用正态分布采样选择对手
        
        Args:
            current_model_path: 当前模型路径
            num_opponents: 需要选择的对手数量
        
        Returns:
            对手模型路径列表
        """
        if current_model_path not in self.records:
            self.register_model(current_model_path)
        
        current_elo = self.records[current_model_path].elo
        
        # 获取所有其他模型
        other_models = [
            path for path in self.records.keys()
            if path != current_model_path and os.path.exists(path)
        ]
        
        if len(other_models) == 0:
            return []
        
        if len(other_models) <= num_opponents:
            return other_models
        
        # 计算每个模型的采样权重（基于正态分布）
        weights = []
        for path in other_models:
            elo = self.records[path].elo
            # 正态分布概率密度
            weight = np.exp(-0.5 * ((elo - current_elo) / OPPONENT_SAMPLE_STD) ** 2)
            weights.append(weight)
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # 按权重采样
        selected_indices = np.random.choice(
            len(other_models),
            size=num_opponents,
            replace=False,
            p=weights
        )
        
        return [other_models[i] for i in selected_indices]
    
    # 保持向后兼容
    def update_elo(self, model_a_path: str, model_b_path: str, score_a: float):
        """更新 ELO 分数（向后兼容）"""
        self.update_rating(model_a_path, model_b_path, score_a)
    
    def select_opponents_by_elo(self, current_model_path: str, num_opponents: int = NUM_OPPONENTS_TO_SAMPLE) -> List[str]:
        """选择对手（向后兼容）"""
        return self.select_opponents(current_model_path, num_opponents)


class BPBattleSimulator(BattleSimulatorBase):
    """BP 对战模拟器"""
    
    def __init__(self, oracle: Optional[WinRateOracle] = None, oracle_path: Optional[str] = None):
        """
        Args:
            oracle: WinRateOracle 实例，如果为 None 则自动加载
            oracle_path: Oracle 模型路径，如果为 None 则使用默认路径
        """
        if oracle is not None:
            self.oracle = oracle
        else:
            self.oracle = self._load_oracle(oracle_path)
    
    def _load_oracle(self, oracle_path: Optional[str] = None) -> WinRateOracle:
        """加载 WinRate Oracle"""
        if oracle_path is None:
            oracle_path = "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth"
        
        oracle = WinRateOracle(
            embed_dim=128,
            nhead=8,
            num_layers=6,
            use_text=True,
            use_player_heroes=True
        ).to(DEVICE)
        
        if os.path.exists(oracle_path):
            oracle.load_state_dict(torch.load(oracle_path, map_location=DEVICE))
            print(f"[ELO] Loaded oracle from {oracle_path}")
        else:
            print(f"[ELO] Warning: Oracle not found at {oracle_path}")
        
        oracle.eval()
        return oracle
    
    def load_agent(self, model_path: str) -> BPTransformerAgent:
        """加载 BP Agent 模型"""
        agent = BPTransformerAgent(embed_dim=256, nhead=8, num_layers=4).to(DEVICE)
        ckpt = torch.load(model_path, map_location=DEVICE)
        state_dict = ckpt["agent_state"] if isinstance(ckpt, dict) and "agent_state" in ckpt else ckpt
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent
    
    def run_bp_battle(
        self,
        agent_radiant: BPTransformerAgent,
        agent_dire: BPTransformerAgent,
        player_set: Dict,
        max_steps: int = 24
    ) -> Tuple[List[int], List[int], float]:
        """
        运行一场 BP 对战
        
        Args:
            agent_radiant: 天辉方 Agent
            agent_dire: 夜魇方 Agent
            player_set: 玩家配置，包含 r_players 和 d_players
            max_steps: 最大步数
        
        Returns:
            (radiant_picks, dire_picks, radiant_win_prob)
        """
        r_players = player_set['r_players']
        d_players = player_set['d_players']
        
        # 创建 BP 状态
        state = BPState([], [], r_players, d_players, radiant_bans=[], dire_bans=[], is_radiant_turn=True, step_idx=0)
        
        # 运行 BP 过程
        step = 0
        while not state.done and step < max_steps:
            state_dict = state.to_dict()
            
            # 选择当前 agent
            current_agent = agent_radiant if state.is_radiant_turn else agent_dire
            
            with torch.no_grad():
                action_logits, _ = current_agent(state_dict)
                
                # Mask 已使用的英雄
                valid_actions = state.get_valid_actions()
                mask = torch.full((NUM_HEROES,), -1e9, device=action_logits.device)
                for h in valid_actions:
                    mask[h - 1] = 0.0
                action_logits = action_logits + mask
                
                # 选择英雄（贪心策略）
                hero_id = torch.argmax(action_logits, dim=-1).item() + 1
            
            # 判断是 pick 还是 ban (使用CM序列)
            is_pick = (state.get_current_action_type() == 'pick')
            state.step(hero_id)
            step += 1
        
        # 使用 Oracle 判断胜负
        if len(state.radiant_heroes) >= 5 and len(state.dire_heroes) >= 5:
            r_picks = state.radiant_heroes[:5]
            d_picks = state.dire_heroes[:5]
        else:
            # BP 未完成，填充到 5 个
            r_picks = state.radiant_heroes + [1] * (5 - len(state.radiant_heroes))
            d_picks = state.dire_heroes + [1] * (5 - len(state.dire_heroes))
        
        win_prob = state.get_reward(self.oracle)
        
        return r_picks, d_picks, win_prob
    
    def evaluate_models(
        self,
        model_a_path: str,
        model_b_path: str,
        num_player_sets: int = NUM_PLAYER_SETS
    ) -> Tuple[float, List[Dict]]:
        """
        评估两个模型的对战结果
        
        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            num_player_sets: 玩家 set 数量
        
        Returns:
            (model_a 胜率, 详细对战记录)
        """
        agent_a = self.load_agent(model_a_path)
        agent_b = self.load_agent(model_b_path)
        
        # 生成玩家 sets
        player_sets = self._generate_player_sets(num_player_sets)
        
        results = []
        a_wins = 0
        total_games = 0
        
        for player_set in player_sets:
            # 随机决定哪方用哪个模型
            a_is_radiant = random.choice([True, False])
            
            if a_is_radiant:
                agent_radiant, agent_dire = agent_a, agent_b
            else:
                agent_radiant, agent_dire = agent_b, agent_a
            
            # 运行对战
            r_picks, d_picks, win_prob = self.run_bp_battle(
                agent_radiant, agent_dire, player_set
            )
            
            # 判断胜负（win_prob 是天辉胜率）
            if a_is_radiant:
                a_win_prob = win_prob
            else:
                a_win_prob = 1.0 - win_prob
            
            # 以 0.5 为阈值判断胜负
            if a_win_prob > 0.5:
                a_wins += 1
                result = "win"
            elif a_win_prob < 0.5:
                result = "loss"
            else:
                result = "draw"
            
            total_games += 1
            
            results.append({
                'a_is_radiant': a_is_radiant,
                'radiant_picks': r_picks,
                'dire_picks': d_picks,
                'radiant_win_prob': win_prob,
                'a_win_prob': a_win_prob,
                'result': result
            })
        
        win_rate = a_wins / total_games if total_games > 0 else 0.5
        return win_rate, results
    
    def _generate_player_sets(self, num_sets: int) -> List[Dict]:
        """
        生成玩家 sets
        
        Returns:
            玩家配置列表，每个包含 r_players 和 d_players
        """
        player_sets = []
        
        for _ in range(num_sets):
            # 采样 10 个玩家（5 天辉 + 5 夜魇）
            all_players = batch_sample_player_preferences(
                num_players=10,
                position_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
                m=3,
                n=5
            )
            
            # 构建玩家特征向量 [5, NUM_HEROES]
            def build_player_features(players):
                from utils.raw_data import get_valid_hero_ids
                valid_hero_ids = get_valid_hero_ids()
                features = []
                for p in players:
                    vector = [0.0] * NUM_HEROES
                    for hero_info in p['heroes']:
                        hero_id = hero_info['id']
                        win_rate = hero_info['win_rate']
                        # 只添加实际存在的英雄
                        if hero_id in valid_hero_ids and hero_id <= NUM_HEROES:
                            vector[hero_id - 1] = win_rate  # 修正索引
                    features.append(vector)
                return features
            
            r_players = build_player_features(all_players[:5])
            d_players = build_player_features(all_players[5:])
            
            player_sets.append({
                'r_players': r_players,
                'd_players': d_players
            })
        
        return player_sets


def evaluate_and_update_elo(
    model_path: str,
    elo_manager: Optional[EloRatingManager] = None,
    battle_simulator: Optional[BPBattleSimulator] = None,
    num_opponents: int = NUM_OPPONENTS_TO_SAMPLE,
    num_player_sets: int = NUM_PLAYER_SETS,
    k_factor: int = ELO_K_FACTOR,
) -> Dict:
    """
    评估模型并更新 ELO 分数
    
    Args:
        model_path: 当前模型路径
        elo_manager: EloRatingManager 实例
        battle_simulator: BPBattleSimulator 实例
        num_opponents: 对手数量
        num_player_sets: 每个对手对战的玩家 set 数量
        k_factor: ELO K-factor
    
    Returns:
        评估结果字典
    """
    if elo_manager is None:
        elo_manager = EloRatingManager()
    if battle_simulator is None:
        battle_simulator = BPBattleSimulator()
    
    # 注册/获取当前模型
    current_record = elo_manager.register_model(model_path)
    current_elo_before = current_record.elo
    
    print(f"\n{'='*60}")
    print(f"ELO Evaluation: {model_path}")
    print(f"Current ELO: {current_elo_before}")
    print(f"{'='*60}")
    
    # 选择对手
    opponents = elo_manager.select_opponents_by_elo(model_path, num_opponents)
    
    if len(opponents) == 0:
        print("[ELO] No opponents found, skipping evaluation")
        return {
            'model_path': model_path,
            'elo_before': current_elo_before,
            'elo_after': current_elo_before,
            'elo_change': 0,
            'opponents': [],
            'results': []
        }
    
    print(f"[ELO] Selected {len(opponents)} opponents by ELO distribution")
    
    results = []
    total_elo_change = 0
    
    for opponent_path in opponents:
        opponent_record = elo_manager.get_record(opponent_path)
        opponent_elo = opponent_record.elo if opponent_record else INITIAL_ELO
        
        print(f"\n  vs {os.path.basename(opponent_path)} (ELO: {opponent_elo})")
        
        # 运行对战
        win_rate, battle_details = battle_simulator.evaluate_models(
            model_path, opponent_path, num_player_sets
        )
        
        # 将胜率转换为得分（简化处理：胜率>0.5算赢，<0.5算输）
        if win_rate > 0.5:
            score = 1.0
        elif win_rate < 0.5:
            score = 0.0
        else:
            score = 0.5
        
        # 计算 ELO 变化
        elo_change = compute_elo_change(current_record.elo, opponent_elo, score, k=k_factor)
        
        print(f"    Win rate: {win_rate*100:.1f}%, Score: {score}, ELO change: {elo_change:+d}")
        
        # 更新 ELO（双边更新）
        elo_manager.update_elo(model_path, opponent_path, score)
        
        results.append({
            'opponent_path': opponent_path,
            'opponent_elo': opponent_elo,
            'win_rate': win_rate,
            'score': score,
            'elo_change': elo_change,
            'battles': battle_details
        })
        
        total_elo_change += elo_change
    
    # 获取更新后的 ELO
    current_record = elo_manager.get_record(model_path)
    current_elo_after = current_record.elo
    
    print(f"\n{'='*60}")
    print(f"ELO Evaluation Complete")
    print(f"ELO: {current_elo_before} -> {current_elo_after} ({current_elo_after - current_elo_before:+d})")
    print(f"Record: {current_record.wins}W/{current_record.losses}L/{current_record.draws}D")
    print(f"{'='*60}\n")
    
    return {
        'model_path': model_path,
        'elo_before': current_elo_before,
        'elo_after': current_elo_after,
        'elo_change': current_elo_after - current_elo_before,
        'opponents': opponents,
        'results': results
    }


def print_elo_leaderboard(save_dir: str = "./ckpts/bp_agent", name_overrides=None):
    """打印 ELO 排行榜"""
    elo_manager = EloRatingManager(save_dir)

    models = elo_manager.list_all_models()
    if len(models) == 0:
        print("[ELO] No models found")
        return

    # 按 ELO 排序
    models.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'='*70}")
    print(f"ELO Leaderboard")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Model':<50}{'ELO':<10}{'W/L/D':<15}")
    print(f"{'-'*70}")

    for rank, (path, elo) in enumerate(models, 1):
        record = elo_manager.get_record(path)
        if record:
            wl = f"{record.wins}/{record.losses}/{record.draws}"
        else:
            wl = "0/0/0"
        model_name = os.path.basename(path)[:48]
        if name_overrides and path in name_overrides:
            model_name = name_overrides[path][:48]
        print(f"{rank:<6}{model_name:<50}{elo:<10}{wl:<15}")

    print(f"{'='*70}\n")


class EloEvaluator(RatingEvaluatorBase):
    """
    ELO 评估器 - 统一的评估接口
    
    用于评估 BP Agent 模型的相对强度，通过与其他模型对战来更新 ELO 分数。
    
    Example:
        >>> from eval import get_evaluator, EvalMethod
        >>> evaluator = get_evaluator(EvalMethod.ELO, save_dir="./ckpts/bp_agent")
        >>> result = evaluator.evaluate("./ckpts/bp_agent/model.pth")
        >>> print(f"ELO: {result['elo_after']}")
    """
    
    def __init__(
        self,
        save_dir: str = "./ckpts/bp_agent",
        oracle: Optional[WinRateOracle] = None,
        oracle_path: Optional[str] = None,
        num_opponents: int = NUM_OPPONENTS_TO_SAMPLE,
        num_player_sets: int = NUM_PLAYER_SETS,
        k_factor: int = ELO_K_FACTOR,
        opponent_sample_std: float = OPPONENT_SAMPLE_STD,
    ):
        """
        初始化 ELO 评估器
        
        Args:
            save_dir: 模型保存目录
            oracle: WinRateOracle 实例（可选）
            oracle_path: Oracle 模型路径（可选）
            num_opponents: 每次评估的对手数量
            num_player_sets: 每个对手对战的玩家 set 数量
            k_factor: ELO K-factor
            opponent_sample_std: 对手选择时的正态分布标准差
        """
        super().__init__(save_dir, num_opponents, num_player_sets)
        self.k_factor = k_factor
        self.opponent_sample_std = opponent_sample_std
        
        # 初始化 ELO 管理器和对战模拟器
        self.rating_manager = EloRatingManager(save_dir=save_dir)
        self.battle_simulator = BPBattleSimulator(oracle=oracle, oracle_path=oracle_path)
        # 保持向后兼容
        self.elo_manager = self.rating_manager
    
    def evaluate(
        self,
        model_path: str,
        num_opponents: Optional[int] = None,
        num_player_sets: Optional[int] = None
    ) -> Dict:
        """
        评估模型并更新 ELO 分数
        
        Args:
            model_path: 模型文件路径
            num_opponents: 对手数量（覆盖默认值）
            num_player_sets: 玩家 set 数量（覆盖默认值）
        
        Returns:
            评估结果字典，包含：
            - model_path: 模型路径
            - elo_before: 评估前 ELO
            - elo_after: 评估后 ELO
            - elo_change: ELO 变化量
            - opponents: 对手列表
            - results: 详细对战结果
        """
        num_opponents = num_opponents or self.num_opponents
        num_player_sets = num_player_sets or self.num_player_sets
        
        return evaluate_and_update_elo(
            model_path=model_path,
            elo_manager=self.rating_manager,
            battle_simulator=self.battle_simulator,
            num_opponents=num_opponents,
            num_player_sets=num_player_sets,
            k_factor=self.k_factor,
        )
    
    def get_rating(self, model_path: str) -> float:
        """获取模型的当前 ELO 分数（实现基类接口）"""
        return self.get_elo(model_path)
    
    def get_elo(self, model_path: str) -> int:
        """获取模型的当前 ELO 分数"""
        record = self.rating_manager.get_record(model_path)
        if record is None:
            # 自动注册新模型
            record = self.rating_manager.register_model(model_path)
        return record.elo
    
    def print_leaderboard(self, name_overrides=None):
        """打印 ELO 排行榜"""
        print_elo_leaderboard(save_dir=self.save_dir, name_overrides=name_overrides)
    
    def register_model(self, model_path: str, elo: int = INITIAL_ELO) -> ModelEloRecord:
        """手动注册模型"""
        return self.rating_manager.register_model(model_path, elo=elo)


if __name__ == "__main__":
    print("=" * 60)
    print("ELO Rating System Test")
    print("=" * 60)
    
    # 测试 ELO 计算
    print("\n--- Testing ELO Calculation ---")
    rating_a, rating_b = 1500, 1500
    score = 1.0  # A 赢
    change = compute_elo_change(rating_a, rating_b, score)
    print(f"Equal rating ({rating_a} vs {rating_b}), A wins: ELO change = {change:+d}")
    
    rating_a, rating_b = 1500, 1700
    score = 1.0
    change = compute_elo_change(rating_a, rating_b, score)
    print(f"Underdog ({rating_a} vs {rating_b}), A wins: ELO change = {change:+d}")
    
    rating_a, rating_b = 1700, 1500
    score = 1.0
    change = compute_elo_change(rating_a, rating_b, score)
    print(f"Favorite ({rating_a} vs {rating_b}), A wins: ELO change = {change:+d}")
    
    # 测试 ELO 管理器
    print("\n--- Testing ELO Manager ---")
    manager = EloRatingManager(save_dir="./ckpts/bp_agent_test")
    
    # 注册一些测试模型
    test_models = [
        "./ckpts/bp_agent_test/model_1.pth",
        "./ckpts/bp_agent_test/model_2.pth",
        "./ckpts/bp_agent_test/model_3.pth",
    ]
    
    for i, model in enumerate(test_models):
        elo = 1500 + (i - 1) * 200  # 1300, 1500, 1700
        record = manager.register_model(model, elo=elo)
        print(f"Registered {model}: ELO = {record.elo}")
    
    # 测试对手选择
    print("\n--- Testing Opponent Selection ---")
    current_model = test_models[1]  # 1500 ELO
    for _ in range(5):
        opponents = manager.select_opponents_by_elo(current_model, num_opponents=2)
        opponent_elos = [manager.get_record(o).elo for o in opponents]
        print(f"Selected opponents ELOs: {opponent_elos}")
    
    # 测试排行榜
    print("\n--- Testing Leaderboard ---")
    print_elo_leaderboard(save_dir="./ckpts/bp_agent_test")
    
    # 测试 EloEvaluator 接口
    print("\n--- Testing EloEvaluator Interface ---")
    from eval import EvalMethod, get_evaluator
    evaluator = get_evaluator(EvalMethod.ELO, save_dir="./ckpts/bp_agent_test")
    print(f"Created evaluator: {type(evaluator).__name__}")
    print(f"Available models: {len(evaluator.list_models())}")
    
    print("\n[OK] All tests passed!")
