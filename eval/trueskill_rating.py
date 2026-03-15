"""
TrueSkill Rating System for BP Agent
用于评估 BP Agent 的相对强度

TrueSkill 是微软开发的评分系统，比 ELO 更适用于团队游戏。
它使用贝叶斯推断来维护每个玩家技能水平的高斯分布（均值 mu 和标准差 sigma）。
"""

import os
import json
import math
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
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


# TrueSkill 参数
INITIAL_MU = 25.0          # 初始平均技能值
INITIAL_SIGMA = 25.0 / 3   # 初始标准差（约 8.33）
BETA = INITIAL_SIGMA / 2   # 性能标准差（约 4.17）
TAU = INITIAL_SIGMA / 100  # 动态因子（约 0.083）
EPSILON = 1e-6             # 数值稳定性常数
DRAW_PROBABILITY = 0.0     # 平局概率（BP 游戏一般无平局）

# 对手选择参数
OPPONENT_SAMPLE_STD = 2.0  # 正态分布采样标准差（TrueSkill 尺度）
NUM_OPPONENTS_TO_SAMPLE = 5

# 对战参数
NUM_PLAYER_SETS = 16
BATTLES_PER_MATCHUP = 2


def gaussian_cdf(x: float) -> float:
    """标准正态分布的累积分布函数"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def gaussian_pdf(x: float) -> float:
    """标准正态分布的概率密度函数"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def v_exceeds_margin(t: float, margin: float) -> float:
    """
    V 函数：当 t 超过 margin 时的调整值
    用于 TrueSkill 的因子图计算
    """
    if abs(t - margin) < EPSILON:
        return -t
    denom = gaussian_cdf(t - margin)
    if denom < EPSILON:
        return -t + margin
    return gaussian_pdf(t - margin) / denom


def w_exceeds_margin(t: float, margin: float) -> float:
    """
    W 函数：V 函数的方差调整
    """
    if abs(t - margin) < EPSILON:
        return 1.0
    vt = v_exceeds_margin(t, margin)
    return vt * (vt + t - margin)


@dataclass
class ModelTrueSkillRecord(ModelRatingRecord):
    """模型 TrueSkill 记录"""
    mu: float = INITIAL_MU
    sigma: float = INITIAL_SIGMA
    staleness: int = 0  # 陈旧度：距离上次参与对战的轮数
    
    @property
    def rating(self) -> float:
        """
        计算保守评分 = mu - 3*sigma
        这是 TrueSkill 中常用的展示分数
        """
        return self.mu - 3 * self.sigma
    
    def to_dict(self) -> dict:
        return {
            'model_path': self.model_path,
            'mu': self.mu,
            'sigma': self.sigma,
            'staleness': self.staleness,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'total_games': self.total_games,
            'last_eval_time': self.last_eval_time,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelTrueSkillRecord":
        return cls(
            model_path=data['model_path'],
            mu=data.get('mu', INITIAL_MU),
            sigma=data.get('sigma', INITIAL_SIGMA),
            staleness=data.get('staleness', 0),
            wins=data.get('wins', 0),
            losses=data.get('losses', 0),
            draws=data.get('draws', 0),
            total_games=data.get('total_games', 0),
            last_eval_time=data.get('last_eval_time', ''),
        )


class TrueSkillRatingManager(RatingManagerBase):
    """TrueSkill 评分管理器"""
    
    # Staleness 相关配置
    STALENESS_THRESHOLD = 5  # 超过此阈值需要强制刷新
    NUM_ACTIVE_MODELS = 5    # 选择最活跃的模型数量
    NUM_REFRESH_BATTLES = 3  # 每个陈旧模型与活跃模型对战的场次
    
    def _get_db_path(self) -> Path:
        """获取数据库文件路径"""
        return self.save_dir / "trueskill_ratings.json"
    
    def _create_record(self, model_path: str, **kwargs) -> ModelTrueSkillRecord:
        """创建新的评分记录"""
        mu = kwargs.get('mu', INITIAL_MU)
        sigma = kwargs.get('sigma', INITIAL_SIGMA)
        staleness = kwargs.get('staleness', 0)
        return ModelTrueSkillRecord(
            model_path=model_path,
            mu=mu,
            sigma=sigma,
            staleness=staleness,
            last_eval_time=datetime.now().strftime("%Y%m%d%H%M%S")
        )
    
    def _record_from_dict(self, data: dict) -> ModelTrueSkillRecord:
        """从字典创建记录对象"""
        return ModelTrueSkillRecord.from_dict(data)
    
    def increment_staleness(self, excluded_model: str = None):
        """
        增加所有模型的 staleness（排除指定模型）
        
        Args:
            excluded_model: 不参与staleness增加的模型路径（通常是当前评估的模型）
        """
        updated = False
        for path, record in self.records.items():
            if path != excluded_model:
                record.staleness += 1
                updated = True
        if updated:
            self._save_records()
    
    def reset_staleness(self, model_path: str):
        """重置模型的 staleness 为 0"""
        if model_path in self.records:
            self.records[model_path].staleness = 0
            self._save_records()
    
    def get_stale_models(self, threshold: int = None) -> List[str]:
        """
        获取超过 staleness 阈值的模型列表
        
        Args:
            threshold: 阈值，默认使用 STALENESS_THRESHOLD
            
        Returns:
            超过阈值的模型路径列表
        """
        threshold = threshold or self.STALENESS_THRESHOLD
        stale_models = []
        for path, record in self.records.items():
            if record.staleness >= threshold and os.path.exists(path):
                stale_models.append(path)
        return stale_models
    
    def get_most_active_models(self, num_models: int = None) -> List[str]:
        """
        获取最活跃的模型列表
        
        活跃度由以下因素综合决定：
        1. total_games（总对战次数）- 主要因素
        2. last_eval_time（最近评估时间）- 次要因素
        
        Args:
            num_models: 返回的模型数量，默认使用 NUM_ACTIVE_MODELS
            
        Returns:
            最活跃的模型路径列表
        """
        num_models = num_models or self.NUM_ACTIVE_MODELS
        
        # 获取所有存在的模型
        existing_models = [
            (path, record) for path, record in self.records.items()
            if os.path.exists(path)
        ]
        
        if len(existing_models) == 0:
            return []
        
        # 计算活跃度分数
        # 以 total_games 为主，以 last_eval_time 为次
        def calc_activity(item) -> float:
            path, record = item
            # 基础分数：总对战次数
            score = record.total_games * 1.0
            
            # 时间衰减因子（越近评估的越活跃）
            if record.last_eval_time:
                try:
                    eval_time = datetime.strptime(record.last_eval_time, "%Y%m%d%H%M%S")
                    # 计算距离现在的天数差
                    days_ago = (datetime.now() - eval_time).days
                    # 时间衰减：每天衰减 10% 的加成
                    time_bonus = max(0, 10 - days_ago) * 0.5
                    score += time_bonus
                except:
                    pass
            
            return score
        
        # 按活跃度排序
        existing_models.sort(key=calc_activity, reverse=True)
        
        return [path for path, _ in existing_models[:num_models]]
    
    def update_rating(self, model_a_path: str, model_b_path: str, score_a: float):
        """
        更新两个模型的 TrueSkill 评分
        
        使用 TrueSkill 的因子图近似算法进行更新。
        score_a: 1.0 = A 赢, 0.0 = B 赢, 0.5 = 平
        """
        if model_a_path not in self.records:
            self.register_model(model_a_path)
        if model_b_path not in self.records:
            self.register_model(model_b_path)
        
        record_a = self.records[model_a_path]
        record_b = self.records[model_b_path]
        
        # 应用动态因子（sigma 稍微增大，模拟技能随时间变化）
        record_a.sigma = math.sqrt(record_a.sigma ** 2 + TAU ** 2)
        record_b.sigma = math.sqrt(record_b.sigma ** 2 + TAU ** 2)
        
        # 计算团队性能分布
        # 这里简化为 1v1，所以团队就是单个玩家
        mu_a, sigma_a = record_a.mu, record_a.sigma
        mu_b, sigma_b = record_b.mu, record_b.sigma
        
        # 计算性能差异分布
        c = math.sqrt(2 * BETA ** 2 + sigma_a ** 2 + sigma_b ** 2)
        
        # 根据比赛结果确定 margin
        # score_a = 1.0: A 赢，margin = 0
        # score_a = 0.0: B 赢，margin = 0
        # score_a = 0.5: 平局，margin = epsilon（很小的值表示平局）
        if abs(score_a - 0.5) < EPSILON:
            margin = EPSILON  # 平局
        else:
            margin = 0.0
        
        # 计算标准化差异
        if score_a > 0.5:
            # A 赢
            t = (mu_a - mu_b) / c
        elif score_a < 0.5:
            # B 赢
            t = (mu_b - mu_a) / c
        else:
            # 平局
            t = 0.0
        
        # 计算更新量
        v = v_exceeds_margin(t, margin)
        w = w_exceeds_margin(t, margin)
        
        # 更新 mu 和 sigma
        if score_a > 0.5:
            # A 赢
            sigma_a_sq_update = (sigma_a / c) ** 2
            sigma_b_sq_update = (sigma_b / c) ** 2
            
            record_a.mu += sigma_a_sq_update * v * c
            record_b.mu -= sigma_b_sq_update * v * c
            
            record_a.sigma *= math.sqrt(1 - sigma_a_sq_update * w)
            record_b.sigma *= math.sqrt(1 - sigma_b_sq_update * w)
            
            record_a.wins += 1
            record_b.losses += 1
            
        elif score_a < 0.5:
            # B 赢
            sigma_a_sq_update = (sigma_a / c) ** 2
            sigma_b_sq_update = (sigma_b / c) ** 2
            
            record_a.mu -= sigma_a_sq_update * v * c
            record_b.mu += sigma_b_sq_update * v * c
            
            record_a.sigma *= math.sqrt(1 - sigma_a_sq_update * w)
            record_b.sigma *= math.sqrt(1 - sigma_b_sq_update * w)
            
            record_a.losses += 1
            record_b.wins += 1
            
        else:
            # 平局
            sigma_a_sq_update = (sigma_a / c) ** 2
            sigma_b_sq_update = (sigma_b / c) ** 2
            
            # 平局时，两个玩家的 mu 都向中间靠拢
            delta = sigma_a_sq_update * v * c
            record_a.mu += delta if mu_a < mu_b else -delta
            record_b.mu += delta if mu_b < mu_a else -delta
            
            record_a.sigma *= math.sqrt(1 - sigma_a_sq_update * w)
            record_b.sigma *= math.sqrt(1 - sigma_b_sq_update * w)
            
            record_a.draws += 1
            record_b.draws += 1
        
        # 确保 sigma 不会太小（最小不确定性）
        record_a.sigma = max(record_a.sigma, INITIAL_SIGMA / 100)
        record_b.sigma = max(record_b.sigma, INITIAL_SIGMA / 100)
        
        # 更新统计
        record_a.total_games += 1
        record_b.total_games += 1
        
        # 重置参与对战的模型的 staleness
        record_a.staleness = 0
        record_b.staleness = 0
        
        record_a.last_eval_time = datetime.now().strftime("%Y%m%d%H%M%S")
        record_b.last_eval_time = datetime.now().strftime("%Y%m%d%H%M%S")
        
        self._save_records()
    
    def get_rating_value(self, model_path: str) -> float:
        """
        获取模型的评分值（保守评分）
        
        Returns:
            mu - 3*sigma（TrueSkill 推荐的展示分数）
        """
        record = self.records.get(model_path)
        if record is None:
            return INITIAL_MU - 3 * INITIAL_SIGMA
        return record.rating
    
    def get_exposure(self, model_path: str) -> float:
        """
        获取模型的暴露评分（mu）
        
        Returns:
            mu 值
        """
        record = self.records.get(model_path)
        if record is None:
            return INITIAL_MU
        return record.mu
    
    def select_opponents(self, current_model_path: str, num_opponents: int = NUM_OPPONENTS_TO_SAMPLE) -> List[str]:
        """
        根据当前模型 TrueSkill，使用正态分布采样选择对手
        """
        if current_model_path not in self.records:
            self.register_model(current_model_path)
        
        current_rating = self.records[current_model_path].mu
        
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
            mu = self.records[path].mu
            # 正态分布概率密度
            weight = np.exp(-0.5 * ((mu - current_rating) / OPPONENT_SAMPLE_STD) ** 2)
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
    
    def refresh_stale_models(
        self,
        battle_simulator: 'BPBattleSimulator',
        num_player_sets: int = NUM_PLAYER_SETS
    ) -> Dict:
        """
        刷新陈旧的模型
        
        找出超过 staleness 阈值的模型，让它们与最活跃的模型对战，
        更新其分数并重置 staleness。
        
        这需要在当前轮次新模型评估之前执行，以防止评分漂移。
        
        Args:
            battle_simulator: 对战模拟器实例
            num_player_sets: 每个对战的玩家 set 数量
            
        Returns:
            刷新结果字典
        """
        stale_models = self.get_stale_models()
        if len(stale_models) == 0:
            return {'refreshed': [], 'results': []}
        
        active_models = self.get_most_active_models(self.NUM_ACTIVE_MODELS)
        if len(active_models) == 0:
            return {'refreshed': [], 'results': []}
        
        print(f"\n[TrueSkill Staleness] Found {len(stale_models)} stale models, refreshing...")
        print(f"[TrueSkill Staleness] Active models: {[os.path.basename(p) for p in active_models]}")
        
        refresh_results = []
        
        for stale_model in stale_models:
            stale_record = self.records[stale_model]
            print(f"\n  Refreshing {os.path.basename(stale_model)} "
                  f"(staleness={stale_record.staleness}, μ={stale_record.mu:.2f})")
            
            model_results = []
            total_mu_change = 0
            
            # 与每个活跃模型对战
            for active_model in active_models:
                if stale_model == active_model:
                    continue
                    
                active_record = self.records[active_model]
                print(f"    vs {os.path.basename(active_model)} "
                      f"(μ={active_record.mu:.2f}, games={active_record.total_games})")
                
                # 运行对战
                win_rate, battle_details = battle_simulator.evaluate_models(
                    stale_model, active_model, num_player_sets
                )
                
                # 将胜率转换为得分
                if win_rate > 0.5:
                    score = 1.0
                elif win_rate < 0.5:
                    score = 0.0
                else:
                    score = 0.5
                
                # 记录更新前的值
                mu_before = self.records[stale_model].mu
                
                # 更新 TrueSkill
                self.update_rating(stale_model, active_model, score)
                
                # 计算变化
                mu_after = self.records[stale_model].mu
                mu_change = mu_after - mu_before
                
                print(f"      Win rate: {win_rate*100:.1f}%, Score: {score}, "
                      f"μ change: {mu_change:+.2f}")
                
                model_results.append({
                    'active_model': active_model,
                    'active_mu': active_record.mu,
                    'win_rate': win_rate,
                    'score': score,
                    'mu_change': mu_change
                })
                
                total_mu_change += mu_change
            
            # 重置该模型的 staleness
            self.reset_staleness(stale_model)
            
            refreshed_record = self.records[stale_model]
            print(f"    Refreshed: μ={stale_record.mu:.2f} -> {refreshed_record.mu:.2f} "
                  f"({total_mu_change:+.2f}), staleness reset to 0")
            
            refresh_results.append({
                'stale_model': stale_model,
                'stale_mu_before': stale_record.mu - total_mu_change,
                'stale_mu_after': refreshed_record.mu,
                'total_mu_change': total_mu_change,
                'battles': model_results
            })
        
        print(f"[TrueSkill Staleness] Refresh complete. {len(stale_models)} models updated.")
        
        return {
            'refreshed': stale_models,
            'active_models_used': active_models,
            'results': refresh_results
        }


class BPBattleSimulator(BattleSimulatorBase):
    """BP 对战模拟器（复用 ELO 的实现）"""
    
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
            print(f"[TrueSkill] Loaded oracle from {oracle_path}")
        else:
            print(f"[TrueSkill] Warning: Oracle not found at {oracle_path}")
        
        oracle.eval()
        return oracle
    
    def load_agent(self, model_path: str) -> BPTransformerAgent:
        """加载 BP Agent 模型"""
        agent = BPTransformerAgent(embed_dim=128, nhead=8, num_layers=4).to(DEVICE)
        agent.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
        """
        r_players = player_set['r_players']
        d_players = player_set['d_players']
        
        # 创建 BP 状态
        state = BPState([], [], r_players, d_players, is_radiant_turn=True)
        
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
            
            # 判断是 pick 还是 ban
            is_pick = state.pick_count['radiant'] + state.pick_count['dire'] < 10
            state.step(hero_id, is_pick)
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
        """生成玩家 sets"""
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
                features = []
                for p in players:
                    vector = [0.0] * NUM_HEROES
                    for hero_info in p['heroes']:
                        hero_id = hero_info['id']
                        win_rate = hero_info['win_rate']
                        if 0 < hero_id < NUM_HEROES:
                            vector[hero_id] = win_rate
                    features.append(vector)
                return features
            
            r_players = build_player_features(all_players[:5])
            d_players = build_player_features(all_players[5:])
            
            player_sets.append({
                'r_players': r_players,
                'd_players': d_players
            })
        
        return player_sets


def evaluate_and_update_trueskill(
    model_path: str,
    rating_manager: Optional[TrueSkillRatingManager] = None,
    battle_simulator: Optional[BPBattleSimulator] = None,
    num_opponents: int = NUM_OPPONENTS_TO_SAMPLE,
    num_player_sets: int = NUM_PLAYER_SETS
) -> Dict:
    """
    评估模型并更新 TrueSkill 评分
    """
    if rating_manager is None:
        rating_manager = TrueSkillRatingManager()
    if battle_simulator is None:
        battle_simulator = BPBattleSimulator()
    
    # 注册/获取当前模型
    current_record = rating_manager.register_model(model_path)
    current_rating_before = current_record.rating
    current_mu_before = current_record.mu
    current_sigma_before = current_record.sigma
    
    print(f"\n{'='*60}")
    print(f"TrueSkill Evaluation: {model_path}")
    print(f"Current: μ={current_mu_before:.2f}, σ={current_sigma_before:.2f}, Rating={current_rating_before:.2f}")
    print(f"{'='*60}")
    
    # 选择对手
    opponents = rating_manager.select_opponents(model_path, num_opponents)
    
    if len(opponents) == 0:
        print("[TrueSkill] No opponents found, skipping evaluation")
        return {
            'model_path': model_path,
            'mu_before': current_mu_before,
            'sigma_before': current_sigma_before,
            'rating_before': current_rating_before,
            'mu_after': current_mu_before,
            'sigma_after': current_sigma_before,
            'rating_after': current_rating_before,
            'opponents': [],
            'results': []
        }
    
    print(f"[TrueSkill] Selected {len(opponents)} opponents by TrueSkill distribution")
    
    results = []
    total_mu_change = 0
    
    for opponent_path in opponents:
        opponent_record = rating_manager.get_record(opponent_path)
        opponent_mu = opponent_record.mu if opponent_record else INITIAL_MU
        opponent_sigma = opponent_record.sigma if opponent_record else INITIAL_SIGMA
        opponent_rating = opponent_record.rating if opponent_record else (INITIAL_MU - 3 * INITIAL_SIGMA)
        
        print(f"\n  vs {os.path.basename(opponent_path)} (μ={opponent_mu:.2f}, σ={opponent_sigma:.2f})")
        
        # 运行对战
        win_rate, battle_details = battle_simulator.evaluate_models(
            model_path, opponent_path, num_player_sets
        )
        
        # 将胜率转换为得分
        if win_rate > 0.5:
            score = 1.0
        elif win_rate < 0.5:
            score = 0.0
        else:
            score = 0.5
        
        # 记录更新前的值
        mu_before = rating_manager.get_record(model_path).mu
        
        # 更新 TrueSkill（双边更新）
        rating_manager.update_rating(model_path, opponent_path, score)
        
        # 计算变化
        mu_after = rating_manager.get_record(model_path).mu
        mu_change = mu_after - mu_before
        
        print(f"    Win rate: {win_rate*100:.1f}%, Score: {score}, μ change: {mu_change:+.2f}")
        
        results.append({
            'opponent_path': opponent_path,
            'opponent_mu': opponent_mu,
            'opponent_sigma': opponent_sigma,
            'opponent_rating': opponent_rating,
            'win_rate': win_rate,
            'score': score,
            'mu_change': mu_change,
            'battles': battle_details
        })
        
        total_mu_change += mu_change
    
    # 获取更新后的值
    current_record = rating_manager.get_record(model_path)
    current_mu_after = current_record.mu
    current_sigma_after = current_record.sigma
    current_rating_after = current_record.rating
    
    print(f"\n{'='*60}")
    print(f"TrueSkill Evaluation Complete")
    print(f"μ: {current_mu_before:.2f} -> {current_mu_after:.2f} ({total_mu_change:+.2f})")
    print(f"σ: {current_sigma_before:.2f} -> {current_sigma_after:.2f} ({current_sigma_after - current_sigma_before:+.2f})")
    print(f"Rating: {current_rating_before:.2f} -> {current_rating_after:.2f}")
    print(f"Record: {current_record.wins}W/{current_record.losses}L/{current_record.draws}D")
    print(f"{'='*60}\n")
    
    return {
        'model_path': model_path,
        'mu_before': current_mu_before,
        'sigma_before': current_sigma_before,
        'rating_before': current_rating_before,
        'mu_after': current_mu_after,
        'sigma_after': current_sigma_after,
        'rating_after': current_rating_after,
        'opponents': opponents,
        'results': results
    }


def print_trueskill_leaderboard(save_dir: str = "./ckpts/bp_agent"):
    """打印 TrueSkill 排行榜"""
    rating_manager = TrueSkillRatingManager(save_dir)
    
    models = rating_manager.list_all_models()
    if len(models) == 0:
        print("[TrueSkill] No models found")
        return
    
    # 按 rating 排序
    models.sort(key=lambda x: x[1], reverse=True)
    
    # 统计 staleness 信息
    stale_count = len(rating_manager.get_stale_models())
    
    print(f"\n{'='*90}")
    print(f"TrueSkill Leaderboard (Staleness Threshold: {rating_manager.STALENESS_THRESHOLD}, "
          f"Stale Models: {stale_count})")
    print(f"{'='*90}")
    print(f"{'Rank':<6}{'Model':<40}{'μ':<9}{'σ':<9}{'Rating':<9}{'W/L/D':<12}{'Stale':<6}")
    print(f"{'-'*90}")
    
    for rank, (path, rating) in enumerate(models, 1):
        record = rating_manager.get_record(path)
        if record:
            wl = f"{record.wins}/{record.losses}/{record.draws}"
            mu_str = f"{record.mu:.1f}"
            sigma_str = f"{record.sigma:.1f}"
            rating_str = f"{rating:.1f}"
            stale_str = f"{record.staleness}"
            # 标记超过阈值的 staleness
            if record.staleness >= rating_manager.STALENESS_THRESHOLD:
                stale_str = f"{record.staleness}*"
        else:
            wl = "0/0/0"
            mu_str = "-"
            sigma_str = "-"
            rating_str = "-"
            stale_str = "-"
        model_name = os.path.basename(path)[:38]
        print(f"{rank:<6}{model_name:<40}{mu_str:<9}{sigma_str:<9}{rating_str:<9}{wl:<12}{stale_str:<6}")
    
    print(f"{'='*90}")
    print(f"* = Staleness >= threshold ({rating_manager.STALENESS_THRESHOLD}), will be refreshed before next eval")
    print(f"\n")


class TrueSkillEvaluator(RatingEvaluatorBase):
    """
    TrueSkill 评估器 - 统一的评估接口
    
    用于评估 BP Agent 模型的相对强度，通过与其他模型对战来更新 TrueSkill 评分。
    
    特性：
    - Staleness 追踪：防止评分漂移，自动刷新长期未评估的模型
    
    Example:
        >>> from eval import get_evaluator, EvalMethod
        >>> evaluator = get_evaluator(EvalMethod.TRUESKILL, save_dir="./ckpts/bp_agent")
        >>> result = evaluator.evaluate("./ckpts/bp_agent/model.pth")
        >>> print(f"Rating: {result['rating_after']}")
    """
    
    def __init__(
        self,
        save_dir: str = "./ckpts/bp_agent",
        oracle: Optional[WinRateOracle] = None,
        oracle_path: Optional[str] = None,
        num_opponents: int = NUM_OPPONENTS_TO_SAMPLE,
        num_player_sets: int = NUM_PLAYER_SETS,
        staleness_threshold: int = 5,
        num_active_models: int = 5,
    ):
        """
        初始化 TrueSkill 评估器
        
        Args:
            save_dir: 模型保存目录
            oracle: WinRateOracle 实例（可选）
            oracle_path: Oracle 模型路径（可选）
            num_opponents: 每次评估的对手数量
            num_player_sets: 每个对手对战的玩家 set 数量
            staleness_threshold: staleness 阈值，超过此值的模型会被强制刷新
            num_active_models: 刷新时选择的活跃模型数量
        """
        super().__init__(save_dir, num_opponents, num_player_sets)
        
        # 初始化 TrueSkill 管理器和对战模拟器
        self.rating_manager = TrueSkillRatingManager(save_dir=save_dir)
        self.battle_simulator = BPBattleSimulator(oracle=oracle, oracle_path=oracle_path)
        
        # 设置 staleness 相关配置
        self.rating_manager.STALENESS_THRESHOLD = staleness_threshold
        self.rating_manager.NUM_ACTIVE_MODELS = num_active_models
    
    def evaluate(
        self,
        model_path: str,
        num_opponents: Optional[int] = None,
        num_player_sets: Optional[int] = None,
        skip_staleness_refresh: bool = False
    ) -> Dict:
        """
        评估模型并更新 TrueSkill 评分
        
        在评估新模型之前，会先检查并刷新陈旧的模型（staleness 超过阈值），
        以防止评分系统长期漂移。
        
        Args:
            model_path: 要评估的模型路径
            num_opponents: 对手数量
            num_player_sets: 每个对手对战的玩家 set 数量
            skip_staleness_refresh: 是否跳过 staleness 刷新（用于特殊场景）
            
        Returns:
            评估结果字典，包含 'staleness_refresh' 字段记录刷新结果
        """
        num_opponents = num_opponents or self.num_opponents
        num_player_sets = num_player_sets or self.num_player_sets
        
        # 步骤 1: 刷新陈旧的模型（在评估新模型之前）
        refresh_result = {'refreshed': [], 'results': []}
        if not skip_staleness_refresh:
            refresh_result = self.rating_manager.refresh_stale_models(
                self.battle_simulator, num_player_sets
            )
        
        # 步骤 2: 增加所有模型的 staleness（排除当前要评估的模型）
        self.rating_manager.increment_staleness(excluded_model=model_path)
        
        # 步骤 3: 评估当前模型
        eval_result = evaluate_and_update_trueskill(
            model_path=model_path,
            rating_manager=self.rating_manager,
            battle_simulator=self.battle_simulator,
            num_opponents=num_opponents,
            num_player_sets=num_player_sets
        )
        
        # 合并结果
        eval_result['staleness_refresh'] = refresh_result
        
        return eval_result
    
    def get_rating(self, model_path: str) -> float:
        """获取模型的当前保守评分（mu - 3*sigma）"""
        record = self.rating_manager.get_record(model_path)
        if record is None:
            # 自动注册新模型
            record = self.rating_manager.register_model(model_path)
        return record.rating
    
    def get_mu(self, model_path: str) -> float:
        """获取模型的 mu 值"""
        record = self.rating_manager.get_record(model_path)
        if record is None:
            record = self.rating_manager.register_model(model_path)
        return record.mu
    
    def get_sigma(self, model_path: str) -> float:
        """获取模型的 sigma 值"""
        record = self.rating_manager.get_record(model_path)
        if record is None:
            record = self.rating_manager.register_model(model_path)
        return record.sigma
    
    def print_leaderboard(self):
        """打印 TrueSkill 排行榜"""
        print_trueskill_leaderboard(save_dir=self.save_dir)
    
    def register_model(self, model_path: str, mu: float = INITIAL_MU, sigma: float = INITIAL_SIGMA) -> ModelTrueSkillRecord:
        """手动注册模型"""
        return self.rating_manager.register_model(model_path, mu=mu, sigma=sigma)


if __name__ == "__main__":
    print("=" * 60)
    print("TrueSkill Rating System Test")
    print("=" * 60)
    
    # 测试 TrueSkill 计算
    print("\n--- Testing TrueSkill Calculation ---")
    manager = TrueSkillRatingManager(save_dir="./ckpts/bp_agent_test_trueskill")
    
    # 注册一些测试模型
    test_models = [
        "./ckpts/bp_agent_test_trueskill/model_1.pth",
        "./ckpts/bp_agent_test_trueskill/model_2.pth",
        "./ckpts/bp_agent_test_trueskill/model_3.pth",
    ]
    
    for i, model in enumerate(test_models):
        mu = 25.0 + (i - 1) * 5  # 20, 25, 30
        record = manager.register_model(model, mu=mu, sigma=INITIAL_SIGMA)
        print(f"Registered {model}: μ={record.mu:.2f}, σ={record.sigma:.2f}, Rating={record.rating:.2f}")
    
    # 测试对战更新
    print("\n--- Testing Rating Update ---")
    print("Simulating: model_2 beats model_1")
    
    record_before_a = manager.get_record(test_models[1])
    record_before_b = manager.get_record(test_models[0])
    print(f"Before: model_2 μ={record_before_a.mu:.2f}, model_1 μ={record_before_b.mu:.2f}")
    
    manager.update_rating(test_models[1], test_models[0], 1.0)  # model_2 wins
    
    record_after_a = manager.get_record(test_models[1])
    record_after_b = manager.get_record(test_models[0])
    print(f"After:  model_2 μ={record_after_a.mu:.2f}, model_1 μ={record_after_b.mu:.2f}")
    
    # 测试对手选择
    print("\n--- Testing Opponent Selection ---")
    current_model = test_models[1]  # mu = 25
    for _ in range(5):
        opponents = manager.select_opponents(current_model, num_opponents=2)
        opponent_mus = [manager.get_record(o).mu for o in opponents]
        print(f"Selected opponents μ: {opponent_mus}")
    
    # 测试排行榜
    print("\n--- Testing Leaderboard ---")
    print_trueskill_leaderboard(save_dir="./ckpts/bp_agent_test_trueskill")
    
    # 测试 TrueSkillEvaluator 接口
    print("\n--- Testing TrueSkillEvaluator Interface ---")
    from eval import EvalMethod, get_evaluator
    evaluator = get_evaluator(EvalMethod.TRUESKILL, save_dir="./ckpts/bp_agent_test_trueskill")
    print(f"Created evaluator: {type(evaluator).__name__}")
    print(f"Available models: {len(evaluator.list_models())}")
    
    print("\n[OK] All tests passed!")
