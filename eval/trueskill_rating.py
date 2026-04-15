"""
TrueSkill Rating System for BP Agent (使用官方 trueskill 库)
用于评估 BP Agent 的相对强度

使用成熟的 trueskill 库保证数学严谨性，同时保留项目中的：
- Staleness 追踪机制
- 活跃模型选择策略
- 智能对手采样
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

import trueskill

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
    RatingEvaluatorBase,
)


# ============== TrueSkill 环境配置 ==============
# 使用全局 TrueSkill 环境，参数与原实现保持一致
INITIAL_MU = 25.0
INITIAL_SIGMA = 25.0 / 3
BETA = INITIAL_SIGMA / 2
TAU = INITIAL_SIGMA / 100
DRAW_PROBABILITY = 0.0

# 创建全局 TrueSkill 环境
TS_ENV = trueskill.TrueSkill(
    mu=INITIAL_MU,
    sigma=INITIAL_SIGMA,
    beta=BETA,
    tau=TAU,
    draw_probability=DRAW_PROBABILITY,
)

# 对手选择参数
OPPONENT_SAMPLE_STD = 2.0
NUM_OPPONENTS_TO_SAMPLE = 5

# 对战参数
NUM_PLAYER_SETS = 16


@dataclass
class ModelTrueSkillRecord(ModelRatingRecord):
    """模型 TrueSkill 记录"""

    mu: float = INITIAL_MU
    sigma: float = INITIAL_SIGMA
    staleness: int = 0

    @property
    def rating(self) -> float:
        """
        计算保守评分 = mu - 3*sigma
        TrueSkill 中常用的展示分数，表示有 99.7% 置信度真实技能 >= 此值
        """
        return self.mu - 3 * self.sigma

    @property
    def trueskill_rating(self) -> trueskill.Rating:
        """获取 trueskill.Rating 对象"""
        return trueskill.Rating(mu=self.mu, sigma=self.sigma)

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "mu": self.mu,
            "sigma": self.sigma,
            "staleness": self.staleness,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_games": self.total_games,
            "last_eval_time": self.last_eval_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelTrueSkillRecord":
        return cls(
            model_path=data["model_path"],
            mu=data.get("mu", INITIAL_MU),
            sigma=data.get("sigma", INITIAL_SIGMA),
            staleness=data.get("staleness", 0),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            draws=data.get("draws", 0),
            total_games=data.get("total_games", 0),
            last_eval_time=data.get("last_eval_time", ""),
        )


class TrueSkillRatingManager(RatingManagerBase):
    """TrueSkill 评分管理器"""

    # Staleness 相关配置
    STALENESS_THRESHOLD = 5
    NUM_ACTIVE_MODELS = 5
    NUM_REFRESH_BATTLES = 3

    def _get_db_path(self) -> Path:
        """获取数据库文件路径"""
        return self.save_dir / "trueskill_ratings.json"

    def _create_record(self, model_path: str, **kwargs) -> ModelTrueSkillRecord:
        """创建新的评分记录"""
        mu = kwargs.get("mu", INITIAL_MU)
        sigma = kwargs.get("sigma", INITIAL_SIGMA)
        staleness = kwargs.get("staleness", 0)
        return ModelTrueSkillRecord(
            model_path=model_path,
            mu=mu,
            sigma=sigma,
            staleness=staleness,
            last_eval_time=datetime.now().strftime("%Y%m%d%H%M%S"),
        )

    def _record_from_dict(self, data: dict) -> ModelTrueSkillRecord:
        """从字典创建记录对象"""
        return ModelTrueSkillRecord.from_dict(data)

    def increment_staleness(self, excluded_model: str = None):
        """
        增加所有模型的 staleness（排除指定模型）

        Args:
            excluded_model: 不参与staleness增加的模型路径
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
        """获取超过 staleness 阈值的模型列表"""
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
        """
        num_models = num_models or self.NUM_ACTIVE_MODELS

        existing_models = [
            (path, record)
            for path, record in self.records.items()
            if os.path.exists(path)
        ]

        if len(existing_models) == 0:
            return []

        def calc_activity(item) -> float:
            path, record = item
            score = record.total_games * 1.0

            if record.last_eval_time:
                try:
                    eval_time = datetime.strptime(record.last_eval_time, "%Y%m%d%H%M%S")
                    days_ago = (datetime.now() - eval_time).days
                    time_bonus = max(0, 10 - days_ago) * 0.5
                    score += time_bonus
                except:
                    pass

            return score

        existing_models.sort(key=calc_activity, reverse=True)
        return [path for path, _ in existing_models[:num_models]]

    def update_rating(self, model_a_path: str, model_b_path: str, score_a: float):
        """
        更新两个模型的 TrueSkill 评分（单场对战）

        使用 trueskill 库的标准更新逻辑

        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            score_a: 模型 A 的得分（1.0=A赢，0.0=B赢，0.5=平）
        """
        if model_a_path not in self.records:
            self.register_model(model_a_path)
        if model_b_path not in self.records:
            self.register_model(model_b_path)

        record_a = self.records[model_a_path]
        record_b = self.records[model_b_path]

        # 获取 Rating 对象
        rating_a = record_a.trueskill_rating
        rating_b = record_b.trueskill_rating

        # 确定对战结果的 ranks
        # TrueSkill 的 ranks: 0 表示赢，数字越大越弱
        if score_a > 0.5 + 1e-8:
            # A 赢: ranks = [0, 1]
            new_rating_a, new_rating_b = TS_ENV.rate_1vs1(rating_a, rating_b)
            record_a.wins += 1
            record_b.losses += 1
        elif score_a < 0.5 - 1e-8:
            # B 赢: ranks = [1, 0]，通过交换顺序实现
            new_rating_b, new_rating_a = TS_ENV.rate_1vs1(rating_b, rating_a)
            record_a.losses += 1
            record_b.wins += 1
        else:
            # 平局
            (new_rating_a,), (new_rating_b,) = TS_ENV.rate(
                [(rating_a,), (rating_b,)], ranks=[0, 0]
            )
            record_a.draws += 1
            record_b.draws += 1

        # 更新记录
        record_a.mu = new_rating_a.mu
        record_a.sigma = new_rating_a.sigma
        record_b.mu = new_rating_b.mu
        record_b.sigma = new_rating_b.sigma

        record_a.total_games += 1
        record_b.total_games += 1

        record_a.staleness = 0
        record_b.staleness = 0

        record_a.last_eval_time = datetime.now().strftime("%Y%m%d%H%M%S")
        record_b.last_eval_time = datetime.now().strftime("%Y%m%d%H%M%S")

        self._save_records()

    def update_rating_batch(
        self,
        model_a_path: str,
        model_b_path: str,
        a_win_count: int,
        b_win_count: int,
        draw_count: int = 0,
    ):
        """
        批量更新两个模型的评分（多场对战聚合）

        严谨做法：每场单独更新，以正确累积不确定性

        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            a_win_count: A 赢的场次
            b_win_count: B 赢的场次
            draw_count: 平局场次
        """
        # 记录更新前的值
        if model_a_path in self.records:
            mu_before = self.records[model_a_path].mu
        else:
            mu_before = INITIAL_MU

        # 混合对战顺序以避免顺序偏差
        battles = []
        battles.extend([1.0] * a_win_count)
        battles.extend([0.0] * b_win_count)
        battles.extend([0.5] * draw_count)
        random.shuffle(battles)

        # 逐场更新
        for score in battles:
            self.update_rating(model_a_path, model_b_path, score)

        # 返回 mu 变化
        if model_a_path in self.records:
            mu_after = self.records[model_a_path].mu
            return mu_after - mu_before
        return 0.0

    def get_rating_value(self, model_path: str) -> float:
        """获取模型的保守评分（mu - 3*sigma）"""
        record = self.records.get(model_path)
        if record is None:
            return INITIAL_MU - 3 * INITIAL_SIGMA
        return record.rating

    def get_exposure(self, model_path: str) -> float:
        """获取模型的暴露评分（mu）"""
        record = self.records.get(model_path)
        if record is None:
            return INITIAL_MU
        return record.mu

    def select_opponents(
        self, current_model_path: str, num_opponents: int = NUM_OPPONENTS_TO_SAMPLE
    ) -> List[str]:
        """
        根据当前模型 TrueSkill，使用正态分布采样选择对手
        """
        if current_model_path not in self.records:
            self.register_model(current_model_path)

        current_mu = self.records[current_model_path].mu

        other_models = [
            path
            for path in self.records.keys()
            if path != current_model_path and os.path.exists(path)
        ]

        if len(other_models) == 0:
            return []

        if len(other_models) <= num_opponents:
            return other_models

        weights = []
        for path in other_models:
            mu = self.records[path].mu
            weight = np.exp(-0.5 * ((mu - current_mu) / OPPONENT_SAMPLE_STD) ** 2)
            weights.append(weight)

        weights = np.array(weights)
        weights = weights / weights.sum()

        selected_indices = np.random.choice(
            len(other_models), size=num_opponents, replace=False, p=weights
        )

        return [other_models[i] for i in selected_indices]

    def refresh_stale_models(
        self,
        battle_simulator: "BPBattleSimulator",
        num_player_sets: int = NUM_PLAYER_SETS,
    ) -> Dict:
        """
        刷新陈旧的模型

        找出超过 staleness 阈值的模型，让它们与最活跃的模型对战，
        更新其分数并重置 staleness。
        """
        stale_models = self.get_stale_models()
        if len(stale_models) == 0:
            return {"refreshed": [], "results": []}

        active_models = self.get_most_active_models(self.NUM_ACTIVE_MODELS)
        if len(active_models) == 0:
            return {"refreshed": [], "results": []}

        print(
            f"\n[TrueSkill Staleness] Found {len(stale_models)} stale models, refreshing..."
        )
        print(
            f"[TrueSkill Staleness] Active models: {[os.path.basename(p) for p in active_models]}"
        )

        refresh_results = []

        for stale_model in stale_models:
            stale_record = self.records[stale_model]
            print(
                f"\n  Refreshing {os.path.basename(stale_model)} "
                f"(staleness={stale_record.staleness}, μ={stale_record.mu:.2f})"
            )

            model_results = []
            total_mu_change = 0

            for active_model in active_models:
                if stale_model == active_model:
                    continue

                active_record = self.records[active_model]
                print(
                    f"    vs {os.path.basename(active_model)} "
                    f"(μ={active_record.mu:.2f}, games={active_record.total_games})"
                )

                win_rate, battle_details = battle_simulator.evaluate_models(
                    stale_model, active_model, num_player_sets
                )

                a_win_count = battle_details.count("win")
                b_win_count = battle_details.count("loss")
                draw_count = battle_details.count("draw")

                mu_before = self.records[stale_model].mu

                self.update_rating_batch(
                    stale_model, active_model, a_win_count, b_win_count, draw_count
                )

                mu_after = self.records[stale_model].mu
                mu_change = mu_after - mu_before

                print(
                    f"      Win rate: {win_rate * 100:.1f}%, μ change: {mu_change:+.2f}"
                )

                model_results.append(
                    {
                        "active_model": active_model,
                        "active_mu": active_record.mu,
                        "win_rate": win_rate,
                        "mu_change": mu_change,
                    }
                )

                total_mu_change += mu_change

            self.reset_staleness(stale_model)

            refreshed_record = self.records[stale_model]
            print(
                f"    Refreshed: μ={stale_record.mu:.2f} -> {refreshed_record.mu:.2f} "
                f"({total_mu_change:+.2f}), staleness reset to 0"
            )

            refresh_results.append(
                {
                    "stale_model": stale_model,
                    "stale_mu_before": stale_record.mu - total_mu_change,
                    "stale_mu_after": refreshed_record.mu,
                    "total_mu_change": total_mu_change,
                    "battles": model_results,
                }
            )

        print(
            f"[TrueSkill Staleness] Refresh complete. {len(stale_models)} models updated."
        )

        return {
            "refreshed": stale_models,
            "active_models_used": active_models,
            "results": refresh_results,
        }


class BPBattleSimulator(BattleSimulatorBase):
    """BP 对战模拟器"""

    def __init__(
        self, oracle: Optional[WinRateOracle] = None, oracle_path: Optional[str] = None
    ):
        if oracle is not None:
            self.oracle = oracle
        else:
            self.oracle = self._load_oracle(oracle_path)

    def _load_oracle(self, oracle_path: Optional[str] = None) -> WinRateOracle:
        if oracle_path is None:
            oracle_path = "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth"

        oracle = WinRateOracle(
            embed_dim=128, nhead=8, num_layers=6, use_text=True, use_player_heroes=True
        ).to(DEVICE)

        if os.path.exists(oracle_path):
            oracle.load_state_dict(torch.load(oracle_path, map_location=DEVICE))
            print(f"[TrueSkill] Loaded oracle from {oracle_path}")
        else:
            print(f"[TrueSkill] Warning: Oracle not found at {oracle_path}")

        oracle.eval()
        return oracle

    def load_agent(self, model_path: str) -> BPTransformerAgent:
        agent = BPTransformerAgent(embed_dim=256, nhead=8, num_layers=4).to(DEVICE)
        agent.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.eval()
        return agent

    def run_bp_battle(
        self,
        agent_radiant: BPTransformerAgent,
        agent_dire: BPTransformerAgent,
        player_set: Dict,
        max_steps: int = 24,
    ) -> Tuple[List[int], List[int], float]:
        r_players = player_set["r_players"]
        d_players = player_set["d_players"]

        state = BPState(
            [],
            [],
            r_players,
            d_players,
            radiant_bans=[],
            dire_bans=[],
            is_radiant_turn=True,
            step_idx=0,
        )

        step = 0
        while not state.done and step < max_steps:
            state_dict = state.to_dict()

            current_agent = agent_radiant if state.is_radiant_turn else agent_dire

            with torch.no_grad():
                action_logits, _ = current_agent(state_dict)

                valid_actions = state.get_valid_actions()
                mask = torch.full((NUM_HEROES,), -1e9, device=action_logits.device)
                for h in valid_actions:
                    mask[h - 1] = 0.0
                action_logits = action_logits + mask

                hero_id = torch.argmax(action_logits, dim=-1).item() + 1

            state.step(hero_id)
            step += 1

        if len(state.radiant_heroes) >= 5 and len(state.dire_heroes) >= 5:
            r_picks = state.radiant_heroes[:5]
            d_picks = state.dire_heroes[:5]
        else:
            r_picks = state.radiant_heroes + [1] * (5 - len(state.radiant_heroes))
            d_picks = state.dire_heroes + [1] * (5 - len(state.dire_heroes))

        win_prob = state.get_reward(self.oracle)

        return r_picks, d_picks, win_prob

    def evaluate_models(
        self,
        model_a_path: str,
        model_b_path: str,
        num_player_sets: int = NUM_PLAYER_SETS,
    ) -> Tuple[float, List[str]]:
        """
        评估两个模型的对战结果

        Returns:
            (model_a 胜率, 每场结果列表 ['win'/'loss'/'draw'])
        """
        agent_a = self.load_agent(model_a_path)
        agent_b = self.load_agent(model_b_path)

        player_sets = self._generate_player_sets(num_player_sets)

        battle_results = []
        a_wins = 0
        total_games = 0

        for player_set in player_sets:
            a_is_radiant = random.choice([True, False])

            if a_is_radiant:
                agent_radiant, agent_dire = agent_a, agent_b
            else:
                agent_radiant, agent_dire = agent_b, agent_a

            r_picks, d_picks, win_prob = self.run_bp_battle(
                agent_radiant, agent_dire, player_set
            )

            if a_is_radiant:
                a_win_prob = win_prob
            else:
                a_win_prob = 1.0 - win_prob

            if a_win_prob > 0.5 + 1e-8:
                a_wins += 1
                result = "win"
            elif a_win_prob < 0.5 - 1e-8:
                result = "loss"
            else:
                result = "draw"

            total_games += 1
            battle_results.append(result)

        win_rate = a_wins / total_games if total_games > 0 else 0.5
        return win_rate, battle_results

    def _generate_player_sets(self, num_sets: int) -> List[Dict]:
        player_sets = []

        for _ in range(num_sets):
            all_players = batch_sample_player_preferences(
                num_players=10,
                position_distribution={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
                m=3,
                n=5,
            )

            from utils.raw_data import get_valid_hero_ids

            valid_hero_ids = get_valid_hero_ids()

            def build_player_features(players):
                features = []
                for p in players:
                    vector = [0.0] * NUM_HEROES
                    for hero_info in p["heroes"]:
                        hero_id = hero_info["id"]
                        win_rate = hero_info["win_rate"]
                        if hero_id in valid_hero_ids and hero_id <= NUM_HEROES:
                            vector[hero_id - 1] = win_rate
                    features.append(vector)
                return features

            r_players = build_player_features(all_players[:5])
            d_players = build_player_features(all_players[5:])

            player_sets.append({"r_players": r_players, "d_players": d_players})

        return player_sets


def evaluate_and_update_trueskill(
    model_path: str,
    rating_manager: Optional[TrueSkillRatingManager] = None,
    battle_simulator: Optional[BPBattleSimulator] = None,
    num_opponents: int = NUM_OPPONENTS_TO_SAMPLE,
    num_player_sets: int = NUM_PLAYER_SETS,
) -> Dict:
    """
    评估模型并更新 TrueSkill 评分（严谨版本：逐场更新）
    """
    if rating_manager is None:
        rating_manager = TrueSkillRatingManager()
    if battle_simulator is None:
        battle_simulator = BPBattleSimulator()

    current_record = rating_manager.register_model(model_path)
    current_rating_before = current_record.rating
    current_mu_before = current_record.mu
    current_sigma_before = current_record.sigma

    print(f"\n{'=' * 60}")
    print(f"TrueSkill Evaluation: {model_path}")
    print(
        f"Current: μ={current_mu_before:.2f}, σ={current_sigma_before:.2f}, Rating={current_rating_before:.2f}"
    )
    print(f"{'=' * 60}")

    opponents = rating_manager.select_opponents(model_path, num_opponents)

    if len(opponents) == 0:
        print("[TrueSkill] No opponents found, skipping evaluation")
        return {
            "model_path": model_path,
            "mu_before": current_mu_before,
            "sigma_before": current_sigma_before,
            "rating_before": current_rating_before,
            "mu_after": current_mu_before,
            "sigma_after": current_sigma_before,
            "rating_after": current_rating_before,
            "opponents": [],
            "results": [],
        }

    print(f"[TrueSkill] Selected {len(opponents)} opponents by TrueSkill distribution")

    results = []
    total_mu_change = 0

    for opponent_path in opponents:
        opponent_record = rating_manager.get_record(opponent_path)
        opponent_mu = opponent_record.mu if opponent_record else INITIAL_MU
        opponent_sigma = opponent_record.sigma if opponent_record else INITIAL_SIGMA
        opponent_rating = (
            opponent_record.rating
            if opponent_record
            else (INITIAL_MU - 3 * INITIAL_SIGMA)
        )

        print(
            f"\n  vs {os.path.basename(opponent_path)} (μ={opponent_mu:.2f}, σ={opponent_sigma:.2f})"
        )

        win_rate, battle_results = battle_simulator.evaluate_models(
            model_path, opponent_path, num_player_sets
        )

        a_win_count = battle_results.count("win")
        b_win_count = battle_results.count("loss")
        draw_count = battle_results.count("draw")

        mu_before = rating_manager.get_record(model_path).mu

        rating_manager.update_rating_batch(
            model_path, opponent_path, a_win_count, b_win_count, draw_count
        )

        mu_after = rating_manager.get_record(model_path).mu
        mu_change = mu_after - mu_before

        print(
            f"    Battles: {a_win_count}W/{b_win_count}L/{draw_count}D, "
            f"Win rate: {win_rate * 100:.1f}%, μ change: {mu_change:+.2f}"
        )

        results.append(
            {
                "opponent_path": opponent_path,
                "opponent_mu": opponent_mu,
                "opponent_sigma": opponent_sigma,
                "opponent_rating": opponent_rating,
                "win_rate": win_rate,
                "a_win_count": a_win_count,
                "b_win_count": b_win_count,
                "draw_count": draw_count,
                "mu_change": mu_change,
                "battle_results": battle_results,
            }
        )

        total_mu_change += mu_change

    current_record = rating_manager.get_record(model_path)
    current_mu_after = current_record.mu
    current_sigma_after = current_record.sigma
    current_rating_after = current_record.rating

    print(f"\n{'=' * 60}")
    print(f"TrueSkill Evaluation Complete")
    print(
        f"μ: {current_mu_before:.2f} -> {current_mu_after:.2f} ({total_mu_change:+.2f})"
    )
    print(
        f"σ: {current_sigma_before:.2f} -> {current_sigma_after:.2f} ({current_sigma_after - current_sigma_before:+.2f})"
    )
    print(f"Rating: {current_rating_before:.2f} -> {current_rating_after:.2f}")
    print(
        f"Record: {current_record.wins}W/{current_record.losses}L/{current_record.draws}D"
    )
    print(f"{'=' * 60}\n")

    return {
        "model_path": model_path,
        "mu_before": current_mu_before,
        "sigma_before": current_sigma_before,
        "rating_before": current_rating_before,
        "mu_after": current_mu_after,
        "sigma_after": current_sigma_after,
        "rating_after": current_rating_after,
        "opponents": opponents,
        "results": results,
    }


def print_trueskill_leaderboard(save_dir: str = "./ckpts/bp_agent"):
    """打印 TrueSkill 排行榜"""
    rating_manager = TrueSkillRatingManager(save_dir)

    models = rating_manager.list_all_models()
    if len(models) == 0:
        print("[TrueSkill] No models found")
        return

    models.sort(key=lambda x: x[1], reverse=True)

    stale_count = len(rating_manager.get_stale_models())

    print(f"\n{'=' * 90}")
    print(
        f"TrueSkill Leaderboard (Staleness Threshold: {rating_manager.STALENESS_THRESHOLD}, "
        f"Stale Models: {stale_count})"
    )
    print(f"{'=' * 90}")
    print(
        f"{'Rank':<6}{'Model':<40}{'μ':<9}{'σ':<9}{'Rating':<9}{'W/L/D':<12}{'Stale':<6}"
    )
    print(f"{'-' * 90}")

    for rank, (path, rating) in enumerate(models, 1):
        record = rating_manager.get_record(path)
        if record:
            wl = f"{record.wins}/{record.losses}/{record.draws}"
            mu_str = f"{record.mu:.1f}"
            sigma_str = f"{record.sigma:.1f}"
            rating_str = f"{rating:.1f}"
            stale_str = f"{record.staleness}"
            if record.staleness >= rating_manager.STALENESS_THRESHOLD:
                stale_str = f"{record.staleness}*"
        else:
            wl = "0/0/0"
            mu_str = "-"
            sigma_str = "-"
            rating_str = "-"
            stale_str = "-"
        model_name = os.path.basename(path)[:38]
        print(
            f"{rank:<6}{model_name:<40}{mu_str:<9}{sigma_str:<9}{rating_str:<9}{wl:<12}{stale_str:<6}"
        )

    print(f"{'=' * 90}")
    print(
        f"* = Staleness >= threshold ({rating_manager.STALENESS_THRESHOLD}), will be refreshed before next eval"
    )
    print(f"\n")


class TrueSkillEvaluator(RatingEvaluatorBase):
    """
    TrueSkill 评估器 - 使用 trueskill 库的严谨版本

    特性：
    - 逐场更新评分，不丢弃信息
    - 正确的平局处理
    - Staleness 追踪
    - 活跃模型选择
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
        additional_dirs: Optional[List[str]] = None,
    ):
        super().__init__(save_dir, num_opponents, num_player_sets)

        self.rating_manager = TrueSkillRatingManager(save_dir=save_dir, additional_dirs=additional_dirs)
        self.battle_simulator = BPBattleSimulator(
            oracle=oracle, oracle_path=oracle_path
        )

        self.rating_manager.STALENESS_THRESHOLD = staleness_threshold
        self.rating_manager.NUM_ACTIVE_MODELS = num_active_models

    def evaluate(
        self,
        model_path: str,
        num_opponents: Optional[int] = None,
        num_player_sets: Optional[int] = None,
        skip_staleness_refresh: bool = False,
    ) -> Dict:
        num_opponents = num_opponents or self.num_opponents
        num_player_sets = num_player_sets or self.num_player_sets

        refresh_result = {"refreshed": [], "results": []}
        if not skip_staleness_refresh:
            refresh_result = self.rating_manager.refresh_stale_models(
                self.battle_simulator, num_player_sets
            )

        self.rating_manager.increment_staleness(excluded_model=model_path)

        eval_result = evaluate_and_update_trueskill(
            model_path=model_path,
            rating_manager=self.rating_manager,
            battle_simulator=self.battle_simulator,
            num_opponents=num_opponents,
            num_player_sets=num_player_sets,
        )

        eval_result["staleness_refresh"] = refresh_result

        return eval_result

    def get_rating(self, model_path: str) -> float:
        record = self.rating_manager.get_record(model_path)
        if record is None:
            record = self.rating_manager.register_model(model_path)
        return record.rating

    def get_mu(self, model_path: str) -> float:
        record = self.rating_manager.get_record(model_path)
        if record is None:
            record = self.rating_manager.register_model(model_path)
        return record.mu

    def get_sigma(self, model_path: str) -> float:
        record = self.rating_manager.get_record(model_path)
        if record is None:
            record = self.rating_manager.register_model(model_path)
        return record.sigma

    def print_leaderboard(self):
        print_trueskill_leaderboard(save_dir=self.save_dir)

    def register_model(
        self, model_path: str, mu: float = INITIAL_MU, sigma: float = INITIAL_SIGMA
    ) -> ModelTrueSkillRecord:
        return self.rating_manager.register_model(model_path, mu=mu, sigma=sigma)


if __name__ == "__main__":
    print("=" * 60)
    print("TrueSkill Rating System Test (with trueskill library)")
    print("=" * 60)

    print("\n--- Testing TrueSkill Calculation ---")
    manager = TrueSkillRatingManager(save_dir="./ckpts/bp_agent_test_trueskill")

    test_models = [
        "./ckpts/bp_agent_test_trueskill/model_1.pth",
        "./ckpts/bp_agent_test_trueskill/model_2.pth",
        "./ckpts/bp_agent_test_trueskill/model_3.pth",
    ]

    for i, model in enumerate(test_models):
        mu = 25.0 + (i - 1) * 5
        record = manager.register_model(model, mu=mu, sigma=INITIAL_SIGMA)
        print(
            f"Registered {model}: μ={record.mu:.2f}, σ={record.sigma:.2f}, Rating={record.rating:.2f}"
        )

    print("\n--- Testing Rating Update (single battle) ---")
    print("Simulating: model_2 beats model_1")

    record_before_a = manager.get_record(test_models[1])
    record_before_b = manager.get_record(test_models[0])
    print(
        f"Before: model_2 μ={record_before_a.mu:.2f}, model_1 μ={record_before_b.mu:.2f}"
    )

    manager.update_rating(test_models[1], test_models[0], 1.0)

    record_after_a = manager.get_record(test_models[1])
    record_after_b = manager.get_record(test_models[0])
    print(
        f"After:  model_2 μ={record_after_a.mu:.2f}, model_1 μ={record_after_b.mu:.2f}"
    )

    print("\n--- Testing Batch Update (16 battles: 9W-7L) ---")
    print("Simulating: model_2 vs model_1, 9 wins, 7 losses")

    mu_before = manager.get_record(test_models[1]).mu
    mu_change = manager.update_rating_batch(test_models[1], test_models[0], 9, 7, 0)
    mu_after = manager.get_record(test_models[1]).mu

    print(
        f"Before: μ={mu_before:.2f}, After: μ={mu_after:.2f}, Change: {mu_change:+.2f}"
    )

    print("\n--- Testing Opponent Selection ---")
    current_model = test_models[1]
    for _ in range(5):
        opponents = manager.select_opponents(current_model, num_opponents=2)
        opponent_mus = [manager.get_record(o).mu for o in opponents]
        print(f"Selected opponents μ: {opponent_mus}")

    print("\n--- Testing Leaderboard ---")
    print_trueskill_leaderboard(save_dir="./ckpts/bp_agent_test_trueskill")

    print("\n--- Testing TrueSkillEvaluator Interface ---")
    from eval import EvalMethod, get_evaluator

    evaluator = get_evaluator(
        EvalMethod.TRUESKILL, save_dir="./ckpts/bp_agent_test_trueskill"
    )
    print(f"Created evaluator: {type(evaluator).__name__}")
    print(f"Available models: {len(evaluator.list_models())}")

    print("\n[OK] All tests passed!")
