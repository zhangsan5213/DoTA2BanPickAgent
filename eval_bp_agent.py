"""
BP Agent 评估脚本 - 让最强的模型对战，输出阵容展示训练效果

Usage:
    python eval_bp_agent.py --top_n 3 --matches 5
    python eval_bp_agent.py --models model1.pth model2.pth --matches 3
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F

from model.bp_agent import BPTransformerAgent
from model.win_rate_oracle import WinRateOracle
from utils.device import DEVICE
from utils.raw_data import NUM_HEROES, get_valid_hero_ids, get_valid_hero_ids
from utils.bp_env import BPState
from eval import TrueSkillRatingManager, EloRatingManager


# ============== 英雄名称映射 ==============

def load_hero_id_to_name() -> Dict[int, str]:
    """从 hero_features.xlsx 加载英雄ID到名称的映射"""
    try:
        import pandas as pd
        df = pd.read_excel("./data/hero_features.xlsx")
        return {int(row['id']): row['name'] for _, row in df.iterrows()}
    except Exception as e:
        print(f"[!] Warning: Could not load hero names: {e}")
        return {}

# 全局缓存
_HERO_ID_TO_NAME: Dict[int, str] = {}

def get_hero_name(hero_id: int) -> str:
    """获取英雄名称"""
    global _HERO_ID_TO_NAME
    if not _HERO_ID_TO_NAME:
        _HERO_ID_TO_NAME = load_hero_id_to_name()
    return _HERO_ID_TO_NAME.get(hero_id, f"Hero_{hero_id}")


# ============== 对战模拟 ==============

class BPDuelSimulator:
    """BP 对战模拟器 - 用于展示模型对战过程"""
    
    def __init__(self, oracle: Optional[WinRateOracle] = None, oracle_path: Optional[str] = None):
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
            print(f"[+] Loaded oracle from {oracle_path}")
        else:
            print(f"[!] Warning: Oracle not found at {oracle_path}")
        
        oracle.eval()
        return oracle
    
    def load_agent(self, model_path: str) -> BPTransformerAgent:
        """加载 BP Agent 模型"""
        agent = BPTransformerAgent(embed_dim=128, nhead=8, num_layers=4).to(DEVICE)
        agent.load_state_dict(torch.load(model_path, map_location=DEVICE))
        agent.eval()
        return agent
    
    def run_bp_duel(
        self,
        agent_radiant: BPTransformerAgent,
        agent_dire: BPTransformerAgent,
        r_players: List[List[float]],
        d_players: List[List[float]],
        verbose: bool = True
    ) -> Tuple[List[int], List[int], float, List[Dict]]:
        """
        运行一场完整的 BP 对战，并输出过程
        
        Returns:
            radiant_picks, dire_picks, win_prob, action_history
        """
        state = BPState([], [], r_players, d_players, radiant_bans=[], dire_bans=[], is_radiant_turn=True, step_idx=0)
        action_history = []
        
        if verbose:
            print("\n" + "="*80)
            print("🎮 BP 对战开始")
            print("="*80)
        
        step = 0
        # CM BP顺序（20步）已在BPState中定义
        
        while not state.done and step < 24:
            state_dict = state.to_dict()
            current_agent = agent_radiant if state.is_radiant_turn else agent_dire
            team_name = "🔴 Radiant" if state.is_radiant_turn else "🔵 Dire"
            
            with torch.no_grad():
                action_logits, _ = current_agent(state_dict)
                
                # Mask 已使用的英雄（包括ban和pick）和不存在英雄
                valid_actions = state.get_valid_actions()
                mask = torch.full((NUM_HEROES,), -1e9, device=action_logits.device)
                # 先标记所有有效英雄为可选
                valid_hero_ids = get_valid_hero_ids()
                for h in valid_hero_ids:
                    if h <= NUM_HEROES:
                        mask[h - 1] = 0.0
                # 再屏蔽已使用的英雄（包括ban和pick）
                used = set(state.radiant_heroes + state.dire_heroes + state.radiant_bans + state.dire_bans)
                for h in used:
                    if h <= NUM_HEROES:
                        mask[h - 1] = -1e9
                action_logits = action_logits + mask
                
                # 贪心选择
                hero_id = torch.argmax(action_logits, dim=-1).item() + 1
                hero_name = get_hero_name(hero_id)
            
            # 使用BPState的CM序列判断当前是ban还是pick
            is_pick = (state.get_current_action_type() == 'pick')
            action_type = "PICK" if is_pick else "BAN"
            
            action_info = {
                'step': step,
                'team': 'radiant' if state.is_radiant_turn else 'dire',
                'action_type': action_type,
                'hero_id': hero_id,
                'hero_name': hero_name
            }
            action_history.append(action_info)
            
            if verbose:
                ban_count = state.ban_count['radiant'] + state.ban_count['dire']
                pick_count = state.pick_count['radiant'] + state.pick_count['dire']
                progress = f"[Ban:{ban_count}/10 Pick:{pick_count}/10]"
                print(f"  {team_name:<12} {action_type:<5} {hero_name:<20} {progress}")
            
            state.step(hero_id)  # is_pick由BPState自动判断
            step += 1
        
        # 获取胜率
        if len(state.radiant_heroes) >= 5 and len(state.dire_heroes) >= 5:
            r_picks = state.radiant_heroes[:5]
            d_picks = state.dire_heroes[:5]
        else:
            r_picks = state.radiant_heroes + [1] * (5 - len(state.radiant_heroes))
            d_picks = state.dire_heroes + [1] * (5 - len(state.dire_heroes))
        
        win_prob = state.get_reward(self.oracle)
        
        if verbose:
            print("-"*80)
            print("📊 最终结果")
            print("-"*80)
            print(f"  🔴 Radiant Picks: {[get_hero_name(h) for h in r_picks]}")
            print(f"  🔵 Dire Picks:    {[get_hero_name(h) for h in d_picks]}")
            print(f"  📈 Radiant Win Probability: {win_prob*100:.1f}%")
            print(f"  📉 Dire Win Probability: {(1-win_prob)*100:.1f}%")
            winner = "🔴 Radiant" if win_prob > 0.5 else "🔵 Dire"
            print(f"  🏆 Winner: {winner}")
            print("="*80)
        
        return r_picks, d_picks, win_prob, action_history


# ============== 模型选择 ==============

def get_top_models_by_trueskill(save_dir: str = "./ckpts/bp_agent", top_n: int = 3) -> List[Tuple[str, float]]:
    """从 TrueSkill 评分数据库获取评分最高的模型"""
    db_path = Path(save_dir) / "trueskill_ratings.json"
    
    if not db_path.exists():
        print(f"[!] TrueSkill database not found at {db_path}")
        return []
    
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    # 按 rating (mu - 3*sigma) 排序
    models = []
    for path, record in data.items():
        if os.path.exists(path):
            rating = record.get('mu', 25.0) - 3 * record.get('sigma', 8.33)
            models.append((path, rating, record.get('mu', 25.0), record.get('sigma', 8.33)))
    
    models.sort(key=lambda x: x[1], reverse=True)
    return models[:top_n]


def get_top_models_by_elo(save_dir: str = "./ckpts/bp_agent", top_n: int = 3) -> List[Tuple[str, float]]:
    """从 ELO 评分数据库获取评分最高的模型"""
    db_path = Path(save_dir) / "elo_ratings.json"
    
    if not db_path.exists():
        print(f"[!] ELO database not found at {db_path}")
        return []
    
    with open(db_path, 'r') as f:
        data = json.load(f)
    
    models = []
    for path, record in data.items():
        if os.path.exists(path):
            rating = record.get('rating', 1500)
            models.append((path, rating))
    
    models.sort(key=lambda x: x[1], reverse=True)
    return [(m[0], m[1], None, None) for m in models[:top_n]]


def generate_random_players(num_samples: int = 1) -> List[Dict]:
    """生成随机玩家数据"""
    try:
        from utils.player_preference_sampler import batch_sample_player_preferences
        
        player_sets = []
        for _ in range(num_samples):
            all_players = batch_sample_player_preferences(
                num_players=10,
                m=3,
                n=5
            )
            
            def build_features(players):
                features = []
                valid_hero_ids = get_valid_hero_ids()
                for p in players:
                    vector = [0.0] * NUM_HEROES
                    for hero_info in p['heroes']:
                        hero_id = hero_info['id']
                        win_rate = hero_info['win_rate']
                        # 只添加实际存在的英雄
                        if hero_id in valid_hero_ids and hero_id <= NUM_HEROES:
                            vector[hero_id - 1] = win_rate  # 修正：使用hero_id - 1作为索引
                    features.append(vector)
                return features
            
            player_sets.append({
                'r_players': build_features(all_players[:5]),
                'd_players': build_features(all_players[5:])
            })
        
        return player_sets
    except Exception as e:
        print(f"[!] Error generating players: {e}")
        # 返回默认空玩家
        return [{
            'r_players': [[0.0] * NUM_HEROES for _ in range(5)],
            'd_players': [[0.0] * NUM_HEROES for _ in range(5)]
        }] * num_samples


# ============== 主函数 ==============

def run_tournament(
    model_paths: List[str],
    num_matches: int = 3,
    oracle_path: Optional[str] = None,
    verbose: bool = True
):
    """运行模型之间的循环赛"""
    simulator = BPDuelSimulator(oracle_path=oracle_path)
    agents = {path: simulator.load_agent(path) for path in model_paths}
    
    results = {path: {'wins': 0, 'losses': 0, 'total_win_prob': 0.0, 'matches': 0} for path in model_paths}
    
    print("\n" + "🎮"*40)
    print(" "*30 + "BP AGENT TOURNAMENT" + " "*30)
    print("🎮"*40)
    
    print(f"\n[+] Loaded {len(model_paths)} models:")
    for i, path in enumerate(model_paths, 1):
        print(f"    {i}. {os.path.basename(path)}")
    
    # 循环赛
    for i, model_a in enumerate(model_paths):
        for j, model_b in enumerate(model_paths):
            if i >= j:  # 避免重复对战
                continue
            
            print(f"\n{'='*80}")
            print(f"⚔️  MATCH: {os.path.basename(model_a)} vs {os.path.basename(model_b)}")
            print(f"{'='*80}")
            
            for match_idx in range(num_matches):
                player_set = generate_random_players(1)[0]
                
                # 随机决定哪方先手
                if random.choice([True, False]):
                    agent_r, agent_d = agents[model_a], agents[model_b]
                    model_r, model_d = model_a, model_b
                else:
                    agent_r, agent_d = agents[model_b], agents[model_a]
                    model_r, model_d = model_b, model_a
                
                r_picks, d_picks, win_prob, _ = simulator.run_bp_duel(
                    agent_r, agent_d,
                    player_set['r_players'],
                    player_set['d_players'],
                    verbose=verbose
                )
                
                # 更新结果
                if model_r == model_a:
                    a_win_prob = win_prob
                else:
                    a_win_prob = 1.0 - win_prob
                
                if a_win_prob > 0.5:
                    results[model_a]['wins'] += 1
                    results[model_b]['losses'] += 1
                elif a_win_prob < 0.5:
                    results[model_a]['losses'] += 1
                    results[model_b]['wins'] += 1
                else:
                    # 平局
                    pass
                
                results[model_a]['total_win_prob'] += a_win_prob
                results[model_b]['total_win_prob'] += (1 - a_win_prob)
                results[model_a]['matches'] += 1
                results[model_b]['matches'] += 1
    
    # 打印最终排行榜
    print("\n" + "🏆"*40)
    print(" "*35 + "FINAL STANDINGS" + " "*35)
    print("🏆"*40)
    print(f"{'Rank':<6}{'Model':<45}{'W/L':<10}{'Win%':<10}{'Avg WinProb':<12}")
    print("-"*80)
    
    standings = []
    for path, stats in results.items():
        if stats['matches'] > 0:
            win_rate = stats['wins'] / stats['matches']
            avg_win_prob = stats['total_win_prob'] / stats['matches']
            standings.append((path, stats['wins'], stats['losses'], win_rate, avg_win_prob))
    
    standings.sort(key=lambda x: x[3], reverse=True)
    
    for rank, (path, wins, losses, win_rate, avg_win_prob) in enumerate(standings, 1):
        model_name = os.path.basename(path)[:43]
        print(f"{rank:<6}{model_name:<45}{wins}/{losses:<8}{win_rate*100:>6.1f}%    {avg_win_prob*100:>6.1f}%")
    
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="BP Agent Evaluation - Watch top models battle!")
    parser.add_argument("--top_n", type=int, default=3, help="Select top N models from leaderboard")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Specific model paths to evaluate")
    parser.add_argument("--matches", type=int, default=3, help="Number of matches per pair")
    parser.add_argument("--oracle_path", type=str, default=None, help="Path to WinRateOracle checkpoint")
    parser.add_argument("--rating", type=str, choices=["trueskill", "elo"], default="trueskill", 
                        help="Rating system to use for selecting top models")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    # 确定要使用的模型
    if args.models:
        model_paths = args.models
    else:
        if args.rating == "trueskill":
            top_models = get_top_models_by_trueskill(top_n=args.top_n)
        else:
            top_models = get_top_models_by_elo(top_n=args.top_n)
        
        if not top_models:
            print("[!] No models found in rating database!")
            print("[*] Please train some models first or specify model paths with --models")
            return
        
        print(f"\n[+] Top {len(top_models)} models by {args.rating}:")
        for i, (path, rating, mu, sigma) in enumerate(top_models, 1):
            if mu is not None:
                print(f"    {i}. {os.path.basename(path)} (μ={mu:.2f}, σ={sigma:.2f}, rating={rating:.2f})")
            else:
                print(f"    {i}. {os.path.basename(path)} (rating={rating:.0f})")
        
        model_paths = [m[0] for m in top_models]
    
    # 验证模型文件存在
    valid_paths = [p for p in model_paths if os.path.exists(p)]
    if len(valid_paths) != len(model_paths):
        missing = set(model_paths) - set(valid_paths)
        print(f"[!] Warning: {len(missing)} model(s) not found:")
        for m in missing:
            print(f"    - {m}")
    
    if len(valid_paths) < 2:
        print("[!] Error: Need at least 2 models to run a tournament!")
        return
    
    # 运行锦标赛
    run_tournament(
        valid_paths,
        num_matches=args.matches,
        oracle_path=args.oracle_path,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
