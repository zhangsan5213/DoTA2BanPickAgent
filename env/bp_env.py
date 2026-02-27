"""
BP Environment for Dota2 Ban/Pick

模拟Dota2的BP流程，支持随机玩家偏好采样
"""
import random
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from utils.raw_data import NUM_HEROES


class BPEnvironment:
    """
    Dota2 BP 环境模拟
    
    支持标准Dota2 BP流程：
    - Ban Phase 1: 4 bans (交替)
    - Pick Phase 1: 4 picks (R, D, D, R)
    - Ban Phase 2: 4 bans (交替)
    - Pick Phase 2: 4 picks (D, R, D, R)
    - Final Pick: 2 picks (R, D)
    """
    
    def __init__(self, matches_data, player_data_enabled=True, player_sampler=None, use_sampled_players=False):
        """
        Args:
            matches_data: 比赛数据列表
            player_data_enabled: 是否包含玩家数据
            player_sampler: 玩家偏好采样器（用于生成虚拟玩家）
            use_sampled_players: 是否使用采样器生成玩家偏好（而非从数据读取）
        """
        self.matches_data = matches_data
        self.player_data_enabled = player_data_enabled
        self.player_sampler = player_sampler
        self.use_sampled_players = use_sampled_players
        
        # BP 状态（将在reset中初始化）
        self.reset()

    def reset(self, match_data=None):
        """重置环境到初始状态"""
        if match_data is None:
            match_data = random.choice(self.matches_data)

        self.match_data = match_data
        self.radiant_picks = []  # 天辉已选英雄
        self.dire_picks = []     # 夜魇已选英雄
        self.radiant_bans = []   # 天辉已Ban英雄
        self.dire_bans = []      # 夜魇已Ban英雄
        
        # 提取目标阵容（用于计算 reward）
        self.target_radiant = self._extract_picks(match_data, team=0)
        self.target_dire = self._extract_picks(match_data, team=1)
        self.radiant_win = match_data.get('radiant_win', False)

        # 玩家数据
        self.radiant_player_feats = None
        self.dire_player_feats = None
        
        if self.player_data_enabled:
            if self.use_sampled_players and self.player_sampler is not None:
                # 使用采样器生成虚拟玩家偏好
                radiant_prefs = self.player_sampler.sample_player_preferences(n_players=5)
                dire_prefs = self.player_sampler.sample_player_preferences(n_players=5)
                
                self.radiant_player_feats = [
                    self.player_sampler.preferences_to_winrate_vector(p).tolist()
                    for p in radiant_prefs
                ]
                self.dire_player_feats = [
                    self.player_sampler.preferences_to_winrate_vector(p).tolist()
                    for p in dire_prefs
                ]
            else:
                # 从比赛数据读取玩家信息
                players = match_data.get('players', [])
                if players and len(players) > 0:
                    radiant_players, dire_players = self._split_players(players)
                    self.radiant_player_feats = self._build_player_feats(radiant_players)
                    self.dire_player_feats = self._build_player_feats(dire_players)
                elif self.player_sampler is not None:
                    # 数据中没有玩家信息，使用采样器生成
                    radiant_prefs = self.player_sampler.sample_player_preferences(n_players=5)
                    dire_prefs = self.player_sampler.sample_player_preferences(n_players=5)
                    
                    self.radiant_player_feats = [
                        self.player_sampler.preferences_to_winrate_vector(p).tolist()
                        for p in radiant_prefs
                    ]
                    self.dire_player_feats = [
                        self.player_sampler.preferences_to_winrate_vector(p).tolist()
                        for p in dire_prefs
                    ]

        # 当前行动方
        self.current_step = 0
        self.current_team = 0  # 0=天辉, 1=夜魇

        # 有效行动序列 (标准Dota2 BP顺序)
        # Ban Phase 1: r_ban, d_ban, r_ban, d_ban (4 bans)
        # Pick Phase 1: r_pick, d_pick, d_pick, r_pick (4 picks)
        # Ban Phase 2: d_ban, r_ban, d_ban, r_ban (4 bans)
        # Pick Phase 2: d_pick, r_pick, d_pick, r_pick (4 picks)
        # Final Pick: r_pick, d_pick (2 picks)
        # Total: 8 bans, 10 picks
        self.action_sequence = [
            # Ban Phase 1
            (0, 'ban'), (1, 'ban'), (0, 'ban'), (1, 'ban'),
            # Pick Phase 1
            (0, 'pick'), (1, 'pick'), (1, 'pick'), (0, 'pick'),
            # Ban Phase 2
            (1, 'ban'), (0, 'ban'), (1, 'ban'), (0, 'ban'),
            # Pick Phase 2
            (1, 'pick'), (0, 'pick'), (1, 'pick'), (0, 'pick'),
            # Final Pick
            (0, 'pick'), (1, 'pick'),
        ]

        return self._get_state()

    def _extract_picks(self, match_data, team):
        """提取指定队伍的选英雄"""
        picks = []
        for act in match_data.get('picks_bans', []):
            if act.get('is_pick', False) and act.get('team', 0) == team:
                picks.append(act['hero_id'])
        return picks[:5]

    def _split_players(self, players):
        """根据 player_slot 分队"""
        radiant, dire = [], []
        for p in players:
            slot = p.get('player_slot', 0)
            if slot < 128:
                radiant.append(p)
            else:
                dire.append(p)
        return radiant, dire

    def _build_player_feats(self, players):
        """构建玩家特征 [5, NUM_HEROES]"""
        vectors = []
        for player in players[:5]:
            hero_history = player.get('hero_history', {})
            vector = [0.0] * NUM_HEROES
            for hero_id_str, stats in hero_history.items():
                try:
                    hero_id = int(hero_id_str)
                    games = stats.get('games', 0)
                    wins = stats.get('wins', 0)
                    if 0 < hero_id < NUM_HEROES and games >= 3:
                        vector[hero_id] = wins / games
                except (ValueError, TypeError):
                    continue
            vectors.append(vector)
        while len(vectors) < 5:
            vectors.append([0.0] * NUM_HEROES)
        return vectors

    def _get_state(self):
        """获取当前状态"""
        # 构建 BP 序列
        hero_ids = []
        team_flags = []
        action_types = []
        valid_mask = []

        # 跟踪当前索引
        r_ban_idx = 0
        d_ban_idx = 0
        r_pick_idx = 0
        d_pick_idx = 0

        for i, (team, action_type) in enumerate(self.action_sequence[:self.current_step + 1]):
            if action_type == 'ban':
                if team == 0:
                    hero_ids.append(self.radiant_bans[r_ban_idx] if r_ban_idx < len(self.radiant_bans) else 0)
                    r_ban_idx += 1
                else:
                    hero_ids.append(self.dire_bans[d_ban_idx] if d_ban_idx < len(self.dire_bans) else 0)
                    d_ban_idx += 1
            else:
                if team == 0:
                    hero_ids.append(self.radiant_picks[r_pick_idx] if r_pick_idx < len(self.radiant_picks) else 0)
                    r_pick_idx += 1
                else:
                    hero_ids.append(self.dire_picks[d_pick_idx] if d_pick_idx < len(self.dire_picks) else 0)
                    d_pick_idx += 1
            team_flags.append(team)
            action_types.append(0 if action_type == 'ban' else 1)
            valid_mask.append(1)

        # Padding
        max_len = 24
        while len(hero_ids) < max_len:
            hero_ids.append(0)
            team_flags.append(0)
            action_types.append(0)
            valid_mask.append(0)

        return {
            'hero_ids': torch.tensor([hero_ids], dtype=torch.long),
            'team_flags': torch.tensor([team_flags], dtype=torch.long),
            'action_types': torch.tensor([action_types], dtype=torch.long),
            'valid_mask': torch.tensor([valid_mask], dtype=torch.long),
            'radiant_player_feats': torch.tensor([self.radiant_player_feats], dtype=torch.float32) if self.radiant_player_feats else None,
            'dire_player_feats': torch.tensor([self.dire_player_feats], dtype=torch.float32) if self.dire_player_feats else None,
        }

    def get_valid_actions(self):
        """获取当前有效的行动（可选择的英雄）"""
        banned = set(self.radiant_bans + self.dire_bans)
        picked = set(self.radiant_picks + self.dire_picks)
        invalid = banned | picked

        # 返回所有可用英雄的 ID
        valid_heroes = [h for h in range(1, NUM_HEROES + 1) if h not in invalid]
        return valid_heroes

    def step(self, hero_id):
        """
        执行一步行动
        Returns:
            state: 下一个状态
            reward: 即时奖励（中间步骤为0，终局由外部计算）
            done: 是否结束
        """
        team, action_type = self.action_sequence[self.current_step]

        if action_type == 'ban':
            if team == 0:
                self.radiant_bans.append(hero_id)
            else:
                self.dire_bans.append(hero_id)
        else:
            if team == 0:
                self.radiant_picks.append(hero_id)
            else:
                self.dire_picks.append(hero_id)

        self.current_step += 1
        done = self.current_step >= len(self.action_sequence)

        # 中间步骤奖励为0，终局奖励由外部使用 Oracle 计算
        reward = 0.0

        state = self._get_state() if not done else None
        return state, reward, done

    def get_final_picks(self):
        """获取最终阵容"""
        return self.radiant_picks.copy(), self.dire_picks.copy()
    
    def get_player_feats(self):
        """获取玩家特征（用于 Oracle 评估）"""
        r_feats = torch.tensor(self.radiant_player_feats, dtype=torch.float32) if self.radiant_player_feats else None
        d_feats = torch.tensor(self.dire_player_feats, dtype=torch.float32) if self.dire_player_feats else None
        return r_feats, d_feats
