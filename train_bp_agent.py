r"""
BP Agent PPO Training Script

使用预训练的 WinRateOracle 作为 Reward Model，训练 BP Agent。
- 复用 model/win_rate_oracle.py 中的 hero_encoder 结构和权重
- 使用 Oracle 预测作为终局 reward
- 支持 player_preference_sampler 生成虚拟玩家偏好数据
"""

import os
import re
import json
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from model.bp_agent import BPAgent, BPStateEncoder, BPActorNetwork, BPValueNetwork
from model.win_rate_oracle import WinRateOracle
from model.hero_encoder import MultiModalHeroEncoder
from utils.raw_data import NUM_HEROES, NUM_HERO_FEATURES, HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP
from utils.player_preference_sampler import get_player_sampler, sample_random_player_features


def get_latest_oracle_ckpt(oracle_dir: str = "./ckpts/win_rate_oracle-num_heroes_160") -> str:
    """
    自动获取指定目录下时间戳最新的 Oracle checkpoint 文件
    
    文件名格式: win_rate_oracle-{timestamp}-{epoch}-{acc}.pth
    按时间戳排序返回最新的文件
    """
    if not os.path.exists(oracle_dir):
        raise FileNotFoundError(f"Oracle checkpoint 目录不存在: {oracle_dir}")
    
    ckpt_files = []
    for f in os.listdir(oracle_dir):
        if f.startswith("win_rate_oracle-") and f.endswith(".pth"):
            # 提取时间戳 (格式: YYYYMMDDHHMMSS)
            match = re.search(r'win_rate_oracle-(\d{14})-', f)
            if match:
                timestamp = match.group(1)
                ckpt_files.append((timestamp, os.path.join(oracle_dir, f)))
    
    if not ckpt_files:
        raise FileNotFoundError(f"在 {oracle_dir} 目录下未找到 Oracle checkpoint 文件")
    
    # 按时间戳排序，返回最新的
    ckpt_files.sort(key=lambda x: x[0], reverse=True)
    latest_ckpt = ckpt_files[0][1]
    print(f"[*] 自动选择最新的 Oracle checkpoint: {os.path.basename(latest_ckpt)}")
    return latest_ckpt


# ==================== 配置 ====================
class Config:
    # 路径配置
    ORACLE_CKPT_DIR = "./ckpts/win_rate_oracle-num_heroes_160"  # Oracle checkpoint 目录
    ORACLE_CKPT_PATH = None  # 设为 None 时自动抓取最新的，或手动指定路径
    DATA_FILE = "./data/high_mmr_with_stats-rank_40-duration_15.json"
    SAVE_DIR = "./ckpts/bp_agent"
    LOG_DIR = "./runs/bp_agent"

    # 模型配置
    EMBED_DIM = 64
    NHEAD = 4
    NUM_LAYERS = 4
    USE_TEXT = False
    USE_PLAYER_HEROES = True

    # PPO 配置
    PPO_EPOCHS = 4
    CLIP_RATIO = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    GAMMA = 0.99
    LAMBDA = 0.95
    LR = 3e-4
    GRAD_CLIP = 0.5

    # 训练配置
    BATCH_SIZE = 64
    NUM_ENVS = 16  # 并行环境数
    MAX_EPISODES = 64000
    UPDATE_INTERVAL = 16  # 每多少个episode更新一次
    SAVE_INTERVAL = 8  # 每多少个episode保存一次模型

    # BP 配置
    TOTAL_PICKS = 10  # 每队5个英雄
    TOTAL_BANS = 8    # 每队4个Ban
    
    # 玩家偏好采样配置
    USE_SAMPLED_PLAYERS = True  # 是否使用采样器生成虚拟玩家
    SAMPLED_PLAYER_RATIO = 0.5  # 使用采样玩家的比例（其余使用真实数据）
    PLAYER_SAMPLER_TEMP = 0.5   # 采样器 temperature
    PLAYER_SAMPLER_RANDOMNESS = 0.2  # 采样器 randomness
    
    # ELO 评估配置
    ELO_N_OPPONENTS = 8    # 评估时选择的对手数
    ELO_N_GAMES = 4        # 每个对手对战局数
    ELO_JSON_PATH = "./ckpts/bp_agent/elo_ratings.json"  # ELO记录文件


os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)


# ==================== BP 环境 ====================
class BPEnvironment:
    """
    简单的 BP 环境模拟
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

        # BP 状态
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
        self.radiant_ban_idx = 0  # 天辉ban计数
        self.dire_ban_idx = 0     # 夜魇ban计数
        self.radiant_pick_idx = 0  # 天辉pick计数
        self.dire_pick_idx = 0     # 夜魇pick计数

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

        # 当前行动方: 0=天辉ban, 1=夜魇ban, 2=天辉pick, 3=夜魇pick
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


# ==================== 数据集 ====================
class BPTrajectoryDataset(Dataset):
    """BP 轨迹数据集"""

    def __init__(self, trajectories):
        """
        trajectories: list of dict, 每个 dict 包含:
            - states: list of state dicts
            - actions: list of hero_ids
            - rewards: list of rewards
            - dones: list of bools
            - log_probs: list of log probabilities
            - values: list of value estimates
        """
        self.trajectories = trajectories

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        return self.trajectories[idx]


def collate_trajectories(batch):
    """批量处理轨迹数据 - 返回整个batch用于PPO的多epoch训练"""
    return batch


# ==================== 工具函数 ====================
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    计算 GAE (Generalized Advantage Estimation)
    Args:
        rewards: [T]
        values: [T+1]
        dones: [T]
    Returns:
        advantages: [T]
        returns: [T]
    """
    advantages = []
    gae = 0
    values = values.detach().cpu().numpy()
    rewards = rewards.detach().cpu().numpy()
    dones = dones.detach().cpu().numpy()

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
    returns = advantages + torch.tensor(values[:-1], dtype=torch.float32)

    return advantages, returns


def load_matches_from_json(file_path):
    """从 JSON 文件加载比赛数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤有效比赛（两队各有5个pick）
    valid_matches = []
    for m in data:
        r_picks = [act['hero_id'] for act in m.get('picks_bans', [])
                   if act.get('is_pick', False) and act.get('team', 0) == 0]
        d_picks = [act['hero_id'] for act in m.get('picks_bans', [])
                   if act.get('is_pick', False) and act.get('team', 0) == 1]
        if len(r_picks) == 5 and len(d_picks) == 5:
            valid_matches.append(m)

    print(f"[*] 加载了 {len(valid_matches)} 场有效比赛")
    return valid_matches


def load_oracle_and_copy_encoder(config, device):
    """加载预训练的 Oracle 并复制其 encoder"""
    oracle = WinRateOracle(
        embed_dim=config.EMBED_DIM,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        use_text=config.USE_TEXT,
        use_player_heroes=config.USE_PLAYER_HEROES,
        hero_encoder_id_dim=128,
        hero_encoder_attr_dim=64,
        hero_encoder_text_dim=128,
    ).to(device)

    # 确定 checkpoint 路径
    ckpt_path = config.ORACLE_CKPT_PATH
    if ckpt_path is None:
        # 自动抓取最新的 checkpoint
        ckpt_path = get_latest_oracle_ckpt(config.ORACLE_CKPT_DIR)
    
    if os.path.exists(ckpt_path):
        print(f"[*] 加载预训练 Oracle: {ckpt_path}")
        oracle.load_state_dict(torch.load(ckpt_path, map_location=device))
        # 提取并打印准确率
        match = re.search(r'-0\.(.+)\.pth$', ckpt_path)
        if match:
            print(f"[*] Oracle 准确率: 0.{match.group(1)}")
    else:
        print(f"[!] 警告: Oracle 权重文件不存在 {ckpt_path}")

    return oracle


def compute_oracle_reward(oracle, radiant_picks, dire_picks, radiant_player_feats, dire_player_feats, device):
    """
    使用 Oracle 计算终局奖励（返回天辉胜率，线性映射到 [-1, 1]）
    
    Args:
        oracle: WinRateOracle 模型
        radiant_picks: list of 5 hero ids (1-based)
        dire_picks: list of 5 hero ids (1-based)
        radiant_player_feats: [5, NUM_HEROES] tensor or None
        dire_player_feats: [5, NUM_HEROES] tensor or None
        device: torch device
    
    Returns:
        reward: float (映射到 [-1, 1]，1=天辉必胜，-1=天辉必败)
    """
    oracle.eval()
    with torch.no_grad():
        # 转换为 tensor (1-based hero ids)
        r_picks = torch.tensor([radiant_picks], dtype=torch.long, device=device)
        d_picks = torch.tensor([dire_picks], dtype=torch.long, device=device)
        
        # 获取英雄特征 (hero_input_from_ids 内部会转为 0-based)
        r_ids, r_attrs, r_sem = oracle.hero_input_from_ids(r_picks)
        d_ids, d_attrs, d_sem = oracle.hero_input_from_ids(d_picks)
        
        # 处理玩家特征
        if radiant_player_feats is not None:
            r_player = radiant_player_feats.unsqueeze(0).to(device)  # [1, 5, NUM_HEROES]
        else:
            r_player = None
        if dire_player_feats is not None:
            d_player = dire_player_feats.unsqueeze(0).to(device)
        else:
            d_player = None
        
        # 预测胜率
        win_prob = oracle.forward(r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem, r_player, d_player)
        
        # 线性映射 [0, 1] -> [-1, 1]
        reward = 2.0 * win_prob.item() - 1.0
        
    return reward


def compute_elo_expected_score(rating_a, rating_b):
    """计算A对B的期望胜率"""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a, rating_b, score_a, k=32):
    """
    更新ELO分数
    Args:
        rating_a: A当前分数
        rating_b: B当前分数
        score_a: A的实际得分（1=胜, 0.5=平, 0=负）
        k: K因子
    Returns:
        new_rating_a, new_rating_b
    """
    expected_a = compute_elo_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1.0 - score_a) - expected_b)
    
    return new_rating_a, new_rating_b


def run_single_match(agent_a, agent_b, oracle, matches_data, player_sampler, device):
    """
    运行单场比赛，返回agent_a的得分（1=胜, 0=负）
    天辉夜魇随机分配
    """
    # 随机分配
    if random.random() < 0.5:
        radiant_agent, dire_agent = agent_a, agent_b
        a_is_radiant = True
    else:
        radiant_agent, dire_agent = agent_b, agent_a
        a_is_radiant = False
    
    # 创建环境
    env = BPEnvironment(
        matches_data,
        player_data_enabled=Config.USE_PLAYER_HEROES,
        player_sampler=player_sampler,
        use_sampled_players=True
    )
    state = env.reset()
    done = False
    
    # 运行BP
    while not done:
        current_step = env.current_step
        current_team, _ = env.action_sequence[current_step]
        
        active_agent = radiant_agent if current_team == 0 else dire_agent
        
        with torch.no_grad():
            state_feat = active_agent.encode_state(
                hero_ids=state['hero_ids'].to(device),
                team_flags=state['team_flags'].to(device),
                action_types=state['action_types'].to(device),
                valid_mask=state['valid_mask'].to(device),
                radiant_player_feats=state['radiant_player_feats'].to(device) if state['radiant_player_feats'] is not None else None,
                dire_player_feats=state['dire_player_feats'].to(device) if state['dire_player_feats'] is not None else None,
            )
            
            valid_heroes = env.get_valid_actions()
            if len(valid_heroes) == 0:
                break
            
            K = min(32, len(valid_heroes))
            candidate_ids = random.sample(valid_heroes, K) if len(valid_heroes) >= K else valid_heroes + [0] * (K - len(valid_heroes))
            while len(candidate_ids) < 32:
                candidate_ids.append(0)
            candidate_ids = torch.tensor([candidate_ids], dtype=torch.long).to(device)
            
            action, _, _ = active_agent.get_action(
                state_feat=state_feat,
                candidate_hero_ids=candidate_ids,
                deterministic=True,
            )
            actual_action = candidate_ids[0, action[0].item()].item()
        
        state, _, done = env.step(actual_action)
    
    # Oracle判定胜负
    radiant_picks, dire_picks = env.get_final_picks()
    r_player_feats, d_player_feats = env.get_player_feats()
    
    if len(radiant_picks) != 5 or len(dire_picks) != 5:
        return 0.5  # 平局
    
    oracle.eval()
    with torch.no_grad():
        r_picks = torch.tensor([radiant_picks], dtype=torch.long, device=device)
        d_picks = torch.tensor([dire_picks], dtype=torch.long, device=device)
        r_ids, r_attrs, r_sem = oracle.hero_input_from_ids(r_picks)
        d_ids, d_attrs, d_sem = oracle.hero_input_from_ids(d_picks)
        
        r_player = r_player_feats.unsqueeze(0).to(device) if r_player_feats is not None else None
        d_player = d_player_feats.unsqueeze(0).to(device) if d_player_feats is not None else None
        
        radiant_win_prob = oracle.forward(r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem, r_player, d_player)
        radiant_win = radiant_win_prob.item() > 0.5
    
    # 返回agent_a的得分
    if a_is_radiant:
        return 1.0 if radiant_win else 0.0
    else:
        return 0.0 if radiant_win else 1.0


def evaluate_elo_rating(current_agent, oracle, checkpoints, elo_ratings, matches_data, player_sampler, device, n_games_per_opponent=8):
    """
    评估并更新当前模型的ELO分数
    
    与之前最近的若干个checkpoint对战，更新ELO分数
    
    Args:
        current_agent: 当前模型（已设为eval模式）
        oracle: WinRateOracle
        checkpoints: 所有保存的checkpoint路径列表
        elo_ratings: 每个checkpoint的ELO分数字典 {path: rating}
        matches_data: 比赛数据
        player_sampler: 玩家采样器
        device: torch device
        n_games_per_opponent: 每个对手对战局数
    
    Returns:
        current_elo: 当前模型的ELO分数
    """
    if len(checkpoints) < 2:
        return 1500.0
    
    current_ckpt = checkpoints[-1]
    
    # 与最近的最多8个对手对战
    historical_ckpts = checkpoints[:-1]
    n_opponents = min(Config.ELO_N_OPPONENTS, len(historical_ckpts))
    opponent_ckpts = historical_ckpts[-n_opponents:]  # 取最近的
    
    # 加载对手并更新ELO
    for opp_ckpt in opponent_ckpts:
        opp_agent = BPAgent(
            embed_dim=Config.EMBED_DIM,
            nhead=Config.NHEAD,
            num_layers=Config.NUM_LAYERS,
            use_text=Config.USE_TEXT,
            use_player_heroes=Config.USE_PLAYER_HEROES,
        ).to(device)
        opp_agent.load_state_dict(torch.load(opp_ckpt, map_location=device))
        opp_agent.eval()
        
        # 对战n_games_per_opponent局
        wins = 0
        for _ in range(n_games_per_opponent):
            score = run_single_match(current_agent, opp_agent, oracle, matches_data, player_sampler, device)
            wins += score
        
        # 计算平均得分
        avg_score = wins / n_games_per_opponent
        
        # 更新ELO（使用平均得分作为期望）
        current_rating = elo_ratings[current_ckpt]
        opp_rating = elo_ratings[opp_ckpt]
        
        new_current, new_opp = update_elo(current_rating, opp_rating, avg_score, k=32)
        
        elo_ratings[current_ckpt] = new_current
        elo_ratings[opp_ckpt] = new_opp
        
        del opp_agent
    
    return elo_ratings[current_ckpt]


# ==================== PPO 训练 ====================
def collect_trajectories(envs, agent, oracle, device, num_steps=14):
    """
    并行收集多个环境的轨迹（批量处理版本）
    
    将所有环境的状态组成 batch 一次性推理，充分利用 GPU 并行计算
    
    Args:
        envs: list of BPEnvironment
        agent: BPAgent
        oracle: WinRateOracle (用于计算终局 reward)
        device: torch device
        num_steps: BP 步数
    Returns:
        trajectories: 轨迹列表
    """
    n_envs = len(envs)
    
    # 初始化所有环境
    env_states = []
    env_trajectories = [{
        'states': [], 'actions': [], 'rewards': [], 'dones': [], 
        'log_probs': [], 'values': [], 'teams': [], 'done': False
    } for _ in range(n_envs)]
    
    for env in envs:
        state = env.reset()
        env_states.append({
            'state': state,
            'valid': True,
            'env': env,
        })
    
    # 批量处理直到所有环境完成
    while any(not env_trajectories[i]['done'] for i in range(n_envs)):
        # 收集所有活跃环境的状态
        active_indices = [i for i in range(n_envs) if not env_trajectories[i]['done']]
        if not active_indices:
            break
        
        # 构建 batch 输入
        batch_hero_ids = []
        batch_team_flags = []
        batch_action_types = []
        batch_valid_mask = []
        batch_radiant_player_feats = []
        batch_dire_player_feats = []
        batch_candidate_ids = []
        batch_valid_heroes_list = []  # 用于后续 step
        
        for idx in active_indices:
            state = env_states[idx]['state']
            env = env_states[idx]['env']
            
            batch_hero_ids.append(state['hero_ids'])
            batch_team_flags.append(state['team_flags'])
            batch_action_types.append(state['action_types'])
            batch_valid_mask.append(state['valid_mask'])
            batch_radiant_player_feats.append(state['radiant_player_feats'])
            batch_dire_player_feats.append(state['dire_player_feats'])
            
            # 获取候选英雄
            valid_heroes = env.get_valid_actions()
            batch_valid_heroes_list.append(valid_heroes)
            
            K = min(32, len(valid_heroes))
            candidate_ids = random.sample(valid_heroes, K) if len(valid_heroes) > 0 else [0] * K
            while len(candidate_ids) < 32:
                candidate_ids.append(0)
            batch_candidate_ids.append(candidate_ids)
        
        # 拼接成 batch tensor
        batch_hero_ids = torch.cat(batch_hero_ids, dim=0).to(device)
        batch_team_flags = torch.cat(batch_team_flags, dim=0).to(device)
        batch_action_types = torch.cat(batch_action_types, dim=0).to(device)
        batch_valid_mask = torch.cat(batch_valid_mask, dim=0).to(device)
        
        # 处理 player feats（可能为 None）
        if batch_radiant_player_feats[0] is not None:
            batch_radiant_player_feats = torch.cat(batch_radiant_player_feats, dim=0).to(device)
            batch_dire_player_feats = torch.cat(batch_dire_player_feats, dim=0).to(device)
        else:
            batch_radiant_player_feats = None
            batch_dire_player_feats = None
        
        batch_candidate_ids = torch.tensor(batch_candidate_ids, dtype=torch.long).to(device)
        
        # 批量编码状态
        with torch.no_grad():
            batch_state_feat = agent.encode_state(
                hero_ids=batch_hero_ids,
                team_flags=batch_team_flags,
                action_types=batch_action_types,
                valid_mask=batch_valid_mask,
                radiant_player_feats=batch_radiant_player_feats,
                dire_player_feats=batch_dire_player_feats,
            )
            
            # 批量获取行动
            batch_actions = []
            batch_log_probs = []
            batch_values = []
            
            for i in range(len(active_indices)):
                action, log_prob, value = agent.get_action(
                    state_feat=batch_state_feat[i:i+1],
                    candidate_hero_ids=batch_candidate_ids[i:i+1],
                    deterministic=False,
                )
                batch_actions.append(action)
                batch_log_probs.append(log_prob)
                batch_values.append(value)
        
        # 执行环境步骤
        for i, idx in enumerate(active_indices):
            env = env_states[idx]['env']
            state = env_states[idx]['state']
            trajectory = env_trajectories[idx]
            
            valid_heroes = batch_valid_heroes_list[i]
            if len(valid_heroes) == 0:
                # 无有效动作，标记为完成
                trajectory['done'] = True
                continue
            
            # 获取实际英雄ID
            action_idx = batch_actions[i].item()
            actual_action = batch_candidate_ids[i, action_idx].item()
            
            # 获取当前行动的 team (从 action_sequence 中查询)
            current_step = env.current_step - 1  # env.step() 已经增加了计数
            current_team = env.action_sequence[current_step][0] if current_step < len(env.action_sequence) else 0
            
            # 存储转换
            trajectory['states'].append({
                'hero_ids': state['hero_ids'].clone(),
                'team_flags': state['team_flags'].clone(),
                'action_types': state['action_types'].clone(),
                'valid_mask': state['valid_mask'].clone(),
                'radiant_player_feats': state['radiant_player_feats'].clone() if state['radiant_player_feats'] is not None else None,
                'dire_player_feats': state['dire_player_feats'].clone() if state['dire_player_feats'] is not None else None,
                'state_feat': batch_state_feat[i].detach().unsqueeze(0),
                'candidate_ids': batch_candidate_ids[i:i+1].clone(),
                'action_idx': batch_actions[i].clone(),
                'team': current_team,  # 记录当前行动的 team
            })
            trajectory['actions'].append(action_idx)
            trajectory['log_probs'].append(batch_log_probs[i])
            trajectory['values'].append(batch_values[i])
            trajectory['teams'].append(current_team)  # 记录每一步的 team
            
            # 执行步骤
            next_state, reward, done = env.step(actual_action)
            trajectory['rewards'].append(reward)
            trajectory['dones'].append(done)
            
            if done:
                trajectory['done'] = True
                # 使用 Oracle 计算终局奖励（始终用天辉胜率作为客观局势）
                radiant_picks, dire_picks = env.get_final_picks()
                r_player_feats, d_player_feats = env.get_player_feats()
                
                if len(radiant_picks) == 5 and len(dire_picks) == 5:
                    final_reward = compute_oracle_reward(
                        oracle, radiant_picks, dire_picks,
                        r_player_feats, d_player_feats, device
                    )
                    # 只在最后一步设置终局奖励（其余为0，GAE会传播）
                    if len(trajectory['rewards']) > 0:
                        trajectory['rewards'][-1] = final_reward
                else:
                    if len(trajectory['rewards']) > 0:
                        trajectory['rewards'][-1] = 0.5  # 异常结束给予中性奖励
                
                # 添加最终 value
                final_value = torch.zeros(1, 1).to(device)
                trajectory['values'].append(final_value)
            else:
                env_states[idx]['state'] = next_state
    
    # 处理未完成的环境（添加最终 value）
    for i in range(n_envs):
        trajectory = env_trajectories[i]
        if not trajectory.get('done', False):
            state = env_states[i]['state']
            with torch.no_grad():
                final_state_feat = agent.encode_state(
                    hero_ids=state['hero_ids'].to(device),
                    team_flags=state['team_flags'].to(device),
                    action_types=state['action_types'].to(device),
                    valid_mask=state['valid_mask'].to(device),
                    radiant_player_feats=state['radiant_player_feats'].to(device) if state['radiant_player_feats'] is not None else None,
                    dire_player_feats=state['dire_player_feats'].to(device) if state['dire_player_feats'] is not None else None,
                )
                final_value = agent.value(final_state_feat)
            trajectory['values'].append(final_value)
            trajectory['done'] = True
    
    # 构建最终轨迹列表
    trajectories = []
    for trajectory in env_trajectories:
        trajectories.append({
            'states': trajectory['states'],
            'actions': torch.tensor(trajectory['actions'], dtype=torch.long),
            'rewards': torch.tensor(trajectory['rewards'], dtype=torch.float32),
            'dones': torch.tensor(trajectory['dones'], dtype=torch.float32),
            'log_probs': torch.cat(trajectory['log_probs']),
            'values': torch.cat(trajectory['values']),
        })
    
    return trajectories


def train_ppo(agent, trajectories, optimizer, config, device):
    """
    PPO 更新 - 支持多 epoch
    """
    agent.train()
    
    # 合并所有轨迹的数据用于 mini-batch 训练
    all_states = []
    all_actions = []
    all_advantages = []
    all_returns = []
    all_old_log_probs = []
    
    # 首先为每个轨迹计算 GAE
    for traj in trajectories:
        states = traj['states']
        actions = traj['actions'].to(device)
        rewards = traj['rewards'].to(device)
        dones = traj['dones'].to(device)
        old_log_probs = traj['log_probs'].to(device)
        old_values = traj['values'].to(device)

        # 计算 GAE（基于天辉视角的 value）
        advantages, returns = compute_gae(rewards, old_values, dones, config.GAMMA, config.LAMBDA)
        
        # 根据 team 调整 advantage 符号
        # team=0（天辉）: 最大化天辉胜率，advantage 不变
        # team=1（夜魇）: 最小化天辉胜率，advantage 取反
        for i in range(len(states)):
            team = states[i]['team']
            if team == 1:
                advantages[i] = -advantages[i]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 存储
        for i in range(len(states)):
            all_states.append(states[i])
            all_actions.append(actions[i])
            all_advantages.append(advantages[i])
            all_returns.append(returns[i])
            all_old_log_probs.append(old_log_probs[i])
    
    # 转换为 tensor 并移到 device
    all_actions = torch.stack(all_actions).to(device)
    all_advantages = torch.stack(all_advantages).to(device)
    all_returns = torch.stack(all_returns).to(device)
    all_old_log_probs = torch.stack(all_old_log_probs).to(device)
    
    # 多 epoch 训练
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    n_updates = 0
    
    dataset_size = len(all_states)
    
    for epoch in range(config.PPO_EPOCHS):
        # 随机打乱
        indices = torch.randperm(dataset_size)
        
        # Mini-batch 训练
        for start in range(0, dataset_size, config.BATCH_SIZE):
            end = min(start + config.BATCH_SIZE, dataset_size)
            batch_indices = indices[start:end]
            
            # 收集 batch 数据
            batch_states = [all_states[i] for i in batch_indices]
            batch_actions = all_actions[batch_indices]
            batch_advantages = all_advantages[batch_indices]
            batch_returns = all_returns[batch_indices]
            batch_old_log_probs = all_old_log_probs[batch_indices]
            
            # 重新计算 log_prob 和 value
            new_log_probs = []
            new_values = []
            entropies = []
            
            for i, state in enumerate(batch_states):
                state_feat = agent.encode_state(
                    hero_ids=state['hero_ids'].to(device),
                    team_flags=state['team_flags'].to(device),
                    action_types=state['action_types'].to(device),
                    valid_mask=state['valid_mask'].to(device),
                    radiant_player_feats=state['radiant_player_feats'].to(device) if state['radiant_player_feats'] is not None else None,
                    dire_player_feats=state['dire_player_feats'].to(device) if state['dire_player_feats'] is not None else None,
                )

                candidate_ids = state['candidate_ids'].to(device)
                action_idx = state['action_idx'].to(device)  # 候选列表中的索引

                log_prob, value, entropy = agent.evaluate_actions(
                    state_feat=state_feat,
                    candidate_hero_ids=candidate_ids,
                    actions=action_idx,
                )

                new_log_probs.append(log_prob)
                new_values.append(value)
                entropies.append(entropy)
            
            new_log_probs = torch.cat(new_log_probs)
            new_values = torch.cat(new_values).squeeze(-1)  # [B, 1] -> [B]
            entropies = torch.cat(entropies)
            
            # 确保 batch_returns 也是 1D
            batch_returns = batch_returns.squeeze(-1)
            
            # PPO 损失
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - config.CLIP_RATIO, 1 + config.CLIP_RATIO) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(new_values, batch_returns)

            entropy_loss = -entropies.mean()

            loss = policy_loss + config.VALUE_COEF * value_loss + config.ENTROPY_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy_loss.item()
            n_updates += 1
    
    return {
        'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0,
        'value_loss': total_value_loss / n_updates if n_updates > 0 else 0,
        'entropy': total_entropy / n_updates if n_updates > 0 else 0,
    }


def main():
    # 设置随机种子
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    # 加载数据
    matches_data = load_matches_from_json(Config.DATA_FILE)

    # 初始化玩家偏好采样器（如果需要）
    player_sampler = None
    if Config.USE_SAMPLED_PLAYERS or Config.USE_PLAYER_HEROES:
        player_sampler = get_player_sampler(
            temperature=Config.PLAYER_SAMPLER_TEMP,
            randomness=Config.PLAYER_SAMPLER_RANDOMNESS
        )
        print(f"[*] 已初始化 PlayerPreferenceSampler (temp={Config.PLAYER_SAMPLER_TEMP})")

    # 创建环境 - 按比例混合真实玩家和采样玩家
    envs = []
    for i in range(Config.NUM_ENVS):
        use_sampled = Config.USE_SAMPLED_PLAYERS and (random.random() < Config.SAMPLED_PLAYER_RATIO)
        env = BPEnvironment(
            matches_data, 
            player_data_enabled=Config.USE_PLAYER_HEROES,
            player_sampler=player_sampler,
            use_sampled_players=use_sampled
        )
        envs.append(env)
    
    print(f"[*] 创建了 {Config.NUM_ENVS} 个环境 (采样玩家比例: {Config.SAMPLED_PLAYER_RATIO:.0%})")

    # 加载 Oracle 并复制 encoder
    oracle = load_oracle_and_copy_encoder(Config, device)
    oracle.eval()

    # 创建 Agent（复用 oracle 的 encoder 权重）
    agent = BPAgent(
        embed_dim=Config.EMBED_DIM,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        use_text=Config.USE_TEXT,
        use_player_heroes=Config.USE_PLAYER_HEROES,
    ).to(device)

    # 复制 oracle 的 hero_encoder 权重到 agent
    oracle_state_dict = oracle.state_dict()
    agent_state_dict = agent.state_dict()

    # 映射关系：Oracle 的 hero_encoder 对应 Agent 的 state_encoder.hero_encoder
    copied_keys = []
    for key in oracle_state_dict:
        # 只复制 hero_encoder 相关的权重
        if key.startswith('hero_encoder.'):
            # 映射到 agent 中的路径
            agent_key = 'state_encoder.' + key
            if agent_key in agent_state_dict:
                if agent_state_dict[agent_key].shape == oracle_state_dict[key].shape:
                    agent_state_dict[agent_key] = oracle_state_dict[key]
                    copied_keys.append(agent_key)
        # player_encoder 也可以复制
        elif key.startswith('player_encoder.'):
            agent_key = 'state_encoder.' + key
            if agent_key in agent_state_dict:
                if agent_state_dict[agent_key].shape == oracle_state_dict[key].shape:
                    agent_state_dict[agent_key] = oracle_state_dict[key]
                    copied_keys.append(agent_key)

    agent.load_state_dict(agent_state_dict)
    print(f"[*] 已从 Oracle 复制 {len(copied_keys)} 个模块到 Agent")

    # 优化器
    optimizer = optim.Adam(agent.parameters(), lr=Config.LR)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")))

    # 训练循环
    global_step = 0
    episode_rewards = []
    
    # 加载已有ELO记录
    elo_ratings = {}
    if os.path.exists(Config.ELO_JSON_PATH):
        with open(Config.ELO_JSON_PATH, 'r') as f:
            elo_data = json.load(f)
            elo_ratings = elo_data.get('ratings', {})
            print(f"[*] 加载ELO记录: {len(elo_ratings)} 个checkpoint")
    
    # 检查已有checkpoint
    existing_checkpoints = []
    if os.path.exists(Config.SAVE_DIR):
        for f in os.listdir(Config.SAVE_DIR):
            if f.endswith('.pth') and f.startswith('bp_agent-'):
                existing_checkpoints.append(os.path.join(Config.SAVE_DIR, f))
    
    # 找出没有ELO记录的新checkpoint
    new_checkpoints = [ckpt for ckpt in existing_checkpoints if ckpt not in elo_ratings]
    
    # 如果有新ckpt，初始化ELO并通过对战定分
    if new_checkpoints:
        print(f"[*] 发现 {len(new_checkpoints)} 个新checkpoint需要ELO定分...")
        for ckpt in new_checkpoints:
            elo_ratings[ckpt] = 1500.0
        
        # 选择用于定分的ckpt集合（新ckpt + 部分已有ckpt）
        old_ckpts = [c for c in elo_ratings.keys() if c not in new_checkpoints]
        selected_old = random.sample(old_ckpts, min(Config.ELO_N_OPPONENTS, len(old_ckpts))) if old_ckpts else []
        selected_ckpts = new_checkpoints + selected_old
        
        # 让它们互相PK更新ELO
        print(f"[*] ELO定分: {len(new_checkpoints)} 新ckpt + {len(selected_old)} 旧ckpt 互相PK...")
        for i, ckpt_a in enumerate(selected_ckpts):
            agent_a = BPAgent(
                embed_dim=Config.EMBED_DIM, nhead=Config.NHEAD, num_layers=Config.NUM_LAYERS,
                use_text=Config.USE_TEXT, use_player_heroes=Config.USE_PLAYER_HEROES,
            ).to(device)
            agent_a.load_state_dict(torch.load(ckpt_a, map_location=device))
            agent_a.eval()
            
            for j, ckpt_b in enumerate(selected_ckpts[i+1:], i+1):
                agent_b = BPAgent(
                    embed_dim=Config.EMBED_DIM, nhead=Config.NHEAD, num_layers=Config.NUM_LAYERS,
                    use_text=Config.USE_TEXT, use_player_heroes=Config.USE_PLAYER_HEROES,
                ).to(device)
                agent_b.load_state_dict(torch.load(ckpt_b, map_location=device))
                agent_b.eval()
                
                wins_a = 0
                for _ in range(Config.ELO_N_GAMES):
                    score_a = run_single_match(agent_a, agent_b, oracle, matches_data, player_sampler, device)
                    wins_a += score_a
                avg_score = wins_a / Config.ELO_N_GAMES
                
                rating_a, rating_b = update_elo(elo_ratings[ckpt_a], elo_ratings[ckpt_b], avg_score, k=32)
                elo_ratings[ckpt_a] = rating_a
                elo_ratings[ckpt_b] = rating_b
                del agent_b
            del agent_a
        
        # 保存ELO记录
        with open(Config.ELO_JSON_PATH, 'w') as f:
            json.dump({'ratings': elo_ratings, 'last_updated': datetime.now().isoformat()}, f, indent=2)
        print(f"[*] ELO定分完成，已保存到 {Config.ELO_JSON_PATH}")
    
    # 选择ELO最高的checkpoint加载（从已有ELO记录中选择）
    saved_checkpoints = []  # 本次训练保存的checkpoint
    
    if elo_ratings:
        best_ckpt = max(elo_ratings.keys(), key=lambda x: elo_ratings[x])
        best_elo = elo_ratings[best_ckpt]
        # 检查文件是否存在
        if os.path.exists(best_ckpt):
            print(f"[*] 加载最强checkpoint: {os.path.basename(best_ckpt)}, ELO={best_elo:.1f}")
            agent.load_state_dict(torch.load(best_ckpt, map_location=device))
            saved_checkpoints.append(best_ckpt)
        else:
            print(f"[!] 最强checkpoint文件不存在，使用Oracle权重初始化")
    
    print(f"[*] 开始训练，共 {Config.MAX_EPISODES} 个 episode...")

    for episode in tqdm(range(Config.MAX_EPISODES), desc="Training", ncols=90):
        # 收集轨迹
        trajectories = collect_trajectories(envs, agent, oracle, device)

        # 计算本轮 episode 的总奖励
        for traj in trajectories:
            total_reward = traj['rewards'].sum().item()
            episode_rewards.append(total_reward)

        # PPO 更新
        if (episode + 1) % Config.UPDATE_INTERVAL == 0:
            loss_dict = train_ppo(agent, trajectories, optimizer, Config, device)

            # 记录基础训练指标
            writer.add_scalar('Loss/Policy', loss_dict['policy_loss'], global_step)
            writer.add_scalar('Loss/Value', loss_dict['value_loss'], global_step)
            writer.add_scalar('Loss/Entropy', loss_dict['entropy'], global_step)

            global_step += 1

            # 保存检查点
            if (episode + 1) % (Config.UPDATE_INTERVAL * Config.SAVE_INTERVAL) == 0:
                ckpt_path = os.path.join(Config.SAVE_DIR, f"bp_agent-{datetime.now().strftime('%Y%m%d%H%M%S')}-{episode+1}.pth")
                torch.save(agent.state_dict(), ckpt_path)
                print(f"[*] 保存检查点: {ckpt_path}")
                saved_checkpoints.append(ckpt_path)
                
                # 初始化新checkpoint的ELO分数（继承前一个分数或1500）
                if len(saved_checkpoints) == 1:
                    elo_ratings[ckpt_path] = 1500.0
                else:
                    # 继承前一个checkpoint的分数作为初始值
                    prev_ckpt = saved_checkpoints[-2]
                    if prev_ckpt in elo_ratings:
                        elo_ratings[ckpt_path] = elo_ratings[prev_ckpt]
                    else:
                        elo_ratings[ckpt_path] = 1500.0
                
                # 每次保存都进行ELO定级：与随机选择的对手PK
                print(f"[*] 新checkpoint ELO定级中...")
                agent.eval()
                
                # 选择对手（从历史ckpt中随机选）
                all_historical = [c for c in elo_ratings.keys() if c != ckpt_path]
                if all_historical:
                    n_opp = min(Config.ELO_N_OPPONENTS, len(all_historical))
                    opponents = random.sample(all_historical, n_opp)
                    
                    # 与每个对手对战
                    for opp_ckpt in opponents:
                        opp_agent = BPAgent(
                            embed_dim=Config.EMBED_DIM, nhead=Config.NHEAD, num_layers=Config.NUM_LAYERS,
                            use_text=Config.USE_TEXT, use_player_heroes=Config.USE_PLAYER_HEROES,
                        ).to(device)
                        opp_agent.load_state_dict(torch.load(opp_ckpt, map_location=device))
                        opp_agent.eval()
                        
                        wins = 0
                        for _ in range(Config.ELO_N_GAMES):
                            score = run_single_match(agent, opp_agent, oracle, matches_data, player_sampler, device)
                            wins += score
                        avg_score = wins / Config.ELO_N_GAMES
                        
                        # 更新ELO
                        current_rating = elo_ratings[ckpt_path]
                        opp_rating = elo_ratings[opp_ckpt]
                        new_rating, new_opp_rating = update_elo(current_rating, opp_rating, avg_score, k=32)
                        elo_ratings[ckpt_path] = new_rating
                        elo_ratings[opp_ckpt] = new_opp_rating
                        
                        del opp_agent
                
                agent.train()
                
                # 记录ELO
                current_elo = elo_ratings[ckpt_path]
                writer.add_scalar('Elo/Rating', current_elo, global_step)
                avg_elo = sum(elo_ratings.values()) / len(elo_ratings)
                writer.add_scalar('Elo/AverageRating', avg_elo, global_step)
                print(f"[*] ELO评分: 当前={current_elo:.1f}, 平均={avg_elo:.1f}")
                
                # 保存ELO记录到JSON
                with open(Config.ELO_JSON_PATH, 'w') as f:
                    json.dump({'ratings': elo_ratings, 'last_updated': datetime.now().isoformat()}, f, indent=2)

    writer.close()
    print("[*] 训练完成!")


if __name__ == "__main__":
    main()
