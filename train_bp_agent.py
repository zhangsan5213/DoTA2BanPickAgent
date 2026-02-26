r"""
BP Agent PPO Training Script

使用预训练的 WinRateOracle 作为 Reward Model，训练 BP Agent。
- 复用 model/win_rate_oracle.py 中的 hero_encoder 结构和权重
- 使用 ckpts\win_rate_oracle-num_heroes_256\win_rate_oracle-20260202012403-030-0.9835.pth 作为预训练模型
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
    MAX_EPISODES = 10000
    UPDATE_INTERVAL = 16  # 每多少个episode更新一次

    # BP 配置
    TOTAL_PICKS = 10  # 每队5个英雄
    TOTAL_BANS = 8    # 每队4个Ban


os.makedirs(Config.SAVE_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)


# ==================== BP 环境 ====================
class BPEnvironment:
    """
    简单的 BP 环境模拟
    """
    def __init__(self, matches_data, player_data_enabled=True):
        """
        Args:
            matches_data: 比赛数据列表
            player_data_enabled: 是否包含玩家数据
        """
        self.matches_data = matches_data
        self.player_data_enabled = player_data_enabled

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
            players = match_data.get('players', [])
            radiant_players, dire_players = self._split_players(players)
            self.radiant_player_feats = self._build_player_feats(radiant_players)
            self.dire_player_feats = self._build_player_feats(dire_players)

        # 当前行动方: 0=天辉ban, 1=夜魇ban, 2=天辉pick, 3=夜魇pick
        self.current_step = 0
        self.current_team = 0  # 0=天辉, 1=夜魇

        # 有效行动序列 (标准BP顺序)
        # 0: r_ban, 1: d_ban, 2: r_ban, 3: d_ban, 4: r_ban, 5: d_ban
        # 6: r_ban, 7: d_ban, 8: r_pick, 9: d_pick, 10: r_pick, 11: d_pick
        # 12: r_pick, 13: d_pick, 14: r_pick, 15: d_pick
        self.action_sequence = [
            (0, 'ban'), (1, 'ban'), (0, 'ban'), (1, 'ban'), (0, 'ban'), (1, 'ban'),
            (0, 'ban'), (1, 'ban'), (0, 'pick'), (1, 'pick'),
            (0, 'pick'), (1, 'pick'), (0, 'pick'), (1, 'pick')
        ]

        return self._get_state()

    def _extract_picks(self, match_data, team):
        """提取指定队伍的选英雄"""
        picks = []
        for act in match_data.get('picks_bans', []):
            if act.get('is_pick', False) and act.get('team', 0) == team:
                picks.append(act['hero_id'])
        return picks[:5]

    def _extract_bans(self, match_data):
        """提取两队的 ban 英雄"""
        radiant_bans, dire_bans = [], []
        for act in match_data.get('picks_bans', []):
            if not act.get('is_pick', True):  # is_pick=False means ban
                hero_id = act['hero_id']
                team = act.get('team', 0)
                if team == 0:
                    radiant_bans.append(hero_id)
                else:
                    dire_bans.append(hero_id)
        return radiant_bans, dire_bans

    def _extract_both_picks(self, match_data):
        """提取两队的 pick 英雄"""
        radiant_picks, dire_picks = [], []
        for act in match_data.get('picks_bans', []):
            if act.get('is_pick', False):
                hero_id = act['hero_id']
                team = act.get('team', 0)
                if team == 0:
                    radiant_picks.append(hero_id)
                else:
                    dire_picks.append(hero_id)
        return radiant_picks, dire_picks

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
            reward: 即时奖励
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

        # 计算奖励
        reward = 0.0
        if done:
            # 游戏结束，计算最终奖励
            reward = 1.0 if self.radiant_win else 0.0

        state = self._get_state() if not done else None
        return state, reward, done


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
    """批量处理轨迹数据"""
    return batch[0]  # 单个轨迹


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
    values = values.cpu().numpy()
    rewards = rewards.cpu().numpy()
    dones = dones.cpu().numpy()

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.tensor(advantages, dtype=torch.float32)
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
        if found := re.findall(r'-0\.(.+)\.pth$', ckpt_path):
            print(f"[*] Oracle 准确率: 0.{found[0]}")
    else:
        print(f"[!] 警告: Oracle 权重文件不存在 {ckpt_path}")

    return oracle


# ==================== PPO 训练 ====================
def collect_trajectories(envs, agent, oracle, device, num_steps=14):
    """
    收集多个环境的轨迹
    Args:
        envs: list of BPEnvironment
        agent: BPAgent
        oracle: WinRateOracle (用于计算 reward)
        device: torch device
        num_steps: BP 步数
    Returns:
        trajectories: 轨迹列表
    """
    trajectories = []

    for env_idx, env in enumerate(envs):
        # 重置环境
        state = env.reset()
        done = False

        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        while not done:
            # 编码状态
            state_feat = agent.encode_state(
                hero_ids=state['hero_ids'].to(device),
                team_flags=state['team_flags'].to(device),
                action_types=state['action_types'].to(device),
                valid_mask=state['valid_mask'].to(device),
                radiant_player_feats=state['radiant_player_feats'].to(device) if state['radiant_player_feats'] is not None else None,
                dire_player_feats=state['dire_player_feats'].to(device) if state['dire_player_feats'] is not None else None,
            )

            # 获取有效行动
            valid_heroes = env.get_valid_actions()
            if len(valid_heroes) == 0:
                break

            # 随机采样候选英雄（用于减少计算）
            K = min(32, len(valid_heroes))
            candidate_ids = random.sample(valid_heroes, K)
            # Padding
            while len(candidate_ids) < K:
                candidate_ids.append(0)
            candidate_ids = torch.tensor([candidate_ids], dtype=torch.long).to(device)

            # 获取行动
            action, log_prob, value = agent.get_action(
                state_feat=state_feat,
                candidate_hero_ids=candidate_ids,
                deterministic=False,
            )

            # 从有效行动中选择实际的英雄ID
            actual_action = candidate_ids[0, action[0].item()].item()

            # 执行行动
            next_state, reward, done = env.step(actual_action)

            # 存储转换
            states.append({
                'hero_ids': state['hero_ids'].clone(),
                'team_flags': state['team_flags'].clone(),
                'action_types': state['action_types'].clone(),
                'valid_mask': state['valid_mask'].clone(),
                'radiant_player_feats': state['radiant_player_feats'].clone() if state['radiant_player_feats'] is not None else None,
                'dire_player_feats': state['dire_player_feats'].clone() if state['dire_player_feats'] is not None else None,
                'state_feat': state_feat.detach(),
                'candidate_ids': candidate_ids.clone(),
                'action': action.clone(),
            })
            actions.append(actual_action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            state = next_state

        # 添加最终状态的 value
        if not done and state is not None:
            final_state_feat = agent.encode_state(
                hero_ids=state['hero_ids'].to(device),
                team_flags=state['team_flags'].to(device),
                action_types=state['action_types'].to(device),
                valid_mask=state['valid_mask'].to(device),
                radiant_player_feats=state['radiant_player_feats'].to(device) if state['radiant_player_feats'] is not None else None,
                dire_player_feats=state['dire_player_feats'].to(device) if state['dire_player_feats'] is not None else None,
            )
            final_value = agent.value(final_state_feat)
        else:
            final_value = torch.zeros(1, 1).to(device)

        values.append(final_value)

        # 构建轨迹
        trajectory = {
            'states': states,
            'actions': torch.tensor(actions, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'dones': torch.tensor(dones, dtype=torch.float32),
            'log_probs': torch.cat(log_probs),
            'values': torch.cat(values),
        }
        trajectories.append(trajectory)

    return trajectories


def train_ppo(agent, oracle, trajectories, optimizer, config, device):
    """
    PPO 更新
    """
    agent.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0

    for traj in trajectories:
        states = traj['states']
        actions = traj['actions'].to(device)
        rewards = traj['rewards'].to(device)
        dones = traj['dones'].to(device)
        old_log_probs = traj['log_probs'].to(device)
        old_values = traj['values'].to(device)

        # 计算 GAE
        advantages, returns = compute_gae(rewards, old_values, dones, config.GAMMA, config.LAMBDA)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 重新计算当前策略的 log_prob 和 value
        new_log_probs = []
        new_values = []
        entropies = []

        for i, state in enumerate(states):
            state_feat = agent.encode_state(
                hero_ids=state['hero_ids'].to(device),
                team_flags=state['team_flags'].to(device),
                action_types=state['action_types'].to(device),
                valid_mask=state['valid_mask'].to(device),
                radiant_player_feats=state['radiant_player_feats'].to(device) if state['radiant_player_feats'] is not None else None,
                dire_player_feats=state['dire_player_feats'].to(device) if state['dire_player_feats'] is not None else None,
            )

            candidate_ids = state['candidate_ids'].to(device)
            action = actions[i].unsqueeze(0)

            log_prob, value, entropy = agent.evaluate_actions(
                state_feat=state_feat,
                candidate_hero_ids=candidate_ids,
                actions=action,
            )

            new_log_probs.append(log_prob)
            new_values.append(value)
            entropies.append(entropy)

        new_log_probs = torch.cat(new_log_probs)
        new_values = torch.cat(new_values).squeeze(-1)
        entropies = torch.cat(entropies)

        # PPO 损失
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - config.CLIP_RATIO, 1 + config.CLIP_RATIO) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.functional.mse_loss(new_values, returns)

        entropy_loss = -entropies.mean()

        loss = policy_loss + config.VALUE_COEF * value_loss + config.ENTROPY_COEF * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), config.GRAD_CLIP)
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_entropy += entropy_loss.item()

    num_trajs = len(trajectories)
    return {
        'policy_loss': total_policy_loss / num_trajs,
        'value_loss': total_value_loss / num_trajs,
        'entropy': total_entropy / num_trajs,
    }


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    # 加载数据
    matches_data = load_matches_from_json(Config.DATA_FILE)

    # 创建环境
    envs = [BPEnvironment(matches_data, player_data_enabled=Config.USE_PLAYER_HEROES)
            for _ in range(Config.NUM_ENVS)]

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

    # 找出共享的 key 并复制
    for key in agent_state_dict:
        if key in oracle_state_dict:
            # 检查形状是否匹配
            if agent_state_dict[key].shape == oracle_state_dict[key].shape:
                agent_state_dict[key] = oracle_state_dict[key]

    agent.load_state_dict(agent_state_dict)
    print(f"[*] 已从 Oracle 复制 encoder 权重到 Agent")

    # 优化器
    optimizer = optim.Adam(agent.parameters(), lr=Config.LR)

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")))

    # 训练循环
    global_step = 0
    episode_rewards = []

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
            loss_dict = train_ppo(agent, oracle, trajectories, optimizer, Config, device)

            # 记录日志
            writer.add_scalar('Loss/Policy', loss_dict['policy_loss'], global_step)
            writer.add_scalar('Loss/Value', loss_dict['value_loss'], global_step)
            writer.add_scalar('Loss/Entropy', loss_dict['entropy'], global_step)
            writer.add_scalar('Reward/Mean', np.mean(episode_rewards[-Config.UPDATE_INTERVAL:]), global_step)

            global_step += 1

            # 保存检查点
            if (episode + 1) % (Config.UPDATE_INTERVAL * 10) == 0:
                ckpt_path = os.path.join(Config.SAVE_DIR, f"bp_agent-{datetime.now().strftime('%Y%m%d%H%M%S')}-{episode+1}.pth")
                torch.save(agent.state_dict(), ckpt_path)
                print(f"[*] 保存检查点: {ckpt_path}")

    writer.close()
    print("[*] 训练完成!")


if __name__ == "__main__":
    main()
