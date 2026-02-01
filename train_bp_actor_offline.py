import os
import re
import json
import torch
import random
import pathlib
import glob

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP, NUM_HEROES
from utils.get_data_cm_bp import fetch_high_mmr_matches
from model.win_rate_oracle import WinRateOracle
from model.bp_policy import PPODecoderAgent

torch.random.manual_seed(42)

BP_ACTOR_SAVE_DIR = "./ckpts/bp_actor"
WIN_RATE_ORACLE_SAVE_DIR = "./ckpts/win_rate_oracle"
if not os.path.exists(BP_ACTOR_SAVE_DIR):
    pathlib.Path(BP_ACTOR_SAVE_DIR).mkdir(parents=True, exist_ok=True)

def double_match_data(match_list: list[dict]):
    """双向数据增强：交换两队并翻转胜负标签"""
    match_list_copy = []
    for match in match_list:
        match_copy = deepcopy(match)
        match_copy['radiant_win'] = not match_copy['radiant_win']
        for bp in match_copy['picks_bans']:
            bp['team'] = 1 - bp['team']
        # 同时交换玩家数据
        if 'players' in match_copy:
            # player_slot: 0-4 = 天辉, 128-132 = 夜魇
            for player in match_copy['players']:
                player['player_slot'] = 128 + player['player_slot'] if player['player_slot'] < 128 else player['player_slot'] - 128
        match_list_copy.append(match_copy)
    match_list.extend(match_list_copy)


def find_latest_oracle_model(save_dir: str = WIN_RATE_ORACLE_SAVE_DIR):
    """从保存目录中找到最新的 WinRateOracle 模型"""
    pattern = os.path.join(save_dir, "win_rate_oracle-*.pth")
    model_files = glob.glob(pattern)
    if not model_files:
        return None
    # 按修改时间排序，取最新的
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

class BPDataset(Dataset):
    """
    BP数据集，从 high_mmr_with_stats.json 加载
    支持玩家特征提取用于 Oracle 奖励计算
    """
    def __init__(self, json_data, max_len=24, min_total_games=10):
        self.data = json_data
        self.max_len = max_len
        self.min_total_games = min_total_games

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def split_players_by_team(players):
        """根据 player_slot 将玩家分为天辉和夜魇两队"""
        radiant, dire = [], []
        for p in players:
            slot = p.get('player_slot', 0)
            if slot < 128:
                radiant.append(p)
            else:
                dire.append(p)
        return radiant, dire
    
    @staticmethod
    def build_player_feature_vector(players, num_heroes=NUM_HEROES, max_heroes_per_player=10, min_games=3, min_total_games=10):
        """
        从玩家 hero_history 构建胜率特征向量 [5, num_heroes]
        """
        vectors = []
        for player in players[:5]:
            hero_history = player.get('hero_history', {})
            total_games = sum(h.get('games', 0) for h in hero_history.values())
            
            if total_games < min_total_games:
                vectors.append([0.0] * num_heroes)
                continue
            
            hero_list = []
            for hero_id_str, stats in hero_history.items():
                try:
                    hero_id = int(hero_id_str)
                    games = stats.get('games', 0)
                    wins = stats.get('wins', 0)
                    winrate = wins / games if games > 0 else 0
                    hero_list.append((hero_id, games, winrate))
                except (ValueError, TypeError):
                    continue
            
            hero_list.sort(key=lambda x: x[1], reverse=True)
            
            vector = [0.0] * num_heroes
            for hero_id, games, winrate in hero_list[:max_heroes_per_player]:
                if 0 < hero_id < num_heroes and games >= min_games:
                    vector[hero_id] = winrate
            
            vectors.append(vector)
        
        while len(vectors) < 5:
            vectors.append([0.0] * num_heroes)
        return vectors

    def __getitem__(self, idx):
        match = self.data[idx]
        pb = match['picks_bans']
        
        # 准备容器
        hero_ids = torch.zeros(self.max_len, dtype=torch.long)
        team_ids = torch.zeros(self.max_len, dtype=torch.long)
        type_ids = torch.zeros(self.max_len, dtype=torch.long)
        
        # 提取 10 个 pick 用于 Oracle（后续计算奖励用）
        radiant_picks = []
        dire_picks = []

        for i, step in enumerate(pb):
            if i >= self.max_len: break
            h_id = step['hero_id']
            hero_ids[i] = h_id
            # team: 0 (Radiant) -> 1, 1 (Dire) -> 2
            team_ids[i] = 1 if step['team'] == 0 else 2
            # is_pick: False (Ban) -> 1, True (Pick) -> 2
            type_ids[i] = 2 if step['is_pick'] else 1
            
            if step['is_pick']:
                if step['team'] == 0: 
                    radiant_picks.append(h_id)
                else: 
                    dire_picks.append(h_id)

        # 提取玩家特征
        players = match.get('players', [])
        radiant_players, dire_players = self.split_players_by_team(players)
        r_player_feats = self.build_player_feature_vector(radiant_players, min_total_games=self.min_total_games)
        d_player_feats = self.build_player_feature_vector(dire_players, min_total_games=self.min_total_games)

        return {
            "hero_seq": hero_ids,
            "team_seq": team_ids,
            "type_seq": type_ids,
            "radiant_picks": torch.tensor(radiant_picks, dtype=torch.long),
            "dire_picks": torch.tensor(dire_picks, dtype=torch.long),
            "radiant_player_feats": torch.tensor(r_player_feats, dtype=torch.float32),
            "dire_player_feats": torch.tensor(d_player_feats, dtype=torch.float32),
        }

class BP_PPOTrainer:
    def __init__(self, agent: PPODecoderAgent, oracle: WinRateOracle, writer: SummaryWriter, 
                 lr=1e-4, gamma=0.95, lam=0.95, eps_clip=0.2, ent_coef=0.01, critic_loss_factor=1,
                 weight_decay=0.01):
        self.agent = agent
        self.oracle = oracle  # WinRateOracle
        # 使用 AdamW 优化器，与 train_winrate_oracle.py 保持一致
        self.optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=weight_decay)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.ent_coef = ent_coef  # 熵系数，鼓励探索
        self.critic_loss_factor = critic_loss_factor
        self.writer = writer
        self.global_step = 0
    
    def prep_data(self, batch):
        device = next(self.agent.parameters()).device
        
        hero_seq = batch['hero_seq'].to(device)  # [B, 24]
        team_seq = batch['team_seq'].to(device)  # [B, 24]
        type_seq = batch['type_seq'].to(device)  # [B, 24]
        r_picks = batch['radiant_picks'].to(device)         # [B, 5]
        d_picks = batch['dire_picks'].to(device)            # [B, 5]
        r_player_feats = batch['radiant_player_feats'].to(device)  # [B, 5, NUM_HEROES]
        d_player_feats = batch['dire_player_feats'].to(device)     # [B, 5, NUM_HEROES]

        return device, hero_seq, team_seq, type_seq, r_picks, d_picks, r_player_feats, d_player_feats

    def compute_reward(self, radiant_picks, dire_picks, r_player_feats=None, d_player_feats=None):
        """
        利用 Oracle 计算最终奖励 (Radiant视角)
        支持玩家特征输入
        """
        with torch.no_grad():
            # win_prob 范围 0~1 (Radiant 胜率)
            win_prob = self.oracle.predict(
                radiant_picks, dire_picks, 
                radiant_player_feats=r_player_feats, 
                dire_player_feats=d_player_feats,
                return_tensor=True
            )
            # 映射到 -1 ~ 1
            reward = (win_prob - 0.5) * 2
            return reward.squeeze(-1)  # [B]
    
    def train_step_critic_pretrain(self, batch):
        device, hero_seq, team_seq, type_seq, r_picks, d_picks, r_player_feats, d_player_feats = self.prep_data(batch)
        
        # 1. 计算最终奖励 R
        with torch.no_grad():
            # final_reward 范围 [-1, 1]，现在支持玩家特征
            final_reward = self.compute_reward(r_picks, d_picks, r_player_feats, d_player_feats)  # [B]
            
            # 2. 构造指数衰减序列
            T = 24
            steps = torch.arange(T, device=device).float()  # [0, 1, ..., 23]
            
            # 计算每个 step 的权重: gamma^(T - 1 - t)
            weights = self.gamma ** (T - 1 - steps)  # [24]
            
            # 3. 计算 Soft Targets
            # targets 形状: [B, 24]
            targets = final_reward.unsqueeze(1) * weights.unsqueeze(0)  # [B, 1] * [1, 24] = [B, 24]
            
        # 4. 前向传播
        _, curr_values = self.agent(hero_seq, team_seq, type_seq)
        
        # 5. 计算损失
        loss = F.mse_loss(curr_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        # 记录到 TensorBoard
        self.writer.add_scalar('Pretrain/Critic Loss', loss.item(), self.global_step)
        self.global_step += 1
        
        return loss.item()

    def train_step(self, batch, num_ppo_iters: int = 4):
        device, hero_seq, team_seq, type_seq, r_picks, d_picks, r_player_feats, d_player_feats = self.prep_data(batch)

        # 1. 预计算原始 Logits 和 Values (用于 PPO 的 Old Log Probs)
        with torch.no_grad():
            old_logits, values = self.agent(hero_seq, team_seq, type_seq)
            
            # 必须使用 Mask 来计算 old_log_probs，确保合法的英雄选择
            old_log_probs_list = []
            for t in range(24):
                if t > 0:
                    mask_t = torch.full((hero_seq.shape[0], NUM_HEROES + 1), 0.0, device=device)
                    for b in range(hero_seq.shape[0]):
                        for prev_t in range(t):
                            mask_t[b, hero_seq[b, prev_t].item()] = -1e9
                else:
                    mask_t = torch.zeros((hero_seq.shape[0], NUM_HEROES + 1), device=device)
                
                step_logits = old_logits[:, t, :] + mask_t
                old_dist_t = Categorical(logits=step_logits)
                old_log_probs_list.append(old_dist_t.log_prob(hero_seq[:, t]))
            
            old_log_probs = torch.stack(old_log_probs_list, dim=1)  # [B, 24]

        # 2. 计算奖励序列 (只有最后一步有值)，支持玩家特征
        rewards = self.compute_reward(r_picks, d_picks, r_player_feats, d_player_feats).flatten()

        # 3. 计算 GAE (Generalized Advantage Estimation)
        advantages = torch.zeros([rewards.shape[0], 24]).cuda()
        last_gae_lam = 0
        with torch.no_grad():
            for t in reversed(range(24)):
                # 1. 确定当前步的即时奖励 r_t
                # 只有最后一步 (t=23) 有奖励
                curr_reward = rewards if t == 23 else 0
                
                # 2. 确定下一步的价值 V(s_{t+1})
                # 如果是最后一步，后面没状态了，V_next = 0
                # 否则，V_next = values[:, t+1]
                if t == 23:
                    v_next = 0
                else:
                    v_next = values[:, t+1]
                
                # 3. 计算 TD Error: delta = r + gamma * V(s_next) - V(s_curr)
                delta = curr_reward + self.gamma * v_next - values[:, t]
                
                # 4. 计算 GAE
                # A_t = delta_t + gamma * lam * A_{t+1}
                advantages[:, t] = last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
        
        # TD Target = Advantage + Value
        td_targets = advantages + values

        # 4. PPO 核心更新循环
        for _ in range(num_ppo_iters): 
            # 重新获取当前策略的预测
            curr_logits, curr_values = self.agent(hero_seq, team_seq, type_seq)
            # 必须使用 Mask 来计算 log_prob，确保与 old_log_probs 分布一致
            curr_log_probs_list = []
            entropy_list = []
            for t in range(24):
                # 构建当前步的 mask（屏蔽已选英雄）
                if t > 0:
                    mask_t = torch.full((hero_seq.shape[0], NUM_HEROES + 1), 0.0, device=device)
                    # 屏蔽已选英雄（设为极小值）
                    for b in range(hero_seq.shape[0]):
                        for prev_t in range(t):
                            mask_t[b, hero_seq[b, prev_t].item()] = -1e9
                else:
                    mask_t = torch.zeros((hero_seq.shape[0], NUM_HEROES + 1), device=device)
                
                step_logits = curr_logits[:, t, :] + mask_t
                curr_dist_t = Categorical(logits=step_logits)
                curr_log_probs_list.append(curr_dist_t.log_prob(hero_seq[:, t]))
                entropy_list.append(curr_dist_t.entropy())
            
            curr_log_probs = torch.stack(curr_log_probs_list, dim=1)
            entropy = torch.stack(entropy_list, dim=1)

            # 重要性采样比例
            ratio = torch.exp(curr_log_probs - old_log_probs)

            # --- 核心修改：Advantage 方向性调整 ---
            # 我们的 values 和 rewards 是基于 Radiant 视角的。
            # 如果当前是 Dire (team_seq == 2) 在选人，他应该最小化 Radiant 的 Value。
            # 所以对 Dire 而言，真正的 Advantage = -Standard_Advantage
            # team_seq: 1 为 Radiant, 2 为 Dire
            perspective_mask = torch.where(team_seq == 1, 1.0, -1.0)
            adj_advantages = advantages * perspective_mask

            # PPO Actor Loss
            surr1 = ratio * adj_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adj_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value Loss: 使 Critic 预测的“Radiant 胜率”更准
            critic_loss = F.mse_loss(curr_values, td_targets.detach())

            # Total Loss
            loss = actor_loss + self.critic_loss_factor * critic_loss - self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪 (针对 Transformer)
            nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.optimizer.step()

        self.writer.add_scalar('Train/Actor Loss', actor_loss.item(), self.global_step)
        self.writer.add_scalar('Train/Critic Loss', critic_loss.item(), self.global_step)
        self.writer.add_scalar('Train/Total Loss', loss.item(), self.global_step)
        self.writer.add_scalar('Train/Entropy', entropy.mean().item(), self.global_step)
        self.global_step += 1
        return actor_loss.item(), critic_loss.item()

if __name__ == "__main__":
    # 0. 更新比赛数据（可选）
    # print(">>> Updating match data ...")
    # fetch_high_mmr_matches(
    #     output_file='./data/high_mmr_with_stats.json',
    #     target_count=100000,
    #     min_rank=50,
    #     min_duration=18 * 60,
    # )

    # 1. 准备数据
    print(">>> Loading data ...")
    with open("./data/high_mmr_with_stats.json", "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # 双向数据增强
    print(">>> Applying data augmentation ...")
    double_match_data(raw_data)
    
    dataset = BPDataset(raw_data, min_total_games=10)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    print(f"[*] Dataset size: {len(dataset)}")

    # 2. 实例化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Using device: {device}")
    
    # 自动加载最新的 Oracle 模型
    oracle_model_path = find_latest_oracle_model()
    if oracle_model_path is None:
        raise FileNotFoundError(f"No WinRateOracle model found in {WIN_RATE_ORACLE_SAVE_DIR}")
    print(f"[*] Loading Oracle from: {oracle_model_path}")
    
    # 使用与 train_winrate_oracle.py 一致的模型参数
    oracle = WinRateOracle(
        embed_dim=64, 
        nhead=4, 
        num_layers=4, 
        use_text=False, 
        use_player_heroes=True,
        # HeroEncoder 参数
        hero_encoder_id_dim=128,
        hero_encoder_attr_dim=64,
        hero_encoder_text_dim=128,
        hero_encoder_dropout=0.1,
        hero_encoder_res_layers=3,
        hero_encoder_attn_heads=4,
        hero_encoder_modality_dropout=0.1,
    ).to(device)
    
    # 加载模型权重
    oracle.load_state_dict(torch.load(oracle_model_path, map_location=device))
    oracle.eval()  # 设置为评估模式
    print("[*] Oracle loaded successfully")
    
    # 创建 PPO Agent
    agent = PPODecoderAgent(embed_dim=32, nhead=4, num_layers=4, dim_feedforward=64).to(device)
    print(f"[*] Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # 3. 训练
    log_dir = os.path.join("runs", "bp_ppo_agent_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[*] TensorBoard logs: {log_dir}")
    
    epochs_critic_pretrain = 4
    epochs_together = 1280
    trainer = BP_PPOTrainer(
        agent, oracle, writer, 
        lr=1e-4, 
        critic_loss_factor=1,
        weight_decay=0.01  # 与 train_winrate_oracle.py 一致
    )

    print(">>> Stage 1: Pre-training Critic with Supervised Learning...")
    for epoch in range(epochs_critic_pretrain):
        pbar = tqdm(loader, desc=f"Critic Pretrain Epoch {epoch+1}/{epochs_critic_pretrain}", ncols=160)
        for batch in pbar:
            critic_loss = trainer.train_step_critic_pretrain(batch)
            pbar.set_postfix({"Critic loss": f"{critic_loss:.4f}"})

    print(">>> Stage 2: Joint Actor-Critic PPO Training...")
    best_loss = float('inf')
    for epoch in range(epochs_together):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs_together}", ncols=160)
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        num_batches = 0
        
        for batch in pbar:
            actor_loss, critic_loss = trainer.train_step(batch)
            epoch_actor_loss += actor_loss
            epoch_critic_loss += critic_loss
            num_batches += 1
            pbar.set_postfix({
                "Actor loss": f"{actor_loss:.4f}", 
                "Critic loss": f"{critic_loss:.4f}"
            })
        
        # 记录每个 Epoch 的平均损失
        avg_actor_loss = epoch_actor_loss / num_batches
        avg_critic_loss = epoch_critic_loss / num_batches
        writer.add_scalar('Epoch/Actor Loss', avg_actor_loss, epoch)
        writer.add_scalar('Epoch/Critic Loss', avg_critic_loss, epoch)
        
        # 保存最佳模型
        total_loss = avg_actor_loss + avg_critic_loss
        if total_loss < best_loss:
            best_loss = total_loss
            datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
            save_path = os.path.join(BP_ACTOR_SAVE_DIR, f"bp_actor-{datetime_str}-{epoch:04d}-{total_loss:.4f}.pth")
            torch.save(agent.state_dict(), save_path)
            print(f"[*] Saved best model to: {save_path}")
    
    writer.close()
    print(">>> Training completed!")