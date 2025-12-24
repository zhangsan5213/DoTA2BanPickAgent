import os
import re
import json
import torch
import random
import pathlib

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP
from utils.get_data_cm_bp import fetch_high_mmr_matches
from model.win_rate_oracle import WinRateOracle
from model.bp_policy import PPODecoderAgent

torch.random.manual_seed(42)

BP_ACTOR_SAVE_DIR = "./ckpts/bp_actor"
if not os.path.exists(BP_ACTOR_SAVE_DIR):
    pathlib.Path(BP_ACTOR_SAVE_DIR).mkdir(parents=True, exist_ok=True)

class BPDataset(Dataset):
    def __init__(self, json_data, max_len=24):
        self.data = json_data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

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
                if step['team'] == 0: radiant_picks.append(h_id)
                else: dire_picks.append(h_id)

        return {
            "hero_seq": hero_ids,
            "team_seq": team_ids,
            "type_seq": type_ids,
            "radiant_picks": torch.tensor(radiant_picks),
            "dire_picks": torch.tensor(dire_picks)
        }

class BP_PPOTrainer:
    def __init__(self, agent, oracle, lr=1e-4, gamma=0.99, lam=0.95, eps_clip=0.2, ent_coef=0.01):
        self.agent = agent
        self.oracle = oracle  # WinRateOracle
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.ent_coef = ent_coef # 熵系数，鼓励探索

    def compute_reward(self, radiant_picks, dire_picks):
        """
        利用 Oracle 计算最终奖励 (Radiant视角)
        """
        with torch.no_grad():
            # win_prob 范围 0~1 (Radiant 胜率)
            win_prob = self.oracle.predict(radiant_picks, dire_picks) # TODO：去oracle里面定义一个predict，支持这样传入id
            # 映射到 -1 ~ 1
            reward = (win_prob - 0.5) * 2
            return reward

    def train_step(self, batch):
        device = next(self.agent.parameters()).device
        
        # 转换数据
        hero_seq = batch['hero_seq'].to(device) # [B, 24]
        team_seq = batch['team_seq'].to(device) # [B, 24]
        type_seq = batch['type_seq'].to(device) # [B, 24]
        r_picks = batch['radiant_picks']        # [B, 5]
        d_picks = batch['dire_picks']           # [B, 5]

        # 1. 预计算原始 Logits 和 Values (用于 PPO 的 Old Log Probs)
        with torch.no_grad():
            # 注意：此处 agent 内部应实现 [Team, Type, Hero] 的交错拼接逻辑
            old_logits, values = self.agent(hero_seq, team_seq, type_seq)
            old_dist = Categorical(logits=old_logits)
            # 这里的 hero_seq 实际上就是 Actions
            old_log_probs = old_dist.log_prob(hero_seq) # [B, 24]

        # 2. 计算奖励序列 (只有最后一步有值)
        B = hero_seq.size(0)
        rewards = torch.zeros((B, 24), device=device)
        for i in range(B):
            rewards[i, -1] = self.compute_reward(r_picks[i], d_picks[i])

        # 3. 计算 GAE (Generalized Advantage Estimation)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        with torch.no_grad():
            for t in reversed(range(24)):
                next_value = values[:, t+1] if t < 23 else 0
                # TD Error: delta = r + gamma * V(s') - V(s)
                delta = rewards[:, t] + self.gamma * next_value - values[:, t]
                # GAE
                advantages[:, t] = last_gae_lam = delta + self.gamma * self.lam * last_gae_lam
        
        # TD Target = Advantage + Value
        td_targets = advantages + values

        # 4. PPO 核心更新循环
        for epoch in range(4): 
            # 重新获取当前策略的预测
            curr_logits, curr_values = self.agent(hero_seq, team_seq, type_seq)
            curr_dist = Categorical(logits=curr_logits)
            curr_log_probs = curr_dist.log_prob(hero_seq)
            entropy = curr_dist.entropy()

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
            loss = actor_loss + 0.5 * critic_loss - self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪 (针对 Transformer)
            nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            self.optimizer.step()

        return loss.item(), critic_loss.item()

if __name__ == "__main__":
    print('='*20 + ' 更新比赛数据 ' + '='*20)
    fetch_high_mmr_matches(
        output_file='./data/high_mmr_cm_matches.json',
        target_count=100000,
        min_rank=50,
        min_duration=18 * 60,
    )

    print('='*20 + ' 训练 PPODecoderAgent ' + '='*20)
    # 1. 实例化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    oracle = model = WinRateOracle(embed_dim=32, nhead=4, num_layers=4, use_text=False).to(device)
    oracle.load_state_dict(torch.load("./ckpts/win_rate_oracle/win_rate_oracle-20251224124558-126-0.8393.pth"))
    agent = PPODecoderAgent().to(device)

    # 2. 准备数据
    with open("./data/high_mmr_cm_matches.json", "r", encoding='utf-8') as f:
        raw_data = json.load(f)
    dataset = BPDataset(raw_data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. 训练
    trainer = BP_PPOTrainer(agent, oracle)
    for epoch in range(100):
        for batch in loader:
            loss, v_loss = trainer.train_step(batch)
            print(f"Loss: {loss:.4f}, Value Loss (TD Error): {v_loss:.4f}")