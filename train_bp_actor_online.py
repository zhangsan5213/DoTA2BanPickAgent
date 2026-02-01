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

from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP, STATIC_HERO_MASK
from utils.get_data_cm_bp import fetch_high_mmr_matches
from model.hero_encoder import NUM_HEROES
from model.win_rate_oracle import WinRateOracle
from model.bp_policy import PPODecoderAgent

torch.random.manual_seed(42)

BP_ACTOR_SAVE_DIR = "./ckpts/bp_actor"
WIN_RATE_ORACLE_SAVE_DIR = "./ckpts/win_rate_oracle"
if not os.path.exists(BP_ACTOR_SAVE_DIR):
    pathlib.Path(BP_ACTOR_SAVE_DIR).mkdir(parents=True, exist_ok=True)


def find_latest_oracle_model(save_dir: str = WIN_RATE_ORACLE_SAVE_DIR):
    """从保存目录中找到最新的 WinRateOracle 模型"""
    pattern = os.path.join(save_dir, "win_rate_oracle-*.pth")
    model_files = glob.glob(pattern)
    if not model_files:
        return None
    # 按修改时间排序，取最新的
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

# 标准 CM 模式序列 (Radiant=1, Dire=2; Ban=1, Pick=2)
# 简化版示例（24步）：Ban(RDRDRDRD) -> Pick(RDDR) -> Ban(RDRDRD) -> Pick(DRDR) -> Ban(DR)
CM_TEAM_SEQ = torch.tensor([1,2,1,2,1,2,1,2, 1,2,2,1, 1,2,1,2,1,2, 2,1,2,1, 2,1], dtype=torch.long)
CM_TYPE_SEQ = torch.tensor([1,1,1,1,1,1,1,1, 2,2,2,2, 1,1,1,1,1,1, 2,2,2,2, 1,1], dtype=torch.long)

def get_combined_mask(hero_seq_so_far, device):
    """
    hero_seq_so_far: [batch_size, current_step] 已经选过的 hero_id
    """
    batch_size = hero_seq_so_far.shape[0]
    
    # 1. 复制静态掩码到每个 batch [B, MAX_HERO_ID]
    mask = STATIC_HERO_MASK.to(device).clone().repeat(batch_size, 1)
    
    # 2. 动态屏蔽：将本局已经出现过的 hero_id 设为 -1e9
    # 使用 scatter_ 快速填充
    if hero_seq_so_far.shape[1] > 0:
        # 排除 hero_id 为 0 的情况（如果有填充）
        # 我们只操作大于 0 的 ID
        # mask.scatter_(dim, index, value)
        ones = torch.full(hero_seq_so_far.shape, -1e9).to(device)
        mask.scatter_(1, hero_seq_so_far, ones)
        
    return mask

class BP_PPOTrainer_Online:
    def __init__(self, agent: PPODecoderAgent, oracle: WinRateOracle, writer: SummaryWriter, 
                 lr=1e-4, gamma=0.99, weight_decay=0.01, gamma_pretrain=0.95):
        self.agent = agent
        self.oracle = oracle
        # 使用 AdamW 优化器，与 train_winrate_oracle.py 保持一致
        self.optimizer = torch.optim.AdamW(agent.parameters(), lr=lr, weight_decay=weight_decay)
        self.writer = writer
        self.gamma = gamma
        self.gamma_pretrain = gamma_pretrain  # Critic 预训练用的 gamma
        self.eps_clip = 0.2
        self.ent_coef = 0.02  # 在线学习需要更强的探索
        self.global_step = 0
        
    def pretrain_critic_step(self, batch_size=128):
        """
        在线收集轨迹并预训练 Critic（类似 offline 的 train_step_critic_pretrain）
        """
        device = next(self.agent.parameters()).device
        self.agent.eval()
        
        team_seq = CM_TEAM_SEQ.repeat(batch_size, 1).to(device)
        type_seq = CM_TYPE_SEQ.repeat(batch_size, 1).to(device)
        
        obs_hero_seq = torch.zeros(batch_size, 24, dtype=torch.long).to(device)
        
        # 1. 在线收集轨迹（仅用于生成最终阵容，不涉及梯度）
        with torch.no_grad():
            for t in range(24):
                logits, _ = self.agent(obs_hero_seq, team_seq, type_seq)
                step_logits = logits[:, t, :]
                
                # 应用掩码
                current_mask = get_combined_mask(obs_hero_seq[:, :t], device)
                masked_logits = step_logits + current_mask
                
                # 采样
                dist = Categorical(logits=masked_logits)
                action = dist.sample()
                obs_hero_seq[:, t] = action
        
        # 2. 提取 pick 并计算最终奖励
        r_picks, d_picks = self.extract_picks(obs_hero_seq, team_seq, type_seq)
        with torch.no_grad():
            win_prob = self.oracle.predict(r_picks, d_picks, return_tensor=True)
            final_rewards = (win_prob.squeeze(-1) - 0.5) * 2  # [B], 范围 [-1, 1]
        
        # 3. 计算 Critic 当前预测值
        self.agent.train()
        _, curr_values = self.agent(obs_hero_seq, team_seq, type_seq)  # [B, 24]
        
        # 4. 构造指数衰减目标 (与 offline 一致)
        T = 24
        steps = torch.arange(T, device=device).float()  # [0, 1, ..., 23]
        weights = self.gamma_pretrain ** (T - 1 - steps)  # [24]
        targets = final_rewards.unsqueeze(1) * weights.unsqueeze(0)  # [B, 24]
        
        # 5. 计算 MSE Loss
        loss = F.mse_loss(curr_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()
        
        # 记录到 TensorBoard
        self.writer.add_scalar('Pretrain/Critic Loss', loss.item(), self.global_step)
        self.writer.add_scalar('Pretrain/Final Reward Avg', final_rewards.mean().item(), self.global_step)
        self.global_step += 1
        
        return loss.item()

    def collect_trajectories(self, batch_size=64):
        device = next(self.agent.parameters()).device
        self.agent.eval()
        
        team_seq = CM_TEAM_SEQ.repeat(batch_size, 1).to(device)
        type_seq = CM_TYPE_SEQ.repeat(batch_size, 1).to(device)
        
        obs_hero_seq = torch.zeros(batch_size, 24, dtype=torch.long).to(device)
        log_probs = []
        values = []
        
        with torch.no_grad():
            for t in range(24):
                # 获取当前步的 logits
                logits, v = self.agent(obs_hero_seq, team_seq, type_seq)
                step_logits = logits[:, t, :] # [B, MAX_HERO_ID]
                
                # --- 关键修改：应用混合掩码 ---
                # 传入之前 t 步已经选好的英雄
                current_mask = get_combined_mask(obs_hero_seq[:, :t], device)
                # 叠加掩码：合法且未选的位置加 0，非法位置加 -1e9
                masked_logits = step_logits + current_mask
                
                # 采样
                dist = Categorical(logits=masked_logits)
                action = dist.sample() # 此时采样出的 ID 必然在 VALID_HERO_IDS 中
                
                obs_hero_seq[:, t] = action
                log_probs.append(dist.log_prob(action))
                values.append(v[:, t])
                
                # 更新序列供下一步使用
                obs_hero_seq[:, t] = action
        
        # 计算最终奖励
        # 提取 pick 的英雄用于 Oracle
        r_picks, d_picks = self.extract_picks(obs_hero_seq, team_seq, type_seq)
        with torch.no_grad():
            # 使用 return_tensor=True 获取 tensor 形式的预测结果
            win_prob = self.oracle.predict(r_picks, d_picks, return_tensor=True)
            final_rewards = (win_prob.squeeze(-1) - 0.5) * 2  # [B]
            
        return {
            "hero_seq": obs_hero_seq,
            "team_seq": team_seq,
            "type_seq": type_seq,
            "old_log_probs": torch.stack(log_probs, dim=1), # [B, 24]
            "old_values": torch.stack(values, dim=1),       # [B, 24]
            "rewards": final_rewards,                        # [B]
        }

    def extract_picks(self, hero_seq, team_seq, type_seq):
        # 辅助函数：从完整序列中提取 5v5 英雄 ID
        batch_size = hero_seq.shape[0]
        r_picks = []
        d_picks = []
        for b in range(batch_size):
            # team=1 & type=2 是 Radiant Pick
            rp = hero_seq[b][(team_seq[b] == 1) & (type_seq[b] == 2)]
            # team=2 & type=2 是 Dire Pick
            dp = hero_seq[b][(team_seq[b] == 2) & (type_seq[b] == 2)]
            r_picks.append(rp[:5]) # 确保固定长度 5
            d_picks.append(dp[:5])
        return torch.stack(r_picks), torch.stack(d_picks)

    def train_online_step(self, batch_size=128, ppo_epochs=4):
        # 1. 在线采样
        data = self.collect_trajectories(batch_size)
        hero_seq = data['hero_seq']
        team_seq = data['team_seq']
        type_seq = data['type_seq']
        old_log_probs = data['old_log_probs']
        old_values = data['old_values']
        final_rewards = data['rewards']

        # 2. 计算 GAE (同你之前的逻辑)
        # 这里简化处理：因为只有最后一步有奖励，直接计算 TD Target
        advantages = torch.zeros_like(old_values)
        last_gae = 0
        for t in reversed(range(24)):
            reward = final_rewards if t == 23 else 0
            v_next = 0 if t == 23 else old_values[:, t+1]
            delta = reward + self.gamma * v_next - old_values[:, t]
            advantages[:, t] = last_gae = delta + self.gamma * 0.95 * last_gae
        
        td_targets = advantages + old_values

        # 3. PPO 更新
        self.agent.train()
        device = hero_seq.device
        for _ in range(ppo_epochs):
            curr_logits, curr_values = self.agent(hero_seq, team_seq, type_seq)
            
            # 重新计算 log_prob (必须使用与采样时一致的 Mask)
            new_log_probs = []
            entropies = []
            for t in range(24):
                mask_t = get_combined_mask(hero_seq[:, :t], device)
                dist_t = Categorical(logits=curr_logits[:, t, :] + mask_t)
                new_log_probs.append(dist_t.log_prob(hero_seq[:, t]))
                entropies.append(dist_t.entropy())
            
            curr_log_probs = torch.stack(new_log_probs, dim=1)
            entropy = torch.stack(entropies, dim=1).mean()
            
            ratio = torch.exp(curr_log_probs - old_log_probs)
            
            # 方向性调整 (Radiant 视角)
            perspective_mask = torch.where(team_seq == 1, 1.0, -1.0)
            adj_adv = advantages * perspective_mask
            
            surr1 = ratio * adj_adv
            surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * adj_adv
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(curr_values, td_targets.detach())
            entropy_loss = -entropy  # 最大化熵 -> 最小化负熵
            
            total_loss = actor_loss + 0.5 * critic_loss + self.ent_coef * entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
            self.optimizer.step()

        # Log - 增强 TensorBoard 记录，与离线训练一致
        self.writer.add_scalar('Train/Actor Loss', actor_loss.item(), self.global_step)
        self.writer.add_scalar('Train/Critic Loss', critic_loss.item(), self.global_step)
        self.writer.add_scalar('Train/Total Loss', total_loss.item(), self.global_step)
        self.writer.add_scalar('Train/Entropy', entropy.item(), self.global_step)
        self.writer.add_scalar('Online/Final_Win_Prob_Avg', (final_rewards.mean()+1)/2, self.global_step)
        self.global_step += 1
        return actor_loss.item(), critic_loss.item()

if __name__ == "__main__":
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
    
    oracle.load_state_dict(torch.load(oracle_model_path, map_location=device))
    oracle.eval()  # 设置为评估模式
    print("[*] Oracle loaded successfully")
    
    # 创建 PPO Agent
    agent = PPODecoderAgent(embed_dim=32, nhead=4, num_layers=4, dim_feedforward=64).to(device)
    print(f"[*] Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    log_dir = os.path.join("runs", "bp_ppo_agent_online_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[*] TensorBoard logs: {log_dir}")
    
    # 创建训练器，使用 AdamW 和 weight_decay
    trainer = BP_PPOTrainer_Online(agent, oracle, writer, lr=1e-4, weight_decay=0.01)
    
    # ========== Stage 1: Critic 预训练 ==========
    print(">>> Stage 1: Pre-training Critic with Online Trajectories...")
    pretrain_episodes = 200  # 预训练步数
    pbar_pretrain = tqdm(range(pretrain_episodes), desc="Critic Pretrain")
    for _ in pbar_pretrain:
        critic_loss = trainer.pretrain_critic_step(batch_size=128)
        pbar_pretrain.set_postfix({"Critic loss": f"{critic_loss:.4f}"})
    
    # 重置 global_step 用于正式训练记录
    trainer.global_step = 0
    
    # ========== Stage 2: 正式在线 PPO 训练 ==========
    print(">>> Stage 2: Online Self-Play Training...")
    num_episodes = 10000 
    best_loss = float('inf')
    
    pbar = tqdm(range(num_episodes), desc="Online Training")
    for epoch in pbar:
        a_loss, c_loss = trainer.train_online_step(batch_size=128)

        if epoch % 10 == 0:
            pbar.set_postfix({"Actor": f"{a_loss:.3f}", "Critic": f"{c_loss:.3f}"})
        
        # 保存最佳模型（基于总损失）
        if epoch % 100 == 0:
            total_loss = a_loss + c_loss
            if total_loss < best_loss:
                best_loss = total_loss
                datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
                save_path = os.path.join(BP_ACTOR_SAVE_DIR, f"bp_actor_online-{datetime_str}-{epoch:05d}-{total_loss:.4f}.pth")
                torch.save(agent.state_dict(), save_path)
                print(f"[*] Saved best model to: {save_path}")
            # 同时保存一个 latest 版本
            torch.save(agent.state_dict(), os.path.join(BP_ACTOR_SAVE_DIR, "online_agent_latest.pth"))
    
    writer.close()
    print(">>> Training completed!")