"""
BP Agent PPO Training Script (Optimized Version)

针对训练不稳定问题的优化版本：
1. 降低更新频率，增加每次更新的样本量
2. 降低学习率，使用学习率衰减
3. 增加熵正则化系数，促进探索
4. 增加模型容量
5. 添加 KL 散度约束和 Early Stopping
6. 专注 ELO 评估（零和博弈不记录 reward）
"""


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")


import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

from model.bp_agent import BPAgent
from model.win_rate_oracle import WinRateOracle
from env.bp_env import BPEnvironment
from data.dataset import MatchDataset
from training.ppo_v2 import PPOTrainerV2
from training.collector import TrajectoryCollector
from training.rewards import create_oracle_reward_fn
from utils.player_preference_sampler import get_player_sampler
from utils.elo_rating import *
from utils.elo_rating_parallel import (
    evaluate_checkpoints_elo_parallel,
    evaluate_single_checkpoint_elo_parallel
)


class Config:
    # 路径配置
    ORACLE_CKPT_DIR = "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128"
    DATA_FILE = "./data/high_mmr_with_stats-rank_40-duration_15.json"
    SAVE_DIR = "./ckpts/bp_agent"
    LOG_DIR = "./runs/"
    ELO_JSON_PATH = "./ckpts/bp_agent/elo_ratings.json"

    # 模型配置（增大容量）
    EMBED_DIM = 128  # 从 64 增加到 128
    NHEAD = 8        # 从 4 增加到 8
    NUM_LAYERS = 6   # 从 4 增加到 6
    USE_TEXT = True
    USE_PLAYER_HEROES = True

    # PPO 配置（更保守的更新）
    PPO_EPOCHS = 6     # 从 4 增加到 6，每次更新更充分
    CLIP_RATIO = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.05  # 从 0.02 增加到 0.05，强制更多探索
    GAMMA = 0.99
    LAMBDA = 0.95
    LR = 2e-4          # 提高到 2e-4，加速探索
    LR_DECAY = 0.9995  # 减缓学习率衰减，保持长期学习能力
    GRAD_CLIP = 0.5
    MAX_KL = 0.05      # 从 0.02 提高到 0.05，允许更大更新

    # 训练配置（降低更新频率）
    BATCH_SIZE = 256   # 从 64 增加到 256
    NUM_ENVS = 32      # 从 16 增加到 32，增加并行度
    MAX_EPISODES = 64000
    UPDATE_INTERVAL = 64  # 从 16 增加到 64，降低更新频率
    SAVE_INTERVAL = 4     # 相应调整保存间隔

    # ELO 配置
    ELO_N_OPPONENTS = 16
    ELO_N_GAMES = 8
    ELO_N_WORKERS = 8


def get_latest_oracle_ckpt(oracle_dir: str) -> str:
    """获取最新的Oracle checkpoint"""
    import re
    ckpt_files = []
    for f in os.listdir(oracle_dir):
        if f.startswith("win_rate_oracle-") and f.endswith(".pth"):
            match = re.search(r'win_rate_oracle-(\d{14})-', f)
            if match:
                ckpt_files.append((match.group(1), os.path.join(oracle_dir, f)))
    
    if not ckpt_files:
        raise FileNotFoundError(f"未找到Oracle checkpoint")
    
    ckpt_files.sort(key=lambda x: x[0], reverse=True)
    return ckpt_files[0][1]


def load_oracle_and_copy_encoder(config, device):
    """加载Oracle并复制encoder权重到Agent"""
    oracle = WinRateOracle(
        embed_dim=config.EMBED_DIM, nhead=config.NHEAD, num_layers=config.NUM_LAYERS,
        use_text=config.USE_TEXT, use_player_heroes=config.USE_PLAYER_HEROES,
    ).to(device)
    
    ckpt_path = get_latest_oracle_ckpt(config.ORACLE_CKPT_DIR)
    print(f"[*] 加载Oracle: {os.path.basename(ckpt_path)}")
    oracle.load_state_dict(torch.load(ckpt_path, map_location=device))
    oracle.eval()
    
    return oracle


def init_agent_from_oracle(agent, oracle):
    """从Oracle复制encoder权重"""
    oracle_state = oracle.state_dict()
    agent_state = agent.state_dict()
    
    copied = 0
    for key in oracle_state:
        if key.startswith('hero_encoder.') or key.startswith('player_encoder.'):
            agent_key = 'state_encoder.' + key
            if agent_key in agent_state and agent_state[agent_key].shape == oracle_state[key].shape:
                agent_state[agent_key] = oracle_state[key]
                copied += 1
    
    agent.load_state_dict(agent_state)
    print(f"[*] 从Oracle复制 {copied} 个参数")


def init_from_elo(agent, config, device, dataset, player_sampler, oracle):
    """从ELO记录初始化，选择最强checkpoint"""
    elo_ratings, _ = load_elo_ratings(config.ELO_JSON_PATH)
    new_ckpts = find_new_checkpoints(elo_ratings, config.SAVE_DIR)
    
    # 新ckpt定分
    if new_ckpts:
        print(f"[*] {len(new_ckpts)} 个新ckpt需要ELO定分")
        for ckpt in new_ckpts:
            elo_ratings[ckpt] = 1500.0
        
        # 所有可用ckpt（新ckpt已在elo_ratings中）
        all_ckpts = list(elo_ratings.keys())
        
        # 使用并行版本进行ELO定分
        elo_ratings = evaluate_checkpoints_elo_parallel(
            new_ckpts, all_ckpts, elo_ratings, BPAgent, oracle,
            dataset.matches, player_sampler, device, config,
            n_opponents_per_ckpt=getattr(config, 'ELO_N_OPPONENTS', 5),
            n_games_per_match=getattr(config, 'ELO_N_GAMES', 10),
            n_workers=getattr(config, 'ELO_N_WORKERS', 4)
        )
        
        save_elo_ratings(elo_ratings, config.ELO_JSON_PATH)
        print(f"[*] ELO定分完成，已保存")
    
    # 加载最强
    best_ckpt, best_elo = get_best_checkpoint(elo_ratings) or (None, 1500)
    if best_ckpt and os.path.exists(best_ckpt):
        agent.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"[*] 加载最强ckpt: {os.path.basename(best_ckpt)}, ELO={best_elo:.1f}")
        return best_ckpt
    return None


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 设备: {device}")
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # 加载数据
    dataset = MatchDataset(Config.DATA_FILE)
    player_sampler = get_player_sampler()
    
    # 创建Agent和Oracle
    agent = BPAgent(
        embed_dim=Config.EMBED_DIM, nhead=Config.NHEAD, num_layers=Config.NUM_LAYERS,
        use_text=Config.USE_TEXT, use_player_heroes=Config.USE_PLAYER_HEROES,
    ).to(device)
    
    oracle = load_oracle_and_copy_encoder(Config, device)
    init_agent_from_oracle(agent, oracle)
    init_from_elo(agent, Config, device, dataset, player_sampler, oracle)
    
    # 创建环境
    envs = [BPEnvironment(dataset.matches, Config.USE_PLAYER_HEROES, player_sampler, True)
            for _ in range(Config.NUM_ENVS)]
    
    # 训练器
    optimizer = optim.Adam(agent.parameters(), lr=Config.LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=Config.LR_DECAY)
    ppo_trainer = PPOTrainerV2(agent, optimizer, Config, device)
    collector = TrajectoryCollector(envs, device)
    reward_fn = create_oracle_reward_fn(oracle, device)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, 'bp_agent_exp-' + datetime.now().strftime("%Y%m%d-%H%M%S")))
    
    # 训练循环
    global_step = 0
    saved_checkpoints = []
    elo_ratings, _ = load_elo_ratings(Config.ELO_JSON_PATH)
    
    # 统计信息
    recent_kls = []
    
    for episode in tqdm(range(Config.MAX_EPISODES), desc="Training", ncols=90):
        # 计算温度参数（逐渐降低，后期更稳定）
        temperature = max(0.5, 1.0 - episode / Config.MAX_EPISODES)
        
        # 收集轨迹
        trajectories = collector.collect(agent, reward_fn, temperature=temperature)
        
        # PPO更新
        if (episode + 1) % Config.UPDATE_INTERVAL == 0:
            traj_dicts = [t.to_tensors(device) for t in trajectories]
            loss_dict = ppo_trainer.update(traj_dicts)
            
            writer.add_scalar('Loss/Policy', loss_dict['policy_loss'], global_step)
            writer.add_scalar('Loss/Value', loss_dict['value_loss'], global_step)
            writer.add_scalar('Loss/Entropy', loss_dict['entropy'], global_step)
            
            # 记录 KL 散度
            if 'kl_div' in loss_dict:
                writer.add_scalar('Train/KL_Divergence', loss_dict['kl_div'], global_step)
                recent_kls.append(loss_dict['kl_div'])
            
            # 记录学习率和温度
            writer.add_scalar('Train/Learning_Rate', scheduler.get_last_lr()[0], global_step)
            writer.add_scalar('Train/Temperature', temperature, global_step)
            
            global_step += 1
            scheduler.step()
            
            # 保存checkpoint
            if (episode + 1) % (Config.UPDATE_INTERVAL * Config.SAVE_INTERVAL) == 0:
                ckpt_path = os.path.join(Config.SAVE_DIR, 
                    f"bp_agent-{datetime.now().strftime('%Y%m%d%H%M%S')}-{episode+1}.pth")
                torch.save(agent.state_dict(), ckpt_path)
                saved_checkpoints.append(ckpt_path)
                
                # 初始化新ckpt的ELO
                if len(saved_checkpoints) == 1:
                    elo_ratings[ckpt_path] = 1500.0
                else:
                    prev_ckpt = saved_checkpoints[-2]
                    elo_ratings[ckpt_path] = elo_ratings.get(prev_ckpt, 1500.0)
                
                # ELO定分
                print(f"\n[*] ELO定分...")
                agent.eval()
                
                current_elo, elo_ratings = evaluate_single_checkpoint_elo_parallel(
                    ckpt_path, elo_ratings, BPAgent, oracle,
                    dataset.matches, player_sampler, device, Config,
                    n_opponents=Config.ELO_N_OPPONENTS,
                    n_games=Config.ELO_N_GAMES,
                    n_workers=Config.ELO_N_WORKERS
                )
                
                agent.train()
                
                # 记录和保存
                writer.add_scalar('Elo/Rating', current_elo, global_step)
                avg_elo = sum(elo_ratings.values()) / len(elo_ratings)
                writer.add_scalar('Elo/AverageRating', avg_elo, global_step)
                print(f"[*] ELO: 当前={current_elo:.1f}, 平均={avg_elo:.1f}")
                
                # 记录 KL 趋势
                if recent_kls:
                    writer.add_scalar('Train/KL_Mean', np.mean(recent_kls), global_step)
                    recent_kls = []
                
                save_elo_ratings(elo_ratings, Config.ELO_JSON_PATH)
    
    writer.close()
    print("[*] 训练完成")


if __name__ == "__main__":
    main()
