"""BP Agent Training Script"""

import os
import subprocess
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import numpy as np

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from model.win_rate_oracle import WinRateOracle
from utils.bp_env import compute_gae, ppo_loss, collect_rollout, normalize_advantages, compute_value_loss
from utils.device import DEVICE
from utils.raw_data import NUM_HEROES, get_valid_hero_ids
from utils.player_preference_sampler_optimized import sample_player_preferences_batch, _load_hero_data
from eval import EvalMethod, get_evaluator, RatingEvaluatorBase


def compute_entropy(logits, mask=None):
    """
    计算策略的熵（entropy）
    
    Args:
        logits: 原始logits [num_actions]
        mask: 可选的mask，已使用的英雄为-inf
    
    Returns:
        entropy: 策略熵（标量）
    """
    if mask is not None:
        logits = logits + mask
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    # Entropy = -sum(p * log(p))
    entropy = -(probs * log_probs).sum()
    return entropy


def compute_kl_divergence(new_log_probs, old_log_probs):
    """
    计算KL散度（近似值）
    
    Args:
        new_log_probs: 新策略的log概率
        old_log_probs: 旧策略的log概率
    
    Returns:
        KL散度估计值
    """
    # 使用 (old_log_prob - new_log_prob) 作为 KL 的近似
    # 这是 PPO 中常用的近似方法
    ratio = torch.exp(new_log_probs - old_log_probs)
    kl = (ratio - 1) - (new_log_probs - old_log_probs)
    return kl.mean().item()


def start_tensorboard(log_dir="./runs/bp_agent"):
    """
    启动 TensorBoard 进程
    
    Args:
        log_dir: TensorBoard 日志目录
    
    Returns:
        subprocess.Popen: TensorBoard 进程
    """
    try:
        # 检查 tensorboard 是否已安装
        import tensorboard
    except ImportError:
        print("[!] TensorBoard not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
    
    # 启动 TensorBoard
    cmd = [
        "tensorboard",
        "--logdir", log_dir
    ]
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )
    
    print(f"[+] TensorBoard started at http://localhost:6006")
    print(f"[+] Log directory: {log_dir}")
    
    return process

ACTOR_LR = 3e-4
VALUE_LR = 6e-4  # Value network学习率是actor的2倍，加速收敛

# Loss 系数配置 (以 actor 为基准 1.0)
VALUE_LOSS_COEFF = 2.0    # value_loss 是 actor 的 2 倍
ENTROPY_LOSS_COEFF = 0.03  # entropy_loss 是 actor 的 0.03 倍

# TensorBoard 配置（将在 train() 中动态生成带时间戳的目录）
TENSORBOARD_LOG_PREFIX = "bp_agent_exp_"

# 评估配置字典
RATING_CONFIG = {
    # 通用配置
    "method": "trueskill",  # 可选: "elo" 或 "trueskill"
    "eval_interval": 2,      # 每 N 个 epoch 评估一次
    "num_opponents": 8,      # 每次评估对战对手数量
    "num_player_sets": 16,    # 每个对手对战的玩家 set 数量
    
    # ELO 专用配置
    "elo": {
        "k_factor": 32,          # ELO K-factor
        "initial_rating": 1500,  # 初始 ELO 分数
        "scale": 400,            # ELO 比例因子
        "opponent_sample_std": 200,  # 对手选择时的正态分布标准差
    },
    
    # TrueSkill 专用配置
    "trueskill": {
        "initial_mu": 25.0,      # 初始平均技能值
        "initial_sigma": 8.33,   # 初始标准差 (25/3)
        "beta": 4.17,            # 性能标准差 (sigma/2)
        "tau": 0.083,            # 动态因子 (sigma/100)
        "draw_probability": 0.0, # 平局概率
        "opponent_sample_std": 2.0,  # TrueSkill 尺度下的采样标准差
        # Staleness 相关配置（防止评分漂移）
        "staleness_threshold": 5,    # 超过此阈值的模型会被强制刷新
        "num_active_models": 5,      # 刷新时选择的活跃模型数量
    },
}


def generate_samples(num_samples):
    """
    直接生成训练样本
    
    Args:
        num_samples: 样本数量
    Returns:
        list: 样本列表，每个样本包含 r_players 和 d_players
    """
    # 预加载缓存
    _load_hero_data()
    
    # 一次性生成所有玩家（两队各5人）
    total_players = num_samples * 10
    all_players = sample_player_preferences_batch(
        num_players=total_players,
        m=3,
        n=5,
        use_parallel=num_samples > 20
    )
    
    samples = []
    for i in range(num_samples):
        start_idx = i * 10
        r_players = all_players[start_idx:start_idx + 5]
        d_players = all_players[start_idx + 5:start_idx + 10]
        
        sample = {
            'r_players': _player_prefs_to_feats(r_players),
            'd_players': _player_prefs_to_feats(d_players),
        }
        samples.append(sample)
    
    return samples


def _player_prefs_to_feats(player_prefs):
    """将玩家偏好转换为特征向量 [5, NUM_HEROES]"""
    feats = []
    for p in player_prefs:
        vec = [0.0] * NUM_HEROES
        for h in p['heroes']:
            hero_id = h['id']
            win_rate = h['win_rate']
            if 0 < hero_id <= NUM_HEROES:
                vec[hero_id - 1] = win_rate
        feats.append(vec)
    while len(feats) < 5:
        feats.append([0.0] * NUM_HEROES)
    return feats


def train(epochs=32, batch_size=16, rollout_steps=5, samples_per_epoch=1024, 
         rating_config=None, use_tensorboard=True, log_dir=None):
    """
    训练 BP Agent
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        rollout_steps: 每个样本的 rollout 步数
        samples_per_epoch: 每个epoch生成的样本数
        rating_config: 评估配置字典，如果为 None 则使用全局 RATING_CONFIG
        use_tensorboard: 是否使用 TensorBoard
        log_dir: TensorBoard 日志目录
    """
    # 启动 TensorBoard
    tb_process = None
    if use_tensorboard:
        if log_dir is None:
            # 使用带时间戳的目录名，格式与 win_rate_oracle 一致
            log_dir = os.path.join("runs", TENSORBOARD_LOG_PREFIX + datetime.now().strftime("%Y%m%d-%H%M%S"))
        tb_process = start_tensorboard(log_dir)
        # 初始化 TensorBoard writer
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[+] TensorBoard writer initialized")
    else:
        writer = None
    # 使用传入的配置或默认配置
    config = rating_config or RATING_CONFIG
    method = config.get("method", "elo")
    
    if method.lower() == "elo":
        eval_method = EvalMethod.ELO
        method_name = "ELO"
    elif method.lower() == "trueskill":
        eval_method = EvalMethod.TRUESKILL
        method_name = "TrueSkill"
    else:
        raise ValueError(f"Unknown rating method: {method}. Use 'elo' or 'trueskill'")
    
    print(f"[+] Using {method_name} rating system for evaluation")
    
    # 创建带时间戳的保存目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"./ckpts/bp_agent-{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[+] Models will be saved to: {save_dir}")
    
    # 全局步数计数器（用于 TensorBoard）
    global_step = 0
    
    # Load oracle
    oracle = WinRateOracle(embed_dim=128, nhead=8, num_layers=6, use_text=True, use_player_heroes=True).to(DEVICE)
    oracle_path = "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth"
    if os.path.exists(oracle_path):
        oracle.load_state_dict(torch.load(oracle_path, map_location=DEVICE))
        print(f"[+] Loaded oracle from {oracle_path}")
    oracle.eval()

    # Agent
    agent = BPTransformerAgent(embed_dim=EMBED_DIM, nhead=8, num_layers=4).to(DEVICE)
    
    # 使用不同的学习率: value_lr = 2 * actor_lr
    # 将参数分为policy和value两组
    policy_params = []
    value_params = []
    for name, param in agent.named_parameters():
        if 'value_head' in name:
            value_params.append(param)
        else:
            policy_params.append(param)
    
    optimizer = AdamW([
        {'params': policy_params, 'lr': ACTOR_LR},
        {'params': value_params, 'lr': VALUE_LR}
    ])

    # 评分评估器（ELO 或 TrueSkill）
    # 从配置中提取参数
    eval_kwargs = {
        "save_dir": save_dir,
        "oracle": oracle,
        "num_opponents": config.get("num_opponents", 5),
        "num_player_sets": config.get("num_player_sets", 8),
    }
    
    # 添加方法专用参数
    if method.lower() == "elo":
        elo_cfg = config.get("elo", {})
        eval_kwargs.update({
            "k_factor": elo_cfg.get("k_factor", 32),
            "opponent_sample_std": elo_cfg.get("opponent_sample_std", 200),
        })
    elif method.lower() == "trueskill":
        ts_cfg = config.get("trueskill", {})
        # TrueSkillEvaluator 只接受 staleness_threshold 和 num_active_models
        eval_kwargs.update({
            "staleness_threshold": ts_cfg.get("staleness_threshold", 5),
            "num_active_models": ts_cfg.get("num_active_models", 5),
        })
    
    rating_evaluator: RatingEvaluatorBase = get_evaluator(eval_method, **eval_kwargs)
    
    # Training loop
    for epoch in range(epochs):
        agent.train()
        total_loss = 0
        
        # 每个epoch重新生成数据
        samples = generate_samples(samples_per_epoch)
        
        # 按batch_size切分样本
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", ncols=90)
        for batch_idx in pbar:
            # 获取当前batch的样本
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]
            # 收集rollout（限制为rollout_steps个样本）
            current_batch_size = len(batch_samples)
            rollouts = [collect_rollout(agent, oracle, s) for s in batch_samples[:min(current_batch_size, rollout_steps)]]

            batch_actor_loss = 0
            batch_value_loss = 0
            batch_entropy_loss = 0
            batch_kl = 0
            
            for rollout in rollouts:
                actions = rollout['actions'].to(DEVICE)
                old_log_probs = rollout['log_probs'].to(DEVICE)
                values = rollout['values'].to(DEVICE)
                rewards = rollout['rewards'].to(DEVICE)

                T = len(rewards)
                dones = torch.zeros(T, device=DEVICE)
                advantages, returns = compute_gae(
                    rewards.unsqueeze(-1), 
                    values.unsqueeze(-1), 
                    dones.unsqueeze(-1),
                    normalize_returns=True  # 启用return归一化
                )
                advantages = advantages.squeeze(-1)
                returns = returns.squeeze(-1)
                
                # Advantage归一化 - 关键trick！
                advantages = normalize_advantages(advantages)

                new_log_probs_list, new_values = [], []
                # 预加载有效英雄ID集合
                valid_hero_ids = get_valid_hero_ids()
                for i, state in enumerate(rollout['states']):
                    logits, v = agent(state)
                    heroes = state['action_history']['heroes']
                    used = set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
                    # 创建mask：只允许选择实际存在且未被使用的英雄
                    mask = torch.full((NUM_HEROES,), -1e9, device=logits.device)
                    # 先标记所有有效英雄为可选
                    for h in valid_hero_ids:
                        if h <= NUM_HEROES:
                            mask[h - 1] = 0.0
                    # 再屏蔽已使用的英雄
                    for h in used:
                        if h < NUM_HEROES:  # used中的英雄是0-indexed
                            mask[h] = -1e9
                    logits = logits + mask
                    probs = torch.softmax(logits, dim=-1)
                    new_log_probs_list.append(torch.distributions.Categorical(probs).log_prob(actions[i]))
                    new_values.append(v.squeeze(-1))  # squeeze [1,1] -> [1]

                new_log_probs = torch.stack(new_log_probs_list)
                new_values = torch.cat(new_values)

                policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages)
                
                # PPO value loss with clipping
                # new_values: [T] (每个state重新计算的value)
                # values[:-1]: [T] (old values，去掉最后的bootstrap)
                # returns: [T] (GAE计算的returns)
                value_loss = compute_value_loss(
                    new_values,           # 新的value预测 [T]
                    values[:-1],          # 旧的value预测（用于clipping）[T]
                    returns,
                    clip_eps=0.2,
                    use_clipping=True  # 启用value clipping
                )
                
                # 计算 KL 散度
                kl_div = compute_kl_divergence(new_log_probs, old_log_probs)
                
                # 计算 entropy loss（鼓励探索，避免策略过尖）
                entropy_loss = 0
                for i, state in enumerate(rollout['states']):
                    logits, _ = agent(state)
                    heroes = state['action_history']['heroes']
                    used = set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
                    # 创建mask：只允许选择实际存在且未被使用的英雄
                    mask = torch.full((NUM_HEROES,), -1e9, device=logits.device)
                    for h in valid_hero_ids:
                        if h <= NUM_HEROES:
                            mask[h - 1] = 0.0
                    for h in used:
                        if h < NUM_HEROES:
                            mask[h] = -1e9
                    entropy = compute_entropy(logits, mask)
                    entropy_loss -= entropy  # 最大化熵 = 最小化负熵
                entropy_loss = entropy_loss / len(rollout['states'])  # 平均
                
                # 组合loss
                # actor loss 系数为 1.0（基准）
                # value loss 系数为 VALUE_LOSS_COEFF，但已通过不同学习率实现，故保持 1.0
                # entropy loss 系数为 ENTROPY_LOSS_COEFF
                loss = policy_loss + VALUE_LOSS_COEFF * value_loss + ENTROPY_LOSS_COEFF * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch_actor_loss += policy_loss.item()
                batch_value_loss += value_loss.item()
                batch_entropy_loss += entropy_loss.item()
                batch_kl += kl_div
                global_step += 1
            
            # 记录到 TensorBoard
            if writer is not None:
                writer.add_scalar("Loss/actor", batch_actor_loss / len(rollouts), global_step)
                writer.add_scalar("Loss/value", batch_value_loss / len(rollouts), global_step)
                writer.add_scalar("Loss/entropy", batch_entropy_loss / len(rollouts), global_step)
                writer.add_scalar("Loss/total", total_loss / ((batch_idx + 1) * rollout_steps), global_step)
                writer.add_scalar("Loss/kl_divergence", batch_kl / len(rollouts), global_step)
                writer.flush()  # 确保数据立即写入磁盘

            pbar.set_postfix({"Loss": f"{total_loss / ((batch_idx + 1) * rollout_steps):.4f}"})
        
        # 定期保存中间模型并进行评分评估
        eval_interval = config.get("eval_interval", 8)
        if (epoch + 1) % eval_interval == 0:
            checkpoint_path = f"{save_dir}/bp_agent_epoch{epoch+1}.pth"
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"\n[+] Checkpoint saved: {checkpoint_path}")
            
            num_opponents = config.get("num_opponents", 5)
            num_player_sets = config.get("num_player_sets", 8)
            
            print(f"[+] {method_name} evaluation at epoch {epoch+1}...")
            rating_evaluator.evaluate(
                model_path=checkpoint_path,
                num_opponents=num_opponents,
                num_player_sets=num_player_sets
            )
            
            # 打印当前排行榜
            rating_evaluator.print_leaderboard()
            
            # 记录模型评分到 TensorBoard
            if writer is not None:
                rating = rating_evaluator.get_rating(checkpoint_path)
                writer.add_scalar(f"Rating/{method_name.lower()}", rating, epoch + 1)
                writer.flush()  # 确保数据立即写入

    # Save final model
    model_path = f"{save_dir}/bp_agent_final.pth"
    torch.save(agent.state_dict(), model_path)
    print(f"[+] Model saved to {model_path}")
    
    # 训练结束后的最终评分评估
    num_opponents = config.get("num_opponents", 5)
    num_player_sets = config.get("num_player_sets", 8)
    
    print(f"[+] Final {method_name} evaluation...")
    rating_evaluator.evaluate(
        model_path=model_path,
        num_opponents=num_opponents,
        num_player_sets=num_player_sets
    )
    
    # 记录最终模型评分到 TensorBoard
    if writer is not None:
        final_rating = rating_evaluator.get_rating(model_path)
        writer.add_scalar(f"Rating/{method_name.lower()}", final_rating, epochs)
        writer.flush()
    
    # 打印排行榜
    rating_evaluator.print_leaderboard()
    
    # 关闭 TensorBoard
    if writer is not None:
        writer.close()
        print("[+] TensorBoard writer closed")
    
    if tb_process is not None:
        print("[+] TensorBoard process is running in background")
        print(f"[+] You can view logs at http://localhost:6006")
    
    print("[+] Training done!")


if __name__ == "__main__":
    # 使用默认配置 (ELO)
    train(epochs=128, batch_size=128, rollout_steps=10, samples_per_epoch=2048, use_tensorboard=True)
    
    # 使用 TrueSkill 的示例:
    # ts_config = {
    #     "method": "trueskill",
    #     "eval_interval": 8,
    #     "num_opponents": 5,
    #     "num_player_sets": 8,
    #     "trueskill": {
    #         "initial_mu": 25.0,
    #         "initial_sigma": 8.33,
    #     }
    # }
    # train(epochs=32, batch_size=32, rollout_steps=10, rating_config=ts_config)
