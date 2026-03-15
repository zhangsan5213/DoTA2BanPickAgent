"""BP Agent Training Script"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datetime import datetime

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from model.win_rate_oracle import WinRateOracle
from utils.bp_env import compute_gae, ppo_loss, collect_rollout, normalize_advantages, compute_value_loss
from utils.device import DEVICE
from utils.bp_dataset import BPDataset
from utils.raw_data import NUM_HEROES
from eval import EvalMethod, get_evaluator, RatingEvaluatorBase

ACTOR_LR = 3e-4
VALUE_LR = 6e-4  # Value network学习率是actor的2倍，加速收敛

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


def train(epochs=32, batch_size=16, rollout_steps=5, rating_config=None):
    """
    训练 BP Agent
    
    Args:
        epochs: 训练轮数
        batch_size: 批次大小
        rollout_steps: 每个样本的 rollout 步数
        rating_config: 评估配置字典，如果为 None 则使用全局 RATING_CONFIG
    """
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
    
    # Load oracle
    oracle = WinRateOracle(embed_dim=128, nhead=8, num_layers=6, use_text=True, use_player_heroes=True).to(DEVICE)
    oracle_path = "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth"
    if os.path.exists(oracle_path):
        oracle.load_state_dict(torch.load(oracle_path, map_location=DEVICE))
        print(f"[+] Loaded oracle from {oracle_path}")
    oracle.eval()

    # Dataset: 优先使用合成数据，可选加载真实数据
    dataset = BPDataset(data_file="", num_synthetic=32000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
        "save_dir": "./ckpts/bp_agent",
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
        eval_kwargs.update({
            "initial_mu": ts_cfg.get("initial_mu", 25.0),
            "initial_sigma": ts_cfg.get("initial_sigma", 8.33),
            "opponent_sample_std": ts_cfg.get("opponent_sample_std", 2.0),
            "staleness_threshold": ts_cfg.get("staleness_threshold", 5),
            "num_active_models": ts_cfg.get("num_active_models", 5),
        })
    
    rating_evaluator: RatingEvaluatorBase = get_evaluator(eval_method, **eval_kwargs)
    
    # Training loop
    for epoch in range(epochs):
        agent.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", ncols=90)
        for batch in pbar:
            # batch is a dict with keys: r_players, d_players, etc.
            # Convert to list of samples for rollout collection
            samples = []
            # Handle case where DataLoader returns list structure
            r_players_batch = batch['r_players']
            d_players_batch = batch['d_players']
            
            # Transpose if needed: from [5][160][batch_size] to [batch_size][5][160]
            if isinstance(r_players_batch, list) and len(r_players_batch) == 5:
                # DataLoader collates list of lists as [outer][inner][batch]
                batch_size = len(r_players_batch[0][0])
                r_players_batch = [[[r_players_batch[j][k][i] for k in range(len(r_players_batch[0]))] for j in range(5)] for i in range(batch_size)]
                d_players_batch = [[[d_players_batch[j][k][i] for k in range(len(d_players_batch[0]))] for j in range(5)] for i in range(batch_size)]
            else:
                batch_size = len(r_players_batch)
            
            for i in range(batch_size):
                sample = {
                    'r_players': r_players_batch[i],
                    'd_players': d_players_batch[i],
                }
                samples.append(sample)

            rollouts = [collect_rollout(agent, oracle, s) for s in samples[:rollout_steps]]

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

                new_log_probs, new_values = [], []
                for i, state in enumerate(rollout['states']):
                    logits, v = agent(state)
                    heroes = state['action_history']['heroes']
                    used = set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
                    mask = torch.full((NUM_HEROES,), -1e9, device=logits.device)
                    for h in range(1, NUM_HEROES + 1):
                        if (h - 1) not in used:
                            mask[h - 1] = 0.0
                    logits = logits + mask
                    probs = torch.softmax(logits, dim=-1)
                    new_log_probs.append(torch.distributions.Categorical(probs).log_prob(actions[i]))
                    new_values.append(v.squeeze(-1))  # squeeze [1,1] -> [1]

                new_log_probs = torch.stack(new_log_probs)
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
                
                # 组合loss (value_loss的系数已经通过不同学习率实现，这里保持1.0)
                loss = policy_loss + value_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            pbar.set_postfix({"Loss": f"{total_loss / (len(pbar) * rollout_steps):.4f}"})
        
        # 定期保存中间模型并进行评分评估
        eval_interval = config.get("eval_interval", 8)
        if (epoch + 1) % eval_interval == 0:
            checkpoint_path = f"./ckpts/bp_agent/bp_agent_{datetime.now().strftime('%Y%m%d%H%M%S')}_epoch{epoch+1}.pth"
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

    # Save
    os.makedirs("./ckpts/bp_agent", exist_ok=True)
    model_path = f"./ckpts/bp_agent/bp_agent_{datetime.now().strftime('%Y%m%d%H%M%S')}.pth"
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
    
    # 打印排行榜
    rating_evaluator.print_leaderboard()
    
    print("[+] Training done!")


if __name__ == "__main__":
    # 使用默认配置 (ELO)
    train(epochs=32, batch_size=32, rollout_steps=10)
    
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
