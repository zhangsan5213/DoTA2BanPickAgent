"""BP Agent Training Script"""

import os
import random
import subprocess
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import numpy as np
import yaml

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from model.win_rate_oracle import WinRateOracle
from utils.bp_env import (
    compute_gae,
    ppo_loss,
    collect_rollout,
    normalize_advantages,
    compute_value_loss,
)
from utils.device import DEVICE
from utils.raw_data import NUM_HEROES, get_valid_hero_ids
from utils.player_preference_sampler_optimized import (
    sample_player_preferences_batch,
    _load_hero_data,
)
from eval import EvalMethod, get_evaluator, RatingEvaluatorBase


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "bp_agent_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def start_tensorboard(log_dir="./runs", port: int = 6006):
    """Start TensorBoard process."""
    cmd = ["tensorboard", "--logdir", log_dir, "--port", str(port)]
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[+] TensorBoard started at http://localhost:{port}")
        print(f"[+] Log directory: {log_dir}")
        return process
    except FileNotFoundError:
        print(
            "[!] TensorBoard not found. Make sure it's installed: pip install tensorboard"
        )
        return None


def discover_checkpoints(checkpoint_dirs: list) -> list:
    """Discover all .pth checkpoint files from given directories.

    Args:
        checkpoint_dirs: List of directory paths to scan.

    Returns:
        List of (checkpoint_path, mtime) tuples sorted by modification time (newest first).
    """
    checkpoints = []
    for d in checkpoint_dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.endswith(".pth"):
                fpath = os.path.join(d, fname)
                mtime = os.path.getmtime(fpath)
                checkpoints.append((fpath, mtime))
    # Newest first
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    return checkpoints


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
        num_players=total_players, m=3, n=5, use_parallel=num_samples > 20
    )

    samples = []
    for i in range(num_samples):
        start_idx = i * 10
        r_players = all_players[start_idx : start_idx + 5]
        d_players = all_players[start_idx + 5 : start_idx + 10]

        sample = {
            "r_players": _player_prefs_to_feats(r_players),
            "d_players": _player_prefs_to_feats(d_players),
        }
        samples.append(sample)

    return samples


def _player_prefs_to_feats(player_prefs):
    """将玩家偏好转换为特征向量 [5, NUM_HEROES]"""
    feats = []
    for p in player_prefs:
        vec = [0.0] * NUM_HEROES
        for h in p["heroes"]:
            hero_id = h["id"]
            win_rate = h["win_rate"]
            if 0 < hero_id <= NUM_HEROES:
                vec[hero_id - 1] = win_rate
        feats.append(vec)
    while len(feats) < 5:
        feats.append([0.0] * NUM_HEROES)
    return feats


def train(config_path: str = None, **override_kwargs):
    """
    训练 BP Agent

    Args:
        config_path: YAML配置文件路径
        **override_kwargs: 可选，覆盖配置中的参数
            epochs, batch_size, samples_per_epoch,
            use_tensorboard, log_dir
    """
    cfg = load_config(config_path)
    rating_cfg = cfg.get("rating", {})
    training_cfg = cfg.get("training", {})

    # Apply overrides
    epochs = override_kwargs.get("epochs", training_cfg.get("epochs", 32))
    batch_size = override_kwargs.get("batch_size", training_cfg.get("batch_size", 16))
    samples_per_epoch = override_kwargs.get(
        "samples_per_epoch", training_cfg.get("samples_per_epoch", 1024)
    )
    use_tensorboard = override_kwargs.get(
        "use_tensorboard", training_cfg.get("use_tensorboard", True)
    )
    log_dir = override_kwargs.get("log_dir", None)
    historical_opponent_prob = training_cfg.get("historical_opponent_prob", 0.6)
    checkpoint_dirs = training_cfg.get("checkpoint_dirs", [])

    actor_lr = float(cfg.get("actor_lr", 3e-4))
    value_loss_coeff = float(cfg.get("value_loss_coeff", 2.0))
    entropy_loss_coeff = float(cfg.get("entropy_loss_coeff", 0.03))
    tb_log_prefix = cfg.get("tensorboard_log_prefix", "bp_agent_exp_")

    # 启动 TensorBoard
    tb_process = None
    if use_tensorboard:
        if log_dir is None:
            log_dir = os.path.join(
                "runs", tb_log_prefix + datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            log_dir = os.path.join("runs", os.path.basename(log_dir))
        tb_process = start_tensorboard("./runs", port=6006)
        writer = SummaryWriter(log_dir=log_dir)
        print(f"[+] TensorBoard writer initialized")
    else:
        writer = None

    method = rating_cfg.get("method", "elo")

    if method.lower() == "elo":
        eval_method = EvalMethod.ELO
        method_name = "ELO"
    elif method.lower() == "trueskill":
        eval_method = EvalMethod.TRUESKILL
        method_name = "TrueSkill"
    else:
        raise ValueError(f"Unknown rating method: {method}. Use 'elo' or 'trueskill'")

    print(f"[+] Using {method_name} rating system for evaluation")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"./ckpts/bp_agent-{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"[+] Models will be saved to: {save_dir}")

    global_step = 0

    # Load oracle
    oracle = WinRateOracle(
        embed_dim=128, nhead=8, num_layers=6, use_text=True, use_player_heroes=True
    ).to(DEVICE)
    oracle_path = "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth"
    if os.path.exists(oracle_path):
        oracle.load_state_dict(torch.load(oracle_path, map_location=DEVICE))
        print(f"[+] Loaded oracle from {oracle_path}")
    oracle.eval()

    # Agent
    agent = BPTransformerAgent(embed_dim=EMBED_DIM, nhead=8, num_layers=4).to(DEVICE)

    optimizer = AdamW(agent.parameters(), lr=actor_lr)

    eval_kwargs = {
        "save_dir": save_dir,
        "oracle": oracle,
        "num_opponents": rating_cfg.get("num_opponents", 8),
        "num_player_sets": rating_cfg.get("num_player_sets", 16),
    }

    if method.lower() == "elo":
        elo_cfg = rating_cfg.get("elo", {})
        eval_kwargs.update(
            {
                "k_factor": elo_cfg.get("k_factor", 32),
                "opponent_sample_std": elo_cfg.get("opponent_sample_std", 200),
            }
        )
    elif method.lower() == "trueskill":
        ts_cfg = rating_cfg.get("trueskill", {})
        eval_kwargs.update(
            {
                "staleness_threshold": ts_cfg.get("staleness_threshold", 5),
                "num_active_models": ts_cfg.get("num_active_models", 5),
            }
        )

    rating_evaluator: RatingEvaluatorBase = get_evaluator(eval_method, **eval_kwargs)

    # Discover historical checkpoints once per training run
    historical_checkpoints = discover_checkpoints(checkpoint_dirs)
    if historical_checkpoints:
        print(f"[+] Found {len(historical_checkpoints)} historical checkpoints")
        for ckpt, _ in historical_checkpoints[:5]:
            print(f"    {ckpt}")
        if len(historical_checkpoints) > 5:
            print(f"    ... and {len(historical_checkpoints) - 5} more")

    # Training loop
    for epoch in range(epochs):
        agent.train()
        total_loss = 0

        # 每个epoch重新生成数据
        samples = generate_samples(samples_per_epoch)

        # 按batch_size切分样本
        num_batches = (len(samples) + batch_size - 1) // batch_size

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{epochs}", ncols=90)
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            # Decide how many use historical opponent vs self-play
            # Historical: 60%, Self-play: 40%
            current_batch_size = len(batch_samples)
            # Use all samples in batch for rollouts (no wasted computation)
            actual_rollout_steps = current_batch_size
            num_hist = int(actual_rollout_steps * historical_opponent_prob)

            rollouts = []

            # Historical opponent rollouts: current agent randomly plays Radiant or Dire
            # 优化：先计算需要使用哪些历史模型、各几场，然后批量加载重复使用
            if num_hist > 0 and historical_checkpoints:
                # 1. 计算这个 batch 需要与哪些历史模型对战、各几场
                hist_assignments = []  # [(sample_idx, ckpt_idx, ckpt_path), ...]
                for i in range(num_hist):
                    sample_idx = i
                    ckpt_idx = (batch_idx * actual_rollout_steps + i) % len(
                        historical_checkpoints
                    )
                    ckpt_path, _ = historical_checkpoints[ckpt_idx]
                    hist_assignments.append((sample_idx, ckpt_idx, ckpt_path))

                # 2. 按模型分组，统计每个模型需要对战场数
                ckpt_idx_to_samples = {}  # {ckpt_idx: [sample_idx, ...]}
                for sample_idx, ckpt_idx, ckpt_path in hist_assignments:
                    if ckpt_idx not in ckpt_idx_to_samples:
                        ckpt_idx_to_samples[ckpt_idx] = []
                    ckpt_idx_to_samples[ckpt_idx].append((sample_idx, ckpt_path))

                # 3. 逐个加载历史模型，完成所有分配给它的对局
                for ckpt_idx, sample_list in ckpt_idx_to_samples.items():
                    ckpt_path = sample_list[0][1]  # 获取模型路径

                    # 加载模型一次
                    opponent = BPTransformerAgent(
                        embed_dim=EMBED_DIM, nhead=8, num_layers=4
                    ).to(DEVICE)
                    opponent.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                    opponent.eval()

                    # 用这个模型完成所有分配的对局
                    for sample_idx, _ in sample_list:
                        sample = batch_samples[sample_idx]
                        current_side = random.choice(["radiant", "dire"])
                        rollout = collect_rollout(
                            agent,
                            oracle,
                            sample,
                            opponent_agent=opponent,
                            current_side=current_side,
                        )
                        rollouts.append(rollout)

                    # 完成后清理
                    del opponent

            # Self-play rollouts (agent plays both sides)
            for i in range(num_hist, actual_rollout_steps):
                sample = batch_samples[i]
                rollouts.append(collect_rollout(agent, oracle, sample))

            batch_actor_loss = 0
            batch_value_loss = 0
            batch_entropy_loss = 0
            batch_kl = 0

            for rollout in rollouts:
                valid_mask = rollout["valid_mask"].to(DEVICE)
                actions = rollout["actions"].to(DEVICE)
                old_log_probs = rollout["log_probs"].to(DEVICE)
                values = rollout["values"].to(DEVICE)
                rewards = rollout["rewards"].to(DEVICE)

                actions = actions[valid_mask]
                old_log_probs = old_log_probs[valid_mask]
                # Note: values and rewards are NOT filtered by valid_mask here because
                # values has T+1 elements (for GAE bootstrapping) while valid_mask has T elements.
                # GAE needs the full trajectory for correct computation.
                T = len(rewards)
                dones = torch.zeros(T, device=DEVICE)
                advantages, returns = compute_gae(
                    rewards.unsqueeze(-1),
                    values.unsqueeze(-1),
                    dones.unsqueeze(-1),
                    normalize_returns=True,
                )
                advantages = advantages.squeeze(-1)
                returns = returns.squeeze(-1)

                # Advantage归一化
                advantages = normalize_advantages(advantages)

                new_log_probs_list, new_values = [], []
                valid_hero_ids = get_valid_hero_ids()
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
                for idx in valid_indices:
                    state = rollout["states"][idx]
                    logits, v = agent(state)
                    heroes = state["action_history"]["heroes"]
                    used = (
                        set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
                    )
                    mask = torch.full((NUM_HEROES,), -1e9, device=logits.device)
                    # valid_hero_ids are 1-based, so need h - 1 for 0-based index
                    for h in valid_hero_ids:
                        if h <= NUM_HEROES:
                            mask[h - 1] = 0.0
                    # used already contains 0-based hero indices (from history)
                    for h in used:
                        if 0 <= h < NUM_HEROES:
                            mask[h] = -1e9
                    logits = logits + mask
                    probs = torch.softmax(logits, dim=-1)
                    new_log_probs_list.append(
                        torch.distributions.Categorical(probs).log_prob(
                            actions[len(new_log_probs_list)]
                        )
                    )
                    new_values.append(v.squeeze(-1))

                new_log_probs = torch.stack(new_log_probs_list)
                new_values = torch.cat(new_values)

                policy_loss = ppo_loss(new_log_probs, old_log_probs, advantages)

                # Filter values[:-1] and returns to align with valid steps
                old_values_filtered = values[:-1][valid_mask]
                returns_filtered = returns[valid_mask]
                value_loss = compute_value_loss(
                    new_values,
                    old_values_filtered,
                    returns_filtered,
                    clip_eps=0.2,
                    use_clipping=True,
                )

                kl_div = compute_kl_divergence(new_log_probs, old_log_probs)

                entropy_loss = 0
                for idx in valid_indices:
                    state = rollout["states"][idx]
                    logits, _ = agent(state)
                    heroes = state["action_history"]["heroes"]
                    used = (
                        set(heroes.view(-1).tolist()) if heroes.numel() > 0 else set()
                    )
                    mask = torch.full((NUM_HEROES,), -1e9, device=logits.device)
                    for h in valid_hero_ids:
                        if h <= NUM_HEROES:
                            mask[h - 1] = 0.0
                    for h in used:
                        if h < NUM_HEROES:
                            mask[h] = -1e9
                    entropy = compute_entropy(logits, mask)
                    entropy_loss -= entropy
                entropy_loss = entropy_loss / len(valid_indices)  # 平均

                # 组合loss
                # actor loss 系数为 1.0（基准）
                # value loss 系数为 value_loss_coeff，但已通过不同学习率实现，故保持 1.0
                # entropy loss 系数为 entropy_loss_coeff
                loss = (
                    policy_loss
                    + value_loss_coeff * value_loss
                    + entropy_loss_coeff * entropy_loss
                )

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
                writer.add_scalar(
                    "Loss/actor", batch_actor_loss / len(rollouts), global_step
                )
                writer.add_scalar(
                    "Loss/value", batch_value_loss / len(rollouts), global_step
                )
                writer.add_scalar(
                    "Loss/entropy", batch_entropy_loss / len(rollouts), global_step
                )
                writer.add_scalar(
                    "Loss/total",
                    total_loss / ((batch_idx + 1) * actual_rollout_steps),
                    global_step,
                )
                writer.add_scalar(
                    "Loss/kl_divergence", batch_kl / len(rollouts), global_step
                )
                writer.flush()  # 确保数据立即写入磁盘

            pbar.set_postfix(
                {"Loss": f"{total_loss / ((batch_idx + 1) * actual_rollout_steps):.4f}"}
            )

        # 定期保存中间模型并进行评分评估
        eval_interval = rating_cfg.get("eval_interval", 8)
        if (epoch + 1) % eval_interval == 0:
            checkpoint_path = f"{save_dir}/bp_agent_epoch{epoch + 1}.pth"
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"\n[+] Checkpoint saved: {checkpoint_path}")

            num_opponents = rating_cfg.get("num_opponents", 8)
            num_player_sets = rating_cfg.get("num_player_sets", 16)

            print(f"[+] {method_name} evaluation at epoch {epoch + 1}...")
            eval_result = rating_evaluator.evaluate(
                model_path=checkpoint_path,
                num_opponents=num_opponents,
                num_player_sets=num_player_sets,
            )

            # 打印当前排行榜
            rating_evaluator.print_leaderboard()

            # 记录模型评分到 TensorBoard
            if writer is not None:
                # 获取评估结果中的各项指标
                record = rating_evaluator.rating_manager.get_record(checkpoint_path)
                if record is not None:
                    if method.lower() == "trueskill":
                        # TrueSkill: 记录 mu, sigma, rating, avg_winrate
                        writer.add_scalar(
                            f"Rating/{method_name.lower()}_mu", record.mu, epoch + 1
                        )
                        writer.add_scalar(
                            f"Rating/{method_name.lower()}_sigma",
                            record.sigma,
                            epoch + 1,
                        )
                        writer.add_scalar(
                            f"Rating/{method_name.lower()}_rating",
                            record.rating,
                            epoch + 1,
                        )
                    else:
                        # ELO: 只记录 rating
                        writer.add_scalar(
                            f"Rating/{method_name.lower()}_rating",
                            record.elo,
                            epoch + 1,
                        )

                    # 计算并记录本轮平均胜率
                    if eval_result.get("results"):
                        avg_winrate = sum(
                            r["win_rate"] for r in eval_result["results"]
                        ) / len(eval_result["results"])
                        writer.add_scalar("Rating/avg_winrate", avg_winrate, epoch + 1)

                writer.flush()  # 确保数据立即写入

    # Save final model
    model_path = f"{save_dir}/bp_agent_final.pth"
    torch.save(agent.state_dict(), model_path)
    print(f"[+] Model saved to {model_path}")

    # 训练结束后的最终评分评估
    num_opponents = rating_cfg.get("num_opponents", 8)
    num_player_sets = rating_cfg.get("num_player_sets", 16)

    print(f"[+] Final {method_name} evaluation...")
    rating_evaluator.evaluate(
        model_path=model_path,
        num_opponents=num_opponents,
        num_player_sets=num_player_sets,
    )

    # 记录最终模型评分到 TensorBoard
    if writer is not None:
        record = rating_evaluator.rating_manager.get_record(model_path)
        if record is not None:
            if method.lower() == "trueskill":
                writer.add_scalar(f"Rating/{method_name.lower()}_mu", record.mu, epochs)
                writer.add_scalar(
                    f"Rating/{method_name.lower()}_sigma", record.sigma, epochs
                )
                writer.add_scalar(
                    f"Rating/{method_name.lower()}_rating", record.rating, epochs
                )
            else:
                writer.add_scalar(
                    f"Rating/{method_name.lower()}_rating", record.elo, epochs
                )
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
    # 使用 configs/bp_agent_config.yaml 中的默认配置
    train()

    # 使用 TrueSkill 的示例（通过覆盖参数）:
    # train(
    #     config_path="configs/bp_agent_config.yaml",
    #     epochs=32, batch_size=32, rollout_steps=10,
    #     # rating params 需在 yaml 中设置
    # )
