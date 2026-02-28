"""
BP框架演示脚本 - Self-Play模式

展示如何使用BPEngine收集双方（Radiant & Dire）的trajectory
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP冲突

import sys
import pandas as pd
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, '.')

# 加载英雄名字映射
def load_hero_names():
    """加载英雄ID到名字的映射"""
    try:
        hero_df = pd.read_excel("./data/hero_features.xlsx")
        # 确保ID是整数（pandas可能读成float）
        id_to_name = {int(row['id']): row['name'] for _, row in hero_df.iterrows()}
        print(f"Loaded {len(id_to_name)} hero names")
        return id_to_name
    except Exception as e:
        print(f"Warning: Could not load hero names: {e}")
        return {}

HERO_ID_TO_NAME = load_hero_names()

def get_hero_name(hero_id):
    """获取英雄名字，找不到时返回 ID"""
    return HERO_ID_TO_NAME.get(hero_id, f"Hero_{hero_id}")

from model.bp_agent import BPActorCritic
from model.win_rate_oracle import WinRateOracle
from bp_framework.engine import BPEngine
from bp_framework.environment import Team
from utils.player_preference_sampler import sample_player_preference


def generate_team_player_features(num_heroes=160):
    """
    生成一个队伍（5个位置）的玩家特征
    
    Returns:
        player_feats: [5, NUM_HEROES] tensor，每个位置一个玩家
        player_info: list of dict，包含每个玩家的信息
    """
    player_feats = torch.zeros(5, num_heroes)
    player_info = []
    
    for position in range(1, 6):  # 1-5号位
        # 采样该位置玩家的偏好英雄
        heroes = sample_player_preference(
            position=position,
            m=3,  # 3个种子英雄
            n=5,  # 5个扩展英雄
            random_seed=None,  # 随机种子
        )
        
        # 填充到特征矩阵
        for hero in heroes:
            hero_id = hero['id']
            win_rate = hero['win_rate']
            if 1 <= hero_id <= num_heroes:
                player_feats[position - 1, hero_id - 1] = win_rate
        
        # 记录玩家信息
        seed_heroes = [h['name'] for h in heroes if h['is_seed']]
        player_info.append({
            'position': position,
            'seed_heroes': seed_heroes,
            'num_heroes': len(heroes),
        })
    
    return player_feats, player_info


def print_team_players(player_info, team_name="Team"):
    """打印队伍玩家信息"""
    print(f"\n{team_name} Players:")
    for info in player_info:
        print(f"  Position {info['position']}: seed heroes = {', '.join(info['seed_heroes'])}")


def create_dummy_models(device='cpu', use_player_heroes=True):
    """创建未训练的dummy模型用于演示"""
    print("Creating dummy models...")
    
    # 创建Actor-Critic（不使用HeroEncoder以加速演示）
    actor_critic = BPActorCritic(
        embed_dim=64,
        nhead=2,
        num_layers=1,
        num_heroes=160,
        use_hero_encoder=False,  # 简化，不使用HeroEncoder
        hero_encoder_dim=64,
        use_player_heroes=use_player_heroes,  # 是否使用玩家特征
        player_hero_embed_dim=64,
    ).to(device)
    
    # 创建Oracle（同样简化）
    oracle = WinRateOracle(
        embed_dim=64,
        nhead=2,
        num_layers=1,
        use_text=False,  # 不使用文本特征
        use_player_heroes=use_player_heroes,  # 是否使用玩家特征
    ).to(device)
    
    return actor_critic, oracle


def load_trained_oracle(ckpt_path='./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260228235818-083-0.9055.pth', device='cpu'):
    """加载训练好的Oracle"""
    print(f"Loading trained Oracle from: {ckpt_path}")
    
    oracle = WinRateOracle(
        embed_dim=128,
        nhead=8,
        num_layers=6,  # 训练时用的是6层
        use_text=True,  # 训练时使用了文本特征
        use_player_heroes=True,  # 训练时使用了玩家特征
        hero_encoder_id_dim=128,
        hero_encoder_attr_dim=64,
        hero_encoder_text_dim=128,
    ).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 处理可能的不同保存格式
    if 'model_state_dict' in checkpoint:
        oracle.load_state_dict(checkpoint['model_state_dict'])
    else:
        oracle.load_state_dict(checkpoint)
    
    oracle.eval()
    print(f"Oracle loaded! Checkpoint type: {type(checkpoint)}")
    return oracle


def format_picks(picks):
    """格式化输出pick列表（显示名字）"""
    return [f"{get_hero_name(hid)}({hid})" for hid in picks]


def demo_self_play():
    """演示Self-Play（双方用同一个模型，使用训练好的Oracle）"""
    print("\n" + "=" * 70)
    print("DEMO: Self-Play BP Collection (with Trained Oracle)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 生成玩家特征
    print("\nGenerating player features...")
    r_feats, r_info = generate_team_player_features()
    d_feats, d_info = generate_team_player_features()
    
    print_team_players(r_info, "Radiant")
    print_team_players(d_info, "Dire")
    
    # 使用训练好的Oracle
    actor_critic, _ = create_dummy_models(device, use_player_heroes=False)
    oracle = load_trained_oracle(device=device)
    
    # 创建引擎 - Self-Play模式
    engine = BPEngine(
        actor_critic=actor_critic,
        oracle=oracle,
        device=device,
        first_team=Team.RADIANT,  # 天辉先手
        reward_type='final',
        gamma=1.0,
        gae_lambda=0.95,
        use_gae=True,
    )
    
    # 运行一个episode并详细打印
    print("\n>>> Running single episode with verbose output...")
    r_rollout, d_rollout = engine.run_episode(
        deterministic=False, 
        verbose=True, 
        hero_name_fn=get_hero_name,
        radiant_player_feats=r_feats,
        dire_player_feats=d_feats,
    )
    
    print("\n>>> Rollout Statistics:")
    print(f"Radiant transitions: {len(r_rollout)}")
    print(f"Dire transitions:    {len(d_rollout)}")
    print(f"Radiant return:      {sum(t.reward for t in r_rollout.transitions):+.4f}")
    print(f"Dire return:         {sum(t.reward for t in d_rollout.transitions):+.4f}")
    
    # 打印前10个transition的详细信息
    print("\n>>> First 10 Radiant transitions:")
    for i, trans in enumerate(r_rollout.transitions[:10]):
        hero_name = get_hero_name(trans.action)
        print(f"  Step {i}: {hero_name} (ID={trans.action})")
        if len(r_rollout.returns) > 0:
            print(f"    Return: {r_rollout.returns[i]:.4f}, Advantage: {r_rollout.advantages[i]:.4f}")
    
    return r_rollout, d_rollout


def demo_self_play_with_player_features():
    """演示带玩家特征的Self-Play"""
    print("\n" + "=" * 70)
    print("DEMO: Self-Play BP with Player Features (Trained Oracle)")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 生成玩家特征
    print("\nGenerating player features...")
    r_feats, r_info = generate_team_player_features()
    d_feats, d_info = generate_team_player_features()
    
    print_team_players(r_info, "Radiant")
    print_team_players(d_info, "Dire")
    
    # 创建使用玩家特征的模型（Actor-Critic用dummy，Oracle用训练好的）
    actor_critic, _ = create_dummy_models(device, use_player_heroes=True)
    oracle = load_trained_oracle(device=device)
    
    # 创建引擎
    engine = BPEngine(
        actor_critic=actor_critic,
        oracle=oracle,
        device=device,
        first_team=Team.RADIANT,
        reward_type='final',
        gamma=1.0,
        gae_lambda=0.95,
        use_gae=True,
    )
    
    # 运行episode
    print("\n>>> Running episode with player features...")
    r_rollout, d_rollout = engine.run_episode(
        deterministic=False,
        verbose=True,
        hero_name_fn=get_hero_name,
        radiant_player_feats=r_feats,
        dire_player_feats=d_feats,
    )
    
    print("\n>>> Rollout Statistics:")
    print(f"Radiant return: {sum(t.reward for t in r_rollout.transitions):+.4f}")
    print(f"Dire return:    {sum(t.reward for t in d_rollout.transitions):+.4f}")
    
    return r_rollout, d_rollout


def demo_batch_collection():
    """演示批量收集rollouts"""
    print("\n" + "=" * 70)
    print("DEMO: Batch Collection")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor_critic, oracle = create_dummy_models(device)
    
    engine = BPEngine(
        actor_critic=actor_critic,
        oracle=oracle,
        device=device,
        first_team=Team.RADIANT,
    )
    
    # 收集多个episodes
    num_episodes = 5
    print(f"\n>>> Collecting {num_episodes} episodes...")
    
    r_buffer, d_buffer = engine.collect_rollouts(
        num_episodes=num_episodes,
        deterministic=False,
        verbose_interval=1,  # 每轮都打印
    )
    
    print(f"\n>>> Collection Summary:")
    print(f"Radiant buffer: {len(r_buffer)} episodes, {r_buffer.total_transitions()} transitions")
    print(f"Dire buffer:    {len(d_buffer)} episodes, {d_buffer.total_transitions()} transitions")
    
    # 获取batch数据
    r_batches = r_buffer.get_all_batches()
    d_batches = d_buffer.get_all_batches()
    
    if r_batches:
        batch = r_batches[0]
        print(f"\n>>> Sample batch shapes (Radiant):")
        for key, tensor in batch.items():
            print(f"  {key}: {tensor.shape}")
    
    # 合并双方数据（Self-Play训练）
    all_batches = r_batches + d_batches
    print(f"\n>>> Combined for training: {len(all_batches)} batches")
    
    return r_buffer, d_buffer


def demo_evaluation():
    """演示评估"""
    print("\n" + "=" * 70)
    print("DEMO: Policy Evaluation")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor_critic, oracle = create_dummy_models(device)
    
    engine = BPEngine(
        actor_critic=actor_critic,
        oracle=oracle,
        device=device,
        first_team=Team.RADIANT,
    )
    
    print("\n>>> Evaluating policy (deterministic)...")
    stats = engine.evaluate(num_episodes=10, verbose=False)
    
    print("\n>>> Evaluation Results:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    return stats


def demo_different_first_team():
    """演示不同先手的影响"""
    print("\n" + "=" * 70)
    print("DEMO: Different First Team")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actor_critic, oracle = create_dummy_models(device)
    
    for first_team in [Team.RADIANT, Team.DIRE]:
        print(f"\n>>> First team: {first_team.name}")
        
        engine = BPEngine(
            actor_critic=actor_critic,
            oracle=oracle,
            device=device,
            first_team=first_team,
        )
        
        r_rollout, d_rollout = engine.run_episode(deterministic=True, verbose=True, hero_name_fn=get_hero_name)


if __name__ == '__main__':
    print("DOTA2 BP Self-Play Framework Demo")
    print("=" * 70)
    print("Key design: Same model for both Radiant and Dire (team embedding)")
    print("=" * 70)
    
    demos = [
        # ("Self-Play Episode", demo_self_play),
        ("Self-Play with Player Features", demo_self_play_with_player_features),
        # ("Batch Collection", demo_batch_collection),
        # ("Evaluation", demo_evaluation),
        # ("Different First Team", demo_different_first_team),
    ]
    
    for name, demo_fn in demos:
        try:
            demo_fn()
        except Exception as e:
            print(f"\n[!] Demo '{name}' failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
