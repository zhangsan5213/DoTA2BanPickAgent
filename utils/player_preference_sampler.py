"""
玩家偏好采样器
根据指定位置生成玩家的本命英雄池
"""

import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict
from pathlib import Path


# 用于计算相似度的特征列（角色标签 + 主属性）
SIMILARITY_FEATURES = [
    'attr_agi', 'attr_all', 'attr_int', 'attr_str',
    'role_Carry', 'role_Support', 'role_Pusher', 
    'role_Initiator', 'role_Disabler', 'role_Nuker', 
    'role_Durable', 'role_Escape'
]


def load_hero_positions(data_path: str = None) -> Dict[str, Dict[str, bool]]:
    """加载英雄位置映射数据"""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "hero_positions.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_hero_features(data_path: str = None) -> pd.DataFrame:
    """加载英雄特征数据"""
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "hero_features.xlsx"
    df = pd.read_excel(data_path)
    return df


def get_heroes_by_position(df: pd.DataFrame, hero_positions: Dict, position: int) -> pd.DataFrame:
    """
    根据位置筛选英雄
    
    Args:
        df: 英雄特征DataFrame
        hero_positions: 英雄位置映射字典
        position: 位置 (1-5)
    
    Returns:
        该位置的英雄DataFrame
    """
    position_str = str(position)
    valid_heroes = [
        f"npc_dota_hero_{hero}" 
        for hero, positions in hero_positions.items() 
        if positions.get(position_str, False)
    ]
    
    result = df[df['name'].isin(valid_heroes)].copy()
    
    if len(result) == 0:
        raise ValueError(f"位置{position}没有匹配的英雄")
    
    return result


def sample_from_distribution(m: int, min_val: float = 0.45, max_val: float = 0.7) -> np.ndarray:
    """
    从以0.5为中心的对称分布中采样胜率
    离0.5越远概率越低，使用正态分布的绝对值
    
    Args:
        m: 采样数量
        min_val: 最小值
        max_val: 最大值
    
    Returns:
        采样结果数组
    """
    samples = []
    while len(samples) < m:
        # 生成正态分布，然后取绝对值并加上0.5
        val = np.random.normal(0, 0.08)
        winrate = 0.5 + abs(val)
        # 限制在范围内
        if min_val <= winrate <= max_val:
            samples.append(winrate)
    
    return np.array(samples)


def compute_similarities_to_seed(
    candidates_df: pd.DataFrame, 
    seed_heroes_df: pd.DataFrame,
    feature_cols: List[str] = None
) -> np.ndarray:
    """
    计算候选英雄与种子英雄的平均相似度
    
    Args:
        candidates_df: 候选英雄DataFrame
        seed_heroes_df: 种子英雄DataFrame
        feature_cols: 用于计算的特征列
    
    Returns:
        每个候选英雄与种子英雄的平均相似度数组
    """
    if feature_cols is None:
        feature_cols = SIMILARITY_FEATURES
    
    # 种子英雄的平均特征向量
    seed_features = seed_heroes_df[feature_cols].mean(axis=0).values
    seed_norm = np.linalg.norm(seed_features)
    if seed_norm > 0:
        seed_features = seed_features / seed_norm
    
    # 候选英雄特征
    candidate_features = candidates_df[feature_cols].values
    candidate_norms = np.linalg.norm(candidate_features, axis=1, keepdims=True)
    candidate_norms[candidate_norms == 0] = 1
    candidate_features_normalized = candidate_features / candidate_norms
    
    # 计算余弦相似度
    similarities = candidate_features_normalized @ seed_features
    
    return similarities


def sample_similar_heroes(
    df: pd.DataFrame,
    seed_heroes: pd.DataFrame,
    n: int,
    exclude_ids: List[int],
    allow_divergence: bool = True
) -> pd.DataFrame:
    """
    根据种子英雄采样相似英雄
    
    Args:
        df: 全部英雄DataFrame
        seed_heroes: 种子英雄DataFrame
        n: 需要采样的数量
        exclude_ids: 需要排除的英雄ID
        allow_divergence: 是否允许较大程度的发散（跨位置采样时设为True）
    
    Returns:
        采样的相似英雄DataFrame
    """
    # 计算种子英雄的平均特征向量
    seed_features = seed_heroes[SIMILARITY_FEATURES].mean(axis=0).values
    
    # 计算所有候选英雄与种子英雄的相似度
    candidates = df[~df['id'].isin(exclude_ids)].copy()
    if len(candidates) == 0:
        return candidates
    
    candidate_features = candidates[SIMILARITY_FEATURES].values
    
    # 归一化
    seed_norm = np.linalg.norm(seed_features)
    if seed_norm > 0:
        seed_features = seed_features / seed_norm
    
    candidate_norms = np.linalg.norm(candidate_features, axis=1, keepdims=True)
    candidate_norms[candidate_norms == 0] = 1
    candidate_features_normalized = candidate_features / candidate_norms
    
    # 计算余弦相似度
    similarities = candidate_features_normalized @ seed_features
    
    # 转换为概率
    if allow_divergence:
        # 允许较大发散：给低相似度英雄更多机会
        # 使用较温和的softmax
        exp_sim = np.exp(similarities * 2)
    else:
        # 同位置内聚类：高相似度英雄概率更高
        exp_sim = np.exp(similarities * 5)
    
    probs = exp_sim / exp_sim.sum()
    
    # 采样n个
    n = min(n, len(candidates))
    selected_indices = np.random.choice(
        len(candidates), 
        size=n, 
        replace=False, 
        p=probs
    )
    
    return candidates.iloc[selected_indices]


def sample_player_preference(
    position: int,
    m: int = 3,
    n: int = 5,
    data_path: str = None,
    positions_path: str = None,
    random_seed: int = None
) -> List[Dict]:
    """
    采样玩家的本命英雄池
    
    算法步骤：
    1. 根据位置筛选候选英雄（同位置英雄）
    2. 在该范围内采样m个特征相似的英雄作为种子
    3. 根据这m个种子英雄，在全部英雄中找寻相似英雄，采样n个（跨位置，允许发散）
    4. 为m+n个英雄分配0.45~0.7的胜率，离0.5越远概率越低
    
    Args:
        position: 玩家主要位置 (1-5)
        m: 同位置种子英雄数量
        n: 跨位置扩展英雄数量
        data_path: 英雄特征数据路径，默认使用data/hero_features.xlsx
        positions_path: 英雄位置映射文件路径，默认使用data/hero_positions.json
        random_seed: 随机种子
    
    Returns:
        List[Dict]: 包含英雄信息和胜率的字典列表
        每个字典包含：id, name, win_rate, is_seed (是否种子英雄)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 加载数据
    df = load_hero_features(data_path)
    hero_positions = load_hero_positions(positions_path)
    
    # 第一步：根据位置筛选英雄
    position_heroes = get_heroes_by_position(df, hero_positions, position)
    
    if len(position_heroes) < m:
        raise ValueError(f"位置{position}的英雄数量({len(position_heroes)})不足{m}个")
    
    # 第二步：在同位置英雄中采样m个特征相似的英雄
    # 先随机选一个种子
    first_seed_idx = np.random.choice(len(position_heroes))
    seed_heroes_list = [position_heroes.iloc[first_seed_idx]]
    
    if m > 1:
        # 计算与第一个种子的相似度，采样其余m-1个
        remaining = position_heroes.drop(position_heroes.index[first_seed_idx])
        
        # 计算相似度
        seed_heroes_df = pd.DataFrame([position_heroes.iloc[first_seed_idx]])
        similarities = compute_similarities_to_seed(remaining, seed_heroes_df)
        
        # 基于相似度采样m-1个（同位置内聚类，使用较高区分度）
        exp_sim = np.exp(similarities * 5)
        probs = exp_sim / exp_sim.sum()
        
        selected_indices = np.random.choice(
            len(remaining), 
            size=min(m-1, len(remaining)), 
            replace=False, 
            p=probs
        )
        
        for idx in selected_indices:
            seed_heroes_list.append(remaining.iloc[idx])
    
    seed_heroes = pd.DataFrame(seed_heroes_list)
    seed_ids = seed_heroes['id'].tolist()
    
    # 第三步：在全部英雄中找寻相似英雄，采样n个（跨位置，允许发散）
    expansion_heroes = sample_similar_heroes(df, seed_heroes, n, seed_ids, allow_divergence=True)
    expansion_ids = expansion_heroes['id'].tolist()
    
    # 合并结果
    all_heroes = pd.concat([seed_heroes, expansion_heroes], ignore_index=True)
    
    # 第四步：分配胜率
    win_rates = sample_from_distribution(len(all_heroes))
    
    # 构建结果
    results = []
    for i, (_, row) in enumerate(all_heroes.iterrows()):
        # 提取英雄名称（去掉前缀）
        hero_name = row['name'].replace('npc_dota_hero_', '')
        
        results.append({
            'id': int(row['id']),
            'name': hero_name,
            'full_name': row['name'],
            'win_rate': round(win_rates[i], 4),
            'is_seed': row['id'] in seed_ids,
            'position': position
        })
    
    return results


def batch_sample_player_preferences(
    num_players: int,
    position_distribution: Dict[int, float] = None,
    m: int = 3,
    n: int = 5,
    data_path: str = None,
    positions_path: str = None,
    random_seed: int = None
) -> List[Dict]:
    """
    批量采样多个玩家的偏好
    
    Args:
        num_players: 玩家数量
        position_distribution: 位置分布概率，默认均匀分布
        m: 每个玩家的种子英雄数量
        n: 每个玩家的扩展英雄数量
        data_path: 英雄特征数据路径
        positions_path: 英雄位置映射文件路径
        random_seed: 随机种子
    
    Returns:
        包含所有玩家偏好的列表
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if position_distribution is None:
        position_distribution = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
    
    positions = list(position_distribution.keys())
    probs = list(position_distribution.values())
    
    all_players = []
    for i in range(num_players):
        position = np.random.choice(positions, p=probs)
        player_pref = sample_player_preference(
            position=position,
            m=m,
            n=n,
            data_path=data_path,
            positions_path=positions_path,
            random_seed=None  # 不设置，继续使用当前随机状态
        )
        all_players.append({
            'player_id': i,
            'position': position,
            'heroes': player_pref
        })
    
    return all_players


def get_position_heroes(position: int, positions_path: str = None) -> List[str]:
    """
    获取指定位置的所有英雄名称
    
    Args:
        position: 位置 (1-5)
        positions_path: 英雄位置映射文件路径
    
    Returns:
        英雄名称列表（不含npc_dota_hero_前缀）
    """
    hero_positions = load_hero_positions(positions_path)
    position_str = str(position)
    return [
        hero for hero, positions in hero_positions.items() 
        if positions.get(position_str, False)
    ]


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("玩家偏好采样器测试")
    print("=" * 60)
    
    # 显示各位置英雄数量
    print("\n各位置英雄数量:")
    for pos in range(1, 6):
        heroes = get_position_heroes(pos)
        print(f"  {pos}号位: {len(heroes)}个")
    
    # 测试各个位置
    for pos in range(1, 6):
        print(f"\n{'='*40}")
        print(f"位置 {pos} 号位玩家示例:")
        print(f"{'='*40}")
        
        try:
            heroes = sample_player_preference(
                position=pos,
                m=3,
                n=5,
                random_seed=42 + pos
            )
            
            print(f"共 {len(heroes)} 个本命英雄:")
            for h in heroes:
                seed_mark = " [种子]" if h['is_seed'] else ""
                print(f"  - {h['name']}: {h['win_rate']*100:.2f}%{seed_mark}")
        except Exception as e:
            print(f"错误: {e}")
    
    # 测试批量采样
    print(f"\n{'='*40}")
    print("批量采样 10 个玩家:")
    print(f"{'='*40}")
    
    players = batch_sample_player_preferences(
        num_players=10,
        random_seed=123
    )
    
    for p in players:
        seed_heroes = [h['name'] for h in p['heroes'] if h['is_seed']]
        expansion_heroes = [h['name'] for h in p['heroes'] if not h['is_seed']]
        avg_winrate = np.mean([h['win_rate'] for h in p['heroes']])
        print(f"\n玩家 {p['player_id']} ({p['position']}号位):")
        print(f"  种子英雄: {', '.join(seed_heroes)}")
        print(f"  扩展英雄: {', '.join(expansion_heroes)}")
        print(f"  平均胜率: {avg_winrate*100:.2f}%")
