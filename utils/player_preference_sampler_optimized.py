"""
玩家偏好采样器 - 优化版本
基于特征相似度的玩家本命英雄池生成

优化点：
1. 全局缓存英雄数据，避免重复IO
2. 预计算特征向量归一化，加速相似度计算
3. 真正的批量采样接口
4. 使用进程池替代线程池绕过GIL
"""

import numpy as np
import pandas as pd
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ============== 全局缓存 ==============
_hero_features_cache: Optional[pd.DataFrame] = None
_hero_positions_cache: Optional[Dict] = None
_feature_matrix_cache: Optional[np.ndarray] = None  # 预计算的特征矩阵
_hero_ids_cache: Optional[np.ndarray] = None
_similarity_features = [
    'attr_agi', 'attr_all', 'attr_int', 'attr_str',
    'role_Carry', 'role_Support', 'role_Pusher', 
    'role_Initiator', 'role_Disabler', 'role_Nuker', 
    'role_Durable', 'role_Escape'
]


def _load_hero_data(data_path: str = None, positions_path: str = None):
    """加载并缓存英雄数据"""
    global _hero_features_cache, _hero_positions_cache, _feature_matrix_cache, _hero_ids_cache
    
    if _hero_features_cache is None:
        if data_path is None:
            data_path = Path(__file__).parent.parent / "data" / "hero_features.xlsx"
        _hero_features_cache = pd.read_excel(data_path)
        
        # 预计算特征矩阵和归一化
        features = _hero_features_cache[_similarity_features].values.astype(np.float32)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        _feature_matrix_cache = features / norms
        _hero_ids_cache = _hero_features_cache['id'].values
    
    if _hero_positions_cache is None:
        if positions_path is None:
            positions_path = Path(__file__).parent.parent / "data" / "hero_positions.json"
        with open(positions_path, 'r', encoding='utf-8') as f:
            _hero_positions_cache = json.load(f)
    
    return _hero_features_cache, _hero_positions_cache, _feature_matrix_cache, _hero_ids_cache


def clear_cache():
    """清除缓存，用于内存管理或数据更新后"""
    global _hero_features_cache, _hero_positions_cache, _feature_matrix_cache, _hero_ids_cache
    _hero_features_cache = None
    _hero_positions_cache = None
    _feature_matrix_cache = None
    _hero_ids_cache = None


# ============== 核心采样函数 ==============

def sample_from_distribution(m: int, min_val: float = 0.45, max_val: float = 0.7) -> np.ndarray:
    """
    从以0.5为中心的对称分布中采样胜率
    使用向量化操作替代循环
    """
    # 批量生成，然后过滤
    batch_size = m * 3  # 多生成一些以提高效率
    samples = []
    
    while len(samples) < m:
        vals = np.random.normal(0, 0.08, size=batch_size)
        winrates = 0.5 + np.abs(vals)
        valid = winrates[(winrates >= min_val) & (winrates <= max_val)]
        samples.extend(valid[:m - len(samples)])
    
    return np.array(samples[:m])


def compute_similarities_vectorized(
    candidate_features: np.ndarray,
    seed_features: np.ndarray
) -> np.ndarray:
    """
    向量化相似度计算
    
    Args:
        candidate_features: [N, D] 已归一化的候选特征
        seed_features: [M, D] 已归一化的种子特征
    Returns:
        similarities: [N] 每个候选与种子集的平均相似度
    """
    # 种子特征的平均向量
    mean_seed = seed_features.mean(axis=0)
    mean_seed = mean_seed / (np.linalg.norm(mean_seed) + 1e-8)
    
    # 批量计算余弦相似度
    similarities = candidate_features @ mean_seed
    return similarities


def sample_similar_heroes_fast(
    hero_df: pd.DataFrame,
    feature_matrix: np.ndarray,
    hero_ids: np.ndarray,
    seed_heroes_df: pd.DataFrame,
    seed_indices: np.ndarray,
    n: int,
    exclude_ids: np.ndarray,
    allow_divergence: bool = True
) -> pd.DataFrame:
    """
    优化的相似英雄采样（使用预计算的特征矩阵）
    """
    # 获取排除掩码
    exclude_set = set(exclude_ids)
    valid_mask = ~np.isin(hero_ids, list(exclude_set))
    
    if not valid_mask.any():
        return hero_df.iloc[:0]
    
    # 筛选候选
    candidate_indices = np.where(valid_mask)[0]
    candidate_features = feature_matrix[candidate_indices]
    
    # 种子特征
    seed_features = feature_matrix[seed_indices]
    
    # 计算相似度
    similarities = compute_similarities_vectorized(candidate_features, seed_features)
    
    # 转换为概率
    if allow_divergence:
        exp_sim = np.exp(similarities * 2)
    else:
        exp_sim = np.exp(similarities * 5)
    
    probs = exp_sim / exp_sim.sum()
    
    # 采样
    n = min(n, len(candidate_indices))
    if n <= 0:
        return hero_df.iloc[:0]
    
    selected_local_indices = np.random.choice(
        len(candidate_indices), 
        size=n, 
        replace=False, 
        p=probs
    )
    selected_global_indices = candidate_indices[selected_local_indices]
    
    return hero_df.iloc[selected_global_indices]


def get_heroes_by_position_fast(
    hero_df: pd.DataFrame,
    hero_positions: Dict,
    position: int
) -> pd.DataFrame:
    """快速位置筛选"""
    position_str = str(position)
    valid_names = {
        f"npc_dota_hero_{hero}" 
        for hero, positions in hero_positions.items() 
        if positions.get(position_str, False)
    }
    
    mask = hero_df['name'].isin(valid_names)
    result = hero_df[mask]
    
    if len(result) == 0:
        raise ValueError(f"位置{position}没有匹配的英雄")
    
    return result


# ============== 主要API ==============

def sample_player_preference_fast(
    position: int,
    m: int = 3,
    n: int = 5,
    data_path: str = None,
    positions_path: str = None,
    random_seed: int = None
) -> List[Dict]:
    """
    优化的玩家偏好采样（单条）
    
    性能对比（预估）：
    - 原版本: ~50-100ms/次
    - 优化版: ~5-10ms/次 (主要提升来自缓存和向量化)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 加载缓存数据
    hero_df, hero_positions, feature_matrix, hero_ids = _load_hero_data(data_path, positions_path)
    
    # 第一步：位置筛选
    position_heroes = get_heroes_by_position_fast(hero_df, hero_positions, position)
    position_indices = position_heroes.index.values
    
    if len(position_heroes) < m:
        raise ValueError(f"位置{position}的英雄数量({len(position_heroes)})不足{m}个")
    
    # 第二步：采样种子英雄
    # 随机选第一个种子
    first_seed_local_idx = np.random.choice(len(position_indices))
    first_seed_global_idx = position_indices[first_seed_local_idx]
    
    seed_indices = [first_seed_global_idx]
    
    if m > 1:
        # 计算与第一个种子的相似度
        remaining_mask = np.ones(len(position_indices), dtype=bool)
        remaining_mask[first_seed_local_idx] = False
        remaining_indices = position_indices[remaining_mask]
        
        if len(remaining_indices) > 0:
            remaining_features = feature_matrix[remaining_indices]
            seed_feature = feature_matrix[[first_seed_global_idx]]
            
            similarities = compute_similarities_vectorized(remaining_features, seed_feature)
            
            # 基于相似度采样
            exp_sim = np.exp(similarities * 5)
            probs = exp_sim / exp_sim.sum()
            
            num_to_sample = min(m - 1, len(remaining_indices))
            selected = np.random.choice(len(remaining_indices), size=num_to_sample, replace=False, p=probs)
            seed_indices.extend(remaining_indices[selected])
    
    seed_indices = np.array(seed_indices)
    seed_heroes = hero_df.iloc[seed_indices]
    seed_ids = hero_ids[seed_indices]
    
    # 第三步：跨位置采样相似英雄
    expansion_heroes = sample_similar_heroes_fast(
        hero_df, feature_matrix, hero_ids,
        seed_heroes, seed_indices, n, seed_ids, allow_divergence=True
    )
    
    # 合并
    all_heroes = pd.concat([seed_heroes, expansion_heroes], ignore_index=True)
    
    # 第四步：分配胜率
    win_rates = sample_from_distribution(len(all_heroes))
    
    # 构建结果
    results = []
    for i, (_, row) in enumerate(all_heroes.iterrows()):
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


def sample_player_preferences_batch(
    num_players: int,
    position_distribution: Dict[int, float] = None,
    m: int = 3,
    n: int = 5,
    data_path: str = None,
    positions_path: str = None,
    random_seed: int = None,
    use_parallel: bool = True,
    n_workers: int = None
) -> List[Dict]:
    """
    真正的批量采样接口
    
    优化策略：
    1. 单进程内批量生成（减少函数调用开销）
    2. 可选多进程并行（绕过GIL）
    
    Args:
        num_players: 玩家数量
        position_distribution: 位置分布，默认均匀
        m, n: 采样参数
        use_parallel: 是否使用多进程
        n_workers: 进程数，默认CPU核心数
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if position_distribution is None:
        position_distribution = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}
    
    positions = list(position_distribution.keys())
    probs = list(position_distribution.values())
    
    # 预分配位置
    player_positions = np.random.choice(positions, size=num_players, p=probs)
    
    # 预加载数据到缓存
    _load_hero_data(data_path, positions_path)
    
    if use_parallel and num_players > 10:
        # 多进程并行
        if n_workers is None:
            n_workers = min(mp.cpu_count(), 8)
        
        # 将任务分批
        batch_size = max(1, num_players // n_workers)
        batches = [
            (player_positions[i:i+batch_size], m, n, data_path, positions_path)
            for i in range(0, num_players, batch_size)
        ]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_sample_batch_worker, batch)
                for batch in batches
            ]
            
            results = []
            player_id = 0
            for future in as_completed(futures):
                batch_results = future.result()
                for player_data in batch_results:
                    player_data['player_id'] = player_id
                    results.append(player_data)
                    player_id += 1
        
        # 按player_id排序
        results.sort(key=lambda x: x['player_id'])
    else:
        # 单进程批量生成
        results = []
        for i, position in enumerate(player_positions):
            heroes = sample_player_preference_fast(
                position=position,
                m=m,
                n=n
            )
            results.append({
                'player_id': i,
                'position': int(position),
                'heroes': heroes
            })
    
    return results


def _sample_batch_worker(args) -> List[Dict]:
    """多进程工作函数"""
    positions, m, n, data_path, positions_path = args
    
    # 每个进程需要加载自己的缓存
    _load_hero_data(data_path, positions_path)
    
    results = []
    for position in positions:
        heroes = sample_player_preference_fast(
            position=int(position),
            m=m,
            n=n
        )
        results.append({
            'position': int(position),
            'heroes': heroes
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
    兼容旧接口的批量采样（使用优化版本）
    """
    return sample_player_preferences_batch(
        num_players=num_players,
        position_distribution=position_distribution,
        m=m,
        n=n,
        data_path=data_path,
        positions_path=positions_path,
        random_seed=random_seed,
        use_parallel=num_players > 20  # 只有大批量才用并行
    )


def sample_player_preference(
    position: int,
    m: int = 3,
    n: int = 5,
    data_path: str = None,
    positions_path: str = None,
    random_seed: int = None
) -> List[Dict]:
    """
    兼容旧接口的单条采样（使用优化版本）
    """
    return sample_player_preference_fast(
        position=position,
        m=m,
        n=n,
        data_path=data_path,
        positions_path=positions_path,
        random_seed=random_seed
    )


# ============== 辅助函数 ==============

def get_position_heroes(position: int, positions_path: str = None) -> List[str]:
    """获取指定位置的所有英雄名称"""
    _, hero_positions, _, _ = _load_hero_data(positions_path=positions_path)
    position_str = str(position)
    return [
        hero for hero, positions in hero_positions.items() 
        if positions.get(position_str, False)
    ]


# ============== 测试 ==============

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("玩家偏好采样器 - 优化版本测试")
    print("=" * 60)
    
    # 测试各位置英雄数量
    print("\n各位置英雄数量:")
    for pos in range(1, 6):
        heroes = get_position_heroes(pos)
        print(f"  {pos}号位: {len(heroes)}个")
    
    # 预热缓存
    print("\n[预热缓存...]")
    _load_hero_data()
    print("[缓存加载完成]")
    
    # 测试单条采样性能
    print(f"\n{'='*40}")
    print("单条采样性能测试")
    print(f"{'='*40}")
    
    n_iters = 100
    start = time.time()
    for _ in range(n_iters):
        _ = sample_player_preference_fast(position=3, m=3, n=5)
    elapsed = time.time() - start
    print(f"采样 {n_iters} 次耗时: {elapsed:.3f}s")
    print(f"平均每次: {elapsed/n_iters*1000:.2f}ms")
    
    # 测试批量采样性能
    print(f"\n{'='*40}")
    print("批量采样性能测试")
    print(f"{'='*40}")
    
    for batch_size in [10, 100, 1000]:
        start = time.time()
        players = sample_player_preferences_batch(
            num_players=batch_size,
            use_parallel=batch_size > 50
        )
        elapsed = time.time() - start
        print(f"批量采样 {batch_size} 个玩家: {elapsed:.3f}s ({elapsed/batch_size*1000:.2f}ms/个)")
    
    # 显示采样结果示例
    print(f"\n{'='*40}")
    print("采样结果示例")
    print(f"{'='*40}")
    
    for pos in range(1, 6):
        print(f"\n位置 {pos} 号位玩家示例:")
        heroes = sample_player_preference_fast(position=pos, m=3, n=5, random_seed=42 + pos)
        
        print(f"  共 {len(heroes)} 个本命英雄:")
        for h in heroes:
            seed_mark = " [种子]" if h['is_seed'] else ""
            print(f"    - {h['name']}: {h['win_rate']*100:.2f}%{seed_mark}")
    
    # 批量采样示例
    print(f"\n{'='*40}")
    print("批量采样 10 个玩家示例")
    print(f"{'='*40}")
    
    players = sample_player_preferences_batch(
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
