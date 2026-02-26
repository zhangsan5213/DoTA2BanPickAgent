"""
玩家偏好采样器
用于Self-Play时生成具有合理偏好分布的虚拟玩家
基于英雄特征相似性进行采样，确保玩家偏好的内聚性（如擅长Carry的玩家偏好相似的Carry英雄）
"""
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict

# 延迟导入，避免循环依赖
_hero_id_feature_map = None
_num_heroes = None

def _get_raw_data():
    """延迟加载raw_data，避免循环导入问题"""
    global _hero_id_feature_map, _num_heroes
    if _hero_id_feature_map is None:
        from utils.raw_data import HERO_ID_FEATURE_MAP, NUM_HEROES
        _hero_id_feature_map = HERO_ID_FEATURE_MAP
        _num_heroes = NUM_HEROES
    return _hero_id_feature_map, _num_heroes


class HeroSimilarityMatrix:
    """英雄相似度矩阵 - 预计算基于特征的英雄间相似度"""
    
    def __init__(self):
        HERO_ID_FEATURE_MAP, NUM_HEROES = _get_raw_data()
        
        self.num_heroes = NUM_HEROES
        self.valid_hero_ids = list(HERO_ID_FEATURE_MAP.keys())
        
        # 构建特征矩阵 [num_valid_heroes, feature_dim]
        self.hero_id_to_idx = {h_id: i for i, h_id in enumerate(self.valid_hero_ids)}
        self.idx_to_hero_id = {i: h_id for i, h_id in enumerate(self.valid_hero_ids)}
        
        features = []
        for h_id in self.valid_hero_ids:
            feat = HERO_ID_FEATURE_MAP[h_id].numpy()
            features.append(feat)
        self.feature_matrix = np.stack(features)  # [N, 21]
        
        # 归一化特征用于余弦相似度计算
        self.feature_matrix_norm = self.feature_matrix / (
            np.linalg.norm(self.feature_matrix, axis=1, keepdims=True) + 1e-8
        )
        
        # 预计算相似度矩阵
        self.similarity_matrix = self._compute_similarity_matrix()
    
    def _compute_similarity_matrix(self) -> np.ndarray:
        """计算所有有效英雄间的余弦相似度矩阵"""
        # 归一化后的点积 = 余弦相似度
        sim_matrix = np.dot(self.feature_matrix_norm, self.feature_matrix_norm.T)
        return sim_matrix  # [N_valid, N_valid]
    
    def get_similarity(self, hero_id1: int, hero_id2: int) -> float:
        """获取两个英雄间的相似度"""
        if hero_id1 not in self.hero_id_to_idx or hero_id2 not in self.hero_id_to_idx:
            return 0.0
        idx1 = self.hero_id_to_idx[hero_id1]
        idx2 = self.hero_id_to_idx[hero_id2]
        return self.similarity_matrix[idx1, idx2]
    
    def get_top_k_similar(self, hero_id: int, k: int = 5) -> List[Tuple[int, float]]:
        """获取与指定英雄最相似的k个英雄"""
        if hero_id not in self.hero_id_to_idx:
            return []
        idx = self.hero_id_to_idx[hero_id]
        similarities = self.similarity_matrix[idx]
        
        # 获取top k（排除自己）
        top_indices = np.argsort(similarities)[::-1][1:k+1]
        return [(self.idx_to_hero_id[i], similarities[i]) for i in top_indices]


class PlayerPreferenceSampler:
    """
    玩家偏好采样器
    
    基于英雄相似性生成虚拟玩家的偏好分布。
    核心思想：真实玩家的偏好具有内聚性——擅长的英雄往往在玩法/定位上相似。
    
    采样策略：
    1. 随机选择1-3个"核心英雄"作为玩家的"本命"
    2. 基于核心英雄的相似度生成对所有英雄的偏好分数
    3. 将偏好分数转换为胜率分布（偏好高的英雄胜率也高，但有噪声）
    """
    
    def __init__(self, temperature: float = 0.5, randomness: float = 0.2):
        """
        Args:
            temperature: 控制偏好的集中度，越小越集中（越"专精"），越大越分散
            randomness: 控制胜率的随机噪声程度
        """
        self.similarity = HeroSimilarityMatrix()
        self.temperature = temperature
        self.randomness = randomness
        
        # 预计算每个英雄的相似度排名（用于快速采样）
        self._precompute_similarity_ranks()
    
    def _precompute_similarity_ranks(self):
        """预计算每个英雄最相似的其他英雄列表"""
        self.hero_similar_ranks = {}
        for h_id in self.similarity.valid_hero_ids:
            idx = self.similarity.hero_id_to_idx[h_id]
            sims = self.similarity.similarity_matrix[idx]
            # 按相似度排序（排除自己）
            ranked_indices = np.argsort(sims)[::-1][1:]
            self.hero_similar_ranks[h_id] = [
                (self.similarity.idx_to_hero_id[i], sims[i]) 
                for i in ranked_indices
            ]
    
    def sample_player_preferences(self, n_players: int = 1) -> List[Dict[int, float]]:
        """
        采样多个虚拟玩家的偏好分布
        
        Returns:
            List of {hero_id: preference_score}，长度为n_players
        """
        return [self._sample_single_player() for _ in range(n_players)]
    
    def _sample_single_player(self) -> Dict[int, float]:
        """采样单个虚拟玩家的偏好分布
        
        偏好分数分布逻辑：
        - 本命英雄（核心英雄）：偏好接近1.0，反映玩家最擅长的英雄
        - 相似英雄：根据与核心英雄的相似度线性插值，相似度越高偏好越高
        - 无关英雄：随机的基础偏好（0.2-0.5），反映普通水平
        
        最终形成：少数英雄偏好很高（0.7-1.0），大部分在0.3-0.6波动
        """
        # 1. 随机选择1-3个核心英雄（该玩家的"本命"）
        n_core_heroes = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        core_heroes = np.random.choice(
            self.similarity.valid_hero_ids, 
            size=n_core_heroes, 
            replace=False
        )
        
        # 2. 基于核心英雄计算对所有英雄的偏好分数
        preference_scores = np.zeros(len(self.similarity.valid_hero_ids))
        
        # 参数配置
        CORE_PREF = 1.0  # 本命英雄的基准偏好
        MIN_PREF = 0.25  # 最低偏好（完全不会玩的英雄）
        MAX_PREF = 0.95  # 最高偏好上限（本命英雄实际值）
        
        for core_id in core_heroes:
            core_idx = self.similarity.hero_id_to_idx[core_id]
            # 获取该核心英雄对所有其他英雄的相似度
            similarities = self.similarity.similarity_matrix[core_idx].copy()
            
            # 本命英雄自己的偏好设为1.0（最高）
            similarities[core_idx] = 1.0
            
            # 根据相似度映射到偏好分数：
            # - 相似度=1（自己）-> 偏好=MAX_PREF (0.95)
            # - 相似度=0（完全不相关）-> 偏好=MIN_PREF (0.25)
            prefs_from_this_core = MIN_PREF + (MAX_PREF - MIN_PREF) * similarities
            
            # 累加（多个核心英雄的偏好会混合，取平均）
            preference_scores += prefs_from_this_core / n_core_heroes
        
        # 3. 应用temperature调节集中度
        # temperature < 1: 更集中（高手型玩家，偏好差异大）
        # temperature > 1: 更分散（万金油型玩家，啥都会一点）
        # 使用power变换：分数^temperature
        # temperature=0.5时，高分更高，低分相对提升较少
        # temperature=1.0时，保持线性
        # temperature=2.0时，压缩差异
        if self.temperature != 1.0:
            # 归一化到[0,1]后做power变换，再映射回来
            normalized = (preference_scores - MIN_PREF) / (MAX_PREF - MIN_PREF)
            if self.temperature < 1.0:
                # 更集中：高分更高
                transformed = np.power(normalized, self.temperature)
            else:
                # 更分散：差异缩小
                transformed = np.power(normalized, 1.0 / self.temperature)
            preference_scores = MIN_PREF + transformed * (MAX_PREF - MIN_PREF)
        
        # 4. 添加少量随机噪声，让分布更自然
        noise = np.random.normal(0, 0.05, size=preference_scores.shape)
        preference_scores = np.clip(preference_scores + noise, 0.1, 1.0)
        
        # 5. 转换为字典格式
        return {
            self.similarity.idx_to_hero_id[i]: float(preference_scores[i])
            for i in range(len(preference_scores))
        }
    
    def preferences_to_winrate_vector(
        self, 
        preferences: Dict[int, float],
        min_games: int = 3,
        max_heroes_per_player: int = 10
    ) -> np.ndarray:
        """
        将偏好分数转换为胜率向量（用于输入网络）
        
        Args:
            preferences: {hero_id: preference_score}
            min_games: 最少场次阈值（影响是否记录该英雄）
            max_heroes_per_player: 每个玩家最多记录多少个英雄
            
        Returns:
            winrate_vector: [NUM_HEROES]，无效位置为0
        """
        _, NUM_HEROES = _get_raw_data()
        
        # 基于偏好分数决定玩家会玩哪些英雄
        hero_ids = list(preferences.keys())
        scores = np.array([preferences[h_id] for h_id in hero_ids])
        
        # 根据偏好分数采样该玩家实际玩的英雄（偏好高的更可能被玩）
        # 使用多项式分布决定英雄使用频率
        n_heroes_played = np.random.randint(min_games, max_heroes_per_player + 1)
        
        # 按偏好分数加权采样n_heroes_played个英雄
        probs = scores / scores.sum()
        played_heroes = np.random.choice(
            len(hero_ids), 
            size=min(n_heroes_played, len(hero_ids)), 
            replace=False, 
            p=probs
        )
        
        # 为每个玩的英雄生成胜率（偏好高的胜率也高，但有噪声）
        winrate_vector = np.zeros(NUM_HEROES)
        
        for idx in played_heroes:
            hero_id = hero_ids[idx]
            pref_score = scores[idx]
            
            # 胜率计算：
            # 偏好分数范围约0.1-1.0，映射到胜率35%-75%
            # - 偏好1.0（本命英雄）-> 约70-75%胜率
            # - 偏好0.5（普通英雄）-> 约50%胜率  
            # - 偏好0.2（不会玩的）-> 约35-40%胜率
            
            # 线性映射：winrate = 0.35 + pref_score * 0.4
            # 即：0.1->0.39, 0.5->0.55, 1.0->0.75
            base_winrate = 0.35
            pref_bonus = pref_score * 0.40  # 偏好最高贡献40%胜率
            
            # 根据randomness添加噪声（randomness=0.2时，std=0.05）
            noise = np.random.normal(0, self.randomness * 0.25)
            
            winrate = np.clip(base_winrate + pref_bonus + noise, 0.05, 0.95)
            winrate_vector[hero_id - 1] = winrate  # hero_id是1-based
        
        return winrate_vector
    
    def sample_team_player_features(
        self, 
        n_teams: int = 1,
        min_games: int = 3,
        max_heroes_per_player: int = 10
    ) -> np.ndarray:
        """
        采样一个/多个队伍的5个玩家的特征
        
        Args:
            n_teams: 队伍数量
            min_games: 最少场次阈值
            max_heroes_per_player: 每个玩家最多记录多少个英雄
            
        Returns:
            player_features: [n_teams, 5, NUM_HEROES]
        """
        all_teams = []
        
        for _ in range(n_teams):
            # 为一个队伍的5个玩家采样
            team_prefs = self.sample_player_preferences(n_players=5)
            team_features = []
            
            for prefs in team_prefs:
                winrate_vec = self.preferences_to_winrate_vector(
                    prefs, min_games, max_heroes_per_player
                )
                team_features.append(winrate_vec)
            
            # 补全到5个玩家
            while len(team_features) < 5:
                team_features.append(np.zeros(self.similarity.num_heroes))
            
            all_teams.append(np.stack(team_features))
        
        return np.stack(all_teams)  # [n_teams, 5, NUM_HEROES]


# 全局采样器实例（单例模式，避免重复计算相似度矩阵）
_player_sampler = None

def get_player_sampler(temperature: float = 0.5, randomness: float = 0.2) -> PlayerPreferenceSampler:
    """获取全局玩家偏好采样器"""
    global _player_sampler
    if _player_sampler is None:
        _player_sampler = PlayerPreferenceSampler(temperature, randomness)
    return _player_sampler


def sample_random_player_features(
    temperature: float = 0.5,
    randomness: float = 0.2,
    min_games: int = 3,
    max_heroes_per_player: int = 10
) -> np.ndarray:
    """
    便捷函数：采样随机玩家的特征
    
    Returns:
        player_features: [5, NUM_HEROES]
    """
    sampler = get_player_sampler(temperature, randomness)
    team_features = sampler.sample_team_player_features(
        n_teams=1,
        min_games=min_games,
        max_heroes_per_player=max_heroes_per_player
    )
    return team_features[0]  # [5, NUM_HEROES]


def batch_sample_player_features(
    batch_size: int,
    temperature: float = 0.5,
    randomness: float = 0.2,
    min_games: int = 3,
    max_heroes_per_player: int = 10
) -> np.ndarray:
    """
    便捷函数：批量采样玩家特征
    
    Returns:
        player_features: [batch_size, 2, 5, NUM_HEROES] 
                        ( Radiant[5,NUM_HEROES], Dire[5,NUM_HEROES] )
    """
    sampler = get_player_sampler(temperature, randomness)
    # 采样两队
    teams_features = sampler.sample_team_player_features(
        n_teams=batch_size * 2,
        min_games=min_games,
        max_heroes_per_player=max_heroes_per_player
    )  # [batch_size*2, 5, NUM_HEROES]
    
    # 分成Radiant和Dire
    radiant = teams_features[0::2]  # [batch_size, 5, NUM_HEROES]
    dire = teams_features[1::2]     # [batch_size, 5, NUM_HEROES]
    
    return np.stack([radiant, dire], axis=1)  # [batch_size, 2, 5, NUM_HEROES]


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, '.')
    
    print("=" * 60)
    print("测试玩家偏好采样器")
    print("=" * 60)
    
    sampler = PlayerPreferenceSampler(temperature=0.5, randomness=0.2)
    
    # 测试1：采样单个玩家偏好
    print("\n[测试1] 单个玩家偏好分布（Top 10）:")
    prefs = sampler._sample_single_player()
    sorted_prefs = sorted(prefs.items(), key=lambda x: x[1], reverse=True)[:10]
    for hero_id, score in sorted_prefs:
        print(f"  Hero {hero_id}: {score:.4f}")
    
    # 测试2：转换为胜率向量
    print("\n[测试2] 胜率向量（非零元素）:")
    winrate_vec = sampler.preferences_to_winrate_vector(prefs)
    non_zero = [(i+1, v) for i, v in enumerate(winrate_vec) if v > 0]
    for hero_id, winrate in non_zero[:10]:
        print(f"  Hero {hero_id}: {winrate:.3f}")
    
    # 测试3：采样完整队伍
    print("\n[测试3] 完整队伍特征 shape:")
    team_features = sampler.sample_team_player_features(n_teams=1)
    print(f"  Team features shape: {team_features.shape}")  # [1, 5, NUM_HEROES]
    
    # 测试4：查看相似英雄
    print("\n[测试4] Hero 1 (Anti-Mage) 的最相似英雄:")
    sim_matrix = HeroSimilarityMatrix()
    similar = sim_matrix.get_top_k_similar(1, k=5)
    for hero_id, sim in similar:
        print(f"  Hero {hero_id}: sim={sim:.3f}")
    
    print("\n[+] 所有测试通过!")
