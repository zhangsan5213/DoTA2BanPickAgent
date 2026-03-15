"""BP Dataset for training"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.raw_data import NUM_HEROES
from utils.player_preference_sampler_optimized import (
    sample_player_preferences_batch,
    _load_hero_data
)


class BPDataset(Dataset):
    def __init__(self, data_file=None, num_synthetic=1000):
        """
        Args:
            data_file: 真实比赛数据文件路径 (可选)
            num_synthetic: 使用采样器生成的合成样本数量
        """
        self.samples = []

        # 方法1: 从真实数据加载
        if data_file and data_file != "":
            try:
                with open(data_file, 'r') as f:
                    raw = json.load(f)

                for m in raw:
                    r_picks, d_picks = [], []
                    for act in m.get('picks_bans', []):
                        if act.get('is_pick', False):
                            if act.get('team', 0) == 0:
                                r_picks.append(act['hero_id'])
                            else:
                                d_picks.append(act['hero_id'])

                    if len(r_picks) != 5 or len(d_picks) != 5:
                        continue

                    players = m.get('players', [])
                    r_players = self._get_player_feats([p for p in players if p.get('player_slot', 0) < 128][:5])
                    d_players = self._get_player_feats([p for p in players if p.get('player_slot', 0) >= 128][:5])

                    self.samples.append({
                        'r_picks': r_picks,
                        'd_picks': d_picks,
                        'r_players': r_players,
                        'd_players': d_players,
                    })
                print(f"[BPDataset] Loaded {len(self.samples)} samples from real data")
            except Exception as e:
                print(f"[BPDataset] Failed to load real data: {e}")

        # 方法2: 使用玩家偏好采样器生成合成数据
        if num_synthetic > 0:
            synthetic_samples = self._generate_synthetic_samples_fast(num_synthetic)
            self.samples.extend(synthetic_samples)
            print(f"[BPDataset] Generated {len(synthetic_samples)} synthetic samples")

        print(f"[BPDataset] Total samples: {len(self.samples)}")

    def _get_player_feats(self, players):
        """从真实比赛数据提取玩家特征"""
        feats = []
        for p in players[:5]:
            history = p.get('hero_history', {})
            total = sum(h.get('games', 0) for h in history.values())
            vec = [0.0] * NUM_HEROES
            if total >= 10:
                for hid, stats in history.items():
                    try:
                        h = int(hid)
                        games = stats.get('games', 0)
                        wins = stats.get('wins', 0)
                        if 0 < h <= NUM_HEROES and games >= 3:
                            vec[h] = wins / games
                    except:
                        pass
            feats.append(vec)
        while len(feats) < 5:
            feats.append([0.0] * NUM_HEROES)
        return feats

    def _generate_synthetic_samples_fast(self, num_samples):
        """
        使用优化后的采样器生成合成样本
        
        优化点：
        1. 预加载缓存数据（只加载一次）
        2. 分批次批量生成（减少函数调用开销）
        3. 控制内存占用（每批最多生成100个样本 = 1000个玩家）
        4. 使用多进程并行（大批量时自动启用）
        
        Args:
            num_samples: 需要生成的样本数量
        Returns:
            list: 样本列表
        """
        # 预加载缓存（避免每个batch重复加载IO）
        _load_hero_data()
        
        samples = []
        
        # 分批次生成，每批100个样本（1000个玩家）
        # 这样既减少函数调用开销，又不会占用过多内存
        BATCH_SIZE = 100
        
        # 使用tqdm显示总体进度
        pbar = tqdm(total=num_samples, desc="Generating synthetic samples", ncols=90)
        
        num_generated = 0
        while num_generated < num_samples:
            # 计算当前batch的大小（处理最后一批可能不足的情况）
            current_batch_size = min(BATCH_SIZE, num_samples - num_generated)
            
            # 一次性生成当前batch的所有玩家（两队各5人）
            total_players_needed = current_batch_size * 10  # 5 radiant + 5 dire per sample
            
            try:
                all_players = sample_player_preferences_batch(
                    num_players=total_players_needed,
                    m=3,
                    n=5,
                    # 大批量时使用多进程并行加速
                    use_parallel=current_batch_size > 20
                )
            except Exception as e:
                # 如果批量生成失败，降级为单条生成（更稳定）
                print(f"\n[WARN] Batch generation failed ({e}), falling back to single generation")
                all_players = []
                for player_idx in range(total_players_needed):
                    from utils.player_preference_sampler_optimized import sample_player_preference_fast
                    # 随机分配位置（1-5均匀分布）
                    position = np.random.randint(1, 6)
                    heroes = sample_player_preference_fast(position=position, m=3, n=5)
                    # 保持与batch接口一致的返回格式
                    all_players.append({
                        'player_id': player_idx,
                        'position': position,
                        'heroes': heroes
                    })
            
            # 验证生成的玩家数量是否正确
            if len(all_players) != total_players_needed:
                raise RuntimeError(
                    f"Batch generation returned wrong number of players: "
                    f"expected {total_players_needed}, got {len(all_players)}"
                )
            
            # 拆分成 samples（每10个玩家 = 1个样本）
            for i in range(current_batch_size):
                start_idx = i * 10
                # 前5个是天辉，后5个是夜魇
                r_players = all_players[start_idx:start_idx + 5]
                d_players = all_players[start_idx + 5:start_idx + 10]
                
                # 转换为特征向量
                sample = {
                    'r_picks': [],
                    'd_picks': [],
                    'r_players': self._player_prefs_to_feats(r_players),
                    'd_players': self._player_prefs_to_feats(d_players),
                }
                samples.append(sample)
            
            num_generated += current_batch_size
            pbar.update(current_batch_size)
        
        pbar.close()
        
        # 最终验证
        if len(samples) != num_samples:
            raise RuntimeError(
                f"Generated wrong number of samples: expected {num_samples}, got {len(samples)}"
            )
        
        return samples

    def _player_prefs_to_feats(self, player_prefs):
        """将玩家偏好转换为特征向量 [5, NUM_HEROES]"""
        feats = []
        for p in player_prefs:
            vec = [0.0] * NUM_HEROES
            for h in p['heroes']:
                hero_id = h['id']
                win_rate = h['win_rate']
                if 0 < hero_id <= NUM_HEROES:
                    vec[hero_id - 1] = win_rate  # hero_id是1-based，转为0-based索引
            feats.append(vec)
        while len(feats) < 5:
            feats.append([0.0] * NUM_HEROES)
        return feats

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    import time
    
    print("=" * 50)
    print("Testing BPDataset (Optimized)")
    print("=" * 50)

    # Test synthetic data generation with timing
    print("\n[测试合成数据生成性能]")
    
    for n_samples in [10, 50, 100]:
        print(f"\n生成 {n_samples} 个样本...")
        start = time.time()
        dataset = BPDataset(data_file="", num_synthetic=n_samples)
        elapsed = time.time() - start
        print(f"  耗时: {elapsed:.3f}s ({elapsed/n_samples*1000:.2f}ms/样本)")
        print(f"  Dataset size: {len(dataset)}")
    
    # Check sample structure
    dataset = BPDataset(data_file="", num_synthetic=10)
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"r_players shape: {len(sample['r_players'])} x {len(sample['r_players'][0])}")
    print(f"d_players shape: {len(sample['d_players'])} x {len(sample['d_players'][0])}")

    # Check player features
    r_players = sample['r_players']
    non_zero_count = sum(1 for p in r_players for v in p if v > 0)
    print(f"Non-zero entries in r_players: {non_zero_count}")

    # Show some hero preferences
    print("\nSample player 0 hero preferences (non-zero):")
    for i, wr in enumerate(r_players[0]):
        if wr > 0:
            print(f"  Hero {i+1}: {wr:.4f}")

    print("\n[OK] Dataset test passed!")
