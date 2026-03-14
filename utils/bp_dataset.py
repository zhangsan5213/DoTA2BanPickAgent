"""BP Dataset for training"""

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.raw_data import NUM_HEROES
from utils.player_preference_sampler import batch_sample_player_preferences


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
            synthetic_samples = self._generate_synthetic_samples(num_synthetic)
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

    def _generate_synthetic_samples(self, num_samples):
        """使用玩家偏好采样器生成合成样本（并行化）"""

        def generate_one_sample(_):
            # TODO: TOO FUCKING SLOW
            r_players = batch_sample_player_preferences(
                num_players=5, m=3, n=5, random_seed=None
            )
            d_players = batch_sample_player_preferences(
                num_players=5, m=3, n=5, random_seed=None
            )
            return {
                'r_picks': [],
                'd_picks': [],
                'r_players': self._player_prefs_to_feats(r_players),
                'd_players': self._player_prefs_to_feats(d_players),
            }

        # 使用线程池并行生成
        samples = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(generate_one_sample, i) for i in range(num_samples)]
            for future in tqdm(as_completed(futures), total=num_samples, desc="Generating synthetic samples", ncols=90):
                samples.append(future.result())

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
    print("=" * 50)
    print("Testing BPDataset")
    print("=" * 50)

    # Test synthetic data generation
    dataset = BPDataset(data_file="", num_synthetic=10)
    print(f"Dataset size: {len(dataset)}")

    # Check sample structure
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
