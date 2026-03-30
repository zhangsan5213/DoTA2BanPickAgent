"""Data generation for training samples."""

from typing import List, Dict, Any
from utils.player_preference_sampler_optimized import (
    sample_player_preferences_batch,
    _load_hero_data,
)
from utils.raw_data import NUM_HEROES


def player_prefs_to_feats(player_prefs) -> List[List[float]]:
    """Convert player preferences to feature vectors [5, NUM_HEROES]."""
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


class DataGenerator:
    """Generates training samples with player preferences."""

    def __init__(self, num_samples: int = 1024):
        self.num_samples = num_samples
        # Preload cache
        _load_hero_data()

    def generate(self, num_samples: int = None) -> List[Dict[str, Any]]:
        """Generate training samples.

        Args:
            num_samples: Number of samples to generate (defaults to self.num_samples)

        Returns:
            List of samples, each containing r_players and d_players
        """
        if num_samples is None:
            num_samples = self.num_samples

        print(
            f"[DataGenerator] Generating {num_samples} training samples with player preferences..."
        )

        # Generate all players at once (10 players per sample: 5 radiant + 5 dire)
        total_players = num_samples * 10
        print(f"[DataGenerator] Generating preferences for {total_players} players...")

        all_players = sample_player_preferences_batch(
            num_players=total_players, m=3, n=5, use_parallel=num_samples > 20
        )

        samples = []
        for i in range(num_samples):
            start_idx = i * 10
            r_players = all_players[start_idx : start_idx + 5]
            d_players = all_players[start_idx + 5 : start_idx + 10]

            sample = {
                "r_players": player_prefs_to_feats(r_players),
                "d_players": player_prefs_to_feats(d_players),
            }
            samples.append(sample)

        print(f"[DataGenerator] Completed generating {len(samples)} samples")
        return samples
