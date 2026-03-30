"""Configuration management for training."""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import yaml


class TrainingConfig:
    """Training configuration container."""

    def __init__(self, config_path: Optional[str] = None):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = (
                Path(__file__).parent.parent / "configs" / "bp_agent_config.yaml"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Rating config
        rating_cfg = cfg.get("rating", {})
        self.rating_method = rating_cfg.get("method", "elo")
        self.rating_num_opponents = rating_cfg.get("num_opponents", 8)
        self.rating_num_player_sets = rating_cfg.get("num_player_sets", 16)
        self.eval_interval = rating_cfg.get("eval_interval", 8)

        # ELO specific
        elo_cfg = rating_cfg.get("elo", {})
        self.elo_k_factor = elo_cfg.get("k_factor", 32)
        self.elo_opponent_sample_std = elo_cfg.get("opponent_sample_std", 200)

        # TrueSkill specific
        ts_cfg = rating_cfg.get("trueskill", {})
        self.ts_staleness_threshold = ts_cfg.get("staleness_threshold", 5)
        self.ts_num_active_models = ts_cfg.get("num_active_models", 5)

        # Training config
        training_cfg = cfg.get("training", {})
        self.epochs = training_cfg.get("epochs", 32)
        self.batch_size = training_cfg.get("batch_size", 16)
        self.samples_per_epoch = training_cfg.get("samples_per_epoch", 1024)
        self.use_tensorboard = training_cfg.get("use_tensorboard", True)
        self.historical_opponent_prob = training_cfg.get(
            "historical_opponent_prob", 0.6
        )
        self.checkpoint_dirs = training_cfg.get("checkpoint_dirs", [])

        # Model config
        self.actor_lr = float(cfg.get("actor_lr", 3e-4))
        self.value_loss_coeff = float(cfg.get("value_loss_coeff", 2.0))
        self.entropy_loss_coeff = float(cfg.get("entropy_loss_coeff", 0.03))
        self.tensorboard_log_prefix = cfg.get("tensorboard_log_prefix", "bp_agent_exp_")

        # Oracle config
        self.oracle_path = cfg.get(
            "oracle_path",
            "./ckpts/win_rate_oracle-num_heroes_160-text-embd_dim_128-player_attention/win_rate_oracle-20260309033516-000-0.9042.pth",
        )
        self.oracle_embed_dim = cfg.get("oracle_embed_dim", 128)
        self.oracle_nhead = cfg.get("oracle_nhead", 8)
        self.oracle_num_layers = cfg.get("oracle_num_layers", 6)

        # Agent config
        self.agent_embed_dim = cfg.get("agent_embed_dim", 256)
        self.agent_nhead = cfg.get("agent_nhead", 8)
        self.agent_num_layers = cfg.get("agent_num_layers", 4)

        # Runtime overrides
        self._overrides: Dict[str, Any] = {}

    def override(self, **kwargs):
        """Apply runtime overrides."""
        self._overrides.update(kwargs)
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_log_dir(self) -> str:
        """Generate log directory path."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join("runs", f"{self.tensorboard_log_prefix}{timestamp}")

    def get_save_dir(self) -> str:
        """Generate model save directory path."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f"./ckpts/bp_agent-{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        return save_dir
