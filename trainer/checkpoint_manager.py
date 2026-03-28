"""Checkpoint discovery and management."""

import os
from typing import List, Tuple, Optional, Dict
from collections import OrderedDict
import torch

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from utils.device import DEVICE


class CheckpointManager:
    """Manages historical checkpoints for opponent sampling with LRU caching."""

    def __init__(self, checkpoint_dirs: List[str], embed_dim: int = EMBED_DIM,
                 nhead: int = 8, num_layers: int = 4, cache_size: int = 8):
        """
        Args:
            checkpoint_dirs: Directories to scan for checkpoints
            embed_dim: Agent embedding dimension
            nhead: Agent number of attention heads
            num_layers: Agent number of transformer layers
            cache_size: Maximum number of models to keep in LRU cache
        """
        self.checkpoint_dirs = checkpoint_dirs
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self._checkpoints: List[Tuple[str, float]] = []

        # LRU cache for loaded models
        self._model_cache: OrderedDict[str, BPTransformerAgent] = OrderedDict()
        self._cache_size = cache_size

    def discover(self) -> List[Tuple[str, float]]:
        """Discover all .pth checkpoint files from given directories.

        Returns:
            List of (checkpoint_path, mtime) tuples sorted by modification time (newest first).
        """
        self._checkpoints = []
        for d in self.checkpoint_dirs:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.endswith(".pth"):
                    fpath = os.path.join(d, fname)
                    mtime = os.path.getmtime(fpath)
                    self._checkpoints.append((fpath, mtime))
        # Newest first
        self._checkpoints.sort(key=lambda x: x[1], reverse=True)
        return self._checkpoints

    @property
    def checkpoints(self) -> List[Tuple[str, float]]:
        """Get cached checkpoints (discover if empty)."""
        if not self._checkpoints:
            self.discover()
        return self._checkpoints

    def has_checkpoints(self) -> bool:
        """Check if any checkpoints are available."""
        return len(self.checkpoints) > 0

    def load_opponent(self, ckpt_path: str) -> Optional[BPTransformerAgent]:
        """Load an opponent model from checkpoint (with LRU caching).

        Args:
            ckpt_path: Path to checkpoint file

        Returns:
            Loaded opponent model or None if loading failed
        """
        # Check cache first
        if ckpt_path in self._model_cache:
            # Move to end (most recently used)
            self._model_cache.move_to_end(ckpt_path)
            return self._model_cache[ckpt_path]

        # Load from disk
        try:
            opponent = BPTransformerAgent(
                embed_dim=self.embed_dim,
                nhead=self.nhead,
                num_layers=self.num_layers
            ).to(DEVICE)
            opponent.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            opponent.eval()

            # Add to cache
            self._add_to_cache(ckpt_path, opponent)
            return opponent
        except Exception as e:
            print(f"[!] Failed to load opponent from {ckpt_path}: {e}")
            return None

    def _add_to_cache(self, ckpt_path: str, model: BPTransformerAgent):
        """Add model to LRU cache, evicting if necessary."""
        if ckpt_path in self._model_cache:
            self._model_cache.move_to_end(ckpt_path)
            return

        # Evict least recently used if cache is full
        if len(self._model_cache) >= self._cache_size:
            self._model_cache.popitem(last=False)

        self._model_cache[ckpt_path] = model

    def clear_cache(self):
        """Clear the model cache to free memory."""
        self._model_cache.clear()

    def print_summary(self, max_show: int = 5):
        """Print summary of discovered checkpoints."""
        if not self._checkpoints:
            print("[+] No historical checkpoints found")
            return

        print(f"[+] Found {len(self._checkpoints)} historical checkpoints")
        for ckpt, _ in self._checkpoints[:max_show]:
            print(f"    {ckpt}")
        if len(self._checkpoints) > max_show:
            print(f"    ... and {len(self._checkpoints) - max_show} more")
        print(f"[+] Model cache: {len(self._model_cache)}/{self._cache_size} models loaded")
