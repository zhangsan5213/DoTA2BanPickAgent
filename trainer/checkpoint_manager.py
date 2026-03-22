"""Checkpoint discovery and management."""

import os
from typing import List, Tuple, Optional
import torch

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from utils.device import DEVICE


class CheckpointManager:
    """Manages historical checkpoints for opponent sampling."""
    
    def __init__(self, checkpoint_dirs: List[str], embed_dim: int = EMBED_DIM, 
                 nhead: int = 8, num_layers: int = 4):
        self.checkpoint_dirs = checkpoint_dirs
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self._checkpoints: List[Tuple[str, float]] = []
    
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
        """Load an opponent model from checkpoint.
        
        Args:
            ckpt_path: Path to checkpoint file
            
        Returns:
            Loaded opponent model or None if loading failed
        """
        try:
            opponent = BPTransformerAgent(
                embed_dim=self.embed_dim, 
                nhead=self.nhead, 
                num_layers=self.num_layers
            ).to(DEVICE)
            opponent.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            opponent.eval()
            return opponent
        except Exception as e:
            print(f"[!] Failed to load opponent from {ckpt_path}: {e}")
            return None
    
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
