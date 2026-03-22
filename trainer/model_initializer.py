"""Model initialization utilities."""

import os
import torch

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from model.win_rate_oracle import WinRateOracle
from utils.device import DEVICE


def initialize_oracle(config) -> WinRateOracle:
    """Initialize and load win rate oracle.
    
    Args:
        config: TrainingConfig instance
        
    Returns:
        Loaded and eval-mode oracle
    """
    oracle = WinRateOracle(
        embed_dim=config.oracle_embed_dim,
        nhead=config.oracle_nhead,
        num_layers=config.oracle_num_layers,
        use_text=True,
        use_player_heroes=True
    ).to(DEVICE)
    
    if os.path.exists(config.oracle_path):
        oracle.load_state_dict(torch.load(config.oracle_path, map_location=DEVICE))
        print(f"[+] Loaded oracle from {config.oracle_path}")
    else:
        print(f"[!] Oracle not found at {config.oracle_path}")
    
    oracle.eval()
    return oracle


def initialize_agent(config) -> BPTransformerAgent:
    """Initialize BP Agent.
    
    Args:
        config: TrainingConfig instance
        
    Returns:
        Initialized agent
    """
    agent = BPTransformerAgent(
        embed_dim=config.agent_embed_dim or EMBED_DIM,
        nhead=config.agent_nhead,
        num_layers=config.agent_num_layers
    ).to(DEVICE)
    
    return agent


def initialize_optimizer(agent, config):
    """Initialize optimizer for agent.
    
    Args:
        agent: Agent model
        config: TrainingConfig instance
        
    Returns:
        Initialized optimizer
    """
    from torch.optim import AdamW
    return AdamW(agent.parameters(), lr=config.actor_lr)
