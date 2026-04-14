"""Model initialization utilities."""

import os
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR, ReduceLROnPlateau

from model.bp_agent import BPTransformerAgent, EMBED_DIM
from model.win_rate_oracle import WinRateOracle
from utils.device import DEVICE


def initialize_scheduler(optimizer, config):
    """Initialize learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        config: TrainingConfig instance
        
    Returns:
        Initialized scheduler or None if disabled
    """
    if not getattr(config, "lr_scheduler_enabled", False):
        return None
    
    scheduler_type = getattr(config, "lr_scheduler_type", "cosine")
    params = getattr(config, "lr_scheduler_params", {})
    
    if scheduler_type == "cosine":
        T_max = params.get("T_max", 32)
        eta_min = params.get("eta_min", 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        print(f"[+] Initialized CosineAnnealingLR scheduler (T_max={T_max}, eta_min={eta_min:.2e})")
        
    elif scheduler_type == "exponential":
        gamma = params.get("gamma", 0.95)
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        print(f"[+] Initialized ExponentialLR scheduler (gamma={gamma})")
        
    elif scheduler_type == "step":
        step_size = params.get("step_size", 10)
        gamma = params.get("gamma", 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"[+] Initialized StepLR scheduler (step_size={step_size}, gamma={gamma})")
        
    elif scheduler_type == "plateau":
        mode = params.get("mode", "min")
        factor = params.get("factor", 0.5)
        patience = params.get("patience", 3)
        min_lr = params.get("min_lr", 1e-7)
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr, verbose=True
        )
        print(f"[+] Initialized ReduceLROnPlateau scheduler (mode={mode}, factor={factor}, patience={patience}, min_lr={min_lr:.2e})")
        
    else:
        print(f"[!] Unknown scheduler type: {scheduler_type}, using no scheduler")
        return None
    
    return scheduler


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
        num_layers=config.agent_num_layers,
        learnable_temperature=config.agent_learnable_temperature,
    ).to(DEVICE)
    
    # Override temperature initial value if config specifies a non-default value
    if config.agent_temperature != 1.0:
        agent.temperature.data.fill_(config.agent_temperature)
    
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
