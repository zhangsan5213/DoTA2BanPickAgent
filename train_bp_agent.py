"""BP Agent Training Script - Refactored

This is the refactored version using the new trainer module.
For the original implementation, see train_bp_agent.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from trainer import BPAgentTrainer


def main():
    """Main entry point for training."""
    # Using configs/bp_agent_config.yaml for default configuration
    # Override parameters can be passed as kwargs
    trainer = BPAgentTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
    
    # Example with overrides:
    # trainer = BPAgentTrainer(
    #     epochs=32,
    #     batch_size=32,
    #     samples_per_epoch=2048,
    #     actor_lr=1e-4
    # )
    # trainer.train()
