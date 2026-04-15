"""BP Agent Training Script - Refactored

This is the refactored version using the new trainer module.
For the original implementation, see train_bp_agent.py
"""

import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from trainer import BPAgentTrainer


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train BP Agent")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from",
    )
    args = parser.parse_args()

    trainer = BPAgentTrainer(resume_from=args.resume)
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
