"""Main BP Agent Trainer."""

import os
import math
from typing import Optional, Dict, Any

import torch
from torch.optim import AdamW

from .config import TrainingConfig
from .data_generator import DataGenerator
from .checkpoint_manager import CheckpointManager
from .rollout_collector import RolloutCollector
from .loss_computer import LossComputer
from .epoch_runner import EpochRunner
from .evaluator import EvaluatorManager, save_checkpoint
from .tensorboard_logger import TensorBoardLogger
from .model_initializer import initialize_oracle, initialize_agent, initialize_optimizer, initialize_scheduler


class EntropyAnnealer:
    """Entropy系数退火器，随训练进行逐渐降低探索度。"""
    
    def __init__(self, config):
        """初始化退火器。
        
        Args:
            config: TrainingConfig实例
        """
        self.enabled = config.entropy_annealing_enabled
        self.initial = config.entropy_initial_coeff
        self.final = config.entropy_final_coeff
        self.type = config.entropy_annealing_type
        self.total_epochs = config.entropy_annealing_epochs
        self.warmup_steps = config.entropy_warmup_steps
        self.annealing_steps = config.entropy_annealing_steps
        self.current_step = 0
        
        if self.enabled:
            print(f"[+] Entropy Annealing enabled: {self.initial:.4f} -> {self.final:.4f} ({self.type})")
    
    def get_coeff(self, epoch=None, step=None):
        """获取当前entropy系数。
        
        Args:
            epoch: 当前epoch（用于按epoch退火）
            step: 当前step（用于按step退火，优先）
        
        Returns:
            当前entropy系数，如果退火禁用则返回None
        """
        if not self.enabled:
            return None  # 使用配置中的固定值
        
        if step is not None:
            self.current_step = step
        
        # Warmup阶段
        if self.current_step < self.warmup_steps:
            return self.initial
        
        # 计算退火进度
        if self.annealing_steps > 0:
            progress = min(1.0, (self.current_step - self.warmup_steps) / self.annealing_steps)
        elif epoch is not None and self.total_epochs > 0:
            progress = min(1.0, epoch / self.total_epochs)
        else:
            return self.final
        
        # 根据类型计算当前系数
        if self.type == "linear":
            return self.initial + (self.final - self.initial) * progress
        elif self.type == "exponential":
            return self.initial * (self.final / self.initial) ** progress
        elif self.type == "cosine":
            return self.final + (self.initial - self.final) * (1 + math.cos(math.pi * progress)) / 2
        else:
            return self.initial
    
    def step(self):
        """增加步数计数器。"""
        self.current_step += 1
    
    def get_progress(self):
        """获取当前退火进度（0-1）。"""
        if not self.enabled:
            return 0.0
        if self.current_step < self.warmup_steps:
            return 0.0
        if self.annealing_steps > 0:
            return min(1.0, (self.current_step - self.warmup_steps) / self.annealing_steps)
        return 0.0


class BPAgentTrainer:
    """Main trainer for BP Agent using PPO."""

    def __init__(self, config_path: Optional[str] = None, **override_kwargs):
        """Initialize trainer.

        Args:
            config_path: Path to YAML config file
            **override_kwargs: Runtime config overrides
        """
        # Load configuration
        self.config = TrainingConfig(config_path)
        self.config.override(**override_kwargs)

        # Initialize components (will be set in setup)
        self.agent = None
        self.oracle = None
        self.optimizer = None
        self.scheduler = None
        self.data_generator = None
        self.checkpoint_manager = None
        self.rollout_collector = None
        self.loss_computer = None
        self.epoch_runner = None
        self.evaluator = None
        self.logger = None
        self.writer = None
        self.save_dir = None

        self._setup_complete = False

    def setup(self):
        """Setup all training components."""
        if self._setup_complete:
            return

        print("[+] Setting up trainer...")

        # Create save directory
        self.save_dir = self.config.get_save_dir()
        print(f"[+] Models will be saved to: {self.save_dir}")

        # Initialize models
        self.oracle = initialize_oracle(self.config)
        self.agent = initialize_agent(self.config)
        self.optimizer = initialize_optimizer(self.agent, self.config)

        # Initialize data generator
        self.data_generator = DataGenerator(self.config.samples_per_epoch)

        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dirs,
            embed_dim=self.config.agent_embed_dim,
            nhead=self.config.agent_nhead,
            num_layers=self.config.agent_num_layers,
        )
        self.checkpoint_manager.discover()
        self.checkpoint_manager.print_summary()

        # Initialize rollout collector
        rating_manager = None
        if hasattr(self.evaluator, 'rating_evaluator') and hasattr(self.evaluator.rating_evaluator, 'rating_manager'):
            rating_manager = self.evaluator.rating_evaluator.rating_manager

        self.rollout_collector = RolloutCollector(
            self.agent,
            self.oracle,
            historical_prob=self.config.historical_opponent_prob,
            embed_dim=self.config.agent_embed_dim,
            nhead=self.config.agent_nhead,
            num_layers=self.config.agent_num_layers,
            temperature=self.config.agent_temperature,
            policy_staleness_tolerance=self.config.policy_staleness_tolerance,
            rating_manager=rating_manager,
            num_strata=getattr(self.config, 'num_strata', 3),
        )

        # Initialize loss computer
        self.loss_computer = LossComputer(
            self.agent,
            value_loss_coeff=self.config.value_loss_coeff,
            entropy_loss_coeff=self.config.entropy_loss_coeff,
            clip_eps=self.config.clip_ratio,
            value_clip_eps=self.config.value_clip_ratio,
        )

        # Initialize scheduler
        self.scheduler = initialize_scheduler(self.optimizer, self.config)

        # Initialize entropy annealer
        self.entropy_annealer = EntropyAnnealer(self.config)

        # Initialize epoch runner
        self.epoch_runner = EpochRunner(
            self.agent,
            self.optimizer,
            self.loss_computer,
            self.rollout_collector,
            self.checkpoint_manager,
            self.config,
            entropy_annealer=self.entropy_annealer,
        )

        # Initialize evaluator
        self.evaluator = EvaluatorManager(self.config, self.oracle, self.save_dir)

        # Initialize logger
        log_dir = self.config.get_log_dir()
        self.logger = TensorBoardLogger(
            log_dir=log_dir, enabled=self.config.use_tensorboard
        )

        print(f"[+] Using {self.evaluator.method_name} rating system for evaluation")
        self._setup_complete = True

    def train(self):
        """Run training loop."""
        self.setup()

        # Start TensorBoard
        self.writer = self.logger.start()
        self.evaluator.set_writer(self.writer)

        epochs = self.config.epochs
        print(f"[+] Training started for {epochs} epochs...")

        # Training loop
        for epoch in range(epochs):
            print(f"\n[Epoch {epoch + 1}/{epochs}]")
            # Generate training data for this epoch
            print(f"[+] Generating {self.config.samples_per_epoch} training samples...")
            samples = self.data_generator.generate()

            # Run epoch
            epoch_stats = self.epoch_runner.run(
                epoch=epoch, total_epochs=epochs, samples=samples, writer=self.writer
            )

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(epoch_stats['total_loss'])
                else:
                    self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']

            print(
                f"[+] Epoch {epoch + 1}/{epochs} - "
                f"Loss: {epoch_stats['total_loss']:.4f}, "
                f"Policy: {epoch_stats['policy_loss']:.4f}, "
                f"Value: {epoch_stats['value_loss']:.4f}, "
                f"Entropy: {epoch_stats['entropy_loss']:.4f}, "
                f"KL: {epoch_stats['kl_div']:.4f}, "
                f"Rollouts: {epoch_stats['num_rollouts']}, "
                f"LR: {current_lr:.2e}"
            )

            # Log epoch summary to TensorBoard
            if self.writer is not None:
                self.writer.add_scalar(
                    "Epoch/total_loss", epoch_stats["total_loss"], epoch + 1
                )
                self.writer.add_scalar(
                    "Epoch/policy_loss", epoch_stats["policy_loss"], epoch + 1
                )
                self.writer.add_scalar(
                    "Epoch/value_loss", epoch_stats["value_loss"], epoch + 1
                )
                self.writer.add_scalar(
                    "Epoch/entropy_loss", epoch_stats["entropy_loss"], epoch + 1
                )
                self.writer.add_scalar("Epoch/kl_div", epoch_stats["kl_div"], epoch + 1)
                self.writer.add_scalar(
                    "Epoch/num_rollouts", epoch_stats["num_rollouts"], epoch + 1
                )
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar("Training/lr_epoch", current_lr, epoch + 1)
                self.writer.flush()

            # Periodic evaluation
            if self.evaluator.should_evaluate(epoch):
                print(f"\n[+] Evaluating at epoch {epoch + 1}...")
                checkpoint_path = save_checkpoint(self.agent, self.save_dir, epoch + 1)
                self.evaluator.evaluate(checkpoint_path, epoch + 1)

        # Save final model
        final_path = self._save_final_model()

        # Final evaluation
        self.evaluator.final_evaluation(final_path, epochs)

        print("\n[+] Training completed successfully!")

    def _save_final_model(self) -> str:
        """Save final model checkpoint.

        Returns:
            Path to saved model
        """
        model_path = f"{self.save_dir}/bp_agent_final.pth"
        torch.save(self.agent.state_dict(), model_path)
        print(f"[+] Model saved to {model_path}")
        return model_path

    def close(self):
        """Cleanup resources."""
        if self.logger:
            self.logger.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


def train(config_path: Optional[str] = None, **override_kwargs):
    """Convenience function to run training.

    Args:
        config_path: Path to YAML config file
        **override_kwargs: Runtime config overrides
    """
    with BPAgentTrainer(config_path, **override_kwargs) as trainer:
        trainer.train()
