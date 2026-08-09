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

    def __init__(self, config_path: Optional[str] = None, resume_from: Optional[str] = None, **override_kwargs):
        """Initialize trainer.

        Args:
            config_path: Path to YAML config file
            resume_from: Path to checkpoint .pth file to resume training from
            **override_kwargs: Runtime config overrides
        """
        # Load configuration
        self.config = TrainingConfig(config_path)
        self.config.override(**override_kwargs)

        self.resume_from = resume_from
        self.start_epoch = 0
        self.start_global_step = 0
        self.start_grad_accum_step = 0
        self.log_dir = None

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

        # Always create new save directory, even when resuming
        self.save_dir = self.config.get_save_dir()
        if self.resume_from is not None:
            print(f"[+] Resuming from: {self.resume_from}")
        print(f"[+] Models will be saved to: {self.save_dir}")

        # Initialize models
        self.oracle = initialize_oracle(self.config)
        self.agent = initialize_agent(self.config)

        # Resume from checkpoint if provided
        ckpt_data = None
        resumed_from_checkpoint = False
        if self.resume_from is not None:
            ckpt_data = torch.load(self.resume_from, map_location='cpu')
            if isinstance(ckpt_data, dict) and "agent_state" in ckpt_data:
                # New-style full checkpoint
                self.agent.load_state_dict(ckpt_data["agent_state"])
                self.start_epoch = ckpt_data.get("epoch", 0)
                self.start_global_step = ckpt_data.get("global_step", 0)
                self.start_grad_accum_step = ckpt_data.get("grad_accum_step", 0)
                resumed_from_checkpoint = True
                # Don't reuse log_dir - always create new
                print(f"[+] Loaded full checkpoint from {self.resume_from}")
                print(f"[+] Resuming training from epoch {self.start_epoch}, global_step={self.start_global_step}")
            else:
                # Old-style checkpoint (agent state_dict only)
                self.agent.load_state_dict(ckpt_data)
                resumed_from_checkpoint = True
                print(f"[+] Loaded agent checkpoint from {self.resume_from}")
                # Parse epoch number from filename (e.g., bp_agent_epoch4.pth -> 4)
                basename = os.path.basename(self.resume_from)
                if basename.startswith("bp_agent_epoch") and basename.endswith(".pth"):
                    try:
                        self.start_epoch = int(basename[len("bp_agent_epoch"):-len(".pth")])
                        print(f"[+] Resuming training from epoch {self.start_epoch}")
                    except ValueError:
                        self.start_epoch = 0

        # Initialize hero_encoder from oracle when starting fresh training
        if not resumed_from_checkpoint:
            self.agent.hero_encoder.load_state_dict(self.oracle.hero_encoder.state_dict())
            print("[+] Initialized agent hero_encoder from oracle")

        self.optimizer = initialize_optimizer(self.agent, self.config)

        # Initialize data generator
        self.data_generator = DataGenerator(self.config.samples_per_epoch)

        # Initialize checkpoint manager
        # When resuming, include the previous checkpoint directory in the search
        checkpoint_dirs = list(self.config.checkpoint_dirs)
        if self.resume_from is not None:
            prev_ckpt_dir = os.path.dirname(os.path.abspath(self.resume_from))
            if prev_ckpt_dir not in checkpoint_dirs:
                checkpoint_dirs.append(prev_ckpt_dir)
                print(f"[+] Adding previous checkpoint dir to scan: {prev_ckpt_dir}")

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dirs,
            embed_dim=self.config.agent_embed_dim,
            nhead=self.config.agent_nhead,
            num_layers=self.config.agent_num_layers,
        )
        self.checkpoint_manager.discover()
        self.checkpoint_manager.print_summary()

        # Initialize evaluator (must be before rollout collector for rating manager)
        additional_rating_dirs = []
        if self.resume_from is not None:
            prev_ckpt_dir = os.path.dirname(os.path.abspath(self.resume_from))
            additional_rating_dirs.append(prev_ckpt_dir)
            print(f"[+] Loading historical ratings from: {prev_ckpt_dir}")

        self.evaluator = EvaluatorManager(
            self.config, self.oracle, self.save_dir,
            additional_dirs=additional_rating_dirs if additional_rating_dirs else None
        )

        # Initialize rollout collector
        rating_manager = None
        if hasattr(self.evaluator, 'rating_evaluator') and hasattr(self.evaluator.rating_evaluator, 'rating_manager'):
            rating_manager = self.evaluator.rating_evaluator.rating_manager

        mcts_config = None
        if getattr(self.config, 'mcts_enabled', False):
            mcts_config = {
                "num_simulations": getattr(self.config, 'mcts_num_simulations', 64),
                "c_puct": getattr(self.config, 'mcts_c_puct', 1.5),
                "top_k": getattr(self.config, 'mcts_top_k', 20),
                "dirichlet_alpha": getattr(self.config, 'mcts_dirichlet_alpha', 0.0),
                "dirichlet_epsilon": getattr(self.config, 'mcts_dirichlet_epsilon', 0.0),
                "max_search_depth": getattr(self.config, 'mcts_max_search_depth', 0),
            }

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
            oracle_embed_dim=self.config.oracle_embed_dim,
            oracle_nhead=self.config.oracle_nhead,
            oracle_num_layers=self.config.oracle_num_layers,
            use_mcts=getattr(self.config, 'mcts_enabled', False),
            mcts_config=mcts_config,
            use_parallel=getattr(self.config, 'mcts_enabled', False),
            num_workers=4,
            use_batched_mcts=getattr(self.config, 'mcts_use_batched', True),
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

        # Restore optimizer / scheduler / entropy annealer state from full checkpoint
        if ckpt_data is not None and isinstance(ckpt_data, dict):
            if "optimizer_state" in ckpt_data:
                self.optimizer.load_state_dict(ckpt_data["optimizer_state"])
                print("[+] Restored optimizer state")
            if "scheduler_state" in ckpt_data and self.scheduler is not None:
                self.scheduler.load_state_dict(ckpt_data["scheduler_state"])
                print("[+] Restored scheduler state")
            if "entropy_step" in ckpt_data and self.entropy_annealer is not None:
                self.entropy_annealer.current_step = ckpt_data["entropy_step"]
                print(f"[+] Restored entropy annealer step: {self.entropy_annealer.current_step}")

        # Initialize epoch runner
        self.epoch_runner = EpochRunner(
            self.agent,
            self.optimizer,
            self.loss_computer,
            self.rollout_collector,
            self.checkpoint_manager,
            self.config,
            entropy_annealer=self.entropy_annealer,
            start_global_step=self.start_global_step,
            start_grad_accum_step=self.start_grad_accum_step,
        )

        # Always create new TensorBoard log directory, even when resuming
        self.log_dir = self.config.get_log_dir()
        self.logger = TensorBoardLogger(
            log_dir=self.log_dir, enabled=self.config.use_tensorboard
        )
        if self.resume_from is not None:
            print(f"[+] Created new TensorBoard log dir: {self.log_dir}")

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
        for epoch in range(self.start_epoch, epochs):
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
                checkpoint_path = save_checkpoint(
                    self.agent,
                    self.save_dir,
                    epoch + 1,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    entropy_annealer=self.entropy_annealer,
                    global_step=self.epoch_runner.global_step,
                    grad_accum_step=self.epoch_runner.grad_accum_step,
                    # Don't save log_dir - we always create new ones when resuming
                )
                self.evaluator.evaluate(checkpoint_path, epoch + 1)

        # Save final model
        final_path = self._save_final_model()

        # Final evaluation
        self.evaluator.final_evaluation(final_path, epochs)

        print("\n[+] Training completed successfully!")

    def _save_final_model(self) -> str:
        """Save final model checkpoint with full training state.

        Returns:
            Path to saved model
        """
        model_path = f"{self.save_dir}/bp_agent_final.pth"
        checkpoint = {
            "agent_state": self.agent.state_dict(),
            "epoch": self.config.epochs,
            "global_step": self.epoch_runner.global_step,
            "grad_accum_step": self.epoch_runner.grad_accum_step,
        }
        if self.optimizer is not None:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()
        if self.entropy_annealer is not None:
            checkpoint["entropy_step"] = self.entropy_annealer.current_step
        # Don't save log_dir - we always create new ones when resuming
        torch.save(checkpoint, model_path)
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
