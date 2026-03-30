"""Epoch training runner."""

from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import torch

from .loss_computer import LossComputer
from utils.device import DEVICE


class EpochRunner:
    """Runs a single training epoch."""

    def __init__(
        self,
        agent,
        optimizer,
        loss_computer: LossComputer,
        rollout_collector,
        checkpoint_manager,
        config,
    ):
        """
        Args:
            agent: Training agent
            optimizer: Optimizer
            loss_computer: LossComputer instance
            rollout_collector: RolloutCollector instance
            checkpoint_manager: CheckpointManager instance
            config: TrainingConfig instance
        """
        self.agent = agent
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.rollout_collector = rollout_collector
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.global_step = 0

    def run(
        self,
        epoch: int,
        total_epochs: int,
        samples: List[Dict[str, Any]],
        writer=None,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """Run a single epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            total_epochs: Total number of epochs
            samples: Training samples for this epoch
            writer: Optional TensorBoard writer
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary of epoch statistics
        """
        self.agent.train()

        batch_size = self.config.batch_size
        num_batches = (len(samples) + batch_size - 1) // batch_size

        epoch_stats = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "kl_div": 0.0,
            "num_rollouts": 0,
        }

        pbar = tqdm(
            range(num_batches), desc=f"Epoch {epoch + 1}/{total_epochs}", ncols=90
        )

        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            # Collect rollouts
            checkpoints = self.checkpoint_manager.checkpoints
            rollouts = self.rollout_collector.collect_batch(
                batch_samples=batch_samples,
                checkpoints=checkpoints,
                checkpoint_manager=self.checkpoint_manager,
                batch_idx=batch_idx,
            )

            # Process each rollout
            batch_stats = self._process_rollouts(rollouts, writer)

            # Update epoch stats
            for key in [
                "total_loss",
                "policy_loss",
                "value_loss",
                "entropy_loss",
                "kl_div",
            ]:
                epoch_stats[key] += batch_stats[key]
            epoch_stats["num_rollouts"] += len(rollouts)

            # Update progress bar
            avg_loss = epoch_stats["total_loss"] / epoch_stats["num_rollouts"]
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

            if progress_callback:
                progress_callback(epoch, batch_idx, num_batches, avg_loss)

        # Compute averages
        if epoch_stats["num_rollouts"] > 0:
            for key in [
                "total_loss",
                "policy_loss",
                "value_loss",
                "entropy_loss",
                "kl_div",
            ]:
                epoch_stats[key] /= epoch_stats["num_rollouts"]

        return epoch_stats

    def _process_rollouts(self, rollouts: List[Dict], writer) -> Dict[str, float]:
        """Process a batch of rollouts and compute losses.

        Args:
            rollouts: List of rollouts
            writer: TensorBoard writer

        Returns:
            Dictionary of batch statistics
        """
        batch_stats = {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "kl_div": 0.0,
        }

        # 累加所有 rollout 的损失
        self.optimizer.zero_grad()
        total_loss = None
        for rollout in rollouts:
            loss, policy_loss, value_loss, entropy_loss, kl_div = (
                self.loss_computer.compute(rollout)
            )

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

            batch_stats["total_loss"] += loss.item()
            batch_stats["policy_loss"] += policy_loss.item()
            batch_stats["value_loss"] += value_loss.item()
            batch_stats["entropy_loss"] += entropy_loss.item()
            batch_stats["kl_div"] += kl_div

            self.global_step += 1

        # 计算平均损失并更新模型
        batch_size = len(rollouts)
        if total_loss is not None and isinstance(total_loss, torch.Tensor):
            avg_loss = total_loss / batch_size
            avg_loss.backward()
            self.optimizer.step()
            avg_total_loss_val = avg_loss.item()
        else:
            print(
                f"[警告] total_loss不是有效tensor类型: {type(total_loss)}, 值: {total_loss}"
            )
            avg_total_loss_val = batch_stats["total_loss"] / batch_size

        # 计算批次平均统计
        avg_total_loss = batch_stats["total_loss"] / batch_size
        avg_policy_loss = batch_stats["policy_loss"] / batch_size
        avg_value_loss = batch_stats["value_loss"] / batch_size
        avg_entropy_loss = batch_stats["entropy_loss"] / batch_size
        avg_kl_div = batch_stats["kl_div"] / batch_size

        # Log to TensorBoard
        if writer is not None and batch_size > 0:
            writer.add_scalar("Loss/actor", avg_policy_loss, self.global_step)
            writer.add_scalar("Loss/value", avg_value_loss, self.global_step)
            writer.add_scalar("Loss/entropy", avg_entropy_loss, self.global_step)
            writer.add_scalar("Loss/total", avg_total_loss, self.global_step)
            writer.add_scalar("Loss/kl_divergence", avg_kl_div, self.global_step)

            # 添加更多TensorBoard日志
            writer.add_scalar(
                "Training/rollouts_per_batch", batch_size, self.global_step
            )
            writer.add_scalar(
                "Training/global_step", self.global_step, self.global_step
            )

            # 记录学习率
            for i, param_group in enumerate(self.optimizer.param_groups):
                writer.add_scalar(
                    f"Training/lr_group_{i}", param_group["lr"], self.global_step
                )

            writer.flush()

        # 添加详细的控制台日志
        print(
            f"[Batch Stats] Global Step: {self.global_step}, Total Loss: {avg_total_loss:.4f}, "
            f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, "
            f"Entropy Loss: {avg_entropy_loss:.4f}, KL Div: {avg_kl_div:.4f}, "
            f"Rollouts: {batch_size}"
        )

        return batch_stats
