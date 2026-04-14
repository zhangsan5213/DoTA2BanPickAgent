"""Epoch training runner with PPO epoch loop (inspired by GITCGRL)."""

from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import torch
import torch.nn.functional as F

from .loss_computer import LossComputer
from utils.device import DEVICE


class EpochRunner:
    """Runs a single training epoch with PPO epoch loop."""

    def __init__(
        self,
        agent,
        optimizer,
        loss_computer: LossComputer,
        rollout_collector,
        checkpoint_manager,
        config,
        entropy_annealer=None,
    ):
        """
        Args:
            agent: Training agent
            optimizer: Optimizer
            loss_computer: LossComputer instance
            rollout_collector: RolloutCollector instance
            checkpoint_manager: CheckpointManager instance
            config: TrainingConfig instance
            entropy_annealer: Optional EntropyAnnealer instance for dynamic entropy coefficient
        """
        self.agent = agent
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.rollout_collector = rollout_collector
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.entropy_annealer = entropy_annealer
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
            "ppo_epochs": 0,
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

            # Process rollouts with PPO epoch loop (GITCGRL style)
            batch_stats = self._process_rollouts_ppo_epochs(rollouts, writer)

            # Update epoch stats
            for key in [
                "total_loss",
                "policy_loss",
                "value_loss",
                "entropy_loss",
                "kl_div",
                "ppo_epochs",
            ]:
                if key in batch_stats:
                    epoch_stats[key] += batch_stats[key]
            epoch_stats["num_rollouts"] += len(rollouts)

            # Update progress bar
            avg_loss = epoch_stats["total_loss"] / max(epoch_stats["num_rollouts"], 1)
            avg_kl = epoch_stats["kl_div"] / max(epoch_stats["num_rollouts"], 1)
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "KL": f"{avg_kl:.4f}"
            })

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
            # Average PPO epochs per batch
            epoch_stats["ppo_epochs"] /= num_batches

        return epoch_stats

    def _process_rollouts_ppo_epochs(
        self, rollouts: List[Dict], writer
    ) -> Dict[str, float]:
        """Process rollouts with true PPO minibatch updates and KL early stopping.
        
        Inspired by standard PPO: flatten all valid steps, shuffle into minibatches,
        and perform multiple epochs of updates. Stop early if KL exceeds threshold.

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
            "ppo_epochs": 0,
            "early_stops": 0,
        }

        if not rollouts:
            return batch_stats

        # Get PPO epoch configuration
        ppo_epochs = getattr(self.config, "ppo_epochs", 4)
        kl_threshold = getattr(self.config, "kl_threshold", 0.15)
        kl_early_stop = getattr(self.config, "kl_early_stop", True)
        max_grad_norm = getattr(self.config, "max_grad_norm", 0.5)
        minibatch_size = getattr(self.config, "minibatch_size", 64)

        # Prepare rollout data (convert to tensors on DEVICE)
        prepared_rollouts = self._prepare_rollout_data(rollouts)

        # Flatten all valid steps across rollouts
        all_flat_data = []
        for rollout_data in prepared_rollouts:
            flat = self.loss_computer.prepare_rollout(rollout_data)
            if flat is not None and len(flat["actions"]) > 0:
                all_flat_data.append(flat)

        if not all_flat_data:
            return batch_stats

        # Merge all data into single tensors / list
        merged_states = []
        for d in all_flat_data:
            merged_states.extend(d["states"])
        merged_actions = torch.cat([d["actions"] for d in all_flat_data])
        merged_old_log_probs = torch.cat([d["old_log_probs"] for d in all_flat_data])
        merged_advantages = torch.cat([d["advantages"] for d in all_flat_data])
        merged_returns = torch.cat([d["returns"] for d in all_flat_data])
        merged_old_values = torch.cat([d["old_values"] for d in all_flat_data])

        total_steps = len(merged_actions)
        num_minibatches = max(1, total_steps // minibatch_size)

        # PPO epoch loop with minibatch shuffle
        for ppo_epoch in range(ppo_epochs):
            epoch_total_loss = 0.0
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy_loss = 0.0
            epoch_kl_div = 0.0
            valid_updates = 0

            # Shuffle all steps at the start of each PPO epoch
            indices = torch.randperm(total_steps, device=DEVICE)

            for mb_idx in range(num_minibatches):
                start = mb_idx * minibatch_size
                end = min(start + minibatch_size, total_steps)
                mb_indices = indices[start:end]

                mb_states = [merged_states[i] for i in mb_indices.tolist()]
                mb_actions = merged_actions[mb_indices]
                mb_old_log_probs = merged_old_log_probs[mb_indices]
                mb_advantages = merged_advantages[mb_indices]
                mb_returns = merged_returns[mb_indices]
                mb_old_values = merged_old_values[mb_indices]

                # Get current entropy coefficient
                entropy_coeff = None
                if self.entropy_annealer is not None:
                    entropy_coeff = self.entropy_annealer.get_coeff(step=self.global_step)
                    self.entropy_annealer.step()

                # Compute loss and perform update
                result = self._compute_and_update_minibatch(
                    mb_states,
                    mb_actions,
                    mb_old_log_probs,
                    mb_advantages,
                    mb_returns,
                    mb_old_values,
                    entropy_coeff=entropy_coeff,
                    max_grad_norm=max_grad_norm,
                )

                if result is None:
                    continue

                loss, policy_loss, value_loss, entropy_loss, kl_div = result
                epoch_total_loss += loss
                epoch_policy_loss += policy_loss
                epoch_value_loss += value_loss
                epoch_entropy_loss += entropy_loss
                epoch_kl_div += kl_div
                valid_updates += 1
                self.global_step += 1

            # Average over minibatches in this epoch
            if valid_updates > 0:
                avg_kl = epoch_kl_div / valid_updates
                batch_stats["total_loss"] += epoch_total_loss / valid_updates
                batch_stats["policy_loss"] += epoch_policy_loss / valid_updates
                batch_stats["value_loss"] += epoch_value_loss / valid_updates
                batch_stats["entropy_loss"] += epoch_entropy_loss / valid_updates
                batch_stats["kl_div"] += avg_kl
                batch_stats["ppo_epochs"] += 1

                # KL early stopping
                if kl_early_stop and avg_kl > kl_threshold:
                    batch_stats["early_stops"] += 1
                    if writer is not None:
                        writer.add_scalar("Training/ppo_early_stop_epoch", ppo_epoch, self.global_step)
                    break

        # Final average over actual PPO epochs executed
        actual_epochs = batch_stats["ppo_epochs"]
        if actual_epochs > 0:
            for key in ["total_loss", "policy_loss", "value_loss", "entropy_loss", "kl_div"]:
                batch_stats[key] /= actual_epochs

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/actor", batch_stats["policy_loss"], self.global_step)
            writer.add_scalar("Loss/value", batch_stats["value_loss"], self.global_step)
            writer.add_scalar("Loss/entropy", batch_stats["entropy_loss"], self.global_step)
            writer.add_scalar("Loss/total", batch_stats["total_loss"], self.global_step)
            writer.add_scalar("Loss/kl_divergence", batch_stats["kl_div"], self.global_step)
            writer.add_scalar("Training/ppo_epochs_per_batch", actual_epochs, self.global_step)
            writer.add_scalar("Training/ppo_early_stops", batch_stats["early_stops"], self.global_step)
            writer.add_scalar("Training/rollouts_processed", len(rollouts), self.global_step)
            
            if self.entropy_annealer is not None:
                current_coeff = self.entropy_annealer.get_coeff(step=self.global_step)
                if current_coeff is not None:
                    writer.add_scalar("Training/entropy_coeff", current_coeff, self.global_step)

            # Log learning rate
            for i, param_group in enumerate(self.optimizer.param_groups):
                writer.add_scalar(f"Training/lr_group_{i}", param_group["lr"], self.global_step)

            writer.flush()

        # Console log
        entropy_info = ""
        if self.entropy_annealer is not None:
            current_coeff = self.entropy_annealer.get_coeff(step=self.global_step)
            if current_coeff is not None:
                entropy_info = f", Entropy Coeff: {current_coeff:.4f}"
        
        early_stop_info = ""
        if batch_stats["early_stops"] > 0:
            early_stop_info = f", Early Stops: {batch_stats['early_stops']}"

        print(
            f"[Batch Stats] Global Step: {self.global_step}, Total Loss: {batch_stats['total_loss']:.4f}, "
            f"Policy Loss: {batch_stats['policy_loss']:.4f}, Value Loss: {batch_stats['value_loss']:.4f}, "
            f"Entropy Loss: {batch_stats['entropy_loss']:.4f}, KL: {batch_stats['kl_div']:.4f}"
            f"({kl_threshold:.2f}), PPO Epochs: {actual_epochs}/{ppo_epochs}, Rollouts: {len(rollouts)}"
            f"{entropy_info}{early_stop_info}"
        )

        return batch_stats

    def _prepare_rollout_data(self, rollouts: List[Dict]) -> List[Dict]:
        """Prepare rollout data for training by moving tensors to DEVICE.
        
        Args:
            rollouts: List of raw rollouts
            
        Returns:
            List of prepared rollout data dictionaries
        """
        prepared = []
        for rollout in rollouts:
            data = {
                "states": rollout["states"],
                "actions": rollout["actions"].to(DEVICE),
                "old_log_probs": rollout["log_probs"].to(DEVICE),
                "values": rollout["values"].to(DEVICE),
                "rewards": rollout["rewards"].to(DEVICE),
                "valid_mask": rollout["valid_mask"].to(DEVICE),
            }
            if "step_teams" in rollout:
                data["step_teams"] = rollout["step_teams"].to(DEVICE)
            prepared.append(data)
        return prepared

    def _compute_and_update_minibatch(
        self,
        states: List[Dict],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_values: torch.Tensor,
        entropy_coeff: Optional[float] = None,
        max_grad_norm: float = 0.5,
    ) -> Optional[tuple]:
        """Compute losses and perform a single minibatch gradient update.
        
        Args:
            states: List of state dicts for the minibatch
            actions: Minibatch action indices
            old_log_probs: Old policy log probabilities
            advantages: Advantages
            returns: Returns
            old_values: Old value estimates
            entropy_coeff: Optional dynamic entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Tuple of (loss, policy_loss, value_loss, entropy_loss, kl_div) or None if failed
        """
        self.optimizer.zero_grad()
        
        # Compute losses using loss_computer
        result = self.loss_computer.compute_minibatch(
            states,
            actions,
            old_log_probs,
            advantages,
            returns,
            old_values,
            entropy_coeff=entropy_coeff,
        )
        
        if result is None:
            return None
        
        loss, policy_loss, value_loss, entropy_loss, kl_div = result
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        return (
            loss.item(),
            policy_loss.item(),
            value_loss.item(),
            entropy_loss.item(),
            kl_div,
        )
