"""Epoch training runner with PPO epoch loop (inspired by GITCGRL)."""

import time
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
        start_global_step: int = 0,
        start_grad_accum_step: int = 0,
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
            start_global_step: Global step to resume from for TensorBoard continuity
            start_grad_accum_step: Gradient accumulation step to resume from
        """
        self.agent = agent
        self.optimizer = optimizer
        self.loss_computer = loss_computer
        self.rollout_collector = rollout_collector
        self.checkpoint_manager = checkpoint_manager
        self.config = config
        self.entropy_annealer = entropy_annealer
        self.global_step = start_global_step
        self.grad_accum_step = start_grad_accum_step

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
        # Rollout collection runs in eval mode: dropout would otherwise inject noise
        # into old_log_probs, value baselines and MCTS priors, corrupting the PPO ratio.
        self.agent.eval()

        batch_size = self.config.batch_size
        num_batches = (len(samples) + batch_size - 1) // batch_size
        grad_accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)

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

        batch_timings = []
        for batch_idx in pbar:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(samples))
            batch_samples = samples[start_idx:end_idx]

            # Collect rollouts
            t0 = time.perf_counter()
            checkpoints = self.checkpoint_manager.checkpoints
            rollouts = self.rollout_collector.collect_batch(
                batch_samples=batch_samples,
                checkpoints=checkpoints,
                checkpoint_manager=self.checkpoint_manager,
                batch_idx=batch_idx,
            )
            t1 = time.perf_counter()

            # Process rollouts with PPO epoch loop (GITCGRL style)
            batch_stats = self._process_rollouts_ppo_epochs(rollouts, writer, epoch=epoch)
            t2 = time.perf_counter()

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

            batch_timings.append({
                'rollout_ms': (t1 - t0) * 1000,
                'ppo_ms': (t2 - t1) * 1000,
                'total_ms': (t2 - t0) * 1000,
                'num_rollouts': len(rollouts),
            })

            # Print timing every 4 batches
            if (batch_idx + 1) % 4 == 0 or batch_idx == 0:
                avg_rollout = sum(t['rollout_ms'] for t in batch_timings[-4:]) / len(batch_timings[-4:])
                avg_ppo = sum(t['ppo_ms'] for t in batch_timings[-4:]) / len(batch_timings[-4:])
                print(f"\n[Profile Batch {batch_idx+1}] rollout={avg_rollout:7.1f}ms | ppo={avg_ppo:7.1f}ms | rollouts={len(rollouts)}")

            # Update progress bar
            avg_loss = epoch_stats["total_loss"] / max(epoch_stats["num_rollouts"], 1)
            avg_kl = epoch_stats["kl_div"] / max(epoch_stats["num_rollouts"], 1)
            pbar.set_postfix({
                "Loss": f"{avg_loss:.4f}",
                "KL": f"{avg_kl:.4f}"
            })

            if progress_callback:
                progress_callback(epoch, batch_idx, num_batches, avg_loss)

        # Print epoch timing summary
        if batch_timings:
            total_rollout = sum(t['rollout_ms'] for t in batch_timings)
            total_ppo = sum(t['ppo_ms'] for t in batch_timings)
            total_all = sum(t['total_ms'] for t in batch_timings)
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1} TIMING SUMMARY")
            print(f"{'='*70}")
            print(f"  Rollout collection: {total_rollout/1000:.1f}s ({total_rollout/total_all*100:.1f}%)")
            print(f"  PPO training:       {total_ppo/1000:.1f}s ({total_ppo/total_all*100:.1f}%)")
            print(f"  Total batch time:   {total_all/1000:.1f}s")
            print(f"  Num batches:        {len(batch_timings)}")
            print(f"  Avg batch time:     {total_all/len(batch_timings)/1000:.1f}s")
            print(f"{'='*70}")

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
        self, rollouts: List[Dict], writer, epoch: int = 0
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

        # PPO update phase runs in train mode (dropout active); rollout collection
        # above used eval mode to keep behavior-policy quantities noise-free.
        self.agent.train()

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

        merged_mcts_policies = None
        if any("mcts_policies" in d and d["mcts_policies"] is not None for d in all_flat_data):
            merged_mcts_policies = torch.cat([
                d["mcts_policies"] for d in all_flat_data if "mcts_policies" in d
            ])

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
                mb_mcts_policies = (
                    merged_mcts_policies[mb_indices]
                    if merged_mcts_policies is not None
                    else None
                )

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
                    mcts_policies=mb_mcts_policies,
                    epoch=epoch,
                )

                if result is None:
                    continue

                loss, policy_loss, value_loss, entropy_loss, kl_div, mcts_policy_loss = result
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
            f"\n"
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
            if "mcts_policies" in rollout and rollout["mcts_policies"] is not None:
                data["mcts_policies"] = rollout["mcts_policies"].to(DEVICE)
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
        mcts_policies: Optional[torch.Tensor] = None,
        epoch: int = 0,
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
            mcts_policies: Optional MCTS visit-count policy [N, NUM_HEROES]
            epoch: Current epoch number (for value warm-up)

        Returns:
            Tuple of (loss, policy_loss, value_loss, entropy_loss, kl_div, mcts_policy_loss) or None if failed
        """
        grad_accum_steps = getattr(self.config, "gradient_accumulation_steps", 1)
        is_accumulation_step = (self.grad_accum_step % grad_accum_steps) != (grad_accum_steps - 1)
        value_warmup_epochs = getattr(self.config, "value_warmup_epochs", 0)

        # Compute losses using loss_computer
        mcts_policy_weight = getattr(self.config, "mcts_policy_loss_weight", 0.0)
        if mcts_policies is None or not getattr(self.config, "mcts_use_policy_loss", False):
            mcts_policy_weight = 0.0

        result = self.loss_computer.compute_minibatch(
            states,
            actions,
            old_log_probs,
            advantages,
            returns,
            old_values,
            entropy_coeff=entropy_coeff,
            mcts_policies=mcts_policies,
            mcts_policy_weight=mcts_policy_weight,
        )

        if result is None:
            return None

        loss, policy_loss, value_loss, entropy_loss, kl_div, mcts_policy_loss = result

        # Value-only warm-up: first N epochs train only value function
        is_warmup = epoch < value_warmup_epochs
        if is_warmup:
            loss = self.loss_computer.value_loss_coeff * value_loss
            policy_loss = torch.tensor(0.0, device=loss.device)
            entropy_loss = torch.tensor(0.0, device=loss.device)
            kl_div = 0.0

        # Scale loss for gradient accumulation
        if grad_accum_steps > 1:
            loss = loss / grad_accum_steps

        # Backward pass
        loss.backward()

        # Optimizer step only on the last accumulation step
        if not is_accumulation_step:
            # Separate gradient clipping: value head vs rest of network
            # This prevents massive value gradients from suppressing policy learning
            if max_grad_norm > 0:
                non_value_params = [
                    p for n, p in self.agent.named_parameters() if not n.startswith("value_head")
                ]
                value_params = [
                    p for n, p in self.agent.named_parameters() if n.startswith("value_head")
                ]
                if non_value_params:
                    torch.nn.utils.clip_grad_norm_(non_value_params, max_grad_norm)
                if value_params:
                    torch.nn.utils.clip_grad_norm_(value_params, max_grad_norm * 4)

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.grad_accum_step += 1

        return (
            loss.item() * grad_accum_steps,
            policy_loss.item(),
            value_loss.item(),
            entropy_loss.item(),
            kl_div,
            mcts_policy_loss.item(),
        )
