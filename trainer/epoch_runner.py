"""Epoch training runner."""

from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
import torch

from .loss_computer import LossComputer
from utils.device import DEVICE


class EpochRunner:
    """Runs a single training epoch."""
    
    def __init__(self, agent, optimizer, loss_computer: LossComputer,
                 rollout_collector, checkpoint_manager, config):
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
        progress_callback: Optional[Callable] = None
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
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_div': 0.0,
            'num_rollouts': 0
        }
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{total_epochs}", ncols=90)
        
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
                global_step=self.global_step
            )
            
            # Process each rollout
            batch_stats = self._process_rollouts(rollouts, writer)
            
            # Update epoch stats
            for key in ['total_loss', 'policy_loss', 'value_loss', 'entropy_loss', 'kl_div']:
                epoch_stats[key] += batch_stats[key]
            epoch_stats['num_rollouts'] += len(rollouts)
            
            # Update progress bar
            avg_loss = epoch_stats['total_loss'] / epoch_stats['num_rollouts']
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})
            
            if progress_callback:
                progress_callback(epoch, batch_idx, num_batches, avg_loss)
        
        # Compute averages
        if epoch_stats['num_rollouts'] > 0:
            for key in ['total_loss', 'policy_loss', 'value_loss', 'entropy_loss', 'kl_div']:
                epoch_stats[key] /= epoch_stats['num_rollouts']
        
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
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_div': 0.0
        }
        
        for rollout in rollouts:
            loss, policy_loss, value_loss, entropy_loss, kl_div = self.loss_computer.compute(rollout)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_stats['total_loss'] += loss.item()
            batch_stats['policy_loss'] += policy_loss.item()
            batch_stats['value_loss'] += value_loss.item()
            batch_stats['entropy_loss'] += entropy_loss.item()
            batch_stats['kl_div'] += kl_div
            
            self.global_step += 1
        
        # Log to TensorBoard
        if writer is not None and len(rollouts) > 0:
            writer.add_scalar("Loss/actor", batch_stats['policy_loss'] / len(rollouts), self.global_step)
            writer.add_scalar("Loss/value", batch_stats['value_loss'] / len(rollouts), self.global_step)
            writer.add_scalar("Loss/entropy", batch_stats['entropy_loss'] / len(rollouts), self.global_step)
            writer.add_scalar("Loss/total", batch_stats['total_loss'] / len(rollouts), self.global_step)
            writer.add_scalar("Loss/kl_divergence", batch_stats['kl_div'] / len(rollouts), self.global_step)
            writer.flush()
        
        return batch_stats
