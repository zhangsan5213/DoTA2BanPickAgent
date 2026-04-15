"""Evaluation management."""

import os
from typing import Dict, Any, Optional, List
import torch

from eval import EvalMethod, get_evaluator, RatingEvaluatorBase
from model.win_rate_oracle import WinRateOracle


class EvaluatorManager:
    """Manages model evaluation during training."""

    def __init__(self, config, oracle: WinRateOracle, save_dir: str, additional_dirs: Optional[List[str]] = None):
        """
        Args:
            config: TrainingConfig instance
            oracle: Win rate oracle model
            save_dir: Directory to save checkpoints
            additional_dirs: Additional directories to load historical records from
        """
        self.config = config
        self.oracle = oracle
        self.save_dir = save_dir
        self.writer = None

        # Determine evaluation method
        if config.rating_method.lower() == "elo":
            self.eval_method = EvalMethod.ELO
            self.method_name = "ELO"
        elif config.rating_method.lower() == "trueskill":
            self.eval_method = EvalMethod.TRUESKILL
            self.method_name = "TrueSkill"
        else:
            raise ValueError(f"Unknown rating method: {config.rating_method}")

        # Build eval kwargs
        self.eval_kwargs = {
            "save_dir": save_dir,
            "oracle": oracle,
            "num_opponents": config.rating_num_opponents,
            "num_player_sets": config.rating_num_player_sets,
        }

        if config.rating_method.lower() == "elo":
            self.eval_kwargs.update(
                {
                    "k_factor": config.elo_k_factor,
                    "opponent_sample_std": config.elo_opponent_sample_std,
                }
            )
        elif config.rating_method.lower() == "trueskill":
            self.eval_kwargs.update(
                {
                    "staleness_threshold": config.ts_staleness_threshold,
                    "num_active_models": config.ts_num_active_models,
                    "additional_dirs": additional_dirs,
                }
            )

        self.rating_evaluator: RatingEvaluatorBase = get_evaluator(
            self.eval_method, **self.eval_kwargs
        )

    def set_writer(self, writer):
        """Set TensorBoard writer."""
        self.writer = writer

    def should_evaluate(self, epoch: int) -> bool:
        """Check if evaluation should be performed at this epoch."""
        return (epoch + 1) % self.config.eval_interval == 0

    def evaluate(self, model_path: str, epoch: int) -> Dict[str, Any]:
        """Evaluate model at given epoch.

        Args:
            model_path: Path to model checkpoint
            epoch: Current epoch number

        Returns:
            Evaluation results
        """
        print(f"\n[+] {self.method_name} evaluation at epoch {epoch}...")
        print(
            f"[+] Evaluating with {self.config.rating_num_opponents} opponents and {self.config.rating_num_player_sets} player sets..."
        )

        eval_result = self.rating_evaluator.evaluate(
            model_path=model_path,
            num_opponents=self.config.rating_num_opponents,
            num_player_sets=self.config.rating_num_player_sets,
        )

        # Print leaderboard
        print("\n[+] Leaderboard:")
        self.rating_evaluator.print_leaderboard()

        # Log to TensorBoard
        if self.writer is not None:
            self._log_ratings(model_path, eval_result, epoch)

        # Log evaluation details
        if "results" in eval_result:
            num_battles = sum(len(r.get("battle_results", [])) for r in eval_result["results"])
            total_win_rate = sum(r["win_rate"] for r in eval_result["results"])
            avg_win_rate = total_win_rate / len(eval_result["results"]) if eval_result["results"] else 0

            print(f"\n[+] Evaluation summary:")
            print(f"[+] Number of battles: {num_battles}")
            print(f"[+] Average win rate: {avg_win_rate:.4f}")

            if self.writer is not None:
                self.writer.add_scalar("Evaluation/num_battles", num_battles, epoch)
                self.writer.add_scalar("Evaluation/avg_win_rate", avg_win_rate, epoch)
                self.writer.flush()

        return eval_result

    def _log_ratings(self, model_path: str, eval_result: Dict, epoch: int):
        """Log rating metrics to TensorBoard."""
        record = self.rating_evaluator.rating_manager.get_record(model_path)
        if record is None:
            return

        if self.config.rating_method.lower() == "trueskill":
            self.writer.add_scalar(
                f"Rating/{self.method_name.lower()}_mu", record.mu, epoch
            )
            self.writer.add_scalar(
                f"Rating/{self.method_name.lower()}_sigma", record.sigma, epoch
            )
            self.writer.add_scalar(
                f"Rating/{self.method_name.lower()}_rating", record.rating, epoch
            )
        else:
            self.writer.add_scalar(
                f"Rating/{self.method_name.lower()}_rating", record.elo, epoch
            )

        # Log average win rate
        if eval_result.get("results"):
            avg_winrate = sum(r["win_rate"] for r in eval_result["results"]) / len(
                eval_result["results"]
            )
            self.writer.add_scalar("Rating/avg_winrate", avg_winrate, epoch)

        self.writer.flush()

    def final_evaluation(self, model_path: str, total_epochs: int):
        """Perform final evaluation after training.

        Args:
            model_path: Path to final model checkpoint
            total_epochs: Total number of training epochs
        """
        print(f"[+] Final {self.method_name} evaluation...")

        self.rating_evaluator.evaluate(
            model_path=model_path,
            num_opponents=self.config.rating_num_opponents,
            num_player_sets=self.config.rating_num_player_sets,
        )

        # Log final ratings
        if self.writer is not None:
            record = self.rating_evaluator.rating_manager.get_record(model_path)
            if record is not None:
                if self.config.rating_method.lower() == "trueskill":
                    self.writer.add_scalar(
                        f"Rating/{self.method_name.lower()}_mu", record.mu, total_epochs
                    )
                    self.writer.add_scalar(
                        f"Rating/{self.method_name.lower()}_sigma",
                        record.sigma,
                        total_epochs,
                    )
                    self.writer.add_scalar(
                        f"Rating/{self.method_name.lower()}_rating",
                        record.rating,
                        total_epochs,
                    )
                else:
                    self.writer.add_scalar(
                        f"Rating/{self.method_name.lower()}_rating",
                        record.elo,
                        total_epochs,
                    )
            self.writer.flush()

        # Print final leaderboard
        self.rating_evaluator.print_leaderboard()


def save_checkpoint(
    agent,
    save_dir: str,
    epoch: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    entropy_annealer: Optional[Any] = None,
    global_step: int = 0,
    grad_accum_step: int = 0,
    log_dir: Optional[str] = None,
) -> str:
    """Save full training checkpoint.

    Args:
        agent: Agent model
        save_dir: Directory to save checkpoint
        epoch: Current epoch number
        optimizer: Optional optimizer state
        scheduler: Optional scheduler state
        entropy_annealer: Optional entropy annealer state
        global_step: Current global training step for TensorBoard continuity
        grad_accum_step: Current gradient accumulation step
        log_dir: TensorBoard log directory

    Returns:
        Path to saved checkpoint
    """
    checkpoint = {
        "agent_state": agent.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "grad_accum_step": grad_accum_step,
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if entropy_annealer is not None:
        checkpoint["entropy_step"] = entropy_annealer.current_step
    if log_dir is not None:
        checkpoint["log_dir"] = log_dir

    checkpoint_path = f"{save_dir}/bp_agent_epoch{epoch}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"[+] Checkpoint saved: {checkpoint_path}")
    return checkpoint_path
