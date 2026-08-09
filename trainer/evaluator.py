"""Evaluation management."""

import os
from typing import Dict, Any, Optional, List, Tuple
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
                    "initial_mu": config.ts_initial_mu,
                    "initial_sigma": config.ts_initial_sigma,
                    "beta": config.ts_beta,
                    "tau": config.ts_tau,
                    "draw_probability": config.ts_draw_probability,
                    "opponent_sample_std": config.ts_opponent_sample_std,
                }
            )

        self.rating_evaluator: RatingEvaluatorBase = get_evaluator(
            self.eval_method, **self.eval_kwargs
        )

        # Track rating history for plotting
        self.rating_history: List[Tuple[int, float]] = []

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

        # Print leaderboard with epoch name for current model
        print("\n[+] Leaderboard:")
        name_overrides = {model_path: f"bp_agent_epoch{epoch}"}
        self.rating_evaluator.print_leaderboard(name_overrides=name_overrides)

        # Log to TensorBoard
        if self.writer is not None:
            self._log_ratings(model_path, eval_result, epoch)

        # Plot and display rating history (overwrite same PNG each time)
        self._plot_rating_history()

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
            self.rating_history.append((epoch, record.rating))
        else:
            self.writer.add_scalar(
                f"Rating/{self.method_name.lower()}_rating", record.elo, epoch
            )
            self.rating_history.append((epoch, float(record.elo)))

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
                    self.rating_history.append((total_epochs, record.rating))
                else:
                    self.writer.add_scalar(
                        f"Rating/{self.method_name.lower()}_rating",
                        record.elo,
                        total_epochs,
                    )
                    self.rating_history.append((total_epochs, float(record.elo)))
            self.writer.flush()

        # Print final leaderboard with epoch name override for final model
        name_overrides = {model_path: f"bp_agent_epoch{total_epochs}"}
        self.rating_evaluator.print_leaderboard(name_overrides=name_overrides)

        # Plot and display rating history
        self._plot_rating_history()

    def _plot_rating_history(self):
        """Plot rating history over epochs and save to tensorboard log dir.
        Overwrites the same PNG file each time so the path stays constant."""
        if not self.rating_history:
            print("[!] No rating history to plot")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not available, skipping rating history plot")
            return

        epochs = [e for e, _ in self.rating_history]
        ratings = [r for _, r in self.rating_history]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, ratings, marker="o", linewidth=2, markersize=6, color="#1f77b4")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(f"{self.method_name} Rating", fontsize=12)
        ax.set_title(f"{self.method_name} Rating vs Epoch", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(epochs) + 1)

        # Annotate final point
        if ratings:
            ax.annotate(
                f"{ratings[-1]:.1f}",
                xy=(epochs[-1], ratings[-1]),
                xytext=(epochs[-1], ratings[-1] + max(abs(min(ratings)), abs(max(ratings))) * 0.05),
                ha="center",
                fontsize=10,
                arrowprops=dict(arrowstyle="->", color="red"),
            )

        plt.tight_layout()

        # Save to tensorboard log dir (same path every time)
        if self.writer is not None:
            log_dir = getattr(self.writer, "log_dir", None)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                plot_path = os.path.join(
                    log_dir, f"{self.method_name.lower()}_rating_history.png"
                )
                plt.savefig(plot_path, dpi=150)
                print(f"[+] Rating history plot saved to {plot_path}")
                self._display_image_in_terminal(plot_path)

        plt.close()

    def _display_image_in_terminal(self, image_path: str):
        """Display an image in the terminal using ANSI colors and half-block characters."""
        try:
            from PIL import Image
        except ImportError:
            print("[!] Pillow not available, cannot display image in terminal")
            return

        try:
            img = Image.open(image_path)
            img = img.convert("RGBA")

            width, height = img.size
            aspect_ratio = height / width / 1.8
            new_width = 80
            new_height = int(new_width * aspect_ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            pixels = img.load()

            print()
            for y in range(0, new_height, 2):
                line = ""
                for x in range(new_width):
                    r1, g1, b1, a1 = pixels[x, y]
                    if y + 1 < new_height:
                        r2, g2, b2, a2 = pixels[x, y + 1]
                    else:
                        r2, g2, b2, a2 = 0, 0, 0, 0

                    if a1 < 128:
                        r1, g1, b1 = 0, 0, 0
                    if a2 < 128:
                        r2, g2, b2 = 0, 0, 0

                    line += f"\033[48;2;{r1};{g1};{b1}m\033[38;2;{r2};{g2};{b2}m\u2584\033[0m"
                print(line)
            print()
        except Exception as e:
            print(f"[!] Failed to display image in terminal: {e}")


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
