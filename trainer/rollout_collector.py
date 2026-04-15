"""Rollout collection for training with optional parallelization."""

import os
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import torch
import torch.multiprocessing as mp

from model.bp_agent import BPTransformerAgent
from model.win_rate_oracle import WinRateOracle
from utils.bp_env import collect_rollout
from utils.device import DEVICE
from eval.trueskill_rating import TrueSkillRatingManager, INITIAL_MU, INITIAL_SIGMA


def _collect_rollout_worker(args: tuple) -> Dict[str, Any]:
    """Worker function for parallel rollout collection.
    
    This function runs in a separate process and needs to reinitialize models
    since PyTorch models cannot be shared across processes.
    
    Args:
        args: Tuple containing:
            - sample: The sample data for rollout
            - agent_state_dict: State dict for BPTransformerAgent
            - oracle_state_dict: State dict for WinRateOracle
            - agent_config: Dict with embed_dim, nhead, num_layers for agent
            - oracle_config: Dict with embed_dim, nhead, num_layers, use_text, use_player_heroes
            - opponent_state_dict: Optional state dict for opponent agent
            - current_side: "radiant" or "dire" for self-play/opponent play
            - temperature: Optional sampling temperature
    
    Returns:
        Rollout dictionary
    """
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    (
        sample,
        agent_state_dict,
        oracle_state_dict,
        agent_config,
        oracle_config,
        opponent_state_dict,
        current_side,
        temperature,
        use_mcts,
        mcts_config,
    ) = args
    
    # Set device to CPU in worker processes (avoid GPU memory conflicts)
    device = torch.device("cpu")
    
    # Reconstruct agent
    agent = BPTransformerAgent(
        embed_dim=agent_config["embed_dim"],
        nhead=agent_config["nhead"],
        num_layers=agent_config["num_layers"],
    )
    agent.load_state_dict(agent_state_dict)
    agent.to(device)
    agent.eval()
    
    # Reconstruct oracle
    oracle = WinRateOracle(
        embed_dim=oracle_config["embed_dim"],
        nhead=oracle_config["nhead"],
        num_layers=oracle_config["num_layers"],
        use_text=oracle_config.get("use_text", True),
        use_player_heroes=oracle_config.get("use_player_heroes", True),
    )
    oracle.load_state_dict(oracle_state_dict)
    oracle.to(device)
    oracle.eval()
    
    # Reconstruct opponent if provided
    opponent = None
    if opponent_state_dict is not None:
        opponent = BPTransformerAgent(
            embed_dim=agent_config["embed_dim"],
            nhead=agent_config["nhead"],
            num_layers=agent_config["num_layers"],
        )
        opponent.load_state_dict(opponent_state_dict)
        opponent.to(device)
        opponent.eval()
    
    # Collect rollout
    rollout = collect_rollout(
        agent, oracle, sample,
        opponent_agent=opponent,
        current_side=current_side,
        temperature=temperature,
        use_mcts=use_mcts,
        mcts_config=mcts_config,
    )
    
    return rollout


class RolloutCollector:
    """Collects rollouts for training batches with optional parallelization."""

    def __init__(
        self,
        agent: BPTransformerAgent,
        oracle: WinRateOracle,
        historical_prob: float = 0.6,
        embed_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        use_parallel: bool = False,
        num_workers: int = 4,
        oracle_embed_dim: int = 128,
        oracle_nhead: int = 4,
        oracle_num_layers: int = 2,
        oracle_use_text: bool = True,
        oracle_use_player_heroes: bool = True,
        temperature: Optional[float] = None,
        policy_staleness_tolerance: int = 2,
        rating_manager = None,
        num_strata: int = 3,
        use_mcts: bool = False,
        mcts_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            agent: Current training agent
            oracle: Win rate oracle model
            historical_prob: Probability of using historical opponent
            embed_dim: Agent embedding dimension
            nhead: Agent number of attention heads
            num_layers: Agent number of transformer layers
            use_parallel: Whether to use parallel rollout collection
            num_workers: Number of worker processes for parallel collection
            oracle_embed_dim: Oracle embedding dimension
            oracle_nhead: Oracle number of attention heads
            oracle_num_layers: Oracle number of transformer layers
            oracle_use_text: Whether oracle uses text embeddings
            oracle_use_player_heroes: Whether oracle uses player hero features
            temperature: Optional sampling temperature for behavior policy
            policy_staleness_tolerance: Recent checkpoints within this tolerance
                contribute training data for their opponent steps as well.
            rating_manager: Optional TrueSkillRatingManager for stratified opponent sampling.
            num_strata: Number of rating strata for stratified sampling.
            use_mcts: Whether to use MCTS for training rollouts.
            mcts_config: Dict with MCTS hyperparameters (c_puct, num_simulations, top_k).
        """
        self.agent = agent
        self.oracle = oracle
        self.historical_prob = historical_prob
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.temperature = temperature
        self.policy_staleness_tolerance = policy_staleness_tolerance
        self.rating_manager = rating_manager
        self.num_strata = num_strata
        self.use_mcts = use_mcts
        self.mcts_config = mcts_config or {}

        # Parallel configuration
        self.use_parallel = use_parallel
        self.num_workers = num_workers

        # Oracle configuration for worker reconstruction
        self.oracle_config = {
            "embed_dim": oracle_embed_dim,
            "nhead": oracle_nhead,
            "num_layers": oracle_num_layers,
            "use_text": oracle_use_text,
            "use_player_heroes": oracle_use_player_heroes,
        }

        # Agent configuration for worker reconstruction
        self.agent_config = {
            "embed_dim": embed_dim,
            "nhead": nhead,
            "num_layers": num_layers,
        }

    def collect_batch(
        self,
        batch_samples: List[Dict[str, Any]],
        checkpoints: List,
        checkpoint_manager,
        batch_idx: int
    ) -> List[Dict[str, Any]]:
        """Collect rollouts for a batch.

        Args:
            batch_samples: List of samples for this batch
            checkpoints: List of available checkpoints
            checkpoint_manager: CheckpointManager instance
            batch_idx: Current batch index

        Returns:
            List of rollouts
        """
        if self.use_parallel and len(batch_samples) > 1:
            if self.use_mcts:
                return self._collect_batch_threaded(
                    batch_samples, checkpoints, checkpoint_manager, batch_idx
                )
            return self._collect_batch_parallel(
                batch_samples, checkpoints, checkpoint_manager, batch_idx
            )
        else:
            return self._collect_batch_sequential(
                batch_samples, checkpoints, checkpoint_manager, batch_idx
            )

    def _collect_batch_sequential(
        self,
        batch_samples: List[Dict[str, Any]],
        checkpoints: List,
        checkpoint_manager,
        batch_idx: int
    ) -> List[Dict[str, Any]]:
        """Sequential rollout collection (original implementation)."""
        batch_size = len(batch_samples)
        num_hist = int(batch_size * self.historical_prob)

        rollouts = []

        # Historical opponent rollouts
        if num_hist > 0 and checkpoints:
            hist_assignments = self._assign_historical_opponents(
                num_hist, checkpoints
            )
            rollouts.extend(self._collect_historical_rollouts(
                hist_assignments, batch_samples, checkpoints, checkpoint_manager
            ))

        # Self-play rollouts
        for i in range(num_hist, batch_size):
            sample = batch_samples[i]
            rollouts.append(collect_rollout(
                self.agent, self.oracle, sample,
                temperature=self.temperature,
                use_mcts=self.use_mcts,
                mcts_config=self.mcts_config,
            ))

        return rollouts

    def _collect_batch_parallel(
        self,
        batch_samples: List[Dict[str, Any]],
        checkpoints: List,
        checkpoint_manager,
        batch_idx: int
    ) -> List[Dict[str, Any]]:
        """Parallel rollout collection using ProcessPoolExecutor.
        
        Note: Historical opponent rollouts are still collected sequentially
        to leverage LRU caching. Only self-play rollouts are parallelized.
        """
        batch_size = len(batch_samples)
        num_hist = int(batch_size * self.historical_prob)

        rollouts = []
        rollout_indices = []  # Track original indices for ordering

        # Historical opponent rollouts - keep sequential for LRU cache efficiency
        if num_hist > 0 and checkpoints:
            hist_assignments = self._assign_historical_opponents(
                num_hist, checkpoints
            )
            hist_rollouts = self._collect_historical_rollouts(
                hist_assignments, batch_samples, checkpoints, checkpoint_manager
            )
            for (sample_idx, _), rollout in zip(hist_assignments, hist_rollouts):
                rollouts.append((sample_idx, rollout))

        # Self-play rollouts - parallelize these
        if num_hist < batch_size:
            # Get state dicts for worker processes
            agent_state_dict = self.agent.state_dict()
            oracle_state_dict = self.oracle.state_dict()
            
            # Prepare arguments for parallel execution
            worker_args = []
            for i in range(num_hist, batch_size):
                sample = batch_samples[i]
                worker_args.append((
                    sample,
                    agent_state_dict,
                    oracle_state_dict,
                    self.agent_config,
                    self.oracle_config,
                    None,  # opponent_state_dict
                    "radiant",  # current_side (self-play)
                    self.temperature,
                    self.use_mcts,
                    self.mcts_config,
                ))

            # Execute in parallel using ProcessPoolExecutor
            try:
                with ProcessPoolExecutor(
                    max_workers=min(self.num_workers, len(worker_args)),
                    mp_context=mp.get_context("spawn")
                ) as executor:
                    futures = {
                        executor.submit(_collect_rollout_worker, args): idx + num_hist
                        for idx, args in enumerate(worker_args)
                    }
                    
                    for future in as_completed(futures):
                        sample_idx = futures[future]
                        try:
                            rollout = future.result(timeout=300)  # 5 minute timeout
                            rollouts.append((sample_idx, rollout))
                        except Exception as e:
                            print(f"[RolloutCollector] Worker failed for sample {sample_idx}: {e}")
                            # Fallback to sequential collection on error
                            sample = batch_samples[sample_idx]
                            rollout = collect_rollout(
                                self.agent, self.oracle, sample,
                                temperature=self.temperature,
                                use_mcts=self.use_mcts,
                                mcts_config=self.mcts_config,
                            )
                            rollouts.append((sample_idx, rollout))
            except Exception as e:
                print(f"[RolloutCollector] Parallel execution failed: {e}")
                print("[RolloutCollector] Falling back to sequential collection for remaining samples")
                # Fallback to sequential for remaining samples
                for i in range(num_hist, batch_size):
                    if not any(idx == i for idx, _ in rollouts):
                        sample = batch_samples[i]
                        rollout = collect_rollout(
                            self.agent, self.oracle, sample,
                            temperature=self.temperature,
                            use_mcts=self.use_mcts,
                            mcts_config=self.mcts_config,
                        )
                        rollouts.append((i, rollout))

        # Sort by original index to maintain order
        rollouts.sort(key=lambda x: x[0])
        return [r for _, r in rollouts]

    def _collect_batch_threaded(
        self,
        batch_samples: List[Dict[str, Any]],
        checkpoints: List,
        checkpoint_manager,
        batch_idx: int
    ) -> List[Dict[str, Any]]:
        """Thread-based parallel rollout collection for MCTS (keeps models on GPU).

        Uses ThreadPoolExecutor so the agent/oracle stay on CUDA and multiple
        MCTS searches can overlap their forward passes on the GPU.
        """
        batch_size = len(batch_samples)
        num_hist = int(batch_size * self.historical_prob)

        rollouts = []

        # Historical opponent rollouts - keep sequential for LRU cache efficiency
        if num_hist > 0 and checkpoints:
            hist_assignments = self._assign_historical_opponents(
                num_hist, checkpoints
            )
            hist_rollouts = self._collect_historical_rollouts(
                hist_assignments, batch_samples, checkpoints, checkpoint_manager
            )
            for (sample_idx, _), rollout in zip(hist_assignments, hist_rollouts):
                rollouts.append((sample_idx, rollout))

        # Self-play rollouts - parallelize with threads (GPU-safe)
        if num_hist < batch_size:
            worker_args = []
            for i in range(num_hist, batch_size):
                sample = batch_samples[i]
                worker_args.append((
                    sample,
                    self.agent,
                    self.oracle,
                    self.temperature,
                    self.use_mcts,
                    self.mcts_config,
                ))

            def _thread_worker(args):
                sample, agent, oracle, temperature, use_mcts, mcts_config = args
                return collect_rollout(
                    agent, oracle, sample,
                    temperature=temperature,
                    use_mcts=use_mcts,
                    mcts_config=mcts_config,
                )

            try:
                with ThreadPoolExecutor(
                    max_workers=min(self.num_workers, len(worker_args))
                ) as executor:
                    futures = {
                        executor.submit(_thread_worker, args): idx + num_hist
                        for idx, args in enumerate(worker_args)
                    }

                    for future in as_completed(futures):
                        sample_idx = futures[future]
                        try:
                            rollout = future.result(timeout=300)
                            rollouts.append((sample_idx, rollout))
                        except Exception as e:
                            print(f"[RolloutCollector] Thread failed for sample {sample_idx}: {e}")
                            sample = batch_samples[sample_idx]
                            rollout = collect_rollout(
                                self.agent, self.oracle, sample,
                                temperature=self.temperature,
                                use_mcts=self.use_mcts,
                                mcts_config=self.mcts_config,
                            )
                            rollouts.append((sample_idx, rollout))
            except Exception as e:
                print(f"[RolloutCollector] Threaded execution failed: {e}")
                print("[RolloutCollector] Falling back to sequential collection for remaining samples")
                for i in range(num_hist, batch_size):
                    if not any(idx == i for idx, _ in rollouts):
                        sample = batch_samples[i]
                        rollout = collect_rollout(
                            self.agent, self.oracle, sample,
                            temperature=self.temperature,
                            use_mcts=self.use_mcts,
                            mcts_config=self.mcts_config,
                        )
                        rollouts.append((i, rollout))

        rollouts.sort(key=lambda x: x[0])
        return [r for _, r in rollouts]

    def _assign_historical_opponents(
        self, num_hist: int, checkpoints: List[tuple]
    ) -> List[tuple]:
        """Assign historical opponents to samples using stratified sampling by TrueSkill rating.

        If rating_manager is available, checkpoints are divided into rating strata
        and sampled uniformly across strata to ensure diverse opponent strengths.
        Otherwise falls back to uniform random sampling.

        Returns:
            List of (sample_idx, ckpt_idx) tuples
        """
        num_checkpoints = len(checkpoints)
        if num_checkpoints == 0 or num_hist <= 0:
            return []

        # Fallback: uniform random if no rating manager
        if self.rating_manager is None:
            assignments = []
            for i in range(num_hist):
                sample_idx = i
                ckpt_idx = random.randrange(num_checkpoints)
                assignments.append((sample_idx, ckpt_idx))
            return assignments

        # Stratified sampling by TrueSkill rating
        ratings = []
        default_rating = INITIAL_MU - 3 * INITIAL_SIGMA
        for idx, (ckpt_path, _) in enumerate(checkpoints):
            record = self.rating_manager.get_record(ckpt_path)
            rating = record.rating if record is not None else default_rating
            ratings.append((idx, rating))

        # Sort by rating and divide into strata
        ratings.sort(key=lambda x: x[1])
        num_strata = min(self.num_strata, num_checkpoints)
        strata_size = max(1, len(ratings) // num_strata)
        strata = []
        for i in range(num_strata):
            start = i * strata_size
            end = len(ratings) if i == num_strata - 1 else (i + 1) * strata_size
            strata.append(ratings[start:end])

        assignments = []
        for i in range(num_hist):
            # Round-robin across strata
            stratum_idx = i % num_strata
            # Find a non-empty stratum
            attempts = 0
            while not strata[stratum_idx] and attempts < num_strata:
                stratum_idx = (stratum_idx + 1) % num_strata
                attempts += 1
            if not strata[stratum_idx]:
                break
            chosen = random.choice(strata[stratum_idx])
            sample_idx = i
            ckpt_idx = chosen[0]
            assignments.append((sample_idx, ckpt_idx))

        return assignments

    def _collect_historical_rollouts(
        self, assignments: List[tuple], batch_samples: List[Dict[str, Any]],
        checkpoints: List, checkpoint_manager
    ) -> List[Dict[str, Any]]:
        """Collect rollouts against historical opponents."""
        rollouts = []

        # Group by checkpoint index
        ckpt_idx_to_samples = {}
        for sample_idx, ckpt_idx in assignments:
            ckpt_path = checkpoints[ckpt_idx][0]
            if ckpt_idx not in ckpt_idx_to_samples:
                ckpt_idx_to_samples[ckpt_idx] = []
            ckpt_idx_to_samples[ckpt_idx].append((sample_idx, ckpt_path))

        # Load each model once and collect all assigned rollouts
        for ckpt_idx, sample_list in ckpt_idx_to_samples.items():
            ckpt_path = sample_list[0][1]

            opponent = checkpoint_manager.load_opponent(ckpt_path)
            if opponent is None:
                # Fallback to self-play if loading fails
                for sample_idx, _ in sample_list:
                    rollouts.append(collect_rollout(
                        self.agent, self.oracle, batch_samples[sample_idx],
                        temperature=self.temperature,
                        use_mcts=self.use_mcts,
                        mcts_config=self.mcts_config,
                    ))
                continue

            for sample_idx, _ in sample_list:
                sample = batch_samples[sample_idx]
                current_side = random.choice(["radiant", "dire"])
                rollout = collect_rollout(
                    self.agent, self.oracle, sample,
                    opponent_agent=opponent, current_side=current_side,
                    temperature=self.temperature,
                    policy_staleness_tolerance=self.policy_staleness_tolerance,
                    opponent_staleness=ckpt_idx,
                    use_mcts=self.use_mcts,
                    mcts_config=self.mcts_config,
                )
                rollouts.append(rollout)

            # Note: opponent is NOT deleted here - it stays in LRU cache

        return rollouts

    def set_parallel(self, use_parallel: bool, num_workers: Optional[int] = None):
        """Enable or disable parallel collection at runtime.
        
        Args:
            use_parallel: Whether to use parallel collection
            num_workers: Optional new number of workers
        """
        self.use_parallel = use_parallel
        if num_workers is not None:
            self.num_workers = num_workers

    def get_stats(self) -> Dict[str, Any]:
        """Get collector statistics."""
        return {
            "use_parallel": self.use_parallel,
            "num_workers": self.num_workers,
            "historical_prob": self.historical_prob,
            "agent_config": self.agent_config,
            "oracle_config": self.oracle_config,
            "use_mcts": self.use_mcts,
            "mcts_config": self.mcts_config,
        }
