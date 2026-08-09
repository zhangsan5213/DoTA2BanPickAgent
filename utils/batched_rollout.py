"""Fully batched rollout collection - runs MCTS for all rollouts in parallel."""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.bp_env import BPState
from utils.raw_data import NUM_HEROES, get_valid_hero_ids, STATIC_HERO_MASK
from search.mcts_batched import BatchMCTSEngine, BatchedMCTSNode


@dataclass
class RolloutState:
    """Tracks the state of an individual rollout during collection."""
    sample: Dict[str, Any]
    state: BPState
    opponent_agent: Optional[Any] = None
    current_side: str = "radiant"
    is_current: bool = True  # Whether this step contributes to training

    # Collected data
    states: List[Dict[str, torch.Tensor]] = None
    actions: List[int] = None
    log_probs: List[torch.Tensor] = None
    values: List[float] = None
    rewards: List[float] = None
    valid_mask: List[bool] = None
    step_teams: List[int] = None
    mcts_policies: List[Optional[torch.Tensor]] = None

    def __post_init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.valid_mask = []
        self.step_teams = []
        self.mcts_policies = []


def collect_batched_rollouts(
    agent,
    oracle,
    samples: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    use_mcts: bool = True,
    mcts_config: Optional[Dict[str, Any]] = None,
    opponent_agents: Optional[List[Tuple[Any, str, int]]] = None,
    policy_staleness_tolerance: int = 0,
) -> List[Dict[str, Any]]:
    """Collect multiple rollouts in parallel using batched MCTS.

    Args:
        agent: Current training agent
        oracle: Win rate oracle
        samples: List of sample data for rollouts
        temperature: Optional sampling temperature
        use_mcts: Whether to use MCTS
        mcts_config: MCTS hyperparameters
        opponent_agents: Optional list of (opponent_agent, current_side, staleness) tuples
        policy_staleness_tolerance: Tolerance for using stale opponent data

    Returns:
        List of rollout dictionaries
    """
    device = next(agent.parameters()).device
    mcts_config = mcts_config or {}

    # Initialize all rollout states
    rollout_states = []
    for idx, sample in enumerate(samples):
        player_feats_r = sample["r_players"]
        player_feats_d = sample["d_players"]

        state = BPState(
            [], [], player_feats_r, player_feats_d,
            radiant_bans=[], dire_bans=[],
            is_radiant_turn=True, step_idx=0,
        )

        # Determine opponent setup
        opponent_agent = None
        current_side = "radiant"
        is_current = True

        if opponent_agents and idx < len(opponent_agents) and opponent_agents[idx] is not None:
            opponent_agent, current_side, staleness = opponent_agents[idx]
            # Determine if this counts as current for training
            if opponent_agent is not None:
                is_current = staleness <= policy_staleness_tolerance

        rollout_states.append(RolloutState(
            sample=sample,
            state=state,
            opponent_agent=opponent_agent,
            current_side=current_side,
            is_current=is_current,
        ))

    # Initialize MCTS engine
    mcts_engine = None
    if use_mcts:
        mcts_engine = BatchMCTSEngine(
            agent, oracle,
            c_puct=mcts_config.get("c_puct", 1.5),
            num_simulations=mcts_config.get("num_simulations", 64),
            top_k=mcts_config.get("top_k", 20),
            dirichlet_alpha=mcts_config.get("dirichlet_alpha", 0.0),
            dirichlet_epsilon=mcts_config.get("dirichlet_epsilon", 0.0),
            max_search_depth=mcts_config.get("max_search_depth", 0),
        )

    # Persistent MCTS trees (GITCGRL style)
    # game_roots[i]  = GameRoot (root_0) for anchor + cross-step backprop
    # pivots[i]      = CurrentNodes (BattleNode) for starting playouts
    game_roots = [None] * len(rollout_states)
    pivots = [None] * len(rollout_states)

    # Main rollout loop
    max_steps = 24
    active_indices = list(range(len(rollout_states)))

    with tqdm(total=max_steps, desc="Rollout Progress", leave=True, ncols=90) as pbar:
        for step in range(max_steps):
            if not active_indices:
                break

            # Collect states that need decisions
            states_for_mcts = []
            states_for_policy = []
            rollout_decision_info = []

            for idx in active_indices:
                rs = rollout_states[idx]
                s = rs.state

                if s.done:
                    continue

                # Determine active agent
                is_radiant_turn = s.is_radiant_turn
                active_agent = agent

                if rs.opponent_agent is not None:
                    if is_radiant_turn:
                        active_agent = agent if rs.current_side == "radiant" else rs.opponent_agent
                    else:
                        active_agent = agent if rs.current_side == "dire" else rs.opponent_agent

                # Check if this is a current agent step (for training)
                if rs.opponent_agent is None:
                    step_is_current = True
                else:
                    step_is_current = (
                        (rs.current_side == "radiant")
                        if is_radiant_turn
                        else (rs.current_side == "dire")
                    )
                    if not step_is_current:
                        # Check staleness
                        step_is_current = rs.is_current

                # Save info for this decision
                state_dict = s.to_dict(device=device)
                current_team = 0 if is_radiant_turn else 1

                rollout_decision_info.append((
                    idx, state_dict, s, active_agent, step_is_current, current_team
                ))

                if use_mcts and step_is_current:
                    states_for_mcts.append((idx, s))
                else:
                    states_for_policy.append((idx, state_dict, active_agent))

            # Make decisions in batch
            decisions = {}  # idx -> (hero_id, log_prob, value, mcts_policy_tensor)

            # Batch MCTS decisions
            if states_for_mcts and mcts_engine:
                mcts_indices = [idx for idx, _ in states_for_mcts]
                mcts_root_states = [s for _, s in states_for_mcts]
                # GITCGRL: game_root for anchor, pivot_node for playout start
                mcts_prev_roots = [game_roots[i] for i in mcts_indices]
                mcts_pivots = [pivots[i] for i in mcts_indices]

                # Run batch MCTS
                mcts_results = mcts_engine.search_batch(
                    mcts_root_states,
                    prev_roots=mcts_prev_roots,
                    pivots=mcts_pivots,
                )

                # Get logits and values for the same states for old_log_prob calculation
                batch_state_dicts = [s.to_dict(device=device) for _, s in states_for_mcts]
                with torch.no_grad():
                    if batch_state_dicts:
                        # Collate and run batch
                        batch_dict = _collate_state_dicts(batch_state_dicts)
                        batch_logits, batch_values = agent(batch_dict)

                # Process MCTS results
                for i, (idx, _) in enumerate(states_for_mcts):
                    hero_id, mcts_policy, next_root = mcts_results[i]

                    # Store game_root on first encounter
                    if game_roots[idx] is None and next_root is not None:
                        top = next_root
                        while top.parent is not None:
                            top = top.parent
                        game_roots[idx] = top

                    # GITCGRL NavigateTree: descend into ActionNode → BattleNode
                    # next_root is the ActionNode child. Navigate to its BattleNode.
                    if next_root is not None and next_root._is_action_node:
                        if "_battle" not in next_root.children:
                            battle = BatchedMCTSNode(parent=next_root, action=None)
                            battle._depth_from_root = next_root._depth_from_root + 1
                            battle._is_action_node = False
                            next_root.children["_battle"] = battle
                        pivots[idx] = next_root.children["_battle"]

                    # Convert MCTS policy to tensor
                    mcts_policy_tensor = torch.zeros(NUM_HEROES, device=device)
                    for h, p in mcts_policy.items():
                        mcts_policy_tensor[h - 1] = p

                    # Get log_prob from original policy (temperature=1.0)
                    logits = batch_logits[i]
                    value = batch_values[i].item()

                    # Get valid actions and mask - use precomputed static mask
                    s = states_for_mcts[i][1]
                    # Start with precomputed static mask (shape: [NUM_HEROES+1], index 0 = hero_id 0)
                    mask = STATIC_HERO_MASK[1:NUM_HEROES+1].to(device).clone()  # [NUM_HEROES], index 0 = hero_id 1
                    # Mask out used heroes
                    used = set(s.radiant_heroes + s.dire_heroes + s.radiant_bans + s.dire_bans)
                    for h in used:
                        if 1 <= h <= NUM_HEROES:
                            mask[h - 1] = -1e9
                    masked_logits = logits + mask

                    # Compute log_prob
                    target_probs = F.softmax(masked_logits, dim=-1)
                    target_dist = torch.distributions.Categorical(target_probs)
                    log_prob = target_dist.log_prob(torch.tensor(hero_id - 1, device=device))

                    decisions[idx] = (hero_id, log_prob, value, mcts_policy_tensor)

            # Batch policy decisions (non-MCTS or opponent steps)
            if states_for_policy:
                # Group by agent for batch processing
                agent_to_states = {}
                for idx, state_dict, active_agent in states_for_policy:
                    agent_key = id(active_agent)
                    if agent_key not in agent_to_states:
                        agent_to_states[agent_key] = (active_agent, [])
                    agent_to_states[agent_key][1].append((idx, state_dict))

                # Process each agent's states in batch
                for active_agent, idx_state_list in agent_to_states.values():
                    indices = [idx for idx, _ in idx_state_list]
                    state_dicts = [sd for _, sd in idx_state_list]

                    # Batch forward pass
                    with torch.no_grad():
                        batch_dict = _collate_state_dicts(state_dicts)
                        batch_logits, batch_values = active_agent(batch_dict)

                    # Process each decision
                    for i, idx in enumerate(indices):
                        logits = batch_logits[i]
                        value = batch_values[i].item()

                        # Get the state to get valid actions
                        rs = rollout_states[idx]
                        s = rs.state
                        valid_actions = s.get_valid_actions()

                        # Build mask - use precomputed static mask
                        # Start with precomputed static mask (shape: [NUM_HEROES+1], index 0 = hero_id 0)
                        mask = STATIC_HERO_MASK[1:NUM_HEROES+1].to(device).clone()  # [NUM_HEROES], index 0 = hero_id 1
                        # Mask out used heroes
                        used = set(s.radiant_heroes + s.dire_heroes + s.radiant_bans + s.dire_bans)
                        for h in used:
                            if 1 <= h <= NUM_HEROES:
                                mask[h - 1] = -1e9
                        masked_logits = logits + mask

                        # Sample action
                        if temperature is None and hasattr(active_agent, 'get_temperature'):
                            temp = active_agent.get_temperature().item()
                        else:
                            temp = temperature if temperature is not None else 1.0

                        sample_probs = F.softmax(masked_logits / temp, dim=-1)
                        sample_dist = torch.distributions.Categorical(sample_probs)
                        hero_id_idx = sample_dist.sample().item()
                        hero_id = hero_id_idx + 1

                        # Compute log_prob with temperature=1.0
                        target_probs = F.softmax(masked_logits, dim=-1)
                        target_dist = torch.distributions.Categorical(target_probs)
                        log_prob = target_dist.log_prob(torch.tensor(hero_id_idx, device=device))

                        decisions[idx] = (hero_id, log_prob, value, None)

            # Build decision info dictionary for O(1) lookups
            decision_info_dict = {di[0]: di for di in rollout_decision_info}

            # Apply decisions to all rollouts
            new_active_indices = []
            for idx in active_indices:
                rs = rollout_states[idx]

                if rs.state.done:
                    continue

                if idx not in decisions:
                    new_active_indices.append(idx)
                    continue

                hero_id, log_prob, value, mcts_policy_tensor = decisions[idx]

                # Reset persistent tree if this step was NOT an MCTS decision
                step_used_mcts = use_mcts and step_is_current and active_agent is agent
                if not step_used_mcts:
                    game_roots[idx] = None
                    pivots[idx] = None

                # Get decision info (O(1) lookup)
                decision_info = decision_info_dict.get(idx)
                if decision_info is None:
                    new_active_indices.append(idx)
                    continue

                _, state_dict, _, _, step_is_current, current_team = decision_info

                # Record data
                rs.states.append(state_dict)
                rs.actions.append(hero_id - 1)
                rs.log_probs.append(log_prob)
                rs.values.append(value)
                rs.valid_mask.append(step_is_current)
                rs.step_teams.append(current_team)
                rs.mcts_policies.append(mcts_policy_tensor)

                # Step the state
                rs.state.step(hero_id)

                if not rs.state.done:
                    new_active_indices.append(idx)

            active_indices = new_active_indices
            pbar.update(1)
            pbar.set_postfix({"Active Rollouts": len(active_indices)})

    # Finalize all rollouts
    rollouts = []
    for rs in rollout_states:
        final_reward = rs.state.get_reward(oracle)
        if final_reward is None:
            final_reward = 0.5

        # Compute rewards
        rewards = [0.0] * (len(rs.states) - 1) + [final_reward]

        # Get final value
        final_state_dict = rs.state.to_dict(device=device)
        with torch.no_grad():
            _, final_value = agent(final_state_dict)
        final_value = final_value.item()

        # Process mcts_policies
        mcts_policies_stacked = None
        if any(p is not None for p in rs.mcts_policies):
            processed = [
                p if p is not None else torch.zeros(NUM_HEROES, device=device)
                for p in rs.mcts_policies
            ]
            mcts_policies_stacked = torch.stack(processed)

        rollout = {
            "states": rs.states,
            "actions": torch.tensor(rs.actions, dtype=torch.long),
            "log_probs": torch.stack(rs.log_probs) if rs.log_probs else torch.tensor([]),
            "values": torch.tensor(rs.values + [final_value], dtype=torch.float32),
            "rewards": torch.tensor(rewards, dtype=torch.float32),
            "valid_mask": torch.tensor(rs.valid_mask, dtype=torch.bool),
            "step_teams": torch.tensor(rs.step_teams, dtype=torch.long),
            "mcts_policies": mcts_policies_stacked,
        }
        rollouts.append(rollout)

    return rollouts


def _collate_state_dicts(state_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate multiple state dicts into a batch."""
    def collate_recursive(dicts_list):
        if not dicts_list:
            return {}

        first = dicts_list[0]
        if isinstance(first, dict):
            batch = {}
            for key in first.keys():
                values = [d[key] for d in dicts_list]
                batch[key] = collate_recursive(values)
            return batch
        elif isinstance(first, torch.Tensor):
            return torch.cat(dicts_list, dim=0)
        else:
            return dicts_list

    return collate_recursive(state_dicts)
