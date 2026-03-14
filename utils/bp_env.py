"""BP Environment and RL utilities"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
from utils.raw_data import NUM_HEROES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BPState:
    """BP environment state"""
    def __init__(self, radiant_heroes, dire_heroes, radiant_players, dire_players, is_radiant_turn):
        self.radiant_heroes = list(radiant_heroes)
        self.dire_heroes = list(dire_heroes)
        self.radiant_players = radiant_players
        self.dire_players = dire_players
        self.history = {'teams': [], 'actions': [], 'heroes': []}
        self.is_radiant_turn = is_radiant_turn
        self.done = False
        self.pick_count = {'radiant': 0, 'dire': 0}
        self.ban_count = {'radiant': 0, 'dire': 0}

    def to_dict(self, device=DEVICE):
        # Handle both list and tensor inputs
        if not isinstance(self.radiant_players, torch.Tensor):
            r_feats = torch.tensor(self.radiant_players).float()
        else:
            r_feats = self.radiant_players.float()
        if not isinstance(self.dire_players, torch.Tensor):
            d_feats = torch.tensor(self.dire_players).float()
        else:
            d_feats = self.dire_players.float()
        return {
            'radiant_player_feats': r_feats.unsqueeze(0).to(device),
            'dire_player_feats': d_feats.unsqueeze(0).to(device),
            'action_history': {
                'teams': torch.tensor(self.history['teams'], dtype=torch.long, device=device).unsqueeze(0) if self.history['teams'] else torch.zeros(1, 0, dtype=torch.long, device=device),
                'actions': torch.tensor(self.history['actions'], dtype=torch.long, device=device).unsqueeze(0) if self.history['actions'] else torch.zeros(1, 0, dtype=torch.long, device=device),
                'heroes': torch.tensor(self.history['heroes'], dtype=torch.long, device=device).unsqueeze(0) if self.history['heroes'] else torch.zeros(1, 0, dtype=torch.long, device=device),
            },
            'current_actor': torch.tensor([0 if self.is_radiant_turn else 1], device=device),
            'current_action': torch.tensor([1 if self.pick_count['radiant'] + self.pick_count['dire'] < 10 else 2], device=device),
        }

    def step(self, hero_id, is_pick):
        self.history['teams'].append(0 if self.is_radiant_turn else 1)
        self.history['actions'].append(1 if is_pick else 2)
        self.history['heroes'].append(hero_id - 1)

        if is_pick:
            if self.is_radiant_turn:
                self.radiant_heroes.append(hero_id)
                self.pick_count['radiant'] += 1
            else:
                self.dire_heroes.append(hero_id)
                self.pick_count['dire'] += 1
        else:
            if self.is_radiant_turn:
                self.ban_count['radiant'] += 1
            else:
                self.ban_count['dire'] += 1

        total_picks = self.pick_count['radiant'] + self.pick_count['dire']
        if total_picks >= 10:
            self.done = True
        else:
            self.is_radiant_turn = not self.is_radiant_turn

    def get_valid_actions(self):
        used = set(self.radiant_heroes + self.dire_heroes)
        return [h for h in range(1, NUM_HEROES + 1) if h not in used]

    def get_reward(self, oracle):
        if not self.done:
            return None
        r_heroes = self.radiant_heroes[:5] + [0] * (5 - len(self.radiant_heroes))
        d_heroes = self.dire_heroes[:5] + [0] * (5 - len(self.dire_heroes))
        pred = oracle.predict(r_heroes, d_heroes, self.radiant_players, self.dire_players)
        return float(pred[0, 0])


# ============ RL Helpers ============
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2


def compute_gae(rewards, values, dones, gamma=GAMMA, lam=LAMBDA):
    """Compute GAE advantages and returns"""
    # Handle 1D input (single trajectory)
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(-1)
        values = values.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    gae = 0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    # Squeeze back to 1D if input was 1D
    if advantages.dim() == 2 and advantages.shape[1] == 1:
        advantages = advantages.squeeze(-1)
        returns = returns.squeeze(-1)

    return advantages, returns


def ppo_loss(log_probs, old_log_probs, advantages, clip_eps=CLIP_EPS):
    """PPO policy loss"""
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def collect_rollout(agent, oracle, sample, max_steps=20):
    """Collect one BP trajectory"""
    s = BPState([], [], sample['r_players'], sample['d_players'], is_radiant_turn=True)

    states, actions, log_probs, values, rewards = [], [], [], [], []

    step = 0
    while not s.done and step < max_steps:
        state_dict = s.to_dict()
        with torch.no_grad():
            action_logits, value = agent(state_dict)

        valid_actions = s.get_valid_actions()
        if not valid_actions:
            break

        mask = torch.full((NUM_HEROES,), -1e9, device=action_logits.device)
        for h in valid_actions:
            mask[h - 1] = 0.0
        action_logits = action_logits + mask

        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        hero_id = dist.sample().item() + 1
        log_prob = dist.log_prob(torch.tensor(hero_id - 1, device=action_logits.device))

        is_pick = s.pick_count['radiant'] + s.pick_count['dire'] < 10
        s.step(hero_id, is_pick)

        states.append(state_dict)
        actions.append(hero_id - 1)
        log_probs.append(log_prob)
        values.append(value.item())
        step += 1

    final_reward = s.get_reward(oracle)
    if final_reward is None:
        final_reward = 0.5

    rewards = [0.0] * (len(states) - 1) + [final_reward]

    return {
        'states': states,
        'actions': torch.tensor(actions, dtype=torch.long),
        'log_probs': torch.stack(log_probs),
        'values': torch.tensor(values + [final_reward], dtype=torch.float32),
        'rewards': torch.tensor(rewards, dtype=torch.float32),
    }


if __name__ == "__main__":
    print("=" * 50)
    print("Testing BP Environment")
    print("=" * 50)

    # Test BPState
    player_feats = [[0.0] * NUM_HEROES for _ in range(5)]
    player_feats[0][10] = 0.6  # 玩家1擅长英雄10

    s = BPState([], [], player_feats, player_feats, is_radiant_turn=True)

    print(f"Initial state - radiant turn: {s.is_radiant_turn}")
    print(f"Valid actions count: {len(s.get_valid_actions())}")

    # Test step
    s.step(10, is_pick=True)
    print(f"After pick hero 10 - radiant heroes: {s.radiant_heroes}")
    print(f"Valid actions count: {len(s.get_valid_actions())}")

    # Test to_dict
    state_dict = s.to_dict()
    print(f"state_dict keys: {state_dict.keys()}")
    print(f"radiant_player_feats shape: {state_dict['radiant_player_feats'].shape}")
    print(f"action_history teams: {state_dict['action_history']['teams']}")

    # Test GAE
    print("\n--- Testing GAE ---")
    rewards = torch.tensor([0.0, 0.0, 0.8, 0.9])
    values = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.85])
    dones = torch.tensor([0.0, 0.0, 0.0, 0.0])
    advantages, returns = compute_gae(rewards, values, dones)
    print(f"advantages: {advantages}")
    print(f"returns: {returns}")

    # Test PPO loss
    print("\n--- Testing PPO Loss ---")
    log_probs = torch.tensor([-1.0, -1.5, -2.0, -1.8])
    old_log_probs = torch.tensor([-1.1, -1.4, -1.9, -1.9])
    advantages = torch.tensor([0.1, 0.2, -0.1, 0.15])
    loss = ppo_loss(log_probs, old_log_probs, advantages)
    print(f"PPO loss: {loss.item():.4f}")

    print("\n[OK] All tests passed!")
