import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch

from model.bp_agent import *
from utils.raw_data import NUM_HEROES

def test_bp_agent():
    """测试BP Agent"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    agent = BPAgent(
        embed_dim=128,
        nhead=4,
        num_layers=2,
        use_text=True,
        use_player_heroes=True,
    ).to(device)

    B = 4
    T = 10
    K = 20

    hero_ids = torch.randint(1, 130, (B, T), device=device)
    team_flags = torch.randint(0, 2, (B, T), device=device)
    action_types = torch.randint(0, 2, (B, T), device=device)
    valid_mask = torch.ones(B, T, dtype=torch.long, device=device)

    radiant_player_feats = torch.rand(B, 5, NUM_HEROES, device=device)
    dire_player_feats = torch.rand(B, 5, NUM_HEROES, device=device)

    state_feat = agent.encode_state(
        hero_ids=hero_ids,
        team_flags=team_flags,
        action_types=action_types,
        valid_mask=valid_mask,
        radiant_player_feats=radiant_player_feats,
        dire_player_feats=dire_player_feats,
    )
    print(f"State feature shape: {state_feat.shape}")

    candidate_ids = torch.randint(1, 130, (B, K), device=device)

    action, log_prob, value = agent.get_action(
        state_feat=state_feat,
        candidate_hero_ids=candidate_ids,
        deterministic=False,
    )
    print(f"Action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value.squeeze()}")

    log_prob_eval, value_eval, entropy = agent.evaluate_actions(
        state_feat=state_feat,
        candidate_hero_ids=candidate_ids,
        actions=action,
    )
    print(f"Eval log_prob: {log_prob_eval}")
    print(f"Eval entropy: {entropy}")

    print("\nTest passed!")

if __name__ == "__main__":
    test_bp_agent()