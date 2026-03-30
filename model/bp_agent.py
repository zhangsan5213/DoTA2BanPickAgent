"""BP Transformer Agent Model"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from utils.raw_data import NUM_HEROES, HERO_ID_FEATURE_MAP, NUM_HERO_FEATURES

ACTOR_DIM = 8
ACTION_DIM = 8
EMBED_DIM = 256


def init_weights(module):
    """
    统一的权重初始化函数
    - Embedding: uniform_(-0.1, 0.1)
    - Linear (policy head): orthogonal_(gain=0.01) + bias=0
    - Linear (value head): xavier_uniform_ + bias=0
    - LayerNorm: weight=1.0, bias=0.0
    """
    if isinstance(module, nn.Linear):
        # 为 value head 使用 Xavier 初始化
        if hasattr(module, "_is_value_head") or "value" in str(module):
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.orthogonal_(module.weight, gain=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1)
    elif isinstance(module, nn.LayerNorm):
        # 只有当 elementwise_affine=True 时才初始化 weight 和 bias
        if module.elementwise_affine:
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)


def init_embedding(module):
    """Embedding 层统一使用 uniform_(-0.1, 0.1)"""
    if isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.1, 0.1)


class ActionEncoder(nn.Module):
    """Encode BP actions: (actor_team, action_type, target_hero)"""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.team_embed = nn.Embedding(3, ACTOR_DIM)
        self.action_embed = nn.Embedding(3, ACTION_DIM)
        self.hero_embed = nn.Embedding(NUM_HEROES + 1, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(ACTOR_DIM + ACTION_DIM + embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, team_ids, action_ids, hero_ids):
        team_emb = self.team_embed(team_ids)
        action_emb = self.action_embed(action_ids)
        hero_emb = self.hero_embed(hero_ids)
        return self.fusion(torch.cat([team_emb, action_emb, hero_emb], dim=-1))


class PlayerEncoder(nn.Module):
    """Encode 5 players per team with their hero preferences"""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.hero_proj = nn.Linear(NUM_HEROES, embed_dim)
        self.player_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU()
        )

    def forward(self, player_feats):
        x = self.hero_proj(player_feats)
        return self.player_fusion(x)


class BPTransformerAgent(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, nhead=8, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heroes = NUM_HEROES

        self.action_encoder = ActionEncoder(embed_dim)
        self.player_encoder = PlayerEncoder(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(embed_dim, NUM_HEROES)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        # 为 value head 的线性层设置识别属性
        for i, module in enumerate(self.value_head):
            if isinstance(module, nn.Linear):
                setattr(module, "_is_value_head", True)

        # Precompute hero features
        self.register_buffer(
            "all_hero_attrs",
            torch.stack(
                [
                    HERO_ID_FEATURE_MAP.get(h, torch.zeros(NUM_HERO_FEATURES))
                    for h in range(1, NUM_HEROES + 1)
                ]
            ),
        )

        # 应用统一的权重初始化，保证初始策略分布均匀，避免初始策略塌缩
        self.apply(init_weights)

    def forward(self, state):
        """
        state: dict with keys:
            - radiant_player_feats: [B, 5, NUM_HEROES]
            - dire_player_feats: [B, 5, NUM_HEROES]
            - action_history: {teams, actions, heroes}
            - current_actor: [B]
            - current_action: [B]
        Returns:
            - action_logits: [B, NUM_HEROES]
            - value: [B, 1]
        """
        B = state["radiant_player_feats"].shape[0]

        r_player_emb = self.player_encoder(state["radiant_player_feats"])
        d_player_emb = self.player_encoder(state["dire_player_feats"])

        T = state["action_history"]["teams"].shape[1]
        if T > 0:
            action_emb = self.action_encoder(
                state["action_history"]["teams"].view(B * T),
                state["action_history"]["actions"].view(B * T),
                state["action_history"]["heroes"].view(B * T),
            ).view(B, T, -1)
        else:
            action_emb = torch.empty(
                B, 0, self.embed_dim, device=state["radiant_player_feats"].device
            )

        current_actor_emb = self.action_encoder.team_embed(state["current_actor"])
        current_action_emb = self.action_encoder.action_embed(state["current_action"])
        dummy_hero = torch.zeros(
            B, dtype=torch.long, device=state["radiant_player_feats"].device
        )
        current_hero_emb = self.action_encoder.hero_embed(dummy_hero)
        current_q = self.action_encoder.fusion(
            torch.cat([current_actor_emb, current_action_emb, current_hero_emb], dim=-1)
        ).unsqueeze(1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        all_players = torch.cat([r_player_emb, d_player_emb], dim=1)
        seq = torch.cat([cls_tokens, all_players, action_emb, current_q], dim=1)

        out = self.transformer(seq)

        policy_feat = out[:, -1, :]
        action_logits = self.policy_head(policy_feat)

        cls_feat = out[:, 0, :]
        value = self.value_head(cls_feat)

        return action_logits, value


if __name__ == "__main__":
    print("=" * 50)
    print("Testing BPTransformerAgent")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = BPTransformerAgent(embed_dim=128, nhead=8, num_layers=4).to(device)

    # Create dummy state
    B = 2
    state = {
        "radiant_player_feats": torch.randn(B, 5, NUM_HEROES).to(device),
        "dire_player_feats": torch.randn(B, 5, NUM_HEROES).to(device),
        "action_history": {
            "teams": torch.tensor([[0, 1, 0, 1]], device=device).repeat(B, 1),
            "actions": torch.tensor([[1, 1, 2, 2]], device=device).repeat(B, 1),
            "heroes": torch.tensor([[10, 20, 30, 40]], device=device).repeat(B, 1),
        },
        "current_actor": torch.tensor([0, 1], device=device),
        "current_action": torch.tensor([1, 2], device=device),
    }

    logits, value = agent(state)
    print(f"action_logits shape: {logits.shape}")  # [B, NUM_HEROES]
    print(f"value shape: {value.shape}")  # [B, 1]
    print(f"action_logits sample: {logits[0, :5]}")
    print(f"value sample: {value[0]}")

    # Test forward pass
    print("\n[OK] Forward pass successful!")
