"""BP Transformer Agent Model"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from model.hero_encoder import MultiModalHeroEncoder
from utils.raw_data import NUM_HEROES, HERO_ID_FEATURE_MAP, NUM_HERO_FEATURES, HERO_ID_SEMANTIC_MAP

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
        elif hasattr(module, "_is_policy_head"):
            # policy head 使用更大的 gain，让初始 logits 有更大方差，避免卡在均匀分布
            nn.init.orthogonal_(module.weight, gain=0.1)
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
    """Encode BP actions: (actor_team, action_type, target_hero_emb)"""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.team_embed = nn.Embedding(3, ACTOR_DIM)
        self.action_embed = nn.Embedding(3, ACTION_DIM)
        self.fusion = nn.Sequential(
            nn.Linear(ACTOR_DIM + ACTION_DIM + embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(self, team_ids, action_ids, hero_embs):
        team_emb = self.team_embed(team_ids)
        action_emb = self.action_embed(action_ids)
        return self.fusion(torch.cat([team_emb, action_emb, hero_embs], dim=-1))


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
    def __init__(self, embed_dim=EMBED_DIM, nhead=8, num_layers=6, learnable_temperature=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heroes = NUM_HEROES

        # Temperature for action sampling (train: high for exploration, eval: low for exploitation)
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))
        self.learnable_temperature = learnable_temperature

        # Hero encoder: 使用多模态英雄编码器（属性 + 语义）
        self.hero_encoder = MultiModalHeroEncoder(
            embed_dim=embed_dim,
            id_hidden_dim=128,
            attr_hidden_dim=64,
            use_text=True,
            text_embed_dim=1024,
            text_hidden_dim=128,
            dropout=0.1,
            num_res_layers=3,
            attn_heads=4,
            modality_dropout=0.1,
        )

        self.action_encoder = ActionEncoder(embed_dim)
        self.player_encoder = PlayerEncoder(embed_dim)
        self.cls_tokens = nn.Parameter(torch.randn(1, 2, embed_dim))  # [radiant, dire]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.policy_head = nn.Linear(embed_dim, NUM_HEROES)
        setattr(self.policy_head, "_is_policy_head", True)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
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
            persistent=False,
        )

        first_hero_sem = next(iter(HERO_ID_SEMANTIC_MAP.values()))
        default_device = first_hero_sem.device
        self.register_buffer(
            "all_hero_sem",
            torch.stack(
                [
                    HERO_ID_SEMANTIC_MAP.get(
                        h, torch.zeros(1024, device=default_device)
                    )
                    for h in range(1, NUM_HEROES + 1)
                ]
            ),
            persistent=False,
        )

        # 应用统一的权重初始化，保证初始策略分布均匀，避免初始策略塌缩
        self.apply(init_weights)

    def get_temperature(self):
        """获取当前温度（用于动作采样），保证最小值防止除零。"""
        return self.temperature.clamp(min=0.1)

    def hero_input_from_ids(self, hero_ids: torch.Tensor):
        """根据英雄ID快速获取预计算的属性和语义特征

        Args:
            hero_ids: 任意形状，英雄ID（1-based, 1-160；0表示无效/padding）
        Returns:
            indices: 0-based 索引 (0-159)
            attrs: 英雄属性
            sem: 英雄语义
        """
        device = hero_ids.device
        indices = hero_ids - 1
        indices = torch.clamp(indices, min=0, max=NUM_HEROES - 1)
        attrs = self.all_hero_attrs.to(device)[indices]
        sem = self.all_hero_sem.to(device)[indices]
        return indices, attrs, sem

    def encode_hero_ids(self, hero_ids: torch.Tensor):
        """将英雄ID编码为统一的多模态嵌入向量

        Args:
            hero_ids: 任意形状，英雄ID（1-based）
        Returns:
            hero_embs: [*original_shape, embed_dim]
        """
        original_shape = hero_ids.shape
        flat_ids = hero_ids.view(-1)
        indices, attrs, sem = self.hero_input_from_ids(flat_ids)

        # hero_encoder 期望 [batch, seq_len]
        indices = indices.unsqueeze(-1)  # [N, 1]
        attrs = attrs.unsqueeze(1)       # [N, 1, F]
        sem = sem.unsqueeze(1)           # [N, 1, S]

        encoded = self.hero_encoder(indices, attrs, sem)  # [N, 1, embed_dim]
        encoded = encoded.squeeze(1)  # [N, embed_dim]

        return encoded.view(*original_shape, -1)

    def load_state_dict(self, state_dict, strict=True):
        """兼容旧 checkpoint：自动补全缺失的参数。"""
        if 'temperature' not in state_dict and hasattr(self, 'temperature'):
            state_dict['temperature'] = torch.ones_like(self.temperature)
        if 'learnable_temperature' not in state_dict:
            pass
        # 兼容旧 checkpoint 的单 cls_token -> 双 cls_tokens
        if 'cls_token' in state_dict and 'cls_tokens' not in state_dict:
            old_cls = state_dict.pop('cls_token')  # [1, 1, embed_dim]
            state_dict['cls_tokens'] = torch.cat([old_cls, old_cls.clone()], dim=1)
        # 兼容旧 checkpoint 缺少 hero_encoder 的情况
        has_hero_encoder = any(k.startswith('hero_encoder') for k in state_dict.keys())
        if not has_hero_encoder:
            strict = False
            print("[!] Loading old checkpoint without hero_encoder. Hero encoder will be randomly initialized.")
        super().load_state_dict(state_dict, strict=strict)

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
            history_hero_embs = self.encode_hero_ids(
                state["action_history"]["heroes"].view(B * T)
            ).view(B, T, -1)
            action_emb = self.action_encoder(
                state["action_history"]["teams"].view(B * T),
                state["action_history"]["actions"].view(B * T),
                history_hero_embs.view(B * T, -1),
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
        current_hero_emb = self.encode_hero_ids(dummy_hero)
        current_q = self.action_encoder.fusion(
            torch.cat([current_actor_emb, current_action_emb, current_hero_emb], dim=-1)
        ).unsqueeze(1)

        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        all_players = torch.cat([r_player_emb, d_player_emb], dim=1)
        seq = torch.cat([cls_tokens, all_players, action_emb, current_q], dim=1)

        out = self.transformer(seq)

        policy_feat = out[:, -1, :]
        action_logits = self.policy_head(policy_feat)

        radiant_cls_feat = out[:, 0, :]
        dire_cls_feat = out[:, 1, :]
        current_actor = state["current_actor"].unsqueeze(-1)  # [B, 1]
        cls_feat = torch.where(current_actor == 0, radiant_cls_feat, dire_cls_feat)
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
