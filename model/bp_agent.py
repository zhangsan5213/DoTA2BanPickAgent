import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hero_encoder import MultiModalHeroEncoder
from model.win_rate_oracle import PlayerHeroEncoder
from utils.raw_data import NUM_HEROES, NUM_HERO_FEATURES


class BPStateEncoder(nn.Module):
    """
    BP状态编码器：将当前BP局面编码成固定维度的向量
    处理不定长的pick/ban序列，使用自注意力聚合信息

    输入:
        - hero_ids: [B, T] 英雄ID序列 (0表示空位)
        - hero_attrs: [B, T, NUM_HERO_FEATURES] 英雄属性
        - hero_semantics: [B, T, text_dim] 英雄语义 (可选)
        - team_flags: [B, T] 团队标记 (0=天辉, 1=夜魇)
        - action_types: [B, T] 行动类型 (0=ban, 1=pick)
        - player_feats: [B, 2, 5, NUM_HEROES] 玩家英雄偏好 (可选)
    输出:
        - state_feat: [B, embed_dim] 状态特征向量
    """
    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        use_text: bool = True,
        use_player_heroes: bool = True,
        # HeroEncoder 参数
        hero_encoder_id_dim: int = 128,
        hero_encoder_attr_dim: int = 64,
        hero_encoder_text_dim: int = 128,
        hero_encoder_dropout: float = 0.1,
        hero_encoder_res_layers: int = 3,
        hero_encoder_attn_heads: int = 4,
        hero_encoder_modality_dropout: float = 0.1,
        # 额外参数
        max_seq_len: int = 24,  # 最大pick/ban序列长度
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_text = use_text
        self.use_player_heroes = use_player_heroes
        self.max_seq_len = max_seq_len

        # 1. Hero Encoder (复用oracle的编码器)
        self.hero_encoder = MultiModalHeroEncoder(
            embed_dim=embed_dim,
            id_hidden_dim=hero_encoder_id_dim,
            attr_hidden_dim=hero_encoder_attr_dim,
            use_text=use_text,
            text_embed_dim=1024,
            text_hidden_dim=hero_encoder_text_dim,
            dropout=hero_encoder_dropout,
            num_res_layers=hero_encoder_res_layers,
            attn_heads=hero_encoder_attn_heads,
            modality_dropout=hero_encoder_modality_dropout,
        )

        # 2. 位置编码 (用于表示pick/ban的顺序)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        # 3. 团队编码 (天辉/夜魇)
        self.team_embed = nn.Embedding(2, embed_dim)

        # 4. 行动类型编码 (ban/pick)
        self.action_type_embed = nn.Embedding(2, embed_dim)

        # 5. 有效位置mask embedding (区分有效英雄和padding)
        self.valid_embed = nn.Embedding(2, embed_dim)

        # 6. 融合投影层
        self.fusion_proj = nn.Linear(embed_dim * 4, embed_dim)

        # 7. Transformer (自注意力处理序列)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 8. 玩家英雄编码器 (可选)
        if self.use_player_heroes:
            self.player_encoder = PlayerHeroEncoder(
                num_heroes=NUM_HEROES,
                hidden_dim=128,
                embed_dim=embed_dim
            )
            # 状态 + 两队玩家特征
            head_input_dim = embed_dim + embed_dim * 2
        else:
            head_input_dim = embed_dim

        # 9. 最终状态投影
        self.state_head = nn.Sequential(
            nn.Linear(head_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )

        # 预计算所有英雄的静态特征
        from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP
        self.register_buffer("all_hero_attrs", torch.stack([
            HERO_ID_FEATURE_MAP.get(hero_id, torch.zeros(NUM_HERO_FEATURES))
            for hero_id in range(1, NUM_HEROES + 1)
        ]), persistent=False)

        if use_text:
            # 确保默认tensor使用与map中相同device
            sample_tensor = next(iter(HERO_ID_SEMANTIC_MAP.values())) if len(HERO_ID_SEMANTIC_MAP) > 0 else None
            default_device = sample_tensor.device if sample_tensor is not None else torch.device('cpu')
            self.register_buffer("all_hero_sem", torch.stack([
                HERO_ID_SEMANTIC_MAP.get(hero_id, torch.zeros(1024, device=default_device))
                for hero_id in range(1, NUM_HEROES + 1)
            ]), persistent=False)
        else:
            self.all_hero_sem = None

    def forward(
        self,
        hero_ids: torch.Tensor,
        hero_attrs: torch.Tensor = None,
        hero_semantics: torch.Tensor = None,
        team_flags: torch.Tensor = None,
        action_types: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
        radiant_player_feats: torch.Tensor = None,
        dire_player_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            hero_ids: [B, T] 英雄ID (0表示空/padding)
            hero_attrs: [B, T, NUM_HERO_FEATURES] 英雄属性
            hero_semantics: [B, T, text_dim] 英雄语义
            team_flags: [B, T] 团队标记 (0=天辉, 1=夜魇)
            action_types: [B, T] 行动类型 (0=ban, 1=pick)
            valid_mask: [B, T] 有效位置标记 (1=有效, 0=padding)
            radiant_player_feats: [B, 5, NUM_HEROES] 天辉玩家偏好
            dire_player_feats: [B, 5, NUM_HEROES] 夜魇玩家偏好
        Returns:
            state_feat: [B, embed_dim] 编码后的状态向量
        """
        B, T = hero_ids.shape
        device = hero_ids.device

        # --- 1. 英雄特征编码 ---
        # 将 1-based hero_ids 转换为 0-based indices 用于 indexing
        # hero_id=0 (padding) -> index=0 (clamp后，实际不会用到因为valid_mask会过滤)
        indices = hero_ids - 1
        indices = torch.clamp(indices, min=0, max=NUM_HEROES - 1)
        
        if hero_attrs is None:
            hero_attrs = self.all_hero_attrs[indices]

        if hero_semantics is None and self.use_text and self.all_hero_sem is not None:
            hero_semantics = self.all_hero_sem[indices]

        # hero_encoder 需要 0-based indices
        hero_emb = self.hero_encoder(indices, hero_attrs, hero_semantics)

        # --- 2. 添加位置编码 ---
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.pos_embedding(positions)

        # --- 3. 添加团队编码 ---
        if team_flags is None:
            team_flags = torch.zeros(B, T, dtype=torch.long, device=device)
        team_emb = self.team_embed(team_flags)

        # --- 4. 添加行动类型编码 ---
        if action_types is None:
            action_types = torch.ones(B, T, dtype=torch.long, device=device)
        action_emb = self.action_type_embed(action_types)

        # --- 5. 添加有效位置标记 ---
        if valid_mask is None:
            valid_mask = (hero_ids != 0).long()
        valid_emb = self.valid_embed(valid_mask)

        # --- 6. 融合所有特征 ---
        token_emb = hero_emb + pos_emb + team_emb + action_emb + valid_emb

        # --- 7. Transformer 自注意力 ---
        attn_mask = (valid_mask == 0)
        encoded_seq = self.transformer(token_emb, src_key_padding_mask=attn_mask)

        # --- 8. 聚合序列特征 (mean pooling) ---
        valid_mask_exp = valid_mask.unsqueeze(-1).float()
        pooled = (encoded_seq * valid_mask_exp).sum(dim=1) / (valid_mask_exp.sum(dim=1) + 1e-8)

        # --- 9. 融合玩家特征 ---
        if self.use_player_heroes:
            r_player_feat = self.player_encoder(radiant_player_feats) if radiant_player_feats is not None else None
            d_player_feat = self.player_encoder(dire_player_feats) if dire_player_feats is not None else None

            if r_player_feat is not None and d_player_feat is not None:
                combined_feat = torch.cat([pooled, r_player_feat, d_player_feat], dim=-1)
            elif r_player_feat is not None:
                combined_feat = torch.cat([pooled, r_player_feat, torch.zeros_like(r_player_feat)], dim=-1)
            else:
                combined_feat = torch.cat([pooled, torch.zeros_like(pooled), torch.zeros_like(pooled)], dim=-1)
        else:
            combined_feat = pooled

        # --- 10. 最终状态向量 ---
        state_feat = self.state_head(combined_feat)

        return state_feat

    def hero_input_from_ids(self, hero_ids: torch.Tensor):
        """根据英雄ID快速获取预计算的属性和语义特征
        
        Args:
            hero_ids: 1-based 英雄ID (0 表示 padding)
        Returns:
            indices: 0-based 索引 (用于 embedding)
            attrs: 英雄属性
            sem: 英雄语义特征
        """
        device = hero_ids.device
        # 转换为 0-based indices
        indices = hero_ids - 1
        indices = torch.clamp(indices, min=0, max=NUM_HEROES - 1)

        attrs = self.all_hero_attrs[indices]

        if self.use_text and self.all_hero_sem is not None:
            sem = self.all_hero_sem[indices]
        else:
            sem = None

        # 返回 0-based indices，保持与 hero_encoder 的输入一致
        return indices, attrs, sem


class BPActorNetwork(nn.Module):
    """
    Actor网络：给定当前状态和可选英雄，输出行动概率分布
    """
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        use_text: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_text = use_text

        # 候选英雄编码器
        self.candidate_encoder = MultiModalHeroEncoder(
            embed_dim=embed_dim,
            id_hidden_dim=128,
            attr_hidden_dim=64,
            use_text=use_text,
            text_embed_dim=1024,
            text_hidden_dim=128,
            dropout=0.1,
            num_res_layers=2,
            attn_heads=4,
            modality_dropout=0.0,
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )

        # 输出层
        self.output_head = nn.Linear(hidden_dim // 2, 1)

        # 预计算英雄特征
        from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP, NUM_HEROES
        self.register_buffer("all_hero_attrs", torch.stack([
            HERO_ID_FEATURE_MAP.get(hero_id, torch.zeros(NUM_HERO_FEATURES))
            for hero_id in range(1, NUM_HEROES + 1)
        ]), persistent=False)

        if use_text:
            sample_tensor = next(iter(HERO_ID_SEMANTIC_MAP.values())) if HERO_ID_SEMANTIC_MAP else None
            default_device = sample_tensor.device if sample_tensor is not None else torch.device('cpu')
            self.register_buffer("all_hero_sem", torch.stack([
                HERO_ID_SEMANTIC_MAP.get(hero_id, torch.zeros(1024, device=default_device))
                for hero_id in range(1, NUM_HEROES + 1)
            ]), persistent=False)
        else:
            self.all_hero_sem = None

    def forward(
        self,
        state_feat: torch.Tensor,
        candidate_hero_ids: torch.Tensor,
        candidate_hero_attrs: torch.Tensor = None,
        candidate_hero_semantics: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            state_feat: [B, embed_dim]
            candidate_hero_ids: [B, K]
            candidate_hero_attrs: [B, K, NUM_HERO_FEATURES]
            candidate_hero_semantics: [B, K, text_dim]
        Returns:
            logits: [B, K]
        """
        B, K = candidate_hero_ids.shape

        # 编码候选英雄 (转换为 0-indexed)
        candidate_hero_ids_0idx = torch.clamp(candidate_hero_ids - 1, min=0, max=NUM_HEROES - 1)

        if candidate_hero_attrs is None:
            candidate_hero_attrs = self.all_hero_attrs[candidate_hero_ids_0idx]

        if candidate_hero_semantics is None and self.use_text and self.all_hero_sem is not None:
            candidate_hero_semantics = self.all_hero_sem[candidate_hero_ids_0idx]

        candidate_emb = self.candidate_encoder(candidate_hero_ids_0idx, candidate_hero_attrs, candidate_hero_semantics)

        # 扩展state到每个candidate
        state_expanded = state_feat.unsqueeze(1).expand(-1, K, -1)

        # 拼接
        combined = torch.cat([state_expanded, candidate_emb], dim=-1)

        # 通过输出层
        fused = self.fusion(combined)
        logits = self.output_head(fused).squeeze(-1)

        return logits


class BPValueNetwork(nn.Module):
    """Value网络：估计当前状态的价值"""
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.network(state_feat)


class BPAgent(nn.Module):
    """
    完整的BP Agent，用于PPO训练
    包含: StateEncoder, Actor, Value
    """
    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        use_text: bool = True,
        use_player_heroes: bool = True,
        hero_encoder_id_dim: int = 128,
        hero_encoder_attr_dim: int = 64,
        hero_encoder_text_dim: int = 128,
    ):
        super().__init__()

        self.state_encoder = BPStateEncoder(
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            use_text=use_text,
            use_player_heroes=use_player_heroes,
            hero_encoder_id_dim=hero_encoder_id_dim,
            hero_encoder_attr_dim=hero_encoder_attr_dim,
            hero_encoder_text_dim=hero_encoder_text_dim,
        )

        self.actor = BPActorNetwork(
            embed_dim=embed_dim,
            hidden_dim=embed_dim * 2,
            use_text=use_text,
        )

        self.value = BPValueNetwork(
            embed_dim=embed_dim,
            hidden_dim=embed_dim * 2,
        )

    def encode_state(
        self,
        hero_ids: torch.Tensor,
        hero_attrs: torch.Tensor = None,
        hero_semantics: torch.Tensor = None,
        team_flags: torch.Tensor = None,
        action_types: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
        radiant_player_feats: torch.Tensor = None,
        dire_player_feats: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.state_encoder(
            hero_ids=hero_ids,
            hero_attrs=hero_attrs,
            hero_semantics=hero_semantics,
            team_flags=team_flags,
            action_types=action_types,
            valid_mask=valid_mask,
            radiant_player_feats=radiant_player_feats,
            dire_player_feats=dire_player_feats,
        )

    def get_action(
        self,
        state_feat: torch.Tensor,
        candidate_hero_ids: torch.Tensor,
        candidate_hero_attrs: torch.Tensor = None,
        candidate_hero_semantics: torch.Tensor = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ):
        """获取行动
        
        Args:
            temperature: 温度参数，>1 增加随机性，<1 减少随机性
        """
        logits = self.actor(
            state_feat=state_feat,
            candidate_hero_ids=candidate_hero_ids,
            candidate_hero_attrs=candidate_hero_attrs,
            candidate_hero_semantics=candidate_hero_semantics,
        )

        # Mask invalid actions
        mask = (candidate_hero_ids != 0).float()
        mask[mask == 0] = -1e9
        logits = logits + mask
        
        # 应用温度参数（temperature scaling）
        if temperature != 1.0:
            logits = logits / temperature

        value = self.value(state_feat)

        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = None
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(
        self,
        state_feat: torch.Tensor,
        candidate_hero_ids: torch.Tensor,
        actions: torch.Tensor,
        candidate_hero_attrs: torch.Tensor = None,
        candidate_hero_semantics: torch.Tensor = None,
    ):
        """评估给定行动的对数概率和价值"""
        logits = self.actor(
            state_feat=state_feat,
            candidate_hero_ids=candidate_hero_ids,
            candidate_hero_attrs=candidate_hero_attrs,
            candidate_hero_semantics=candidate_hero_semantics,
        )

        mask = (candidate_hero_ids != 0).float()
        mask[mask == 0] = -1e9
        logits = logits + mask

        value = self.value(state_feat)

        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_prob, value, entropy

__all__ = [
    'BPAgent',
    'BPStateEncoder',
    'BPActorNetwork',
    'BPValueNetwork',
]
