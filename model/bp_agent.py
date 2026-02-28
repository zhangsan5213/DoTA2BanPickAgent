import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from model.hero_encoder import MultiModalHeroEncoder, NUM_HEROES, NUM_HERO_FEATURES
from model.win_rate_oracle import PlayerHeroEncoder
from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP


class ActionEncoder(nn.Module):
    """
    编码BP过程中的每个动作（ban或pick某个英雄）
    输入: 英雄ID + 动作类型(ban/pick) + 阵营(天辉/夜魇)
    输出: 动作嵌入向量
    """
    def __init__(
        self,
        num_heroes: int = NUM_HEROES,
        embed_dim: int = 128,
        use_hero_encoder: bool = True,
        hero_encoder_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_hero_encoder = use_hero_encoder
        
        # 动作类型嵌入: 0=ban, 1=pick
        self.action_type_embed = nn.Embedding(2, embed_dim // 4)
        
        # 阵营嵌入: 0=radiant, 1=dire
        self.team_embed = nn.Embedding(2, embed_dim // 4)
        
        # 动作顺序位置编码（可学习）- 最大支持32个动作（BP通常24手左右）
        self.position_embed = nn.Embedding(32, embed_dim // 4)
        
        if use_hero_encoder:
            # 使用预训练的HeroEncoder编码英雄
            self.hero_proj = nn.Sequential(
                nn.Linear(hero_encoder_dim, embed_dim // 4),
                nn.LayerNorm(embed_dim // 4),
                nn.SiLU()
            )
        else:
            # 直接使用英雄ID嵌入
            self.hero_embed = nn.Embedding(num_heroes, embed_dim // 4)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
    
    def forward(
        self,
        hero_indices: torch.Tensor,  # [B, seq_len] 英雄ID (0-based)
        action_types: torch.Tensor,  # [B, seq_len] 0=ban, 1=pick
        teams: torch.Tensor,         # [B, seq_len] 0=radiant, 1=dire
        positions: torch.Tensor,     # [B, seq_len] 动作位置(0,1,2,...)
        hero_features: Optional[torch.Tensor] = None,  # [B, seq_len, hero_dim] 预计算的英雄特征
    ) -> torch.Tensor:
        """
        Returns:
            action_embeds: [B, seq_len, embed_dim]
        """
        # 各个组件嵌入
        type_emb = self.action_type_embed(action_types)      # [B, seq_len, embed_dim//4]
        team_emb = self.team_embed(teams)                    # [B, seq_len, embed_dim//4]
        pos_emb = self.position_embed(positions)             # [B, seq_len, embed_dim//4]
        
        if self.use_hero_encoder and hero_features is not None:
            hero_emb = self.hero_proj(hero_features)         # [B, seq_len, embed_dim//4]
        else:
            hero_emb = self.hero_embed(hero_indices)         # [B, seq_len, embed_dim//4]
        
        # 拼接所有嵌入
        combined = torch.cat([hero_emb, type_emb, team_emb, pos_emb], dim=-1)  # [B, seq_len, embed_dim]
        
        return self.fusion(combined)


class StateEncoder(nn.Module):
    """
    编码BP的当前状态（变长历史动作序列）
    使用Transformer处理序列，输出状态表示
    """
    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 可学习的[CLS] token，用于聚合整个序列信息
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # LayerNorm for output
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        action_embeds: torch.Tensor,  # [B, seq_len, embed_dim]
        mask: Optional[torch.Tensor] = None,  # [B, seq_len] 1=有效位置, 0=padding
    ) -> torch.Tensor:
        """
        Returns:
            state_repr: [B, embed_dim] 状态表示（聚合整个序列）
            seq_features: [B, seq_len, embed_dim] 每个位置的表示（可用于注意力分析）
        """
        batch_size, seq_len, _ = action_embeds.shape
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls_tokens, action_embeds], dim=1)       # [B, seq_len+1, embed_dim]
        
        # 处理mask：CLS token始终有效
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device, dtype=mask.dtype)
            extended_mask = torch.cat([cls_mask, mask], dim=1)  # [B, seq_len+1]
            # Transformer需要key_padding_mask: True表示要mask掉的位置
            key_padding_mask = (extended_mask == 0)
        else:
            key_padding_mask = None
        
        # Transformer处理
        out = self.transformer(x, src_key_padding_mask=key_padding_mask)  # [B, seq_len+1, embed_dim]
        out = self.norm(out)
        
        # 提取CLS位置作为状态表示
        state_repr = out[:, 0, :]  # [B, embed_dim]
        seq_features = out[:, 1:, :]  # [B, seq_len, embed_dim] (去掉CLS)
        
        return state_repr, seq_features


class CriticHead(nn.Module):
    """
    Critic: 评估当前BP状态的价值 V(s)
    """
    def __init__(self, state_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_repr: [B, state_dim]
        Returns:
            value: [B, 1] 状态价值估计
        """
        return self.net(state_repr)


class ActorHead(nn.Module):
    """
    Actor: 在当前状态下输出动作概率分布 π(a|s)
    支持mask掉不可用的动作（已被ban/pick的英雄）
    """
    def __init__(
        self,
        state_dim: int = 128,
        hidden_dim: int = 128,
        num_heroes: int = NUM_HEROES,
    ):
        super().__init__()
        self.num_heroes = num_heroes
        
        # 动作概率网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
        )
        
        # 输出层
        self.logits = nn.Linear(hidden_dim // 2, num_heroes)
    
    def forward(
        self,
        state_repr: torch.Tensor,  # [B, state_dim]
        action_mask: Optional[torch.Tensor] = None,  # [B, num_heroes] 1=可用, 0=不可用
    ) -> torch.Tensor:
        """
        Args:
            state_repr: 状态表示 [B, state_dim]
            action_mask: 动作掩码 [B, num_heroes]，1表示该英雄可用
        Returns:
            action_probs: 动作概率分布 [B, num_heroes]
        """
        features = self.policy_net(state_repr)  # [B, hidden_dim//2]
        logits = self.logits(features)          # [B, num_heroes]
        
        # 应用mask: 不可用的动作设为很大的负数
        if action_mask is not None:
            # 避免log(0)，添加小epsilon
            mask = action_mask.float()
            logits = logits.masked_fill(mask == 0, float('-inf'))
        
        # Softmax得到概率分布
        action_probs = F.softmax(logits, dim=-1)
        
        # 处理全mask的情况（理论上不应该发生）
        action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return action_probs
    
    def get_action_and_logprob(
        self,
        state_repr: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并返回对应的对数概率（用于训练）
        
        Returns:
            action: [B] 采样的英雄ID
            log_prob: [B] 对数概率
        """
        probs = self.forward(state_repr, action_mask)  # [B, num_heroes]
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            # 分类采样
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        # 计算对数概率
        log_probs = torch.log(probs + 1e-10)
        log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, log_prob


class BPActorCritic(nn.Module):
    """
    Actor-Critic模型用于Dota 2 BP过程
    
    输入: 变长的BP历史动作序列（已经ban/pick了哪些英雄）+ 玩家英雄偏好（可选）
    输出: 
      - Actor: 在当前状态下选择下一个英雄的概率分布
      - Critic: 当前状态的价值估计 V(s)
    """
    def __init__(
        self,
        embed_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heroes: int = NUM_HEROES,
        # HeroEncoder参数
        use_hero_encoder: bool = True,
        hero_encoder_dim: int = 128,
        hero_encoder_id_dim: int = 128,
        hero_encoder_attr_dim: int = 64,
        hero_encoder_text_dim: int = 128,
        hero_encoder_dropout: float = 0.1,
        hero_encoder_res_layers: int = 3,
        hero_encoder_attn_heads: int = 4,
        hero_encoder_modality_dropout: float = 0.1,
        use_text: bool = True,
        # 玩家特征参数
        use_player_heroes: bool = False,
        player_hero_embed_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heroes = num_heroes
        self.use_hero_encoder = use_hero_encoder
        self.use_player_heroes = use_player_heroes
        
        # 1. 英雄编码器（用于编码动作中的英雄）
        if use_hero_encoder:
            self.hero_encoder = MultiModalHeroEncoder(
                embed_dim=hero_encoder_dim,
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
        else:
            self.hero_encoder = None
        
        # 2. 动作编码器（编码每个ban/pick动作）
        self.action_encoder = ActionEncoder(
            num_heroes=num_heroes,
            embed_dim=embed_dim,
            use_hero_encoder=use_hero_encoder,
            hero_encoder_dim=hero_encoder_dim,
        )
        
        # 3. 状态编码器（Transformer处理变长序列）
        self.state_encoder = StateEncoder(
            embed_dim=embed_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # 4. 玩家特征编码器（可选）
        if use_player_heroes:
            self.player_encoder = PlayerHeroEncoder(
                num_heroes=num_heroes,
                hidden_dim=128,
                embed_dim=player_hero_embed_dim,
            )
            # 状态 + 双方玩家特征
            critic_input_dim = embed_dim + player_hero_embed_dim * 2
            actor_input_dim = embed_dim + player_hero_embed_dim * 2
        else:
            self.player_encoder = None
            critic_input_dim = embed_dim
            actor_input_dim = embed_dim
        
        # 5. Critic头（状态价值）
        self.critic_head = CriticHead(state_dim=critic_input_dim)
        
        # 6. Actor头（动作策略）
        self.actor_head = ActorHead(
            state_dim=actor_input_dim,
            hidden_dim=embed_dim,
            num_heroes=num_heroes,
        )
        
        # 6. 预计算所有英雄的静态特征（加速推理）
        if use_hero_encoder:
            self._precompute_hero_features()
        else:
            self.register_buffer("all_hero_attrs", None)
            self.register_buffer("all_hero_sem", None)
        
        # 权重初始化
        self._init_weights()
    
    def _precompute_hero_features(self):
        """预计算所有英雄的静态特征用于快速查找"""
        self.register_buffer("all_hero_attrs", torch.stack([
            HERO_ID_FEATURE_MAP.get(hero_id, torch.zeros(NUM_HERO_FEATURES))
            for hero_id in range(1, NUM_HEROES + 1)
        ]), persistent=False)
        
        # 使用utils中的语义特征
        try:
            self.register_buffer("all_hero_sem", torch.stack([
                HERO_ID_SEMANTIC_MAP.get(hero_id, torch.zeros(1024))
                for hero_id in range(1, NUM_HEROES + 1)
            ]), persistent=False)
        except:
            # 如果语义特征不可用
            self.register_buffer("all_hero_sem", None)
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
        
        # Critic输出层使用较小的初始化
        nn.init.orthogonal_(self.critic_head.net[-1].weight, gain=0.01)
        
        # Actor输出层使用较小的初始化（避免初始策略过于确定）
        nn.init.orthogonal_(self.actor_head.logits.weight, gain=0.01)
        nn.init.constant_(self.actor_head.logits.bias, 0)
    
    def encode_heroes(self, hero_ids: torch.Tensor) -> torch.Tensor:
        """
        根据英雄ID获取编码特征
        
        Args:
            hero_ids: [..., ] 英雄ID (1-based, 0表示无效)
        Returns:
            hero_embeds: [..., hero_encoder_dim]
        """
        if not self.use_hero_encoder or self.hero_encoder is None:
            return None
        
        original_shape = hero_ids.shape
        hero_ids_flat = hero_ids.reshape(-1)
        
        # 转为0-based索引
        indices = hero_ids_flat - 1
        indices = torch.clamp(indices, min=0, max=self.num_heroes - 1)
        
        # 获取预计算的特征
        attrs = self.all_hero_attrs[indices].to(hero_ids.device)
        if self.all_hero_sem is not None:
            sems = self.all_hero_sem[indices].to(hero_ids.device)
        else:
            sems = None
        
        # 编码英雄
        hero_embeds = self.hero_encoder(indices, attrs, sems)
        return hero_embeds.reshape(*original_shape, -1)
    
    def forward(
        self,
        hero_ids: torch.Tensor,      # [B, seq_len] 英雄ID (1-based, 0=padding)
        action_types: torch.Tensor,  # [B, seq_len] 0=ban, 1=pick
        teams: torch.Tensor,         # [B, seq_len] 0=radiant, 1=dire
        positions: torch.Tensor,     # [B, seq_len] 动作顺序位置
        action_mask: torch.Tensor,   # [B, num_heroes] 1=可用英雄
        seq_mask: Optional[torch.Tensor] = None,  # [B, seq_len] 1=有效位置
        radiant_player_feats: Optional[torch.Tensor] = None,  # [B, 5, NUM_HEROES] 天辉玩家偏好
        dire_player_feats: Optional[torch.Tensor] = None,     # [B, 5, NUM_HEROES] 夜魇玩家偏好
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            hero_ids: BP历史中的英雄ID序列 [B, seq_len]
            action_types: 动作类型 [B, seq_len]
            teams: 阵营 [B, seq_len]
            positions: 动作位置 [B, seq_len]
            action_mask: 当前可用动作掩码 [B, num_heroes]
            seq_mask: 序列有效位置掩码 [B, seq_len]，None时自动计算（hero_ids > 0）
            radiant_player_feats: 天辉玩家英雄偏好 [B, 5, NUM_HEROES]（可选）
            dire_player_feats: 夜魇玩家英雄偏好 [B, 5, NUM_HEROES]（可选）
        
        Returns:
            action_probs: 动作概率分布 [B, num_heroes]
            value: 状态价值 [B, 1]
        """
        batch_size, seq_len = hero_ids.shape
        device = hero_ids.device
        
        # 自动计算序列mask（如果未提供）
        if seq_mask is None:
            seq_mask = (hero_ids > 0).long()  # [B, seq_len]
        
        # 1. 编码每个动作中的英雄特征
        if self.use_hero_encoder:
            # 准备英雄的属性/语义特征
            hero_features = self.encode_heroes(hero_ids)  # [B, seq_len, hero_encoder_dim]
            hero_indices = torch.clamp(hero_ids - 1, min=0, max=self.num_heroes - 1)
        else:
            hero_features = None
            hero_indices = torch.clamp(hero_ids - 1, min=0, max=self.num_heroes - 1)
        
        # 2. 编码动作序列
        action_embeds = self.action_encoder(
            hero_indices, action_types, teams, positions, hero_features
        )  # [B, seq_len, embed_dim]
        
        # 3. 编码状态（变长序列）
        state_repr, _ = self.state_encoder(action_embeds, seq_mask)  # [B, embed_dim]
        
        # 4. 编码玩家特征（可选）
        if self.use_player_heroes and self.player_encoder is not None:
            assert radiant_player_feats is not None and dire_player_feats is not None, \
                "use_player_heroes=True 时必须提供 player_feats"
            r_player = self.player_encoder(radiant_player_feats)  # [B, embed_dim]
            d_player = self.player_encoder(dire_player_feats)     # [B, embed_dim]
            # 融合状态 + 玩家特征
            combined_repr = torch.cat([state_repr, r_player, d_player], dim=-1)
        else:
            combined_repr = state_repr
        
        # 5. Actor输出动作概率
        action_probs = self.actor_head(combined_repr, action_mask)  # [B, num_heroes]
        
        # 6. Critic输出状态价值
        value = self.critic_head(combined_repr)  # [B, 1]
        
        return action_probs, value
    
    def get_value(self, state_repr: torch.Tensor) -> torch.Tensor:
        """单独获取状态价值（用于评估）"""
        return self.critic_head(state_repr)
    
    def select_action(
        self,
        hero_ids: torch.Tensor,
        action_types: torch.Tensor,
        teams: torch.Tensor,
        positions: torch.Tensor,
        action_mask: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        选择动作（用于推理或rollout收集）
        
        Args:
            hero_ids: BP历史序列
            action_types: 动作类型
            teams: 阵营
            positions: 位置
            action_mask: 动作掩码
            seq_mask: 序列掩码
            deterministic: 是否确定性策略
            radiant_player_feats: 天辉玩家特征 [B, 5, NUM_HEROES]（可选）
            dire_player_feats: 夜魇玩家特征 [B, 5, NUM_HEROES]（可选）
        
        Returns:
            action: 选中的英雄ID [B]
            log_prob: 对数概率 [B]
            value: 状态价值 [B]
        """
        # 先forward得到action_probs和value
        action_probs, value = self.forward(
            hero_ids, action_types, teams, positions, action_mask, seq_mask,
            radiant_player_feats, dire_player_feats
        )
        
        # 从state_encoder重新获取state_repr（避免重复计算）
        batch_size, seq_len = hero_ids.shape
        if seq_mask is None:
            seq_mask = (hero_ids > 0).long()
        
        if self.use_hero_encoder:
            hero_features = self.encode_heroes(hero_ids)
            hero_indices = torch.clamp(hero_ids - 1, min=0, max=self.num_heroes - 1)
        else:
            hero_features = None
            hero_indices = torch.clamp(hero_ids - 1, min=0, max=self.num_heroes - 1)
        
        action_embeds = self.action_encoder(
            hero_indices, action_types, teams, positions, hero_features
        )
        state_repr, _ = self.state_encoder(action_embeds, seq_mask)
        
        # 编码玩家特征（如果启用）
        if self.use_player_heroes and self.player_encoder is not None:
            assert radiant_player_feats is not None and dire_player_feats is not None
            r_player = self.player_encoder(radiant_player_feats)
            d_player = self.player_encoder(dire_player_feats)
            combined_repr = torch.cat([state_repr, r_player, d_player], dim=-1)
        else:
            combined_repr = state_repr
        
        # 采样动作
        action, log_prob = self.actor_head.get_action_and_logprob(
            combined_repr, action_mask, deterministic
        )
        
        return action, log_prob, value.squeeze(-1)


__all__ = [
    'BPActorCritic',
    'ActionEncoder',
    'StateEncoder',
    'ActorHead',
    'CriticHead',
    'PlayerHeroEncoder',
]