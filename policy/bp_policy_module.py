import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_HEROES = 256  # 英雄个数，比真实的偏大
NUM_HERO_FEATURES = 21  # 每个英雄的属性特征维度

# ==========================================
# 0. Hero Encoder
# 用于将单个英雄的多模态特征编码成统一的嵌入向量
# ==========================================
class MultiModalHeroEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        id_hidden_dim: int = 128,
        attr_hidden_dim: int = 64,
        use_text: bool = True,
        text_embed_dim: int = 1024,
        text_hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.use_text = use_text
        
        # 1. ID 分支：增加 LayerNorm 稳定 Embedding 初期随机性
        self.id_embedding = nn.Embedding(NUM_HEROES, id_hidden_dim)
        self.id_norm = nn.LayerNorm(id_hidden_dim)
        
        # 2. 属性分支：增加隐藏层深度提升提取能力
        self.attr_net = nn.Sequential(
            nn.Linear(NUM_HERO_FEATURES, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
            nn.SiLU(), # SiLU 比 GELU 更快且效果相当
            nn.Linear(attr_hidden_dim, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
        )

        # 3. 语义分支 (条件启用)
        current_combined_dim = id_hidden_dim + attr_hidden_dim
        
        if self.use_text:
            self.text_net = nn.Sequential(
                nn.Linear(text_embed_dim, text_hidden_dim),
                nn.LayerNorm(text_hidden_dim),
                nn.SiLU(),
                nn.Linear(text_hidden_dim, text_hidden_dim),
                nn.LayerNorm(text_hidden_dim),
            )
            current_combined_dim += text_hidden_dim
        
        # 4. 融合层：使用残差设计
        self.fusion_proj = nn.Linear(current_combined_dim, embed_dim)
        
        # 增加一个残差块，增强特征交互
        self.res_layer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        """ 初始化权重，帮助模型更快收敛 """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight) # 正交初始化对收敛很有帮助
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, hero_ids, hero_attrs, hero_semantics=None):
        """
        hero_ids: [batch, seq_len]
        hero_attrs: [batch, seq_len, NUM_HERO_FEATURES]
        hero_semantics: [batch, seq_len, text_embed_dim] (Optional if use_text=False)
        """
        # --- 处理各分支 ---
        # ID 分支
        id_feat = self.id_norm(self.id_embedding(hero_ids))
        
        # 属性分支
        attr_feat = self.attr_net(hero_attrs)

        # 语义分支
        if self.use_text and hero_semantics is not None:
            text_feat = self.text_net(hero_semantics)
            combined = torch.cat([id_feat, attr_feat, text_feat], dim=-1)
        else:
            combined = torch.cat([id_feat, attr_feat], dim=-1)
        
        # --- 融合与残差 ---
        # 第一层映射
        x = self.fusion_proj(combined)
        
        # 残差连接：让模型更容易学到恒等映射，减少深层训练难度
        x = x + self.res_layer(x)
        
        return self.final_norm(x)

# ==========================================
# 1. Value Network (Oracle - 胜率预测器)
# 用于给最终阵容打分，产生 Reward
# ==========================================
class WinRateOracle(nn.Module):
    def __init__(self, embed_dim=64, nhead=4, num_layers=2, use_text: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hero_encoder = MultiModalHeroEncoder(embed_dim=embed_dim, use_text=use_text)
        
        # 1. 修改团队嵌入：建议维度设置小一点，或者通过拼接处理
        # 这里的 team_dim 可以是 embed_dim 的一部分，例如 16
        self.team_dim = 16 
        self.team_indicator = nn.Embedding(2, self.team_dim)
        
        # 2. 团队融合层：将 hero_emb (embed_dim) + team_emb (team_dim) 融合回 embed_dim
        self.team_fusion = nn.Sequential(
            nn.Linear(embed_dim + self.team_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # 3. 预测 Token (不变)
        self.predict_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 4. Transformer (不变)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1 # 建议加上 dropout 防止过拟合
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64), # 增加归一化层
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, radiant_hero_ids, radiant_hero_attrs, radiant_hero_semantics, dire_hero_ids, dire_hero_attrs, dire_hero_semantics):
        batch_size = radiant_hero_ids.shape[0]
        device = radiant_hero_ids.device
        
        # --- A. 编码阶段 ---
        r_emb = self.hero_encoder(radiant_hero_ids, radiant_hero_attrs, radiant_hero_semantics)
        d_emb = self.hero_encoder(dire_hero_ids, dire_hero_attrs, dire_hero_semantics)
        
        # --- B. 注入团队信息 (改进版) ---
        # 1. 生成团队索引
        r_team_idx = torch.zeros(batch_size, 5, dtype=torch.long, device=device) # 天辉为 0
        d_team_idx = torch.ones(batch_size, 5, dtype=torch.long, device=device)  # 夜魇为 1
        
        # 2. 获取团队嵌入 [B, 5, team_dim]
        r_team_emb = self.team_indicator(r_team_idx)
        d_team_emb = self.team_indicator(d_team_idx)
        
        # 3. 拼接并投影融合 [B, 5, embed_dim]
        # 这样模型能清晰地感知到：(英雄特征) + (所属阵营)
        r_emb = self.team_fusion(torch.cat([r_emb, r_team_emb], dim=-1))
        d_emb = self.team_fusion(torch.cat([d_emb, d_team_emb], dim=-1))
        
        # --- C. 构造输入序列 ---
        predict_tokens = self.predict_token.expand(batch_size, -1, -1)
        combined_seq = torch.cat([predict_tokens, r_emb, d_emb], dim=1) 
        
        # --- D. Transformer 交互 ---
        out_seq = self.transformer(combined_seq)
        
        # --- E. 提取预测结果 ---
        cls_feature = out_seq[:, 0, :] 
        return self.head(cls_feature)

# ==========================================
# 2. PPO Actor-Critic Network
# ==========================================
class PPOAgent(nn.Module):
    def __init__(self, embed_dim=128, nhead=4):
        super().__init__()
        self.hero_encoder = MultiModalHeroEncoder(embed_dim)
        
        # 位置编码 (Positional Encoding)，让模型知道 BP 的顺序
        self.pos_embedding = nn.Parameter(torch.zeros(1, 24, embed_dim)) 
        
        # Transformer Encoder 层：捕捉 BP 序列间的复杂博弈关系
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Actor Head: 输出英雄选择概率
        self.actor_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, NUM_HEROES)
        )
        
        # Critic Head: 评估当前中间状态的价值 (Advantage 的基础)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward(self, bp_sequence, mask):
        """
        bp_sequence: [batch, seq_len] 当前已选/已禁的英雄序列
        mask: [batch, num_heroes] 掩码，1表示可选，0表示不可选
        """
        # 1. Embedding
        x = self.hero_encoder(bp_sequence) # [batch, seq_len, embed_dim]
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # 2. Transformer Contextualize
        x = self.transformer(x)
        
        # 我们取序列的最后一个 token 的输出作为当前局面的特征表示
        last_token_feat = x[:, -1, :] 
        
        # 3. Actor: 获取动作分布
        logits = self.actor_head(last_token_feat)
        # Apply Mask: 将已选过的英雄概率置为极负无穷，Softmax后变为0
        logits = logits.masked_fill(mask == 0, -1e9)
        probs = F.softmax(logits, dim=-1)
        
        # 4. Critic: 获取局面评分 (State Value)
        state_value = self.critic_head(last_token_feat)
        
        return probs, state_value

if __name__ == "__main__":
    # ==========================================
    # 测试 MultiModalHeroEncoder
    # ==========================================

    import numpy as np
    from utils.raw_data import hero_features, hero_semantic_embeddings

    model = MultiModalHeroEncoder().to('cuda')
    test_output = model.forward(
        hero_ids=torch.Tensor(hero_features['id']).int().cuda(),
        hero_attrs=torch.from_numpy(hero_features.drop(columns=['index', 'name', 'id']).values.astype(np.float32)).cuda(),
        hero_semantics=torch.cat([
            hero_semantic_embeddings[hero_name].view(1, -1)
            for hero_name in hero_features['name']
        ], dim=0).cuda(),
    )