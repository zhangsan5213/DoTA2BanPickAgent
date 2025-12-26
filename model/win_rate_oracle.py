import torch
import torch.nn as nn

from model.hero_encoder import MultiModalHeroEncoder, NUM_HEROES, NUM_HERO_FEATURES
from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP

# ==========================================
# Value Network (Oracle - 胜率预测器)
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

        self.register_buffer("all_hero_attrs", torch.stack([
            HERO_ID_FEATURE_MAP.get(hero_id, torch.zeros(NUM_HERO_FEATURES)).cuda()
            for hero_id in range(1, NUM_HEROES + 1)
        ]), persistent=False)
        self.register_buffer("all_hero_sem", torch.stack([
            HERO_ID_SEMANTIC_MAP.get(hero_id, torch.zeros(1024)).cuda()
            for hero_id in range(1, NUM_HEROES + 1)
        ]), persistent=False)

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
        r_emb = self.team_fusion(torch.cat([r_emb, r_team_emb], dim=-1)) # TODO: Sizes of tensors must match except in dimension 2. Expected size 4 but got size 5 for tensor number 1 in the list.
        d_emb = self.team_fusion(torch.cat([d_emb, d_team_emb], dim=-1))
        
        # --- C. 构造输入序列 ---
        predict_tokens = self.predict_token.expand(batch_size, -1, -1)
        combined_seq = torch.cat([predict_tokens, r_emb, d_emb], dim=1) 
        
        # --- D. Transformer 交互 ---
        out_seq = self.transformer(combined_seq)
        
        # --- E. 提取预测结果 ---
        cls_feature = out_seq[:, 0, :] 
        return self.head(cls_feature)

    def hero_input_from_ids(self, hero_ids: torch.Tensor):
        # 无需循环，直接利用 Tensor 索引 (极其快速)
        ids_tensor = hero_ids.cuda()
        attrs_tensor = self.all_hero_attrs[hero_ids] # 自动处理 [5] -> [5, F] 或 [B, 5] -> [B, 5, F]
        sem_tensor = self.all_hero_sem[hero_ids]
        return ids_tensor, attrs_tensor, sem_tensor

    def predict(self, radiant_picks, dire_picks):
        r_ids, r_attrs, r_sem = self.hero_input_from_ids(radiant_picks)
        d_ids, d_attrs, d_sem = self.hero_input_from_ids(dire_picks)
        return self.forward(r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem)