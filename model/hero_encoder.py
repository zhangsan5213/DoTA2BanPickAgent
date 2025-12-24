import torch
import torch.nn as nn

NUM_HEROES = 256  # 英雄个数，比真实的偏大
NUM_HERO_FEATURES = 21  # 每个英雄的属性特征维度

# ==========================================
# Hero Encoder
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