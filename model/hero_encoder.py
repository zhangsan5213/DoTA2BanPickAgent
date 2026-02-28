import torch
import torch.nn as nn

from utils.raw_data import NUM_HEROES, NUM_HERO_FEATURES

# ==========================================
# Hero Encoder
# 用于将单个英雄的多模态特征编码成统一的嵌入向量
# ==========================================

class AttentionFusion(nn.Module):
    """
    注意力融合模块：让不同模态的特征通过自注意力机制交互
    同时通过门控机制学习每个模态的重要性
    """
    def __init__(self, id_dim, attr_dim, text_dim, embed_dim, num_heads=4, use_text=True):
        super().__init__()
        self.use_text = use_text
        self.num_modalities = 3 if use_text else 2
        self.embed_dim = embed_dim
        
        # 投影到统一维度
        self.id_proj = nn.Linear(id_dim, embed_dim)
        self.attr_proj = nn.Linear(attr_dim, embed_dim)
        if use_text:
            self.text_proj = nn.Linear(text_dim, embed_dim)
        
        # 模态类型嵌入（让模型知道每个token来自哪个模态）
        self.modal_type_embed = nn.Embedding(self.num_modalities, embed_dim)
        
        # 自注意力：让模态间交互
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # 门控融合：学习每个模态的重要性权重
        fusion_input_dim = id_dim + attr_dim + (text_dim if use_text else 0)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fusion_input_dim, self.num_modalities),
            nn.Softmax(dim=-1)  # 输出各模态的权重和为1
        )
    
    def forward(self, id_feat, attr_feat, text_feat=None):
        """
        id_feat: [B, L, id_dim]
        attr_feat: [B, L, attr_dim]
        text_feat: [B, L, text_dim] (optional)
        """
        B, L = id_feat.shape[:2]
        
        # 处理空序列情况（如BP初始状态）
        if L == 0:
            fused = id_feat.new_zeros(B, 0, self.embed_dim)
            gates = id_feat.new_zeros(B, 0, self.num_modalities)
            return fused, gates
        
        # 投影到统一维度
        id_emb = self.id_proj(id_feat)      # [B, L, E]
        attr_emb = self.attr_proj(attr_feat) # [B, L, E]
        
        # 堆叠成序列 [B, L, num_modalities, E] -> [B, L*num_modalities, E]
        if self.use_text and text_feat is not None:
            text_emb = self.text_proj(text_feat)  # [B, L, E]
            multi_modal = torch.stack([id_emb, attr_emb, text_emb], dim=2)  # [B, L, 3, E]
            modality_ids = torch.arange(3, device=id_feat.device)
        else:
            multi_modal = torch.stack([id_emb, attr_emb], dim=2)  # [B, L, 2, E]
            modality_ids = torch.arange(2, device=id_feat.device)
        
        # reshape: [B, L*M, E]
        multi_modal = multi_modal.view(B, L * self.num_modalities, -1)
        
        # 添加模态类型嵌入（让注意力知道token类型）
        modal_embeds = self.modal_type_embed(modality_ids)  # [M, E]
        modal_embeds = modal_embeds.unsqueeze(0).unsqueeze(0)  # [1, 1, M, E]
        modal_embeds = modal_embeds.expand(B, L, -1, -1).reshape(B, L * self.num_modalities, -1)
        multi_modal = multi_modal + modal_embeds
        
        # 自注意力交互
        attn_out, _ = self.self_attn(multi_modal, multi_modal, multi_modal)
        attn_out = self.attn_norm(multi_modal + attn_out)
        
        # FFN
        ffn_out = self.ffn(attn_out)
        attn_out = self.ffn_norm(attn_out + ffn_out)
        
        # reshape回 [B, L, M, E]
        attn_out = attn_out.view(B, L, self.num_modalities, -1)
        
        # 门控融合：计算各模态的权重
        if self.use_text and text_feat is not None:
            concat_feats = torch.cat([id_feat, attr_feat, text_feat], dim=-1)
        else:
            concat_feats = torch.cat([id_feat, attr_feat], dim=-1)
        
        gates = self.fusion_gate(concat_feats)  # [B, L, M]
        gates = gates.unsqueeze(-1)  # [B, L, M, 1]
        
        # 加权融合
        fused = (attn_out * gates).sum(dim=2)  # [B, L, E]
        
        return fused, gates  # 返回gates用于可视化/debug


class DeepResBlock(nn.Module):
    """深层残差块，包含多个子残差层"""
    def __init__(self, embed_dim, num_layers=3, expansion=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * expansion),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * expansion, embed_dim),
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class MultiModalHeroEncoder(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        id_hidden_dim: int = 128,
        attr_hidden_dim: int = 64,
        use_text: bool = True,
        text_embed_dim: int = 1024,
        text_hidden_dim: int = 128,
        dropout: float = 0.1,
        num_res_layers: int = 3,
        attn_heads: int = 4,
        modality_dropout: float = 0.1,
    ):
        super().__init__()
        self.use_text = use_text
        self.modality_dropout = modality_dropout
        
        # 1. ID 分支
        self.id_embedding = nn.Embedding(NUM_HEROES, id_hidden_dim)
        self.id_norm = nn.LayerNorm(id_hidden_dim)
        
        # 2. 属性分支
        self.attr_net = nn.Sequential(
            nn.Linear(NUM_HERO_FEATURES, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
            nn.SiLU(),
            nn.Linear(attr_hidden_dim, attr_hidden_dim),
            nn.LayerNorm(attr_hidden_dim),
        )

        # 3. 语义分支
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
        
        # 4. 注意力融合层（替代原来的简单拼接）
        text_input_dim = text_hidden_dim if use_text else 0
        self.attention_fusion = AttentionFusion(
            id_dim=id_hidden_dim,
            attr_dim=attr_hidden_dim,
            text_dim=text_input_dim,
            embed_dim=embed_dim,
            num_heads=attn_heads,
            use_text=use_text
        )
        
        # 5. 深层残差块
        self.res_blocks = DeepResBlock(
            embed_dim=embed_dim,
            num_layers=num_res_layers,
            expansion=2,
            dropout=dropout
        )
        
        self.final_norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

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
        else:
            text_feat = None
        
        # --- 模态 Dropout（仅在训练时）---
        if self.training and self.modality_dropout > 0:
            if torch.rand(1).item() < self.modality_dropout:
                # 随机mask ID模态
                id_feat = torch.zeros_like(id_feat)
            if torch.rand(1).item() < self.modality_dropout:
                # 随机mask属性模态
                attr_feat = torch.zeros_like(attr_feat)
            if text_feat is not None and torch.rand(1).item() < self.modality_dropout:
                text_feat = torch.zeros_like(text_feat)
        
        # --- 注意力融合 ---
        x, gates = self.attention_fusion(id_feat, attr_feat, text_feat)
        
        # --- 深层残差 ---
        x = self.res_blocks(x)
        
        return self.final_norm(x)


# 保持向后兼容
__all__ = ['MultiModalHeroEncoder', 'AttentionFusion', 'DeepResBlock']
