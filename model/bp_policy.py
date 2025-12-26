import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hero_encoder import NUM_HEROES

# --- RoPE ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        # dim 应该是 head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        # t: [seq_len]
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        # freqs: [seq_len, dim/2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # emb: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb # 返回 cos 和 sin 的基础频率

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    # freqs: [seq_len, head_dim] -> [1, seq_len, 1, head_dim]
    # 调整维度以匹配 [B, L, H, D] 或 [B, H, L, D]
    # 这里假设输入是 [B, H, L, D] (来自 RoPEMultiheadAttention)
    freqs = freqs.unsqueeze(0).unsqueeze(0) 
    # 计算 cos 和 sin
    cos = freqs.cos()
    sin = freqs.sin()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- Transformer 组件 ---
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        assert self.head_dim * nhead == embed_dim, "embed_dim must be divisible by nhead"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs, mask=None):
        B, L, E = x.shape
        # qkv: [B, L, 3*E] -> [B, L, 3, H, D]
        qkv = self.qkv(x).view(B, L, 3, self.nhead, self.head_dim)
        # 分离 q, k, v 并转置为 [B, H, L, D]
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 应用 RoPE
        q, k = apply_rotary_pos_emb(q, k, freqs)
        
        # Scaled Dot-Product Attention
        # 使用 PyTorch 优化的算子
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask, 
            dropout_p=self.dropout.p if self.training else 0,
            is_causal=False # mask 已经在外部生成
        )
        
        # 转换回 [B, L, E]
        out = attn_output.transpose(1, 2).contiguous().view(B, L, E)
        return self.out_proj(out)

class RoPETransformerBlock(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiheadAttention(embed_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.SiLU(),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, freqs, mask=None):
        x = x + self.attn(self.norm1(x), freqs, mask)
        x = x + self.mlp(self.norm2(x))
        return x

# --- Agent ---
class PPODecoderAgent(nn.Module):
    def __init__(self, embed_dim=256, nhead=8, num_layers=4, dim_feedforward=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // nhead
        
        # Embedding 层
        self.hero_embedding = nn.Embedding(NUM_HEROES + 1, embed_dim, padding_idx=0)
        self.team_embedding = nn.Embedding(3, embed_dim) 
        self.type_embedding = nn.Embedding(3, embed_dim)
        
        # RoPE 模块
        self.rope = RotaryEmbedding(self.head_dim)
        
        # 使用自定义的 RoPE Transformer 块
        self.layers = nn.ModuleList([
            RoPETransformerBlock(embed_dim, nhead, dim_feedforward)
            for _ in range(num_layers)
        ])

        self.actor_head = nn.Linear(embed_dim, NUM_HEROES + 1)
        self.critic_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

    def _generate_causal_mask(self, sz, device):
        # PyTorch F.scaled_dot_product_attention 接受的布尔 mask 或 float mask
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask # True 表示被屏蔽

    def forward(self, hero_seq, team_seq, type_seq):
        B, L = hero_seq.size()
        device = hero_seq.device
        
        # 1. 构造交错序列 [B, L*3, E]
        print(hero_seq.max())
        t_emb = self.team_embedding(team_seq)
        p_emb = self.type_embedding(type_seq)
        h_emb = self.hero_embedding(hero_seq)
        
        # 组合顺序 [Team_i, Type_i, Hero_i]
        combined = torch.stack([t_emb, p_emb, h_emb], dim=2) 
        x = combined.view(B, L * 3, self.embed_dim)
        
        # 2. 准备 RoPE 频率和 Mask
        seq_len = x.size(1)
        freqs = self.rope(seq_len, device)
        mask = self._generate_causal_mask(seq_len, device)
        
        # 3. 通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, freqs, mask)
        
        # 4. 提取输出
        # Actor: 基于 Type Token (下标 3k+1) 预测 Hero
        type_indices = torch.arange(1, seq_len, 3, device=device)
        actor_features = x[:, type_indices, :]
        hero_logits = self.actor_head(actor_features) 
        
        # Critic: 基于 Hero Token (下标 3k+1) 评估当前局面
        hero_indices = torch.arange(1, seq_len, 3, device=device)
        critic_features = x[:, hero_indices, :]
        values = self.critic_head(critic_features).squeeze(-1) 
        
        return hero_logits, values