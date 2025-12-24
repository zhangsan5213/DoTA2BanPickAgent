import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from model.hero_encoder import NUM_HEROES, NUM_HERO_FEATURES, MultiModalHeroEncoder
from model.win_rate_oracle import WinRateOracle

# ==========================================
# PPO Actor-Critic Decoder-only Network
# ==========================================

# --- RoPE 实现 ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=256):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, freqs):
    freqs = freqs.unsqueeze(0).unsqueeze(0) # [1, 1, seq_len, dim]
    return (q * freqs.cos()) + (rotate_half(q) * freqs.sin()), \
           (k * freqs.cos()) + (rotate_half(k) * freqs.sin())

# --- Transformer 组件 ---
class RoPEMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, nhead, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.head_dim = embed_dim // nhead
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs, mask=None):
        B, L, E = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, L, self.nhead, self.head_dim).transpose(1, 2), qkv)
        q, k = apply_rotary_pos_emb(q, k, freqs)
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == float('-inf'), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (self.dropout(attn) @ v).transpose(1, 2).reshape(B, L, E)
        return self.out_proj(out)

class RoPETransformerBlock(nn.Module):
    def __init__(self, embed_dim, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attn = RoPEMultiheadAttention(embed_dim, nhead, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
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
    def __init__(self, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 统一的 Embedding 层或分开
        self.hero_embedding = nn.Embedding(NUM_HEROES + 1, embed_dim, padding_idx=0)
        self.team_embedding = nn.Embedding(3, embed_dim) # 0:pad, 1:Radiant, 2:Dire
        self.type_embedding = nn.Embedding(3, embed_dim) # 0:pad, 1:Ban, 2:Pick
        
        # 位置编码长度变为 24 * 3 = 72
        self.pos_embedding = nn.Parameter(torch.zeros(1, 72, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Actor 负责从 Type Token 预测 Hero ID
        self.actor_head = nn.Linear(embed_dim, NUM_HEROES + 1)
        # Critic 负责评估当前所有已输入 Token 代表的局面价值
        self.critic_head = nn.Linear(embed_dim, 1)

    def _generate_causal_mask(self, sz, device):
        mask = torch.triu(torch.ones(sz, sz, device=device) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, hero_seq, team_seq, type_seq):
        """
        hero_seq, team_seq, type_seq: [B, 24]
        """
        B, L = hero_seq.size()
        device = hero_seq.device
        
        # 1. 构造交错序列: [B, L, 3, E]
        t_emb = self.team_embedding(team_seq)
        p_emb = self.type_embedding(type_seq)
        h_emb = self.hero_embedding(hero_seq)
        
        # 重点：按照 [Team, Type, Hero] 顺序排列
        # 维度变换: [B, L, E] -> [B, L, 1, E] -> 拼接为 [B, L, 3, E] -> 打平为 [B, L*3, E]
        combined = torch.stack([t_emb, p_emb, h_emb], dim=2) 
        x = combined.view(B, L * 3, self.embed_dim)
        
        # 2. 加上位置编码
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # 3. Transformer 处理
        mask = self._generate_causal_mask(x.size(1), device)
        # out: [B, 72, E]
        out = self.transformer(x, mask=mask)
        
        # 4. 提取输出
        # 我们规定：在输入了 Team_k, Type_k 之后，预测 Hero_k
        # 因此在打平的序列中，Type_k 的下标是 3*k + 1
        # 我们取所有 3k+1 位置的特征来算 Actor Logits
        type_indices = torch.arange(1, x.size(1), 3, device=device)
        hero_logits = self.actor_head(out[:, type_indices, :]) # [B, 24, NUM_HEROES+1]
        
        # Critic 可以对每一步都产生一个价值，但我们通常取 Hero 选完后的时刻 (下标 3k+2)
        hero_indices = torch.arange(2, x.size(1), 3, device=device)
        values = self.critic_head(out[:, hero_indices, :]).squeeze(-1) # [B, 24]
        
        return hero_logits, values