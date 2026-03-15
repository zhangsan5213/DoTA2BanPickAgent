import torch
import torch.nn as nn

from model.hero_encoder import MultiModalHeroEncoder, NUM_HEROES, NUM_HERO_FEATURES
from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP
from utils.device import DEVICE


class PlayerHeroEncoder(nn.Module):
    """
    编码玩家的英雄胜率偏好（擅长英雄）。
    输入: [B, 5, NUM_HEROES] - 5个玩家，每个玩家有NUM_HEROES维的胜率向量
          （每个位置表示该玩家使用该英雄的胜率，0表示未玩过或场次不足）
    输出: [B, player_embed_dim] - 团队玩家特征
    """
    def __init__(self, num_heroes=NUM_HEROES, hidden_dim=128, embed_dim=64, nhead=4, num_layers=2):
        super().__init__()
        self.num_heroes = num_heroes
        self.embed_dim = embed_dim
        
        # 每个玩家的胜率编码器
        self.player_encoder = nn.Sequential(
            nn.Linear(num_heroes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # 双向自注意力：5个玩家之间进行信息交互
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 团队聚合（将5个玩家聚合成一个团队向量）
        self.team_aggregator = nn.Sequential(
            nn.Linear(embed_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, player_feats):
        """
        Args:
            player_feats: [B, 5, NUM_HEROES] - 玩家胜率矩阵
                          player_feats[b, p, h] = 玩家p使用英雄h的胜率 (0-1)
        Returns:
            team_feat: [B, embed_dim]
        """
        B = player_feats.shape[0]
        
        # 编码每个玩家 [B, 5, embed_dim]
        player_feats_flat = player_feats.view(-1, self.num_heroes)
        player_embeds = self.player_encoder(player_feats_flat)
        player_embeds = player_embeds.view(B, 5, self.embed_dim)
        
        # 双向自注意力交互：让玩家之间交换信息 [B, 5, embed_dim]
        player_embeds = self.transformer(player_embeds)
        
        # 聚合成团队特征 [B, embed_dim]
        team_feat = self.team_aggregator(player_embeds.view(B, -1))
        return team_feat


# ==========================================
# Value Network (Oracle - 胜率预测器)
# 用于给最终阵容打分，产生 Reward
# ==========================================

class WinRateOracle(nn.Module):
    def __init__(
        self, 
        embed_dim=128, 
        nhead=4, 
        num_layers=2, 
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
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_text = use_text
        self.use_player_heroes = use_player_heroes
        
        # 1. Hero Encoder
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
        
        # 2. 团队嵌入
        self.team_dim = 16 
        self.team_indicator = nn.Embedding(2, self.team_dim)
        
        # 3. 团队融合层
        self.team_fusion = nn.Sequential(
            nn.Linear(embed_dim + self.team_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU()
        )
        
        # 4. 玩家英雄编码器
        if self.use_player_heroes:
            self.player_encoder = PlayerHeroEncoder(
                num_heroes=NUM_HEROES, 
                hidden_dim=128, 
                embed_dim=embed_dim,
                nhead=4,
                num_layers=2
            )
            # transformer输出 + 两个团队的玩家特征
            head_input_dim = embed_dim + embed_dim * 2
        else:
            head_input_dim = embed_dim
        
        # 5. 预测 Token
        self.predict_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 6. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=nhead, 
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 7. 预测头
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 8. 预计算所有英雄的静态特征
        self.register_buffer("all_hero_attrs", torch.stack([
            HERO_ID_FEATURE_MAP.get(hero_id, torch.zeros(NUM_HERO_FEATURES))
            for hero_id in range(1, NUM_HEROES + 1)
        ]), persistent=False)
        
        if use_text:
            self.register_buffer("all_hero_sem", torch.stack([
                HERO_ID_SEMANTIC_MAP.get(hero_id, torch.zeros(1024)).to(DEVICE)
                for hero_id in range(1, NUM_HEROES + 1)
            ]), persistent=False)
        else:
            self.all_hero_sem = None

    def forward(
        self, 
        radiant_hero_ids, 
        radiant_hero_attrs, 
        radiant_hero_semantics, 
        dire_hero_ids, 
        dire_hero_attrs, 
        dire_hero_semantics,
        radiant_player_feats=None, 
        dire_player_feats=None
    ):
        """
        Args:
            radiant_hero_ids: [B, 5] 天辉英雄ID (0-based, 0-159)
            radiant_hero_attrs: [B, 5, NUM_HERO_FEATURES] 天辉英雄属性
            radiant_hero_semantics: [B, 5, text_dim] 天辉英雄语义
            dire_hero_ids: [B, 5] 夜魇英雄ID (0-based, 0-159)
            dire_hero_attrs: [B, 5, NUM_HERO_FEATURES] 夜魇英雄属性
            dire_hero_semantics: [B, 5, text_dim] 夜魇英雄语义
            radiant_player_feats: [B, 5, NUM_HEROES] 天辉玩家英雄偏好（可选）
            dire_player_feats: [B, 5, NUM_HEROES] 夜魇玩家英雄偏好（可选）
        Returns:
            win_prob: [B, 1] 天辉胜率
        """
        batch_size = radiant_hero_ids.shape[0]
        device = radiant_hero_ids.device
        
        # --- A. 编码阶段 ---
        # hero_encoder 需要 0-based indices (0-159)
        r_emb = self.hero_encoder(radiant_hero_ids, radiant_hero_attrs, radiant_hero_semantics)
        d_emb = self.hero_encoder(dire_hero_ids, dire_hero_attrs, dire_hero_semantics)
        
        # --- B. 注入团队信息 ---
        # 天辉为 0，夜魇为 1
        r_team_idx = torch.zeros(batch_size, 5, dtype=torch.long, device=device)
        d_team_idx = torch.ones(batch_size, 5, dtype=torch.long, device=device)
        
        # 获取团队嵌入 [B, 5, team_dim]
        r_team_emb = self.team_indicator(r_team_idx)
        d_team_emb = self.team_indicator(d_team_idx)
        
        # 拼接并投影融合 [B, 5, embed_dim]
        r_emb = self.team_fusion(torch.cat([r_emb, r_team_emb], dim=-1))
        d_emb = self.team_fusion(torch.cat([d_emb, d_team_emb], dim=-1))
        
        # --- C. 构造输入序列 ---
        predict_tokens = self.predict_token.expand(batch_size, -1, -1)
        combined_seq = torch.cat([predict_tokens, r_emb, d_emb], dim=1) 
        
        # --- D. Transformer 交互 ---
        out_seq = self.transformer(combined_seq)
        
        # --- E. 提取预测结果 ---
        cls_feature = out_seq[:, 0, :]  # [B, embed_dim]
        
        # --- F. 融合玩家特征 ---
        if self.use_player_heroes:
            assert radiant_player_feats is not None and dire_player_feats is not None, \
                "use_player_heroes=True 时必须提供 player_feats"
            r_player_feat = self.player_encoder(radiant_player_feats)  # [B, embed_dim]
            d_player_feat = self.player_encoder(dire_player_feats)     # [B, embed_dim]
            # 拼接所有特征 [B, embed_dim * 3]
            combined_feat = torch.cat([cls_feature, r_player_feat, d_player_feat], dim=-1)
        else:
            combined_feat = cls_feature
        
        return self.head(combined_feat)

    def hero_input_from_ids(self, hero_ids: torch.Tensor):
        """
        根据英雄ID快速获取预计算的属性和语义特征
        
        Args:
            hero_ids: [5] 或 [B, 5] 英雄ID（1-based, 1-160；0表示无效/padding）
        Returns:
            indices: 0-based 索引 (0-159)，用于传入 hero_encoder
            attrs: 英雄属性 [5, F] 或 [B, 5, F]
            sem: 英雄语义 [5, S] 或 [B, 5, S]
        """
        # 确保在正确设备上
        device = hero_ids.device
        
        # 处理ID：减1转为0-based索引（因为预计算buffer和embedding都是0-based）
        # hero_ids: 1-160 -> indices: 0-159
        indices = hero_ids - 1
        
        # 处理边界：确保索引在有效范围内 [0, NUM_HEROES-1]
        indices = torch.clamp(indices, min=0, max=NUM_HEROES - 1)
        
        # 获取属性 [5, F] 或 [B, 5, F]
        attrs = self.all_hero_attrs[indices]
        
        # 获取语义（如果启用）
        if self.use_text and self.all_hero_sem is not None:
            sem = self.all_hero_sem[indices]
        else:
            sem = None
        
        # 返回 0-based indices，保持与 hero_encoder 的输入一致
        return indices, attrs, sem

    def predict(
        self, 
        radiant_picks, 
        dire_picks, 
        radiant_player_feats=None, 
        dire_player_feats=None,
        return_tensor: bool = False
    ):
        """
        便捷预测接口
        
        Args:
            radiant_picks: list[int] 或 tensor - 天辉5个英雄ID
            dire_picks: list[int] 或 tensor - 夜魇5个英雄ID
            radiant_player_feats: list[list[float]] 或 tensor [5, NUM_HEROES]（可选）
            dire_player_feats: list[list[float]] 或 tensor [5, NUM_HEROES]（可选）
            return_tensor: 是否返回tensor而非numpy
        Returns:
            win_prob: float 或 numpy array - 天辉胜率
        """
        self.eval()
        
        # 获取模型所在设备
        device = next(self.parameters()).device
        
        # 处理英雄ID输入
        if not isinstance(radiant_picks, torch.Tensor):
            radiant_picks = torch.tensor(radiant_picks, dtype=torch.long, device=device)
        else:
            radiant_picks = radiant_picks.to(device)
        if not isinstance(dire_picks, torch.Tensor):
            dire_picks = torch.tensor(dire_picks, dtype=torch.long, device=device)
        else:
            dire_picks = dire_picks.to(device)
        
        # 确保batch维度
        if radiant_picks.dim() == 1:
            radiant_picks = radiant_picks.unsqueeze(0)
        if dire_picks.dim() == 1:
            dire_picks = dire_picks.unsqueeze(0)
        
        # 获取英雄特征
        r_ids, r_attrs, r_sem = self.hero_input_from_ids(radiant_picks)
        d_ids, d_attrs, d_sem = self.hero_input_from_ids(dire_picks)
        
        # 处理玩家特征
        r_player = self._process_player_feats(radiant_player_feats, device)
        d_player = self._process_player_feats(dire_player_feats, device)
        
        # 前向传播
        with torch.no_grad():
            pred = self.forward(
                r_ids, r_attrs, r_sem,
                d_ids, d_attrs, d_sem,
                r_player, d_player
            )
        
        if return_tensor:
            return pred
        return pred.cpu().numpy()

    def _process_player_feats(self, player_feats, device=None):
        """
        统一处理玩家胜率特征
        
        Args:
            player_feats: None / list[list[float]] / tensor [5, NUM_HEROES]
                          每个元素表示该玩家使用对应英雄的胜率 (0-1)
            device: 目标设备，如果为None则使用CPU
        Returns:
            tensor [1, 5, NUM_HEROES] 或 None
        """
        if player_feats is None:
            return None
        
        if device is None:
            device = torch.device('cpu')
        
        if not isinstance(player_feats, torch.Tensor):
            player_feats = torch.tensor(player_feats, dtype=torch.float32, device=device)
        else:
            player_feats = player_feats.to(device)
        
        if player_feats.dim() == 2:
            # [5, NUM_HEROES] -> [1, 5, NUM_HEROES]
            player_feats = player_feats.unsqueeze(0)
        elif player_feats.dim() == 3:
            # 已经是 [B, 5, NUM_HEROES]
            pass
        else:
            raise ValueError(f"player_feats维度错误: {player_feats.dim()}")
        
        return player_feats

    def load_player_hero_data(self, match_id: int, data_file: str = './data/player_hero_features.json'):
        """
        从预处理的数据文件中加载指定比赛的玩家英雄特征
        
        Args:
            match_id: 比赛ID
            data_file: 数据文件路径
        Returns:
            (radiant_player_feats, dire_player_feats) 或 None
        """
        import json
        import os
        
        if not os.path.exists(data_file):
            return None
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        match_data = data.get(str(match_id))
        if match_data is None:
            return None
        
        radiant_feats = torch.tensor(match_data['radiant_player_features'], dtype=torch.float32)
        dire_feats = torch.tensor(match_data['dire_player_features'], dtype=torch.float32)
        
        return radiant_feats, dire_feats


# ==========================================
# 训练数据加载工具
# ==========================================

class OracleTrainingDataset(torch.utils.data.Dataset):
    """
    WinRateOracle训练数据集
    从 high_mmr_with_stats.json 加载数据（包含picks_bans、players、hero_history）
    """
    def __init__(
        self, 
        data_file: str = './data/high_mmr_with_stats.json',
        min_total_games: int = 10,
        max_heroes_per_player: int = 10,
        min_games_per_hero: int = 3
    ):
        import json
        import os
        
        self.data_file = data_file
        self.min_total_games = min_total_games
        self.max_heroes_per_player = max_heroes_per_player
        self.min_games_per_hero = min_games_per_hero
        
        # 加载合并数据文件
        with open(data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # 过滤有完整pick数据的比赛
        self.valid_matches = []
        for m in self.raw_data:
            r_picks, d_picks = self._extract_picks(m)
            if len(r_picks) == 5 and len(d_picks) == 5:
                self.valid_matches.append(m)
        
        print(f"[OracleDataset] 加载了 {len(self.valid_matches)} 场有效比赛")
    
    def _extract_picks(self, match: dict):
        """从 picks_bans 中提取两队的pick英雄ID"""
        r_picks, d_picks = [], []
        for act in match.get('picks_bans', []):
            if act.get('is_pick', False):
                if act.get('team', 0) == 0:
                    r_picks.append(act['hero_id'])
                else:
                    d_picks.append(act['hero_id'])
        return r_picks, d_picks
    
    def _split_players_by_team(self, players):
        """根据 player_slot 分队: 0-4=天辉, 128-132=夜魇"""
        radiant, dire = [], []
        for p in players:
            slot = p.get('player_slot', 0)
            if slot < 128:
                radiant.append(p)
            else:
                dire.append(p)
        return radiant, dire
    
    def _build_player_features(self, players):
        """从 hero_history 构建玩家特征向量 [5, NUM_HEROES]"""
        from utils.raw_data import NUM_HEROES
        
        vectors = []
        for player in players[:5]:
            hero_history = player.get('hero_history', {})
            total_games = sum(h.get('games', 0) for h in hero_history.values())
            
            # 数据不足时用零向量
            if total_games < self.min_total_games:
                vectors.append([0.0] * NUM_HEROES)
                continue
            
            # 按场次排序英雄
            hero_list = []
            for hero_id_str, stats in hero_history.items():
                try:
                    hero_id = int(hero_id_str)
                    games = stats.get('games', 0)
                    wins = stats.get('wins', 0)
                    winrate = wins / games if games > 0 else 0
                    hero_list.append((hero_id, games, winrate))
                except (ValueError, TypeError):
                    continue
            
            hero_list.sort(key=lambda x: x[1], reverse=True)
            
            vector = [0.0] * NUM_HEROES
            for hero_id, games, winrate in hero_list[:self.max_heroes_per_player]:
                if 0 < hero_id < NUM_HEROES and games >= self.min_games_per_hero:
                    vector[hero_id] = winrate
            
            vectors.append(vector)
        
        # 补全到5个玩家
        while len(vectors) < 5:
            vectors.append([0.0] * NUM_HEROES)
        
        return vectors

    def __len__(self):
        return len(self.valid_matches)

    def __getitem__(self, idx):
        match = self.valid_matches[idx]
        match_id = str(match.get('match_id', idx))
        
        # 提取英雄选择
        r_picks, d_picks = self._extract_picks(match)
        radiant_picks = torch.tensor(r_picks, dtype=torch.long)
        dire_picks = torch.tensor(d_picks, dtype=torch.long)
        
        # 提取并构建玩家特征
        players = match.get('players', [])
        radiant_players, dire_players = self._split_players_by_team(players)
        r_player_feats = torch.tensor(self._build_player_features(radiant_players), dtype=torch.float32)
        d_player_feats = torch.tensor(self._build_player_features(dire_players), dtype=torch.float32)
        
        # 获取标签
        radiant_win = 1.0 if match.get('radiant_win', False) else 0.0
        label = torch.tensor(radiant_win, dtype=torch.float32)
        
        return {
            'radiant_picks': radiant_picks,
            'dire_picks': dire_picks,
            'radiant_player_feats': r_player_feats,
            'dire_player_feats': d_player_feats,
            'label': label,
            'match_id': match_id
        }


def collate_oracle_batch(batch):
    """Batch collate function for DataLoader
    
    Note: player_features 存储的是胜率信息而非场次
    """
    return {
        'radiant_picks': torch.stack([b['radiant_picks'] for b in batch]),
        'dire_picks': torch.stack([b['dire_picks'] for b in batch]),
        'radiant_player_feats': torch.stack([b['radiant_player_feats'] for b in batch]),
        'dire_player_feats': torch.stack([b['dire_player_feats'] for b in batch]),
        'labels': torch.stack([b['label'] for b in batch]),
        'match_ids': [b['match_id'] for b in batch]
    }


__all__ = [
    'WinRateOracle', 
    'PlayerHeroEncoder', 
    'OracleTrainingDataset',
    'collate_oracle_batch'
]
