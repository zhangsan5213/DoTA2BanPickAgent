import json
import torch

import numpy as np
import pandas as pd

hero_features = pd.read_excel("./data/hero_features.xlsx")
HERO_ID_FEATURE_MAP = {
    row['id']: torch.Tensor(row.drop(labels=['index', 'name', 'id']).values.astype(np.float32))
    for _, row in hero_features.iterrows()
}

hero_semantic_embeddings = torch.load("./data/hero_semantic_embeddings.pt")
HERO_ID_SEMANTIC_MAP = {
    row['id']: hero_semantic_embeddings[row['name']]
    for _, row in hero_features.iterrows()
}

NUM_HEROES = 256  # 英雄个数，比真实的偏大
NUM_HERO_FEATURES = 21  # 每个英雄的属性特征维度

# 获取所有合法的英雄 ID
VALID_HERO_IDS = set(HERO_ID_FEATURE_MAP.keys())

def create_static_mask(max_id=150):
    # 初始化为极小值（屏蔽）
    mask = torch.full((max_id,), -1e9)
    # 将合法英雄的位置设为 0（不屏蔽）
    for h_id in VALID_HERO_IDS:
        if h_id < max_id:
            mask[h_id] = 0.0
    # 确保 ID 0 始终被屏蔽（通常 ID 0 是 padding 或无效位）
    mask[0] = -1e9
    return mask

# 转换为常量 Tensor
STATIC_HERO_MASK = create_static_mask(NUM_HEROES + 1)