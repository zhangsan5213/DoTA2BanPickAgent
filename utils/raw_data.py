import json
import torch
import numpy as np
import pandas as pd
from typing import Dict

# 缓存变量
_HERO_FEATURES = None
_HERO_SEMANTIC_EMBEDDINGS = None
_HERO_ID_FEATURE_MAP: Dict[int, torch.Tensor] = None
_HERO_ID_SEMANTIC_MAP: Dict[int, torch.Tensor] = None

def _load_hero_features():
    """延迟加载英雄特征数据"""
    global _HERO_FEATURES, _HERO_ID_FEATURE_MAP
    if _HERO_FEATURES is None:
        _HERO_FEATURES = pd.read_excel("./data/hero_features.xlsx")
        _HERO_ID_FEATURE_MAP = {
            row['id']: torch.Tensor(row.drop(labels=['index', 'name', 'id']).values.astype(np.float32))
            for _, row in _HERO_FEATURES.iterrows()
        }
    return _HERO_FEATURES, _HERO_ID_FEATURE_MAP

def _load_semantic_embeddings():
    """延迟加载语义嵌入数据"""
    global _HERO_SEMANTIC_EMBEDDINGS, _HERO_ID_SEMANTIC_MAP
    if _HERO_SEMANTIC_EMBEDDINGS is None:
        _HERO_SEMANTIC_EMBEDDINGS = torch.load("./data/hero_semantic_embeddings.pt")
        # 确保特征数据已加载
        hero_features, _ = _load_hero_features()
        _HERO_ID_SEMANTIC_MAP = {
            row['id']: _HERO_SEMANTIC_EMBEDDINGS[row['name']]
            for _, row in hero_features.iterrows()
        }
    return _HERO_SEMANTIC_EMBEDDINGS, _HERO_ID_SEMANTIC_MAP


# 使用类包装器实现延迟加载
class _LazyHeroFeatures:
    """延迟加载的英雄特征"""
    def _ensure_loaded(self):
        if _HERO_ID_FEATURE_MAP is None:
            _load_hero_features()
    
    def __repr__(self):
        return f"<_LazyHeroFeatures (loaded={_HERO_ID_FEATURE_MAP is not None})>"
    
    def __getitem__(self, key):
        self._ensure_loaded()
        return _HERO_ID_FEATURE_MAP[key]
    
    def get(self, key, default=None):
        self._ensure_loaded()
        return _HERO_ID_FEATURE_MAP.get(key, default)
    
    def __contains__(self, key):
        self._ensure_loaded()
        return key in _HERO_ID_FEATURE_MAP
    
    def keys(self):
        self._ensure_loaded()
        return _HERO_ID_FEATURE_MAP.keys()
    
    def values(self):
        self._ensure_loaded()
        return _HERO_ID_FEATURE_MAP.values()
    
    def items(self):
        self._ensure_loaded()
        return _HERO_ID_FEATURE_MAP.items()
    
    def __iter__(self):
        self._ensure_loaded()
        return iter(_HERO_ID_FEATURE_MAP)
    
    def __len__(self):
        self._ensure_loaded()
        return len(_HERO_ID_FEATURE_MAP)


class _LazySemanticMap:
    """延迟加载的语义映射"""
    def _ensure_loaded(self):
        if _HERO_ID_SEMANTIC_MAP is None:
            _load_semantic_embeddings()
    
    def __repr__(self):
        return f"<_LazySemanticMap (loaded={_HERO_ID_SEMANTIC_MAP is not None})>"
    
    def __getitem__(self, key):
        self._ensure_loaded()
        return _HERO_ID_SEMANTIC_MAP[key]
    
    def get(self, key, default=None):
        self._ensure_loaded()
        return _HERO_ID_SEMANTIC_MAP.get(key, default)
    
    def __contains__(self, key):
        self._ensure_loaded()
        return key in _HERO_ID_SEMANTIC_MAP
    
    def keys(self):
        self._ensure_loaded()
        return _HERO_ID_SEMANTIC_MAP.keys()
    
    def values(self):
        self._ensure_loaded()
        return _HERO_ID_SEMANTIC_MAP.values()
    
    def items(self):
        self._ensure_loaded()
        return _HERO_ID_SEMANTIC_MAP.items()
    
    def __iter__(self):
        self._ensure_loaded()
        return iter(_HERO_ID_SEMANTIC_MAP)
    
    def __len__(self):
        self._ensure_loaded()
        return len(_HERO_ID_SEMANTIC_MAP)


# 导出延迟加载的映射对象
HERO_ID_FEATURE_MAP = _LazyHeroFeatures()
HERO_ID_SEMANTIC_MAP = _LazySemanticMap()

NUM_HEROES = 160  # 英雄个数，比真实的偏大（支持到ID 159，当前最大ID为155）
NUM_HERO_FEATURES = 21  # 每个英雄的属性特征维度

# 获取所有合法的英雄 ID（延迟计算）
VALID_HERO_IDS = set(range(1, NUM_HEROES + 1))

def create_static_mask(max_id=150):
    # 初始化为极小值（屏蔽）
    mask = torch.full((max_id,), -1e9)
    # 将合法英雄的位置设为 0（不屏蔽）
    for h_id in range(1, min(max_id, NUM_HEROES + 1)):
        mask[h_id] = 0.0
    # 确保 ID 0 始终被屏蔽（通常 ID 0 是 padding 或无效位）
    mask[0] = -1e9
    return mask

# 转换为常量 Tensor
STATIC_HERO_MASK = create_static_mask(NUM_HEROES + 1)
