import json
import torch
import numpy as np
import pandas as pd
from typing import Dict

from utils.device import DEVICE

# 缓存变量
_HERO_FEATURES = None
_HERO_SEMANTIC_EMBEDDINGS = None
_HERO_ID_FEATURE_MAP: Dict[int, torch.Tensor] = None
_HERO_ID_SEMANTIC_MAP: Dict[int, torch.Tensor] = None
_VALID_HERO_IDS: set = None  # 实际存在的英雄ID集合

def _load_hero_features():
    """延迟加载英雄特征数据"""
    global _HERO_FEATURES, _HERO_ID_FEATURE_MAP, _VALID_HERO_IDS
    if _HERO_FEATURES is None:
        _HERO_FEATURES = pd.read_excel("./data/hero_features.xlsx")
        _HERO_ID_FEATURE_MAP = {
            row['id']: torch.Tensor(row.drop(labels=['index', 'name', 'id']).values.astype(np.float32))
            for _, row in _HERO_FEATURES.iterrows()
        }
        _VALID_HERO_IDS = set(_HERO_ID_FEATURE_MAP.keys())
    return _HERO_FEATURES, _HERO_ID_FEATURE_MAP, _VALID_HERO_IDS

def _load_semantic_embeddings():
    """延迟加载语义嵌入数据"""
    global _HERO_SEMANTIC_EMBEDDINGS, _HERO_ID_SEMANTIC_MAP
    if _HERO_SEMANTIC_EMBEDDINGS is None:
        _HERO_SEMANTIC_EMBEDDINGS = torch.load("./data/hero_semantic_embeddings.pt", map_location=DEVICE)
        # 确保特征数据已加载
        hero_features, _, _ = _load_hero_features()
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
    
    def get_valid_hero_ids(self):
        """获取实际存在的英雄ID集合"""
        self._ensure_loaded()
        return _VALID_HERO_IDS.copy()
    
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

NUM_HEROES = 160  # 动作空间大小（最大英雄ID+1），用于模型输出维度
NUM_HERO_FEATURES = 21  # 每个英雄的属性特征维度

def get_valid_hero_ids():
    """获取实际存在的英雄ID集合（从数据文件加载）"""
    if _VALID_HERO_IDS is None:
        _load_hero_features()
    return _VALID_HERO_IDS.copy()

# 为了向后兼容，保留VALID_HERO_IDS作为函数调用
VALID_HERO_IDS = get_valid_hero_ids

def create_static_mask(max_id=NUM_HEROES):
    """创建静态mask，只保留实际存在的英雄"""
    # 初始化为极小值（屏蔽所有）
    mask = torch.full((max_id + 1,), -1e9)  # +1 是为了包含索引0
    # 只将实际存在的英雄位置设为 0（不屏蔽）
    valid_ids = get_valid_hero_ids()
    for h_id in valid_ids:
        if h_id <= max_id:
            mask[h_id] = 0.0
    # 确保 ID 0 始终被屏蔽（通常 ID 0 是 padding 或无效位）
    mask[0] = -1e9
    return mask

# 转换为常量 Tensor
STATIC_HERO_MASK = create_static_mask(NUM_HEROES)
