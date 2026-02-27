"""
Dataset loading and management for BP Agent training
"""
import json
import os
import random
from typing import List, Dict, Any


def load_matches_from_json(file_path: str) -> List[Dict[str, Any]]:
    """
    从 JSON 文件加载比赛数据
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        有效比赛数据列表（两队各有5个pick）
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤有效比赛（两队各有5个pick）
    valid_matches = []
    for m in data:
        r_picks = [act['hero_id'] for act in m.get('picks_bans', [])
                   if act.get('is_pick', False) and act.get('team', 0) == 0]
        d_picks = [act['hero_id'] for act in m.get('picks_bans', [])
                   if act.get('is_pick', False) and act.get('team', 0) == 1]
        if len(r_picks) == 5 and len(d_picks) == 5:
            valid_matches.append(m)

    print(f"[*] 加载了 {len(valid_matches)} 场有效比赛")
    return valid_matches


class MatchDataset:
    """
    比赛数据集管理器
    
    支持：
    - 加载比赛数据
    - 随机采样
    - 批量获取
    """
    
    def __init__(self, data_file: str):
        """
        Args:
            data_file: 数据文件路径
        """
        self.data_file = data_file
        self.matches = load_matches_from_json(data_file)
        
    def __len__(self) -> int:
        return len(self.matches)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.matches[idx]
    
    def sample(self, n: int = 1) -> List[Dict[str, Any]]:
        """随机采样n场比赛"""
        return random.sample(self.matches, min(n, len(self.matches)))
    
    def get_random_match(self) -> Dict[str, Any]:
        """获取单场比赛"""
        return random.choice(self.matches)
