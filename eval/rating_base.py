"""
Rating System Abstract Base Class
评分系统抽象基类

为 ELO、TrueSkill 等评分系统提供统一的接口规范
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import os


@dataclass
class ModelRatingRecord:
    """模型评分记录基类"""
    model_path: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_games: int = 0
    last_eval_time: str = ""
    
    # 子类需要添加特定字段，如 ELO 分数、TrueSkill mu/sigma 等
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "ModelRatingRecord":
        return cls(**data)


class RatingManagerBase(ABC):
    """评分管理器抽象基类"""
    
    def __init__(self, save_dir: str = "./ckpts/bp_agent"):
        """
        初始化评分管理器
        
        Args:
            save_dir: 模型保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self._get_db_path()
        self.records: Dict[str, ModelRatingRecord] = {}
        self._load_records()
    
    @abstractmethod
    def _get_db_path(self) -> Path:
        """获取数据库文件路径"""
        pass
    
    @abstractmethod
    def _create_record(self, model_path: str, **kwargs) -> ModelRatingRecord:
        """创建新的评分记录"""
        pass
    
    def _load_records(self):
        """加载评分记录"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.records = {
                    path: self._record_from_dict(record)
                    for path, record in data.items()
                }
        else:
            self.records = {}
    
    def _save_records(self):
        """保存评分记录"""
        data = {
            path: record.to_dict()
            for path, record in self.records.items()
        }
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @abstractmethod
    def _record_from_dict(self, data: dict) -> ModelRatingRecord:
        """从字典创建记录对象（子类实现）"""
        pass
    
    def register_model(self, model_path: str, **kwargs) -> ModelRatingRecord:
        """
        注册新模型
        
        Args:
            model_path: 模型文件路径
            **kwargs: 额外参数（如初始分数等）
        
        Returns:
            ModelRatingRecord
        """
        if model_path not in self.records:
            self.records[model_path] = self._create_record(model_path, **kwargs)
            self._save_records()
        return self.records[model_path]
    
    @abstractmethod
    def update_rating(self, model_a_path: str, model_b_path: str, score_a: float):
        """
        更新两个模型的评分
        
        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            score_a: 模型 A 的实际得分（1=赢，0=输，0.5=平）
        """
        pass
    
    def get_record(self, model_path: str) -> Optional[ModelRatingRecord]:
        """获取模型的评分记录"""
        return self.records.get(model_path)
    
    @abstractmethod
    def get_rating_value(self, model_path: str) -> float:
        """
        获取模型的评分值（用于排序和显示）
        
        Args:
            model_path: 模型路径
            
        Returns:
            评分值（如 ELO 分数、TrueSkill mu 等）
        """
        pass
    
    def select_opponents(self, current_model_path: str, num_opponents: int = 5) -> List[str]:
        """
        根据当前模型评分，选择对手
        
        Args:
            current_model_path: 当前模型路径
            num_opponents: 需要选择的对手数量
        
        Returns:
            对手模型路径列表
        """
        if current_model_path not in self.records:
            self.register_model(current_model_path)
        
        # 获取所有其他模型
        other_models = [
            path for path in self.records.keys()
            if path != current_model_path and os.path.exists(path)
        ]
        
        if len(other_models) == 0:
            return []
        
        if len(other_models) <= num_opponents:
            return other_models
        
        # 子类可以实现更复杂的选择策略
        # 默认随机选择
        import random
        return random.sample(other_models, num_opponents)
    
    def list_all_models(self) -> List[Tuple[str, float]]:
        """列出所有模型及其评分值"""
        return [
            (path, self.get_rating_value(path))
            for path in self.records.keys()
        ]


class BattleSimulatorBase(ABC):
    """对战模拟器基类"""
    
    @abstractmethod
    def evaluate_models(
        self,
        model_a_path: str,
        model_b_path: str,
        num_player_sets: int = 16
    ) -> Tuple[float, List[Dict]]:
        """
        评估两个模型的对战结果
        
        Args:
            model_a_path: 模型 A 路径
            model_b_path: 模型 B 路径
            num_player_sets: 玩家 set 数量
        
        Returns:
            (model_a 胜率, 详细对战记录)
        """
        pass


class RatingEvaluatorBase(ABC):
    """
    评分评估器抽象基类
    
    用于评估 BP Agent 模型的相对强度，通过与其他模型对战来更新评分。
    """
    
    def __init__(
        self,
        save_dir: str = "./ckpts/bp_agent",
        num_opponents: int = 5,
        num_player_sets: int = 16,
    ):
        """
        初始化评分评估器
        
        Args:
            save_dir: 模型保存目录
            num_opponents: 每次评估的对手数量
            num_player_sets: 每个对手对战的玩家 set 数量
        """
        self.save_dir = save_dir
        self.num_opponents = num_opponents
        self.num_player_sets = num_player_sets
        
        # 子类需要初始化 rating_manager 和 battle_simulator
        self.rating_manager: Optional[RatingManagerBase] = None
        self.battle_simulator: Optional[BattleSimulatorBase] = None
    
    @abstractmethod
    def evaluate(
        self,
        model_path: str,
        num_opponents: Optional[int] = None,
        num_player_sets: Optional[int] = None
    ) -> Dict:
        """
        评估模型并更新评分
        
        Args:
            model_path: 模型文件路径
            num_opponents: 对手数量（覆盖默认值）
            num_player_sets: 玩家 set 数量（覆盖默认值）
        
        Returns:
            评估结果字典
        """
        pass
    
    @abstractmethod
    def get_rating(self, model_path: str) -> float:
        """获取模型的当前评分值"""
        pass
    
    @abstractmethod
    def print_leaderboard(self):
        """打印排行榜"""
        pass
    
    def register_model(self, model_path: str, **kwargs) -> ModelRatingRecord:
        """手动注册模型"""
        return self.rating_manager.register_model(model_path, **kwargs)
    
    def list_models(self) -> List[Tuple[str, float]]:
        """列出所有已注册模型及其评分"""
        return self.rating_manager.list_all_models()
