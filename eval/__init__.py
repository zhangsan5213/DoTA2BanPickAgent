"""
Eval Module - 评估模块

提供多种评估 BP Agent 的方法，包括：
- ELO Rating: 基于对战结果的相对强度评估
- TrueSkill: 基于贝叶斯推断的评分系统，考虑不确定性
"""

from enum import Enum, auto
from typing import Dict, Callable, Any, Optional
import os


class EvalMethod(Enum):
    """评估方法枚举"""
    ELO = auto()        # ELO Rating 评估
    TRUESKILL = auto()  # TrueSkill 评分评估
    
    def __str__(self):
        return self.name


# 评估方法注册表
EVAL_REGISTRY: Dict[EvalMethod, Dict[str, Any]] = {
    EvalMethod.ELO: {
        'name': 'ELO Rating',
        'description': '基于对战结果的相对强度评估（适合 zero-sum 博弈）',
        'module_path': 'eval.elo_rating',
        'evaluator_class': 'EloEvaluator',
    },
    EvalMethod.TRUESKILL: {
        'name': 'TrueSkill Rating',
        'description': '基于贝叶斯推断的评分系统，使用高斯分布表示技能水平，考虑不确定性',
        'module_path': 'eval.trueskill_rating',
        'evaluator_class': 'TrueSkillEvaluator',
    }
}


def get_evaluator(method: EvalMethod, **kwargs):
    """
    根据评估方法获取对应的评估器
    
    Args:
        method: 评估方法 (EvalMethod 枚举)
        **kwargs: 传递给评估器的参数
    
    Returns:
        评估器实例
    
    Example:
        >>> from eval import get_evaluator, EvalMethod
        >>> evaluator = get_evaluator(EvalMethod.ELO, save_dir="./ckpts/bp_agent")
        >>> result = evaluator.evaluate(model_path)
    """
    if method not in EVAL_REGISTRY:
        raise ValueError(f"Unknown evaluation method: {method}")
    
    config = EVAL_REGISTRY[method]
    module_path = config['module_path']
    class_name = config['evaluator_class']
    
    # 动态导入模块
    module = __import__(module_path, fromlist=[class_name])
    evaluator_class = getattr(module, class_name)
    
    return evaluator_class(**kwargs)


def list_eval_methods() -> Dict[EvalMethod, str]:
    """列出所有可用的评估方法"""
    return {method: config['name'] for method, config in EVAL_REGISTRY.items()}


def register_eval_method(
    method: EvalMethod,
    name: str,
    description: str,
    module_path: str,
    evaluator_class: str
):
    """
    注册新的评估方法
    
    Args:
        method: 评估方法枚举值
        name: 方法名称
        description: 方法描述
        module_path: 模块路径
        evaluator_class: 评估器类名
    """
    EVAL_REGISTRY[method] = {
        'name': name,
        'description': description,
        'module_path': module_path,
        'evaluator_class': evaluator_class,
    }


# 为了方便使用，导出子模块的主要接口
from eval.rating_base import (
    ModelRatingRecord,
    RatingManagerBase,
    BattleSimulatorBase,
    RatingEvaluatorBase,
)

from eval.elo_rating import (
    EloEvaluator,
    EloRatingManager,
    BPBattleSimulator as EloBattleSimulator,
    evaluate_and_update_elo,
    print_elo_leaderboard,
    INITIAL_ELO,
    ELO_K_FACTOR,
    ModelEloRecord,
)

from eval.trueskill_rating import (
    TrueSkillEvaluator,
    TrueSkillRatingManager,
    BPBattleSimulator as TrueSkillBattleSimulator,
    evaluate_and_update_trueskill,
    print_trueskill_leaderboard,
    INITIAL_MU,
    INITIAL_SIGMA,
    ModelTrueSkillRecord,
)

__all__ = [
    # 枚举和注册表
    'EvalMethod',
    'EVAL_REGISTRY',
    'get_evaluator',
    'list_eval_methods',
    'register_eval_method',
    # 抽象基类
    'ModelRatingRecord',
    'RatingManagerBase',
    'BattleSimulatorBase',
    'RatingEvaluatorBase',
    # ELO 相关
    'EloEvaluator',
    'EloRatingManager',
    'EloBattleSimulator',
    'evaluate_and_update_elo',
    'print_elo_leaderboard',
    'INITIAL_ELO',
    'ELO_K_FACTOR',
    'ModelEloRecord',
    # TrueSkill 相关
    'TrueSkillEvaluator',
    'TrueSkillRatingManager',
    'TrueSkillBattleSimulator',
    'evaluate_and_update_trueskill',
    'print_trueskill_leaderboard',
    'INITIAL_MU',
    'INITIAL_SIGMA',
    'ModelTrueSkillRecord',
]
