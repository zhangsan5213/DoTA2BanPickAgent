"""
DOTA2 BP (Ban/Pick) Self-Play 框架

核心设计：
- 同一个Actor-Critic模型服务双方（Radiant & Dire）
- 通过team embedding区分阵营视角
- 分别收集双方的trajectory，独立计算advantage

快速开始:
```python
from model.bp_agent import BPActorCritic
from model.win_rate_oracle import WinRateOracle
from bp_framework.engine import BPEngine
from bp_framework.environment import Team

# 加载模型
actor_critic = BPActorCritic(...)
oracle = WinRateOracle(...)

# 创建引擎（Self-Play模式）
engine = BPEngine(
    actor_critic=actor_critic,
    oracle=oracle,
    device='cuda',
    first_team=Team.RADIANT,  # 天辉先手ban
)

# 收集训练数据
r_buffer, d_buffer = engine.collect_rollouts(num_episodes=100)

# 合并双方数据进行训练（Self-Play）
all_batches = r_buffer.get_all_batches() + d_buffer.get_all_batches()
```
"""

from bp_framework.environment import (
    ActionType,
    Team,
    BPAction,
    BPState,
    BPPhase,
    BPEnvironment,
)

from bp_framework.rollout import (
    BPTransition,
    BPRollout,
    RolloutBuffer,
)

from bp_framework.reward import (
    RewardCalculator,
    DualTeamRewardCalculator,
)

from bp_framework.engine import (
    BPEngine,
)

__all__ = [
    # Environment
    'ActionType',
    'Team',
    'BPAction',
    'BPState',
    'BPPhase',
    'BPEnvironment',
    # Rollout
    'BPTransition',
    'BPRollout',
    'RolloutBuffer',
    # Reward
    'RewardCalculator',
    'DualTeamRewardCalculator',
    # Engine
    'BPEngine',
]
