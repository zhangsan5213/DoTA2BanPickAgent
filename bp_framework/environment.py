"""
DOTA2 BP环境 - 管理BP状态转移和合法性检查
"""
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from enum import IntEnum

# 加载有效英雄ID
def _load_valid_hero_ids() -> Set[int]:
    """从数据文件加载有效英雄ID"""
    try:
        df = pd.read_excel("./data/hero_features.xlsx")
        return set(df['id'].tolist())
    except Exception as e:
        print(f"Warning: Could not load hero features, using default range: {e}")
        return set(range(1, 156))  # 默认1-155

VALID_HERO_IDS = _load_valid_hero_ids()
NUM_HEROES = 160  # 最大英雄ID+1（用于mask维度）


class ActionType(IntEnum):
    BAN = 0
    PICK = 1


class Team(IntEnum):
    RADIANT = 0
    DIRE = 1


@dataclass
class BPAction:
    """单个BP动作"""
    hero_id: int              # 英雄ID (1-based)
    action_type: ActionType   # ban或pick
    team: Team                # 执行动作的阵营
    position: int             # 第几个动作（0-based）
    
    def to_tensor_dict(self, device='cpu') -> Dict[str, torch.Tensor]:
        """转为tensor格式（用于模型输入）"""
        return {
            'hero_id': torch.tensor([self.hero_id], dtype=torch.long, device=device),
            'action_type': torch.tensor([int(self.action_type)], dtype=torch.long, device=device),
            'team': torch.tensor([int(self.team)], dtype=torch.long, device=device),
            'position': torch.tensor([self.position], dtype=torch.long, device=device),
        }


@dataclass
class BPState:
    """BP状态（观测）"""
    # 动作历史（变长）
    action_history: List[BPAction] = field(default_factory=list)
    
    # 当前状态
    banned_heroes: set = field(default_factory=set)      # 已被ban的英雄
    radiant_picks: List[int] = field(default_factory=list)  # 天辉已pick（有序）
    dire_picks: List[int] = field(default_factory=list)     # 夜魇已pick（有序）
    
    # 玩家特征（可选）
    radiant_player_feats: Optional[torch.Tensor] = None  # [5, NUM_HEROES]
    dire_player_feats: Optional[torch.Tensor] = None     # [5, NUM_HEROES]
    
    # 当前步数信息
    current_step: int = 0
    current_team: Team = Team.RADIANT  # 当前该谁行动
    current_action_type: ActionType = ActionType.BAN  # 当前是ban还是pick阶段
    
    # 是否结束
    is_terminal: bool = False
    winner: Optional[Team] = None  # 如果有的话
    
    def get_available_actions(self, num_heroes: int = NUM_HEROES, valid_hero_ids: Optional[Set[int]] = None) -> torch.Tensor:
        """
        获取可用动作掩码
        
        Args:
            num_heroes: mask总维度（通常为160）
            valid_hero_ids: 有效英雄ID集合，None时使用全局VALID_HERO_IDS
        
        Returns:
            mask: [num_heroes] 1=可用, 0=不可用
        """
        if valid_hero_ids is None:
            valid_hero_ids = VALID_HERO_IDS
        
        mask = torch.zeros(num_heroes, dtype=torch.float32)
        
        # 有效且未被ban/pick的英雄可用
        unavailable = self.banned_heroes | set(self.radiant_picks) | set(self.dire_picks)
        available_heroes = valid_hero_ids - unavailable
        
        for hero_id in available_heroes:
            if 1 <= hero_id <= num_heroes:
                mask[hero_id - 1] = 1  # hero_id是1-based，mask是0-based
        
        return mask
    
    def to_tensor_batch(self, device='cpu') -> Dict[str, torch.Tensor]:
        """
        将状态转为模型输入的tensor批次
        Returns:
            包含hero_ids, action_types, teams, positions的tensor字典
            每个tensor形状为 [1, seq_len]
        """
        seq_len = len(self.action_history)
        if seq_len == 0:
            # 空序列的情况
            return {
                'hero_ids': torch.zeros((1, 0), dtype=torch.long, device=device),
                'action_types': torch.zeros((1, 0), dtype=torch.long, device=device),
                'teams': torch.zeros((1, 0), dtype=torch.long, device=device),
                'positions': torch.zeros((1, 0), dtype=torch.long, device=device),
                'seq_mask': torch.zeros((1, 0), dtype=torch.long, device=device),
            }
        
        hero_ids = torch.tensor([[a.hero_id for a in self.action_history]], dtype=torch.long, device=device)
        action_types = torch.tensor([[int(a.action_type) for a in self.action_history]], dtype=torch.long, device=device)
        teams = torch.tensor([[int(a.team) for a in self.action_history]], dtype=torch.long, device=device)
        positions = torch.tensor([[a.position for a in self.action_history]], dtype=torch.long, device=device)
        seq_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
        
        return {
            'hero_ids': hero_ids,
            'action_types': action_types,
            'teams': teams,
            'positions': positions,
            'seq_mask': seq_mask,
        }
    
    def get_final_picks(self) -> Tuple[List[int], List[int]]:
        """获取最终的pick结果"""
        return self.radiant_picks.copy(), self.dire_picks.copy()


class BPPhase:
    """BP阶段定义（CM模式）"""
    def __init__(self, first_team: Team = Team.RADIANT):
        """
        Args:
            first_team: 先手队伍（谁先开始ban）
        """
        # CM标准模式: Ban 1-4, Pick 1-3, Ban 5-6, Pick 4-5
        # 格式: (action_type, team)
        self.phases = []
        
        team = first_team
        other_team = Team.DIRE if first_team == Team.RADIANT else Team.RADIANT
        
        # Ban phase 1: 每边ban 3个，共6个
        for _ in range(3):
            self.phases.append((ActionType.BAN, team))
            self.phases.append((ActionType.BAN, other_team))
        
        # Pick phase 1: 每边pick 2个，共4个
        # radiant先pick的话是 1-2, dire 1-2, radiant 3, dire 3
        # 这里简化为轮流pick
        for _ in range(2):
            self.phases.append((ActionType.PICK, team))
            self.phases.append((ActionType.PICK, other_team))
        self.phases.append((ActionType.PICK, team))
        self.phases.append((ActionType.PICK, other_team))
        
        # Ban phase 2: 每边ban 2个，共4个
        for _ in range(2):
            self.phases.append((ActionType.BAN, other_team))  # 第二轮后手先ban
            self.phases.append((ActionType.BAN, team))
        
        # Pick phase 2: 每边pick 2个，共4个
        self.phases.append((ActionType.PICK, other_team))
        self.phases.append((ActionType.PICK, team))
        self.phases.append((ActionType.PICK, other_team))
        self.phases.append((ActionType.PICK, team))
        
        self.total_steps = len(self.phases)
    
    def get_step_info(self, step: int) -> Tuple[ActionType, Team]:
        """获取第step步的动作类型和执行阵营"""
        if step >= self.total_steps:
            raise ValueError(f"Step {step} exceeds total steps {self.total_steps}")
        return self.phases[step]
    
    def is_terminal(self, step: int) -> bool:
        """检查是否已结束"""
        return step >= self.total_steps


class BPEnvironment:
    """BP环境：管理状态转移"""
    
    def __init__(
        self,
        num_heroes: int = NUM_HEROES,
        first_team: Team = Team.RADIANT,
        device: str = 'cpu',
        valid_hero_ids: Optional[Set[int]] = None,
    ):
        self.num_heroes = num_heroes
        self.phase_schedule = BPPhase(first_team)
        self.device = device
        self.valid_hero_ids = valid_hero_ids if valid_hero_ids is not None else VALID_HERO_IDS
        self.reset()
    
    def reset(
        self,
        radiant_player_feats: Optional[torch.Tensor] = None,
        dire_player_feats: Optional[torch.Tensor] = None,
    ) -> BPState:
        """重置环境
        
        Args:
            radiant_player_feats: 天辉玩家英雄偏好 [5, NUM_HEROES]（可选）
            dire_player_feats: 夜魇玩家英雄偏好 [5, NUM_HEROES]（可选）
        """
        self.state = BPState(
            radiant_player_feats=radiant_player_feats,
            dire_player_feats=dire_player_feats,
        )
        self._update_current_phase()
        return self.state
    
    def _update_current_phase(self):
        """更新当前阶段信息"""
        if self.phase_schedule.is_terminal(self.state.current_step):
            self.state.is_terminal = True
            return
        
        action_type, team = self.phase_schedule.get_step_info(self.state.current_step)
        self.state.current_action_type = action_type
        self.state.current_team = team
    
    def step(self, hero_id: int) -> Tuple[BPState, bool, dict]:
        """
        执行一步BP动作
        
        Args:
            hero_id: 选择的英雄ID (1-based)
        
        Returns:
            next_state: 下一个状态
            is_terminal: 是否结束
            info: 额外信息
        """
        if self.state.is_terminal:
            raise RuntimeError("Episode already terminated")
        
        # 检查英雄ID是否有效
        if hero_id not in self.valid_hero_ids:
            raise ValueError(f"Hero {hero_id} is not a valid hero ID")
        
        # 检查动作合法性
        available = self.state.get_available_actions(self.num_heroes, self.valid_hero_ids)
        if available[hero_id - 1] == 0:
            raise ValueError(f"Hero {hero_id} is not available (already banned/picked or invalid)")
        
        # 执行动作
        action = BPAction(
            hero_id=hero_id,
            action_type=self.state.current_action_type,
            team=self.state.current_team,
            position=self.state.current_step,
        )
        self.state.action_history.append(action)
        
        # 更新状态
        if action.action_type == ActionType.BAN:
            self.state.banned_heroes.add(hero_id)
        else:  # PICK
            if action.team == Team.RADIANT:
                self.state.radiant_picks.append(hero_id)
            else:
                self.state.dire_picks.append(hero_id)
        
        # 更新步数
        self.state.current_step += 1
        self._update_current_phase()
        
        info = {
            'action': action,
            'valid': True,
        }
        
        return self.state, self.state.is_terminal, info
    
    def get_valid_actions(self) -> torch.Tensor:
        """获取当前可用动作"""
        return self.state.get_available_actions(self.num_heroes, self.valid_hero_ids).to(self.device)
    
    def get_state_for_agent(self) -> Dict[str, torch.Tensor]:
        """获取用于agent输入的状态"""
        state_tensors = self.state.to_tensor_batch(self.device)
        state_tensors['action_mask'] = self.get_valid_actions().unsqueeze(0)  # [1, num_heroes]
        return state_tensors
