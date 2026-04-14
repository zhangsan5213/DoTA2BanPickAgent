# Dota 2 Ban Pick Agent - AI 编码助手指南

## 项目概述

Dota 2 Ban Pick Agent 是一个基于强化学习的项目，用于训练 AI 代理在 Dota 2 队长模式（Captain Mode）中进行智能阵容选择博弈（Ban/Pick）。该项目通过考虑玩家英雄偏好、英雄协同效应和克制关系，学习做出最优决策以最大化获胜概率。

### 核心特点

- **玩家感知的阵容选择**：与通用 Ban/Pick 代理不同，此代理会整合玩家在各英雄上的历史胜率
- **多模态英雄编码**：结合结构化英雄属性与来自技能描述的文本嵌入
- **两阶段训练**：监督预训练的 WinRateOracle 为强化学习提供奖励信号，无需完整游戏对局
- **持续改进**：使用 ELO/TrueSkill 评分系统评估代理版本与历史检查点的对比
- **遵循官方竞技规则**：实现精确的队长模式 Ban/Pick 序列
- **现代 RL**：使用带广义优势估计（GAE）的近端策略优化（PPO）

## 技术栈

- **深度学习框架**: PyTorch + Transformers
- **强化学习算法**: PPO (Proximal Policy Optimization) + GAE
- **数据处理**: Pandas, NumPy
- **评分系统**: ELO, TrueSkill
- **日志记录**: TensorBoard
- **配置管理**: YAML

## 项目结构

```
├── configs/                    # 配置文件目录
│   ├── bp_agent_config.yaml    # BP Agent 训练配置（主配置）
│   └── bp_agent_config_debug.yaml  # 调试配置
│
├── data/                       # 数据文件目录
│   ├── hero_features.xlsx      # 英雄属性特征表
│   ├── hero_semantic_embeddings.pt  # 英雄语义嵌入（来自技能描述）
│   ├── hero_static_features.pt # 英雄静态特征
│   ├── hero_ability_descriptions.json  # 英雄技能描述
│   ├── hero_positions.json     # 英雄位置定义
│   ├── hero_winrates.json      # 英雄胜率数据
│   └── high_mmr_with_stats*.json  # 高 MMR 比赛数据（训练数据）
│
├── model/                      # 神经网络模型定义
│   ├── bp_agent.py            # 主 BP Transformer Agent（策略/价值网络）
│   ├── win_rate_oracle.py     # 胜率预测 Oracle
│   └── hero_encoder.py        # 多模态英雄编码器
│
├── utils/                      # 工具函数
│   ├── bp_env.py              # RL 环境实现（BPState、GAE、PPO Loss）
│   ├── raw_data.py            # 数据加载工具（延迟加载英雄特征）
│   ├── device.py              # 设备管理（CUDA/CPU）
│   ├── player_preference_sampler_optimized.py  # 玩家偏好采样器
│   ├── opendota_api.py        # OpenDota API 接口
│   └── get_data_*.py          # 数据获取脚本
│
├── eval/                       # 评估和评分系统
│   ├── __init__.py            # 评估方法注册和工厂
│   ├── rating_base.py         # 评分基类定义
│   ├── elo_rating.py          # ELO 评分实现
│   └── trueskill_rating.py    # TrueSkill 评分实现
│
├── trainer/                    # 模块化训练组件（新架构）
│   ├── config.py              # 配置管理类
│   ├── bp_agent_trainer.py    # 主训练器
│   ├── epoch_runner.py        # 轮次运行器
│   ├── rollout_collector.py   # 轨迹收集
│   ├── evaluator.py           # 评估器
│   ├── loss_computer.py       # 损失计算
│   └── ...
│
├── ckpts/                      # 模型检查点保存目录
│   ├── win_rate_oracle-*/     # WinRateOracle 检查点
│   └── bp_agent-*/            # BP Agent 检查点
│
├── runs/                       # TensorBoard 日志目录
│
├── train_winrate_oracle.py     # WinRateOracle 训练脚本
├── train_bp_agent.py           # BP Agent 主训练脚本
├── train_bp_agent_new.py       # 新架构训练脚本（使用 trainer/ 模块）
└── eval_bp_agent.py            # 评估脚本（锦标赛模式）
```

## 架构详解

### 1. WinRateOracle（胜率预言机）

预训练的神经网络，用于预测给定最终阵容和玩家偏好时的胜率。为强化学习提供奖励信号。

**关键配置**:
- `embed_dim`: 128
- `nhead`: 8
- `num_layers`: 6
- `use_text`: True（使用文本嵌入）
- `use_player_heroes`: True（使用玩家英雄偏好）

**当前性能**: 在留出高 MMR 比赛数据上达到约 **90.4%** 的预测准确率。

### 2. BPTransformerAgent（主代理）

基于 Transformer 的策略/价值网络，处理：
- **玩家偏好**: 编码每位玩家在各英雄上的历史胜率
- **动作历史**: 编码之前的 Ban/Pick 动作
- **输出**: 生成下一动作的策略 logits 和状态价值估计

**网络结构**:
- ActionEncoder: 编码 (actor_team, action_type, target_hero)
- PlayerEncoder: 编码 5 位玩家的英雄偏好
- TransformerEncoder: 4 层，8 头，256 维嵌入
- Policy Head: 输出 NUM_HEROES (160) 维动作概率
- Value Head: 输出状态价值估计

### 3. 环境（BPState）

实现标准 Dota 2 队长模式 Ban/Pick 顺序（共 20 步）：

1. **Ban Phase 1**: R, D, R, D (4 bans)
2. **Pick Phase 1**: R, D, D, R (4 picks)
3. **Ban Phase 2**: D, R, D, R (4 bans)
4. **Pick Phase 2**: D, R, R, D (4 picks)
5. **Ban Phase 3**: R, D (2 bans)
6. **Pick Phase 3**: R, D (2 picks)

总计：**10 bans + 10 picks**（每队 5 ban 5 pick）

## 运行命令

### 环境要求

**默认执行环境**: `E:naconda	eachs	eshin`

**依赖包**:
```bash
pip install torch pandas numpy openpyxl tqdm pyyaml tensorboard trueskill
```

### 1. 训练 WinRateOracle

```bash
python train_winrate_oracle.py
```

此脚本会：
1. 从 OpenDota API 获取高 MMR 比赛数据（如需要）
2. 训练胜率预测模型
3. 保存检查点到 `./ckpts/win_rate_oracle-*/`

### 2. 训练 BP Agent

```bash
# 使用默认配置
python train_bp_agent.py

# 使用调试配置
python train_bp_agent.py --config configs/bp_agent_config_debug.yaml
```

配置参数（`configs/bp_agent_config.yaml`）:
- `actor_lr`: 3e-4（策略网络学习率）
- `value_loss_coeff`: 2.0（价值损失系数）
- `entropy_loss_coeff`: 0.03（熵损失系数）
- `rating.method`: "trueskill" 或 "elo"
- `training.epochs`: 32（训练轮数）
- `training.batch_size`: 32（批次大小）
- `training.historical_opponent_prob`: 0.6（60% 对局使用历史对手）

### 3. 评估代理

```bash
# 让评分最高的前 3 个模型对战，每对进行 3 场比赛
python eval_bp_agent.py --top_n 3 --matches 3

# 评估指定模型
python eval_bp_agent.py --models ./ckpts/model1.pth ./ckpts/model2.pth --matches 5

# 使用 ELO 评分选择模型
python eval_bp_agent.py --top_n 3 --rating elo
```

### 4. 查看 TensorBoard 日志

```bash
tensorboard --logdir ./runs --port 6006
```

## 代码规范

### 文件组织原则

1. **模型定义**放在 `model/` 目录
2. **工具函数**放在 `utils/` 目录
3. **训练脚本**放在根目录
4. **配置**放在 `configs/` 目录

### 命名规范

- **类名**: PascalCase（如 `BPTransformerAgent`, `WinRateOracle`）
- **函数名**: snake_case（如 `compute_gae`, `collect_rollout`）
- **常量**: UPPER_SNAKE_CASE（如 `NUM_HEROES`, `EMBED_DIM`）
- **私有函数**: 下划线前缀（如 `_load_hero_data`, `_process_player_feats`）

### 注释规范

- 使用中文注释（项目主要使用中文）
- 类和方法使用文档字符串说明功能、参数和返回值
- 关键算法步骤添加行内注释

### 设备管理

所有模型和数据应使用 `utils.device.DEVICE`：

```python
from utils.device import DEVICE

model = MyModel().to(DEVICE)
tensor = tensor.to(DEVICE)
```

### 环境变量

所有 Python 文件开头应设置：
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```

这是为了解决 OpenMP 相关的库冲突问题。

## 数据文件说明

### hero_features.xlsx
包含英雄的结构化属性特征（21维）：
- 基础属性: attr_str, attr_agi, attr_int, attr_all
- 角色标签: role_Carry, role_Support, role_Pusher 等

### hero_semantic_embeddings.pt
英雄语义嵌入（1024维），通过处理英雄技能描述文本获得。

### high_mmr_with_stats-rank_40-duration_15.json
训练数据，包含：
- 比赛 ID
- picks_bans: Ban/Pick 序列
- players: 玩家信息，包含 hero_history（各英雄场次/胜场）
- radiant_win: 比赛结果

## 训练流程

### 两阶段训练方法

1. **监督预训练阶段**: WinRateOracle 在真实高 MMR 比赛数据上训练，预测胜率
2. **强化学习微调阶段**: 
   - 代理与自己对战（40%）或与历史版本对战（60%）
   - Oracle 在每个 Ban/Pick 序列结束时提供奖励
   - 每 N 个 epoch 进行评估，新代理与现有检查点对战并评分
   - 评分较高的代理更可能被采样为对手，推动持续改进

### PPO 训练参数

- **Gamma**: 0.99（折扣因子）
- **Lambda**: 0.95（GAE 参数）
- **Clip Epsilon**: 0.2（PPO 裁剪参数）
- **Value Clip Epsilon**: 0.2（价值裁剪参数）

## 评分系统

支持两种评分方法：

### ELO Rating
- 初始评分: 1500
- K 因子: 32
- 适用于零和博弈的相对强度评估

### TrueSkill
- 初始 Mu: 25.0
- 初始 Sigma: 8.33
- Beta: 4.17
- 使用高斯分布表示技能水平，考虑不确定性

## 重要注意事项

### Python 环境使用规则

**绝对禁止擅自操作**

1. **禁止擅自操作**：
   - 永远不会擅自使用 pip、conda 等包管理工具
   - 禁止对 Python 环境进行任何未授权的修改

2. **默认执行环境**：
   - 所有代码执行默认使用：`E:naconda	eachs	eshin` 环境
   - 如需切换环境必须明确获得授权

3. **执行失败处理**：
   - 如果命令执行失败或无法执行，立即停止操作
   - 清晰地向用户报告问题，等待指示
   - 不进行任何尝试性的修复或替代方案

### 关键实现细节

1. **英雄 ID 处理**: 数据文件中使用 1-based ID（1-160），模型内部使用 0-based ID（0-159），注意转换
2. **有效英雄过滤**: 使用 `get_valid_hero_ids()` 获取实际存在的英雄 ID 集合
3. **动作掩码**: 在策略输出上应用掩码，防止选择已使用或不存在的英雄
4. **奖励计算**: 对于 Dire 方，奖励为 `1.0 - radiant_win_prob`

## 调试技巧

1. **检查数据加载**: 运行 `utils/raw_data.py` 测试英雄特征加载
2. **测试环境**: 运行 `utils/bp_env.py` 测试 BP 环境和 GAE 计算
3. **测试模型**: 各模型文件包含 `if __name__ == "__main__"` 测试代码
4. **TensorBoard**: 实时监控损失、准确率、评分等指标

## 相关文件快速参考

| 功能 | 文件 |
|------|------|
| 主训练脚本 | `train_bp_agent.py` |
| Oracle 训练 | `train_winrate_oracle.py` |
| 评估脚本 | `eval_bp_agent.py` |
| BP Agent 模型 | `model/bp_agent.py` |
| Oracle 模型 | `model/win_rate_oracle.py` |
| 英雄编码器 | `model/hero_encoder.py` |
| 环境实现 | `utils/bp_env.py` |
| 主配置 | `configs/bp_agent_config.yaml` |
| 评分系统 | `eval/` 目录 |
