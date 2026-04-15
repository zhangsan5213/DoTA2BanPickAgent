# DOTA 2 BP Agent 技术实现分析与改进方案

> 分析日期：2026年4月  
> 分析对象：当前 RL 仓库 + 《DOTA 2 智能英雄禁用/选择技术调研》报告

---

## 一、当前算法实现解读

### 1.1 整体架构：两阶段训练（Oracle + RL）

项目采用**监督预训练 → 强化学习微调**的两阶段范式：

1. **WinRateOracle**（监督学习）
   - 输入：双方最终阵容（5+5 英雄）+ 玩家英雄胜率偏好
   - 输出：天辉胜率预测值
   - 架构：MultiModalHeroEncoder（ID + 属性 + 语义三模态融合）+ Transformer + MLP Head
   - 宣称性能：验证集准确率 ~90.4%

2. **BPTransformerAgent**（强化学习）
   - 输入：双方玩家偏好矩阵 `[B, 5, NUM_HEROES]` + BP 动作历史
   - 输出：策略 logits `[B, NUM_HEROES]` + 状态价值 `[B, 1]`
   - 架构：ActionEncoder + PlayerEncoder（独立 MLP）+ 6 层 Transformer + Policy/Value Head
   - 算法：PPO + GAE，支持 KL 早停、梯度裁剪、Entropy 退火

3. **评估系统**
   - ELO / TrueSkill 双评分体系
   - 历史对手采样（historical opponent prob = 0.3）
   - TrueSkill 分层抽样（stratified sampling）
   - Staleness 刷新机制

### 1.2 环境设计

- `BPState` 严格遵循 DOTA 2 队长模式（CM）的 20 步序列：
  - Ban Phase 1: R, D, R, D (4 bans)
  - Pick Phase 1: R, D, D, R (4 picks)
  - Ban Phase 2: D, R, D, R (4 bans)
  - Pick Phase 2: D, R, R, D (4 picks)
  - Ban Phase 3: R, D (2 bans)
  - Pick Phase 3: R, D (2 picks)
- 动作掩码：已选/已 Ban/不存在的英雄被 `-1e9` mask 掉
- 奖励：仅终局获得 Oracle 预测的天辉胜率，Dire 方奖励取反（zero-sum）

### 1.3 训练流程（PPO）

1. 每 epoch 生成 4096 组玩家偏好样本
2. 按 batch（64）采集 rollout：30% 对历史模型，70% 自对弈
3. 对有效 step 打平，按 history 长度分组做 batch forward
4. 执行 4 轮 PPO epoch，minibatch=64，KL>0.1 早停
5. 每隔 1 个 epoch 做 TrueSkill 评估并保存 checkpoint

---

## 二、实现问题（Bugs & 代码缺陷）

### 2.1 Temperature 不一致导致的 off-policy 偏差

**位置**：`utils/bp_env.py:395` vs `trainer/loss_computer.py:278`

- **Rollout 采样**时使用 `probs = F.softmax(action_logits / temp, dim=-1)` 采样动作并记录 `old_log_prob`
- **Loss 计算**时再次除以 temperature：`batch_logits_temp = batch_logits / temp`，然后计算新 log_prob 和 PPO ratio

**问题**：`old_log_prob` 是在 rollout 阶段用 temperature 缩放后的分布采样的，而 loss 计算时又一次除以 temperature。若 `temp ≠ 1.0`，则新旧 log_prob 基于不同温度参数，ratio 计算存在系统性偏差，PPO 的 trust region 被破坏。正确做法应该是：
- rollout 采样用 temperature 控制探索
- old_log_prob 应记录**未缩放**（或统一缩放后）的 logits 概率
- PPO 更新应在统一温度下计算 ratio

### 2.2 `_build_batch_action_mask` 的维度 Bug

**位置**：`trainer/loss_computer.py:356`

```python
heroes = state["action_history"]["heroes"].squeeze(0)  # [T]
```

**问题**：`squeeze(0)` 假设了 history tensor 的 batch 维度为 1。虽然 `_pack_states` 中已将多个 state 拼接成 batch，但单个 state 传入时 shape 是 `[1, T]`。更危险的是，如果 T=1，`squeeze(0)` 会把 `[1, 1]` squeeze 成标量，导致 `for h in heroes` 迭代的是整数而非 tensor，引发逻辑错误。应改为 `heroes = state["action_history"]["heroes"].flatten()`。

### 2.3 Value Loss 中 `old_values` 截断的潜在错位

**位置**：`trainer/loss_computer.py:161` 与 `utils/bp_env.py:289`

- `prepare_rollout` 中：`old_values_valid = all_old_values[valid_mask]`，而 `all_old_values` 来自 `step_values`（长度 T，即有效 step 的 value）
- `compute_value_loss` 中：`old_values_clipped = old_values[:-1] if old_values.shape[0] > returns.shape[0] else old_values`

**问题**：`old_values` 传入的是已经被 valid_mask 过滤后的数据，长度与 returns 一致。`compute_value_loss` 里仍尝试截断最后一个元素（期望传入的是 `values + [bootstrap]`），导致实际截断的是最后一个有效 step 的 old_value，使得 value clipping 的目标错位一个 step。

### 2.4 Bootstrap Value 的阵营偏差

**位置**：`utils/bp_env.py:440-452`

终局 bootstrap value 用 `final_active_agent`（根据 `s.is_radiant_turn` 决定）计算。但此时 BP 已经结束，`is_radiant_turn` 被取反了（`done=True` 时的逻辑）。这导致 bootstrap value 的 team perspective 可能与实际最后一步的决策者不一致。对于 PPO/GAE 来说，bootstrap 应该使用**最后一步实际执行者的 value estimate**，而不是回合结束后翻转的视角。

### 2.5 Oracle 训练数据存在泄漏风险

**位置**：`model/win_rate_oracle.py:496-537`

Oracle 的 `OracleTrainingDataset` 从 `high_mmr_with_stats.json` 中加载玩家 `hero_history`，并构建玩家特征向量。如果 `hero_history` 中的胜率/场次统计包含了**当前这场比赛本身**的数据（或时间窗口内有重叠），则 Oracle 在预测这场比赛结果时，实际上已经"知道"了这些玩家在这场的表现，存在数据泄漏风险。理想情况下，玩家特征应使用比赛时间戳**之前**的历史数据。

### 2.6 非法动作没有显式惩罚

当前仅通过 logits mask 阻止选择已用英雄，但如果模型输出概率质量高度集中在非法动作上（mask 前），梯度信号主要作用于被 mask 掉的位置，合法动作的梯度反而很弱。尤其当模型容量大、训练早期随机性高时，模型可能学会"瞄准"某些热门英雄但无法选择它们，造成策略更新低效。

---

## 三、设计缺陷（Architecture & Methodology）

### 3.1 BP Agent 缺乏英雄属性编码（与 Oracle 能力断层）

**问题**：BP Agent 的 `PlayerEncoder` 只编码"玩家胜率偏好"，完全没有接入英雄的任何属性/语义特征。这意味着：
- Agent 不知道某个英雄是 Carry、Support 还是 Ganker
- Agent 不知道英雄间的协同/克制关系（除非从历史动作中隐式学习）
- Oracle 拥有强大的多模态英雄理解能力，但 Agent 无法利用这些知识做决策

**与调研报告的差距**：报告中的 SOTA 方法（DraftMaster、HeroGNN）都明确将英雄 embedding 作为核心输入。当前 Agent 实际上是在"盲选"——仅基于玩家偏好和历史动作做决策。

### 3.2 缺少位置（Role）分配机制

DOTA 2 阵容的核心约束之一是 1-5 号位分配。当前实现：
- Agent 选出 5 个英雄后直接交给 Oracle
- Oracle 预测胜率时也不显式考虑"谁玩哪个英雄"
- 没有为英雄分配位置的 head 或 loss

**后果**：Agent 可能选出 5 个 Carry 或 5 个 Support，虽然 Oracle 的训练数据可能隐含了位置信息，但 Agent 并未被显式教导要考虑位置平衡。

### 3.3 奖励过于稀疏，无中间奖励塑造（Reward Shaping）

- 仅在终局获得单一标量奖励（Oracle 胜率）
- 22 步决策中只有最后一步有非零奖励
- 没有针对：
  - 成功 Ban 掉对方核心英雄
  - 抢到版本强势英雄
  - 组成合理的技能组合（控制、爆发、推进）
  - 克制对方已选英雄

**与调研报告的差距**：HierarchicalDraft、DraftMaster 等工作都使用了丰富的中间奖励或分层目标函数。

### 3.4 无搜索机制，纯策略梯度（Policy-Only）

**问题**：当前 Agent 每一步直接 softmax 采样，没有任何 MCTS、Minimax 或 lookahead 搜索。

**与调研报告的差距**：
- 调研明确指出 DraftMaster（网易伏羲）使用 **Policy Network + Value Network + MCTS** 达到 >60% 职业选手胜率
- 统计+博弈论方法也使用 Alpha-Beta 剪枝做 3-5 步搜索
- 纯 PPO 在这么大的动作空间（120+）和这么长的 horizon（20 步）下，很容易陷入局部最优

### 3.5 对手建模（Opponent Modeling）缺失

**问题**：Agent 在训练时假设对手要么是自己、要么是历史 checkpoint，但：
- 没有显式建模对手的偏好/策略分布
- 没有 meta-prediction 模块预测对手下一步可能选什么
- 没有针对不同对手类型动态调整策略的能力

**与调研报告的差距**：清华 BP-Agent 工作明确引入了 opponent modeling 和 meta 预测模块。

### 3.6 历史对手比例过低（30%）

配置中 `historical_opponent_prob = 0.3`，这意味着 70% 的时间 Agent 在和自己对弈。对于复杂的非平稳博弈问题：
- 自对弈容易陷入循环策略（cyclic strategies）
- 低比例的历史对手不足以提供策略多样性
- 参考 AlphaStar、OpenAI Five 等系统，通常会维护一个巨大的历史模型池，并从中高频采样对手

### 3.7 缺少层次化决策（Hierarchical RL）

**问题**：Agent 直接输出英雄级别的动作（flat action space），没有高层战术决策。

**与调研报告的差距**：腾讯 AI Lab 的 HierarchicalDraft 将 BP 分为：
- 高层：选择战术类型（poke / 团战 / 速推）
- 低层：在战术约束下选英雄

这种结构不仅提升可解释性，还能显著加速收敛。

---

## 四、改进方案

### 4.1 模型架构改进

#### 4.1.1 将英雄多模态编码接入 BP Agent

**修改文件**：`model/bp_agent.py`

在 `BPTransformerAgent` 中引入 `MultiModalHeroEncoder`，将英雄属性/语义作为可选项加入状态表示：

```python
class BPTransformerAgent(nn.Module):
    def __init__(self, ..., use_hero_encoder=True):
        # 新增：全局英雄知识库 embedding
        self.hero_encoder = MultiModalHeroEncoder(embed_dim=embed_dim, ...)
        # 预计算所有英雄的属性/语义（类似 Oracle）
        self.register_buffer("all_hero_attrs", ...)
        self.register_buffer("all_hero_sem", ...)
```

在 `forward` 中：
1. 用 `hero_encoder` 编码当前**可选英雄池**（或所有英雄）
2. 将动作历史中已选英雄的 embedding 替换为编码后的特征（而非简单的 lookup embedding）
3. 在 Policy Head 前加入一个 cross-attention：当前 query（state representation） attend 到所有可选英雄的 embedding 上

**预期收益**：Agent 能显式利用英雄属性做决策，缩小与 Oracle 之间的知识鸿沟。

#### 4.1.2 增加位置分配 Head（Auxiliary Task）

**修改文件**：`model/bp_agent.py`, `utils/bp_env.py`

在 Agent 的 `forward` 中增加一个辅助输出：

```python
self.position_head = nn.Sequential(
    nn.Linear(embed_dim, 128),
    nn.SiLU(),
    nn.Linear(128, 5)  # 5 个位置的 logits
)
```

在 Pick 动作时，不仅输出选哪个英雄，还输出该英雄分配给哪个位置。环境 `BPState.step()` 可记录位置分配。Loss 中加入辅助监督信号（例如从 Oracle 训练数据中提取每场比赛的英雄-位置对应关系作为 pseudo-label）。

**预期收益**：避免选出 5 Carry 的荒谬阵容，提升阵容合理性。

#### 4.1.3 增加英雄间交互模块（Hero-Hero Cross Attention / GNN）

**修改文件**：`model/bp_agent.py`

在 Transformer Encoder 之后、Policy Head 之前，加入一个轻量级图注意力层：

```python
# 将己方已选英雄和对方已选英雄作为图节点
# 用 cross-attention 计算当前状态下英雄间的协同/克制强度
self.hero_interaction_attn = nn.MultiheadAttention(embed_dim, nhead, batch_first=True)
```

或者引入一个可学习的英雄关系矩阵（类似 HeroGNN 的边权重），在 attention 中作为 bias 加入。

**预期收益**：显式建模英雄关系，避免纯从长序列历史中隐式学习的低效性。

### 4.2 训练算法改进

#### 4.2.1 修复 Temperature Bug 与 Value Loss Bug

**修改文件**：`utils/bp_env.py`, `trainer/loss_computer.py`

1. **Rollout 阶段**：采样使用 temperature，但记录 `old_log_prob` 时应基于 **temperature=1.0** 的 logits（或统一标准）
   ```python
   # 修改前
   probs = F.softmax(action_logits / temp, dim=-1)
   # 修改后
   probs = F.softmax(action_logits / temp, dim=-1)  # 采样仍用 temp
   log_prob_for_storage = F.log_softmax(action_logits, dim=-1).gather(...)  # 存储用 temp=1
   ```

2. **LossComputer**：
   - 删除 `compute_value_loss` 中对 `old_values` 的 `[:-1]` 截断
   - 确保 `prepare_rollout` 传进去的 `old_values` 与 `new_values`、`returns` 三者严格对齐

#### 4.2.2 引入中间奖励塑造（Reward Shaping）

**修改文件**：`utils/bp_env.py`

在 `get_reward` 中不仅返回终局 Oracle 胜率，还可以基于规则计算中间奖励：

```python
def compute_shaped_reward(self, oracle):
    final_reward = self.get_reward(oracle)
    
    # 1. 位置多样性奖励
    position_diversity_bonus = self._compute_role_diversity()
    
    # 2. 协同奖励（基于 OpenDota 胜率矩阵）
    synergy_bonus = self._compute_synergy_bonus()
    
    # 3. 克制奖励
    counter_bonus = self._compute_counter_bonus()
    
    # 4. 终局奖励（主信号）
    return 0.7 * final_reward + 0.1 * position_diversity_bonus + 0.1 * synergy_bonus + 0.1 * counter_bonus
```

**注意**：PPO 的回报计算需使用**塑形后的总奖励**，避免 credit assignment 问题。也可以只在终局 reward 上叠加一个基于规则的中途 penalty（如重复位置惩罚），减少 non-Markovian 风险。

#### 4.2.3 提升历史对手比例 + 多样化对手池

**修改文件**：`configs/bp_agent_config.yaml`

```yaml
training:
  historical_opponent_prob: 0.6  # 从 0.3 提升到 0.6
  policy_staleness_tolerance: 3  # 适当放宽
  num_strata: 5  # 增加分层数
```

同时可考虑维护一个更大的对手池（如最近 50 个 checkpoint 而非默认的几个），并引入基于 TrueSkill 的优先级采样（优先与更强或风格迥异的对手对弈）。

#### 4.2.4 增加对手建模（Opponent Modeling）辅助任务

**修改文件**：`model/bp_agent.py`

在 Agent 中增加一个辅助 head，预测对手下一步选择的英雄：

```python
self.opponent_prediction_head = nn.Linear(embed_dim, NUM_HEROES)
```

在 rollout 收集时，记录对手的实际动作作为 label。Loss 中加入辅助损失（类似自监督）：

```python
opp_loss = F.cross_entropy(opponent_logits, opponent_actual_action)
total_loss = ppo_loss + 0.1 * opp_loss
```

**预期收益**：强迫模型学习对手策略的隐式表示，提升对抗性决策能力。

### 4.3 推理时改进：引入 MCTS / Minimax 搜索

#### 4.3.1 实现轻量级 MCTS（Evaluation 阶段）

**修改文件**：新增 `search/mcts_draft.py`

参考 DraftMaster，在评估/实际使用时引入 MCTS：

```python
class DraftMCTS:
    def __init__(self, policy_net, value_net, oracle, num_simulations=100):
        ...
    
    def search(self, bp_state):
        # 1. Selection: UCB1
        # 2. Expansion: 用 policy_net 的先验概率初始化子节点
        # 3. Simulation: 用 value_net / oracle 快速评估
        # 4. Backpropagation
```

由于动作空间 120+ 较大，可以采用：
- **Policy Pruning**：只扩展 policy net 概率 top-k（如 top-20）的动作
- **Oracle 作为快速 rollout**：模拟到终局直接用 Oracle 打分，无需完整游戏

**预期收益**：推理时突破训练时策略的局限，实现 AlphaGo 风格的搜索增强。

#### 4.3.2 训练时加入 Search-Based Policy Improvement（可选）

如果计算资源允许，可以像 AlphaZero 一样：
- 用 MCTS 的 visit count 作为 improved policy target
- 替代原始 PPO 的 policy gradient，改用 cross-entropy 拟合 MCTS policy

这会彻底改变训练范式，但收益也最大。

### 4.4 与 LLM 结合的可解释性增强（长期方向）

基于调研报告中的 LLM-based Agent 趋势，可设计一个混合架构：

```
┌─────────────────────────────────────┐
│         LLM 解释层（可选）            │
│  "我方需要补充控制，建议选 Lion"      │
├─────────────────────────────────────┤
│         RL Agent (PPO + MCTS)       │
│         实际决策核心                  │
├─────────────────────────────────────┤
│      Oracle + 统计数据层              │
└─────────────────────────────────────┘
```

具体实施：
- 用 LLM 预处理英雄技能描述，生成更丰富的语义 embedding（替代当前简单的 sentence embedding）
- 在评估界面增加 LLM 解说模块，接收 RL Agent 的决策和人类可理解的 board state，生成战术解释

### 4.5 数据与工程改进

#### 4.5.1 修复 Oracle 数据泄漏

**修改文件**：`model/win_rate_oracle.py`

在构建 `hero_history` 特征时，过滤掉时间戳晚于当前比赛的数据。如果 JSON 中缺少时间戳，至少应使用 match_id 做随机 mask（如 10% 场次置零）作为数据清洗兜底。

#### 4.5.2 增加 Invalid Action 的显式惩罚

**修改文件**：`trainer/loss_computer.py`

在 `compute_minibatch` 中加入一个辅助 loss，惩罚模型在未 mask 前对非法动作的高概率：

```python
invalid_probs = batch_probs * invalid_mask.float()  # mask 非法位置为 1，合法为 0
invalid_action_loss = invalid_probs.sum() / batch_size
total_loss = ppo_loss + ... + 0.05 * invalid_action_loss
```

这能帮助模型更快地学会尊重 action mask。

---

## 五、优先级排序与实施建议

| 优先级 | 改进项 | 预期影响 | 实施难度 |
|--------|--------|----------|----------|
| P0 | 修复 Temperature Bug、Value Loss Bug、Mask Bug | 训练稳定性 | 低 |
| P0 | 将 Hero Encoder 接入 BP Agent | 决策质量大幅提升 | 中 |
| P1 | 增加中间奖励塑造 | 收敛速度、策略质量 | 中 |
| P1 | 提升历史对手比例至 50-60% | 避免循环策略 | 低 |
| P1 | 增加位置分配 Auxiliary Head | 阵容合理性 | 中 |
| P2 | 实现轻量级 MCTS（评估阶段） | 推理时突破策略上限 | 高 |
| P2 | 增加对手建模辅助任务 | 对抗能力 | 中 |
| P3 | 引入 LLM 增强可解释性 | 产品化、人机协作 | 高 |

---

## 六、结论

当前仓库实现了一个**工程上较为完整**的 DOTA 2 BP RL 系统，具备现代 RL（PPO+GAE）、两阶段训练、TrueSkill 评估等良好基础。但与技术调研报告中揭示的 SOTA 方法相比，存在三个核心差距：

1. **知识鸿沟**：BP Agent 没有利用英雄属性/语义信息，决策近乎"盲选"
2. **搜索缺失**：纯策略梯度无 MCTS/lookahead，难以在这么大的组合空间中寻优
3. **对手建模薄弱**：缺乏显式的 opponent modeling 和足够多样化的对手池

建议按照上述优先级逐步实施改进，尤其是先修复训练 bug 并接入英雄编码器，这是投入产出比最高的两步。
