"""
Reward computation using Oracle

将Oracle胜率预测作为RL的终局奖励
"""
import torch


def compute_oracle_reward(env, agent, oracle, device):
    """
    使用Oracle计算终局奖励（返回天辉胜率，映射到 [-1, 1]）
    
    Args:
        env: BPEnvironment（已完成BP）
        agent: BPAgent（用于获取最终状态，可选）
        oracle: WinRateOracle
        device: torch设备
    
    Returns:
        float: 奖励值（[-1, 1]，1=天辉必胜，-1=天辉必败）
    """
    radiant_picks, dire_picks = env.get_final_picks()
    r_player_feats, d_player_feats = env.get_player_feats()
    
    if len(radiant_picks) != 5 or len(dire_picks) != 5:
        return 0.0  # 异常结束给予中性奖励
    
    oracle.eval()
    with torch.no_grad():
        # 转换为tensor（1-based hero ids）
        r_picks = torch.tensor([radiant_picks], dtype=torch.long, device=device)
        d_picks = torch.tensor([dire_picks], dtype=torch.long, device=device)
        
        # 获取英雄特征
        r_ids, r_attrs, r_sem = oracle.hero_input_from_ids(r_picks)
        d_ids, d_attrs, d_sem = oracle.hero_input_from_ids(d_picks)
        
        # 处理玩家特征
        r_player = r_player_feats.unsqueeze(0).to(device) if r_player_feats is not None else None
        d_player = d_player_feats.unsqueeze(0).to(device) if d_player_feats is not None else None
        
        # 预测胜率
        win_prob = oracle.forward(r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem, r_player, d_player)
        
        # 线性映射 [0, 1] -> [-1, 1]
        reward = 2.0 * win_prob.item() - 1.0
        
    return reward


def create_oracle_reward_fn(oracle, device):
    """
    创建Oracle奖励函数（适配TrajectoryCollector接口）
    
    Args:
        oracle: WinRateOracle
        device: torch设备
    
    Returns:
        function: 奖励函数 fn(env, agent, device) -> float
    """
    def reward_fn(env, agent, dev):
        return compute_oracle_reward(env, agent, oracle, dev)
    return reward_fn
