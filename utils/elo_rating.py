"""
ELO Rating System for BP Agent

用于Self-Play训练中的模型强度评估
"""
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Callable
import torch


def compute_expected_score(rating_a: float, rating_b: float) -> float:
    """计算A对B的期望胜率"""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a: float, rating_b: float, score_a: float, k: int = 32) -> Tuple[float, float]:
    """
    更新ELO分数
    
    Args:
        rating_a: A当前分数
        rating_b: B当前分数
        score_a: A的实际得分（1=胜, 0.5=平, 0=负）
        k: K因子
    
    Returns:
        new_rating_a, new_rating_b
    """
    expected_a = compute_expected_score(rating_a, rating_b)
    expected_b = 1.0 - expected_a
    
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1.0 - score_a) - expected_b)
    
    return new_rating_a, new_rating_b


def load_elo_ratings(json_path: str) -> Tuple[Dict[str, float], str]:
    """
    加载ELO记录
    
    Returns:
        ratings: {ckpt_path: elo_score}
        last_updated: ISO格式时间字符串
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            return data.get('ratings', {}), data.get('last_updated', '')
    return {}, ''


def save_elo_ratings(ratings: Dict[str, float], json_path: str):
    """保存ELO记录到JSON"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({
            'ratings': ratings,
            'last_updated': datetime.now().isoformat()
        }, f, indent=2)


def get_best_checkpoint(elo_ratings: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """
    获取ELO分数最高的checkpoint
    
    Returns:
        (ckpt_path, elo_score) 或 None
    """
    if not elo_ratings:
        return None
    
    # 过滤存在的文件
    valid_ratings = {k: v for k, v in elo_ratings.items() if os.path.exists(k)}
    if not valid_ratings:
        return None
    
    best_ckpt = max(valid_ratings.keys(), key=lambda x: valid_ratings[x])
    return best_ckpt, valid_ratings[best_ckpt]


def find_new_checkpoints(elo_ratings: Dict[str, float], save_dir: str, prefix: str = 'bp_agent-') -> List[str]:
    """
    找出没有ELO记录的新checkpoint
    
    Args:
        elo_ratings: 已有ELO记录
        save_dir: checkpoint保存目录
        prefix: 文件名前缀
    
    Returns:
        新checkpoint路径列表
    """
    new_ckpts = []
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            if f.endswith('.pth') and f.startswith(prefix):
                full_path = os.path.join(save_dir, f)
                if full_path not in elo_ratings:
                    new_ckpts.append(full_path)
    return sorted(new_ckpts)  # 按文件名排序，确保确定性


def select_elo_opponents(all_checkpoints: List[str], new_checkpoints: List[str], 
                         n_opponents: int) -> List[str]:
    """
    选择ELO定分的对手
    
    策略：优先选择新ckpt，不足时从旧ckpt随机补充
    
    Args:
        all_checkpoints: 所有有ELO记录的ckpt
        new_checkpoints: 新ckpt（无ELO记录）
        n_opponents: 需要的对手数量
    
    Returns:
        选中的ckpt路径列表
    """
    if len(all_checkpoints) <= n_opponents:
        return all_checkpoints
    
    # 优先选新ckpt
    selected = list(new_checkpoints)
    
    # 从旧ckpt中补充
    old_ckpts = [c for c in all_checkpoints if c not in new_checkpoints]
    remaining = n_opponents - len(selected)
    
    if remaining > 0 and old_ckpts:
        selected.extend(random.sample(old_ckpts, min(remaining, len(old_ckpts))))
    
    return selected


def run_single_match(agent_a, agent_b, oracle, matches_data, player_sampler, device, config):
    """
    运行单场比赛，返回agent_a的得分（1=胜, 0=负）
    天辉夜魇随机分配
    
    Args:
        agent_a: 第一个agent
        agent_b: 第二个agent
        oracle: WinRateOracle
        matches_data: 比赛数据
        player_sampler: 玩家采样器
        device: torch设备
        config: 配置对象（需包含USE_PLAYER_HEROES）
    
    Returns:
        float: agent_a的得分（1=胜, 0=负, 0.5=平局）
    """
    # 延迟导入避免循环依赖
    from env.bp_env import BPEnvironment
    
    # 随机分配
    if random.random() < 0.5:
        radiant_agent, dire_agent = agent_a, agent_b
        a_is_radiant = True
    else:
        radiant_agent, dire_agent = agent_b, agent_a
        a_is_radiant = False
    
    # 创建环境
    env = BPEnvironment(
        matches_data,
        player_data_enabled=config.USE_PLAYER_HEROES,
        player_sampler=player_sampler,
        use_sampled_players=True
    )
    state = env.reset()
    done = False
    
    # 运行BP
    while not done:
        current_step = env.current_step
        current_team, _ = env.action_sequence[current_step]
        
        active_agent = radiant_agent if current_team == 0 else dire_agent
        
        with torch.no_grad():
            state_feat = active_agent.encode_state(
                hero_ids=state['hero_ids'].to(device),
                team_flags=state['team_flags'].to(device),
                action_types=state['action_types'].to(device),
                valid_mask=state['valid_mask'].to(device),
                radiant_player_feats=state['radiant_player_feats'].to(device) if state['radiant_player_feats'] is not None else None,
                dire_player_feats=state['dire_player_feats'].to(device) if state['dire_player_feats'] is not None else None,
            )
            
            valid_heroes = env.get_valid_actions()
            if len(valid_heroes) == 0:
                break
            
            K = min(32, len(valid_heroes))
            candidate_ids = random.sample(valid_heroes, K) if len(valid_heroes) >= K else valid_heroes + [0] * (K - len(valid_heroes))
            while len(candidate_ids) < 32:
                candidate_ids.append(0)
            candidate_ids = torch.tensor([candidate_ids], dtype=torch.long).to(device)
            
            action, _, _ = active_agent.get_action(
                state_feat=state_feat,
                candidate_hero_ids=candidate_ids,
                deterministic=True,
            )
            actual_action = candidate_ids[0, action[0].item()].item()
        
        state, _, done = env.step(actual_action)
    
    # Oracle判定胜负
    radiant_picks, dire_picks = env.get_final_picks()
    r_player_feats, d_player_feats = env.get_player_feats()
    
    if len(radiant_picks) != 5 or len(dire_picks) != 5:
        return 0.5  # 平局
    
    oracle.eval()
    with torch.no_grad():
        r_picks = torch.tensor([radiant_picks], dtype=torch.long, device=device)
        d_picks = torch.tensor([dire_picks], dtype=torch.long, device=device)
        r_ids, r_attrs, r_sem = oracle.hero_input_from_ids(r_picks)
        d_ids, d_attrs, d_sem = oracle.hero_input_from_ids(d_picks)
        
        r_player = r_player_feats.unsqueeze(0).to(device) if r_player_feats is not None else None
        d_player = d_player_feats.unsqueeze(0).to(device) if d_player_feats is not None else None
        
        radiant_win_prob = oracle.forward(r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem, r_player, d_player)
        radiant_win = radiant_win_prob.item() > 0.5
    
    # 返回agent_a的得分
    if a_is_radiant:
        return 1.0 if radiant_win else 0.0
    else:
        return 0.0 if radiant_win else 1.0


def evaluate_checkpoints_elo(
    new_checkpoints: List[str],
    all_checkpoints: List[str],
    elo_ratings: Dict[str, float],
    agent_class,
    oracle,
    matches_data,
    player_sampler,
    device,
    config,
    n_opponents_per_ckpt: int = 5,
    n_games_per_match: int = 10,
    verbose: bool = True
) -> Dict[str, float]:
    """
    对一组新checkpoint进行ELO定分
    
    每个新ckpt与多个对手（新ckpt + 历史ckpt）各打多局比赛，
    获得更准确的ELO评分
    
    Args:
        new_checkpoints: 需要定分的新ckpt路径列表
        all_checkpoints: 所有可用的ckpt（包括新旧）
        elo_ratings: ELO分数字典（会被修改）
        agent_class: Agent类（如BPAgent）
        oracle: WinRateOracle
        matches_data: 比赛数据
        player_sampler: 玩家采样器
        device: torch设备
        config: 配置对象
        n_opponents_per_ckpt: 每个新ckpt要交战的对手数量
        n_games_per_match: 每对组合的对战局数
        verbose: 是否打印进度
    
    Returns:
        更新后的elo_ratings
    """
    if not new_checkpoints or len(all_checkpoints) < 2:
        return elo_ratings
    
    # 收集所有需要加载的ckpt（新ckpt + 被选中的对手）
    ckpts_to_load = set(new_checkpoints)
    match_schedule = []  # [(new_ckpt, opponent_ckpt), ...]
    
    for new_ckpt in new_checkpoints:
        # 可选对手（不包括自己）
        candidates = [c for c in all_checkpoints if c != new_ckpt]
        if not candidates:
            continue
        
        # 选择对手：优先选其他新ckpt，不足时从历史ckpt补充
        n_select = min(n_opponents_per_ckpt, len(candidates))
        
        # 分离新ckpt和历史ckpt
        other_new = [c for c in candidates if c in new_checkpoints]
        historical = [c for c in candidates if c not in new_checkpoints]
        
        selected = []
        # 优先选其他新ckpt
        if other_new:
            selected.extend(random.sample(other_new, min(len(other_new), n_select)))
        # 从历史ckpt补充
        if len(selected) < n_select and historical:
            need = n_select - len(selected)
            # 优先选ELO分数高的历史ckpt（更强的对手更有信息量）
            historical_by_elo = sorted(historical, key=lambda c: elo_ratings.get(c, 1500), reverse=True)
            selected.extend(historical_by_elo[:need])
        
        for opp in selected:
            match_schedule.append((new_ckpt, opp))
    
    if not match_schedule:
        return elo_ratings
    
    # 收集所有需要加载的ckpt
    for new_ckpt, opp_ckpt in match_schedule:
        ckpts_to_load.add(opp_ckpt)
    
    # 加载所有需要的agent
    agents = {}
    for ckpt in ckpts_to_load:
        agent = agent_class(
            embed_dim=config.EMBED_DIM,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            use_text=config.USE_TEXT,
            use_player_heroes=config.USE_PLAYER_HEROES,
        ).to(device)
        agent.load_state_dict(torch.load(ckpt, map_location=device))
        agent.eval()
        agents[ckpt] = agent
    
    total_matches = len(match_schedule)
    if verbose:
        print(f"[*] ELO定分: {len(new_checkpoints)} 个新ckpt vs {len(ckpts_to_load)-len(new_checkpoints)} 个对手")
        print(f"[*] 总对战组合: {total_matches}, 每组合{n_games_per_match}局, 总局数: {total_matches * n_games_per_match}")
    
    # 执行对战
    for idx, (ckpt_a, ckpt_b) in enumerate(match_schedule):
        if verbose and (idx + 1) % 5 == 0:
            print(f"[*] 进度: {idx+1}/{total_matches}")
        
        # 对战n_games_per_match局
        wins_a = 0
        for _ in range(n_games_per_match):
            score_a = run_single_match(
                agents[ckpt_a], agents[ckpt_b], oracle,
                matches_data, player_sampler, device, config
            )
            wins_a += score_a
        
        avg_score = wins_a / n_games_per_match
        
        # 更新ELO（更新双方的评分）
        rating_a, rating_b = update_elo(
            elo_ratings[ckpt_a], elo_ratings[ckpt_b], avg_score, k=32
        )
        elo_ratings[ckpt_a] = rating_a
        elo_ratings[ckpt_b] = rating_b
    
    if verbose:
        print(f"[*] ELO定分完成")
    
    # 清理内存
    for agent in agents.values():
        del agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return elo_ratings


def evaluate_single_checkpoint_elo(
    ckpt_path: str,
    elo_ratings: Dict[str, float],
    agent_class,
    oracle,
    matches_data,
    player_sampler,
    device,
    config,
    n_opponents: int = 8,
    n_games: int = 4
) -> Tuple[float, Dict[str, float]]:
    """
    对单个checkpoint进行ELO定分
    
    与随机选择的历史对手对战
    
    Args:
        ckpt_path: 新ckpt路径
        elo_ratings: ELO分数字典（会被修改）
        agent_class: Agent类
        oracle: WinRateOracle
        matches_data: 比赛数据
        player_sampler: 玩家采样器
        device: torch设备
        config: 配置对象
        n_opponents: 对手数量
        n_games: 每对手对战局数
    
    Returns:
        (新ckpt的ELO分数, 更新后的elo_ratings)
    """
    # 选择对手（不包括自己）
    historical = [c for c in elo_ratings.keys() if c != ckpt_path]
    if not historical:
        return elo_ratings.get(ckpt_path, 1500.0), elo_ratings
    
    n_opp = min(n_opponents, len(historical))
    opponents = random.sample(historical, n_opp)
    
    # 加载当前agent
    current_agent = agent_class(
        embed_dim=config.EMBED_DIM,
        nhead=config.NHEAD,
        num_layers=config.NUM_LAYERS,
        use_text=config.USE_TEXT,
        use_player_heroes=config.USE_PLAYER_HEROES,
    ).to(device)
    current_agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    current_agent.eval()
    
    # 与每个对手对战
    for opp_ckpt in opponents:
        opp_agent = agent_class(
            embed_dim=config.EMBED_DIM,
            nhead=config.NHEAD,
            num_layers=config.NUM_LAYERS,
            use_text=config.USE_TEXT,
            use_player_heroes=config.USE_PLAYER_HEROES,
        ).to(device)
        opp_agent.load_state_dict(torch.load(opp_ckpt, map_location=device))
        opp_agent.eval()
        
        wins = 0
        for _ in range(n_games):
            score = run_single_match(
                current_agent, opp_agent, oracle,
                matches_data, player_sampler, device, config
            )
            wins += score
        
        avg_score = wins / n_games
        
        # 更新ELO
        current_rating = elo_ratings[ckpt_path]
        opp_rating = elo_ratings[opp_ckpt]
        new_rating, new_opp_rating = update_elo(current_rating, opp_rating, avg_score, k=32)
        elo_ratings[ckpt_path] = new_rating
        elo_ratings[opp_ckpt] = new_opp_rating
        
        del opp_agent
    
    del current_agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return elo_ratings[ckpt_path], elo_ratings
