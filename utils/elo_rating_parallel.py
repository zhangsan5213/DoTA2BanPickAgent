"""
ELO Rating System - High-Performance Parallel Evaluation

使用 ProcessPoolExecutor 实现真正的多进程并行，充分利用多核 CPU 和 GPU 显存
"""
import os
import random
import json
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import torch


# 全局变量，在子进程中初始化
g_agent_class = None
g_config = None
g_device = None
g_matches_data = None
g_player_sampler = None


def init_worker(agent_class, config, device_str, matches_data, player_sampler):
    """
    子进程初始化函数
    每个子进程独立加载模型到 GPU/CPU
    """
    global g_agent_class, g_config, g_device, g_matches_data, g_player_sampler
    
    g_agent_class = agent_class
    g_config = config
    g_device = torch.device(device_str)
    g_matches_data = matches_data
    g_player_sampler = player_sampler
    
    # 设置 CUDA 可见设备（如果多 GPU，可以分配不同 GPU 给不同进程）
    if torch.cuda.is_available() and device_str.startswith('cuda'):
        # 提取设备索引，如 'cuda:0' -> 0, 'cuda' -> 0
        if ':' in device_str:
            device_idx = int(device_str.split(':')[1])
        else:
            device_idx = 0
        torch.cuda.set_device(device_idx)


def run_matches_for_pair(args):
    """
    在子进程中执行一对 checkpoint 的多局对战
    
    Args:
        args: (ckpt_a_path, ckpt_b_path, n_games, oracle_ckpt_path)
    
    Returns:
        (ckpt_a_path, ckpt_b_path, wins_a, n_games)
    """
    global g_agent_class, g_config, g_device, g_matches_data, g_player_sampler
    
    ckpt_a_path, ckpt_b_path, n_games, oracle_ckpt_path = args
    device = g_device
    
    # 延迟导入避免循环依赖
    from env.bp_env import BPEnvironment
    from utils.elo_rating import run_single_match
    
    # 加载 Agent A
    agent_a = g_agent_class(
        embed_dim=g_config.EMBED_DIM,
        nhead=g_config.NHEAD,
        num_layers=g_config.NUM_LAYERS,
        use_text=g_config.USE_TEXT,
        use_player_heroes=g_config.USE_PLAYER_HEROES,
    ).to(device)
    agent_a.load_state_dict(torch.load(ckpt_a_path, map_location=device))
    agent_a.eval()
    
    # 加载 Agent B
    agent_b = g_agent_class(
        embed_dim=g_config.EMBED_DIM,
        nhead=g_config.NHEAD,
        num_layers=g_config.NUM_LAYERS,
        use_text=g_config.USE_TEXT,
        use_player_heroes=g_config.USE_PLAYER_HEROES,
    ).to(device)
    agent_b.load_state_dict(torch.load(ckpt_b_path, map_location=device))
    agent_b.eval()
    
    # 加载 Oracle
    from model.win_rate_oracle import WinRateOracle
    oracle = WinRateOracle(
        embed_dim=g_config.EMBED_DIM,
        nhead=g_config.NHEAD,
        num_layers=g_config.NUM_LAYERS,
        use_text=g_config.USE_TEXT,
        use_player_heroes=g_config.USE_PLAYER_HEROES,
    ).to(device)
    oracle.load_state_dict(torch.load(oracle_ckpt_path, map_location=device))
    oracle.eval()
    
    # 执行 n_games 局对战
    wins_a = 0
    for _ in range(n_games):
        score_a = run_single_match(
            agent_a, agent_b, oracle,
            g_matches_data, g_player_sampler, device, g_config
        )
        wins_a += score_a
    
    # 清理显存
    del agent_a, agent_b, oracle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return ckpt_a_path, ckpt_b_path, wins_a, n_games


def evaluate_checkpoints_elo_parallel(
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
    n_workers: int = 8,  # 默认增加到 8 进程
    verbose: bool = True
) -> Dict[str, float]:
    """
    高性能并行版本：对一组新checkpoint进行ELO定分
    
    使用 ProcessPoolExecutor 实现真正的多进程并行，每个进程独立运行多局对战
    
    Args:
        new_checkpoints: 需要定分的新ckpt路径列表
        all_checkpoints: 所有可用的ckpt（包括新旧）
        elo_ratings: ELO分数字典（会被修改）
        agent_class: Agent类（如BPAgent）
        oracle: WinRateOracle
        matches_data: 比赛数据
        player_sampler: 玩家采样器
        device: torch设备（主进程）
        config: 配置对象
        n_opponents_per_ckpt: 每个新ckpt要交战的对手数量
        n_games_per_match: 每对组合的对战局数
        n_workers: 并行工作进程数（默认8，可根据CPU核心数调整）
        verbose: 是否打印进度
    
    Returns:
        更新后的elo_ratings
    """
    if not new_checkpoints or len(all_checkpoints) < 2:
        return elo_ratings
    
    # 保存 oracle 的临时路径
    oracle_ckpt_path = os.path.join(config.SAVE_DIR, ".temp_oracle_for_eval.pth")
    torch.save(oracle.state_dict(), oracle_ckpt_path)
    
    # 构建对战调度表（按对战组合聚合，减少进程切换开销）
    # 每对组合只需提交一次，在子进程中跑完 n_games_per_match 局
    match_pairs = []  # [(ckpt_a, ckpt_b, n_games), ...]
    
    for new_ckpt in new_checkpoints:
        candidates = [c for c in all_checkpoints if c != new_ckpt]
        if not candidates:
            continue
        
        n_select = min(n_opponents_per_ckpt, len(candidates))
        other_new = [c for c in candidates if c in new_checkpoints]
        historical = [c for c in candidates if c not in new_checkpoints]
        
        selected = []
        if other_new:
            selected.extend(random.sample(other_new, min(len(other_new), n_select)))
        if len(selected) < n_select and historical:
            need = n_select - len(selected)
            historical_by_elo = sorted(historical, key=lambda c: elo_ratings.get(c, 1500), reverse=True)
            selected.extend(historical_by_elo[:need])
        
        for opp in selected:
            match_pairs.append((new_ckpt, opp, n_games_per_match))
    
    if not match_pairs:
        return elo_ratings
    
    total_pairs = len(match_pairs)
    total_games = total_pairs * n_games_per_match
    
    if verbose:
        print(f"[*] ELO多进程定分: {len(new_checkpoints)} 个新ckpt, {n_workers} 进程")
        print(f"[*] 对战组合: {total_pairs}, 总局数: {total_games}")
    
    # 准备参数列表
    args_list = [
        (ckpt_a, ckpt_b, n_games, oracle_ckpt_path)
        for ckpt_a, ckpt_b, n_games in match_pairs
    ]
    
    # 多进程并行执行 - 确保设备字符串包含索引
    if device.type == 'cuda':
        device_idx = device.index if device.index is not None else 0
        device_str = f'cuda:{device_idx}'
    else:
        device_str = 'cpu'
    results_by_pair = {}  # {(ckpt_a, ckpt_b): (wins, total_games)}
    completed = 0
    
    # Windows 必须使用 spawn，且需要保护好入口点
    ctx = mp.get_context('spawn')
    
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(agent_class, config, device_str, matches_data, player_sampler)
    ) as executor:
        
        futures = {executor.submit(run_matches_for_pair, args): args for args in args_list}
        
        for future in as_completed(futures):
            try:
                ckpt_a, ckpt_b, wins_a, n_games = future.result()
                results_by_pair[(ckpt_a, ckpt_b)] = (wins_a, n_games)
                
                completed += 1
                if verbose and completed % max(1, total_pairs // 100) == 0:
                    print(f"[*] 进度: {completed}/{total_pairs} ({completed/total_pairs*100:.1f}%)")
                    
            except Exception as e:
                print(f"[!] 对局执行出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 更新 ELO 分数
    for (ckpt_a, ckpt_b), (wins_a, n_games) in results_by_pair.items():
        avg_score = wins_a / n_games
        rating_a, rating_b = update_elo(
            elo_ratings[ckpt_a], elo_ratings[ckpt_b], avg_score, k=32
        )
        elo_ratings[ckpt_a] = rating_a
        elo_ratings[ckpt_b] = rating_b
    
    if verbose:
        print(f"[*] ELO定分完成")
    
    return elo_ratings


def evaluate_single_checkpoint_elo_parallel(
    ckpt_path: str,
    elo_ratings: Dict[str, float],
    agent_class,
    oracle,
    matches_data,
    player_sampler,
    device,
    config,
    n_opponents: int = 8,
    n_games: int = 4,
    n_workers: int = 8,
    verbose: bool = True
) -> Tuple[float, Dict[str, float]]:
    """
    并行版本：对单个checkpoint进行ELO定分
    
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
        n_workers: 并行工作进程数
        verbose: 是否打印进度
    
    Returns:
        (新ckpt的ELO分数, 更新后的elo_ratings)
    """
    historical = [c for c in elo_ratings.keys() if c != ckpt_path]
    if not historical:
        return elo_ratings.get(ckpt_path, 1500.0), elo_ratings
    
    n_opp = min(n_opponents, len(historical))
    opponents = random.sample(historical, n_opp)
    
    elo_ratings = evaluate_checkpoints_elo_parallel(
        new_checkpoints=[ckpt_path],
        all_checkpoints=[ckpt_path] + opponents,
        elo_ratings=elo_ratings,
        agent_class=agent_class,
        oracle=oracle,
        matches_data=matches_data,
        player_sampler=player_sampler,
        device=device,
        config=config,
        n_opponents_per_ckpt=n_opponents,
        n_games_per_match=n_games,
        n_workers=n_workers,
        verbose=verbose
    )
    
    return elo_ratings[ckpt_path], elo_ratings


def update_elo(rating_a: float, rating_b: float, score_a: float, k: int = 32) -> Tuple[float, float]:
    """
    更新ELO分数（从原模块复制，避免循环导入）
    """
    expected_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))
    expected_b = 1.0 - expected_a
    
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * ((1.0 - score_a) - expected_b)
    
    return new_rating_a, new_rating_b
