import os
import re
import json
import torch
import random
import pathlib

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP, NUM_HEROES
from utils.get_data_cm_bp import fetch_high_mmr_matches
from model.win_rate_oracle import *

torch.random.manual_seed(42)

WIN_RATE_ORACLE_SAVE_DIR = "./ckpts/win_rate_oracle"
if not os.path.exists(WIN_RATE_ORACLE_SAVE_DIR):
    pathlib.Path(WIN_RATE_ORACLE_SAVE_DIR).mkdir(parents=True, exist_ok=True)


class DOTAMatchDataset(Dataset):
    """适配 high_mmr_with_stats.json 的数据集"""
    
    @staticmethod
    def get_teams_hero_ids(match: dict):
        """从 picks_bans 中提取两队的pick英雄ID"""
        team_hero_ids = {'radiant': [], 'dire': []}
        for act in match.get('picks_bans', []):
            if act.get('is_pick', False):
                team = 'radiant' if act.get('team', 0) == 0 else 'dire'
                team_hero_ids[team].append(act['hero_id'])
        return team_hero_ids['radiant'], team_hero_ids['dire']
    
    @staticmethod
    def split_players_by_team(players):
        """根据 player_slot 将玩家分为天辉和夜魇两队
        player_slot: 0-4 = 天辉, 128-132 = 夜魇
        """
        radiant = []
        dire = []
        for p in players:
            slot = p.get('player_slot', 0)
            if slot < 128:
                radiant.append(p)
            else:
                dire.append(p)
        return radiant, dire
    
    @staticmethod
    def build_player_feature_vector(
        players, 
        num_heroes=NUM_HEROES, 
        max_heroes_per_player=10, 
        min_games=3,
        min_total_games=10
    ):
        """
        从玩家 hero_history 构建胜率特征向量 [5, num_heroes]
        hero_history 格式: {hero_id_str: {games: int, wins: int}}
        
        Returns:
            list[list[float]]: 5个玩家的胜率向量，无效玩家用全0向量表示
        """
        vectors = []
        for player in players[:5]:  # 最多取5个玩家
            hero_history = player.get('hero_history', {})
            
            # 计算总场次
            total_games = sum(
                h.get('games', 0) for h in hero_history.values()
            )
            
            # 总场次不足，使用全0向量（表示匿名或数据不足的玩家）
            if total_games < min_total_games:
                vectors.append([0.0] * num_heroes)
                continue
            
            # 构建胜率向量：按场次排序，取前max_heroes_per_player个
            hero_list = []
            for hero_id_str, stats in hero_history.items():
                try:
                    hero_id = int(hero_id_str)
                    games = stats.get('games', 0)
                    wins = stats.get('wins', 0)
                    winrate = wins / games if games > 0 else 0
                    hero_list.append((hero_id, games, winrate))
                except (ValueError, TypeError):
                    continue
            
            # 按场次排序，取前N个
            hero_list.sort(key=lambda x: x[1], reverse=True)
            
            vector = [0.0] * num_heroes
            for hero_id, games, winrate in hero_list[:max_heroes_per_player]:
                # 只记录有足够场次的英雄胜率，且hero_id在有效范围内
                if 0 < hero_id < num_heroes and games >= min_games:
                    vector[hero_id] = winrate
            
            vectors.append(vector)
        
        # 补全到5个玩家
        while len(vectors) < 5:
            vectors.append([0.0] * num_heroes)
        return vectors
    
    def __init__(self, json_path, HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP, 
                 min_total_games=10, use_data_augmentation=True):
        """
        Args:
            json_path: high_mmr_with_stats.json 路径
            HERO_ID_FEATURE_MAP: 英雄属性映射
            HERO_ID_SEMANTIC_MAP: 英雄语义嵌入映射
            min_total_games: 玩家最少总场次阈值
            use_data_augmentation: 是否使用双向数据增强
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.matches = []
        skipped_count = 0
        invalid_team_count = 0
        
        for m in raw_data:
            # 提取pick英雄
            r_ids, d_ids = self.get_teams_hero_ids(m)
            if len(r_ids) != 5 or len(d_ids) != 5:
                invalid_team_count += 1
                continue
            
            # 根据 player_slot 分队
            players = m.get('players', [])
            radiant_players, dire_players = self.split_players_by_team(players)
            
            # 构建玩家特征向量
            r_player_feats = self.build_player_feature_vector(
                radiant_players, min_total_games=min_total_games
            )
            d_player_feats = self.build_player_feature_vector(
                dire_players, min_total_games=min_total_games
            )
            
            # 检查是否有足够数量的有效玩家（可选：跳过玩家数据不足的比赛）
            # 这里改为保留所有比赛，玩家数据不足时用全0向量表示
            has_valid_r_players = any(sum(1 for v in vec if v > 0) > 0 for vec in r_player_feats)
            has_valid_d_players = any(sum(1 for v in vec if v > 0) > 0 for vec in d_player_feats)
            
            if not (has_valid_r_players and has_valid_d_players):
                skipped_count += 1
                # 继续保留这场比赛，只是玩家特征为0
            
            label = 1.0 if m.get('radiant_win', False) else 0.0
            
            # 添加正向样本
            self.matches.append({
                'r_ids': r_ids,
                'd_ids': d_ids,
                'r_player_feats': r_player_feats,
                'd_player_feats': d_player_feats,
                'has_valid_player_data': has_valid_r_players and has_valid_d_players,
                'label': label
            })
            
            # 双向数据增强（交换两队）
            if use_data_augmentation:
                self.matches.append({
                    'r_ids': d_ids,
                    'd_ids': r_ids,
                    'r_player_feats': d_player_feats,
                    'd_player_feats': r_player_feats,
                    'has_valid_player_data': has_valid_r_players and has_valid_d_players,
                    'label': 1.0 - label
                })
        
        if invalid_team_count > 0:
            print(f"[*] 跳过了 {invalid_team_count} 场队伍不完整(pick≠5)的比赛")
        if skipped_count > 0:
            print(f"[*] 警告: {skipped_count} 场比赛玩家数据不足（将使用零向量）")
        
        # 统计有有效玩家数据的比赛比例
        has_data_count = sum(1 for m in self.matches if m['has_valid_player_data'])
        print(f"[*] 数据集大小: {len(self.matches)} {'(双向增强后)' if use_data_augmentation else ''}")
        print(f"[*] 有有效玩家数据的比赛: {has_data_count}/{len(self.matches)} ({100*has_data_count/len(self.matches):.1f}%)")
        
        self.HERO_ID_FEATURE_MAP = HERO_ID_FEATURE_MAP
        self.HERO_ID_SEMANTIC_MAP = HERO_ID_SEMANTIC_MAP

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        
        def format_input(hero_ids):
            ids_tensor = torch.tensor(hero_ids, dtype=torch.long)
            attrs_tensor = torch.stack([self.HERO_ID_FEATURE_MAP[hid] for hid in hero_ids])
            sem_tensor = torch.stack([self.HERO_ID_SEMANTIC_MAP[hid] for hid in hero_ids])
            return ids_tensor, attrs_tensor, sem_tensor

        r_ids = match['r_ids']
        d_ids = match['d_ids']
        random.shuffle(r_ids)
        random.shuffle(d_ids)
        r_inputs = format_input(r_ids)
        d_inputs = format_input(d_ids)
        
        # 玩家特征 [5, NUM_HEROES]
        r_player_feats = torch.tensor(match['r_player_feats'], dtype=torch.float32)
        d_player_feats = torch.tensor(match['d_player_feats'], dtype=torch.float32)
        
        label = torch.tensor([match['label']], dtype=torch.float32)
        
        return (*r_inputs, *d_inputs, r_player_feats, d_player_feats, label)
    
def train(load_model_path: str = None, epochs: int = 32):
    # 1. 初始化 TensorBoard Writer
    # 日志会保存在 ./runs 目录下，按时间戳区分实验
    log_dir = os.path.join("runs", "win_rate_exp_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[+] TensorBoard 日志将保存至: {log_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DOTAMatchDataset(
        "./data/high_mmr_with_stats.json",  # 使用新的合并数据文件
        HERO_ID_FEATURE_MAP, 
        HERO_ID_SEMANTIC_MAP,
        min_total_games=10,  # 玩家最少总场次阈值
        use_data_augmentation=True
    )
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = WinRateOracle(
        embed_dim=64, 
        nhead=4, 
        num_layers=4, 
        use_text=False, 
        use_player_heroes=True,
        # HeroEncoder 参数
        hero_encoder_id_dim=128,
        hero_encoder_attr_dim=64,
        hero_encoder_text_dim=128,
        hero_encoder_dropout=0.1,
        hero_encoder_res_layers=3,
        hero_encoder_attn_heads=4,
        hero_encoder_modality_dropout=0.1,
    ).to(device)
    if load_model_path is not None and os.path.exists(load_model_path):
        print(f"[+] 加载预训练模型 ...")
        if found := re.findall(r'win_rate_oracle-(\d+)-(\d+)-(.+).pth$', load_model_path):
            acc = float(found[0][-1])
            print(f"[+] 初始准确率: {acc}")
        model.load_state_dict(torch.load(load_model_path))
    else:
        print(f"[+] 未加载预训练模型 ...")
        acc = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCELoss()

    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    global_step = 0
    print(f"[+] 训练开始 ...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=160)
        
        for batch_idx, (r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem, r_player_feats, d_player_feats, labels) in enumerate(pbar):
            r_ids, r_attrs, r_sem = r_ids.to(device), r_attrs.to(device), r_sem.to(device)
            d_ids, d_attrs, d_sem = d_ids.to(device), d_attrs.to(device), d_sem.to(device)
            r_player_feats = r_player_feats.to(device)
            d_player_feats = d_player_feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem, r_player_feats, d_player_feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            # 计算统计数据
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total_samples += labels.size(0)
            
            # 2. 记录每个 Batch 的数据到 TensorBoard
            writer.add_scalar('Batch/Loss', loss.item(), global_step)
            writer.add_scalar('Batch/Accuracy', batch_correct / labels.size(0), global_step)
            # writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], global_step)
            
            global_step += 1
            accuracy = correct/total_samples
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{accuracy:.4f}"})

        # 3. 记录每个 Epoch 的平均数据
        avg_loss = total_loss / len(train_loader)
        avg_acc = correct / total_samples
        writer.add_scalar('Epoch/Loss', avg_loss, epoch)
        writer.add_scalar('Epoch/Accuracy', avg_acc, epoch)

        # 保存模型
        if avg_acc > acc:
            acc = avg_acc
            epoch_str = str(epoch).rjust(len(str(epochs)), '0')
            torch.save(model.state_dict(), os.path.join(WIN_RATE_ORACLE_SAVE_DIR, f"win_rate_oracle-{datetime_str}-{epoch_str}-{avg_acc:.4f}.pth"))

    # 4. 训练结束关闭 writer
    writer.close()

if __name__ == "__main__":
    print('='*20 + ' 更新比赛数据 ' + '='*20)
    fetch_high_mmr_matches(
        output_file='./data/high_mmr_with_stats.json',  # 使用合并后的数据文件
        target_count=100000,
        min_rank=50,
        min_duration=18 * 60,
    )

    print('='*20 + ' 训练 WinRateOracle ' + '='*20)
    train(
        # load_model_path=os.path.join(WIN_RATE_ORACLE_SAVE_DIR, 'win_rate_oracle-20251225233207-050-0.8595.pth'),
        epochs=128,
    )