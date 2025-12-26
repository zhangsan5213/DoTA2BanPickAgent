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

from utils.raw_data import HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP
from utils.get_data_cm_bp import fetch_high_mmr_matches
from model.win_rate_oracle import *

torch.random.manual_seed(42)

WIN_RATE_ORACLE_SAVE_DIR = "./ckpts/win_rate_oracle"
if not os.path.exists(WIN_RATE_ORACLE_SAVE_DIR):
    pathlib.Path(WIN_RATE_ORACLE_SAVE_DIR).mkdir(parents=True, exist_ok=True)

class DOTAMatchDataset(Dataset):
    @staticmethod
    def get_teams_hero_ids(match: dict):
        team_hero_ids = dict()
        for act in match['picks_bans']:
            if act['is_pick']:
                team = 'radiant' if act['team'] == 0 else 'dire'
                if team not in team_hero_ids:
                    team_hero_ids[team] = []
                team_hero_ids[team].append(act['hero_id'])
        return team_hero_ids['radiant'], team_hero_ids['dire']
    
    def __init__(self, json_path, HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP):
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        self.matches = []
        for m in raw_data:
            r_ids, d_ids = self.get_teams_hero_ids(m)
            if not (r_ids and d_ids):
                # 过滤无效比赛
                continue

            # 双向数据增强
            self.matches.append({
                'r_ids': r_ids,
                'd_ids': d_ids,
                'label': 1.0 if m['radiant_win'] else 0.0
            })
            self.matches.append({
                'r_ids': d_ids,
                'd_ids': r_ids,
                'label': 0.0 if m['radiant_win'] else 1.0
            })
        
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
        label = torch.tensor([match['label']], dtype=torch.float32)
        
        return (*r_inputs, *d_inputs, label)
    
def train(load_model_path: str = None, epochs: int = 32):
    # 1. 初始化 TensorBoard Writer
    # 日志会保存在 ./runs 目录下，按时间戳区分实验
    log_dir = os.path.join("runs", "win_rate_exp_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    print(f"[+] TensorBoard 日志将保存至: {log_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = DOTAMatchDataset("./data/high_mmr_cm_matches.json", HERO_ID_FEATURE_MAP, HERO_ID_SEMANTIC_MAP)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = WinRateOracle(embed_dim=32, nhead=4, num_layers=4, use_text=False).to(device)
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
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=epochs)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
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
        
        for batch_idx, (r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem, labels) in enumerate(pbar):
            r_ids, r_attrs, r_sem = r_ids.to(device), r_attrs.to(device), r_sem.to(device)
            d_ids, d_attrs, d_sem = d_ids.to(device), d_attrs.to(device), d_sem.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(r_ids, r_attrs, r_sem, d_ids, d_attrs, d_sem)
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
        output_file='./data/high_mmr_cm_matches.json',
        target_count=100000,
        min_rank=50,
        min_duration=18 * 60,
    )

    print('='*20 + ' 训练 WinRateOracle ' + '='*20)
    train(
        load_model_path=os.path.join(WIN_RATE_ORACLE_SAVE_DIR, 'win_rate_oracle-20251224124558-126-0.8393.pth'),
        epochs=128,
    )