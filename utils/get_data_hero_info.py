import os
import re
import json
import requests

import numpy as np
import pandas as pd

# ==========================================
# 获取英雄属性
# ==========================================

def fetch_hero_data():
    print("正在从 OpenDota 获取英雄数据...")
    
    # 获取英雄基础属性和统计数据
    # 这个接口包含了：角色定位、基础属性、攻击距离、各分段胜率等
    stats_url = "https://api.opendota.com/api/heroStats"
    resp = requests.get(stats_url)
    if resp.status_code != 200:
        raise Exception("无法连接到 OpenDota API")
    
    data = resp.json()
    
    # 转换为 DataFrame 方便处理
    df = pd.DataFrame(data)
    
    # 按照 hero_id 排序，确保索引一致 (1 到 130+，中间有空缺)
    df = df.sort_values('id')
    
    # 1. 处理主属性 (primary_attr) -> One-hot
    # str, agi, int, all (全能)
    attr_dummies = pd.DataFrame.astype(pd.get_dummies(df['primary_attr'], prefix='attr'), int)
    
    # 2. 处理角色标签 (roles) -> Multi-hot
    # 提取所有可能的角色名
    all_roles = set()
    for roles in df['roles']:
        all_roles.update(roles)
    
    role_features = []
    for roles in df['roles']:
        row = {f"role_{r}": (1 if r in roles else 0) for r in all_roles}
        role_features.append(row)
    role_df = pd.DataFrame(role_features)
    
    # 3. 处理攻击类型 (attack_type) -> Binary
    attack_binary = (df['attack_type'] == 'Melee').astype(int).to_frame(name='is_melee')
    
    # 4. 基础数值归一化 (基础护甲, 移速, 攻击距离等)
    df['base_attack_avg'] = (df['base_attack_min'] + df['base_attack_max']) / 2
    numerical_cols = [
        'base_armor', 'base_attack_avg',
        'base_str', 'base_agi', 'base_int', 
        'attack_range', 'move_speed'
    ]
    # 简单的 Min-Max 归一化
    numeric_df = df[numerical_cols].copy()
    numeric_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
    
    # 5. 胜率数据 (选取高分段胜率: 冠绝一世 8_win / 8_pick)
    # 处理除零风险
    df['pro_win_rate'] = df['pro_win'] / df['pro_pick'].replace(0, 1)
    win_rate_df = df[['pro_win_rate']].fillna(0.5)

    # 合并所有特征
    final_features = pd.concat([
        df[['name']], # 保持 NAME 对应
        df[['id']], # 保持 ID 对应
        attr_dummies.reset_index(drop=True),
        role_df.reset_index(drop=True),
        attack_binary.reset_index(drop=True),
        numeric_df.reset_index(drop=True),
        win_rate_df.reset_index(drop=True)
    ], axis=1)
    
    return final_features

# ==========================================
# 获取英雄技能描述文本
# ==========================================

def get_hero_full_descriptions():
    # 1. 获取英雄与技能的对应关系 (Hero ID -> Ability List)
    abilities_map_url = "https://api.opendota.com/api/constants/hero_abilities"
    # 2. 获取技能的详细信息 (Ability Name -> Description)
    ability_details_url = "https://api.opendota.com/api/constants/abilities"
    
    print("正在下载技能数据...")
    hero_to_abilities = requests.get(abilities_map_url).json()
    all_abilities = requests.get(ability_details_url).json()
    
    hero_text_data = {}

    print("正在提取描述并清洗文本...")
    for hero_name, data in hero_to_abilities.items():
        # 获取该英雄的所有技能名
        ability_names = data.get('abilities', [])
        
        full_desc = dict()
        for a_name in ability_names:
            ability_info = all_abilities.get(a_name)
            if ability_info and 'desc' in ability_info:
                # 提取描述内容
                desc = ability_info['desc']
                
                # 清洗文本：
                # 1. 去除 HTML 标签 (如 <br>)
                desc = re.sub(r'<[^>]+>', ' ', desc)
                # 2. 去除游戏内的变量占位符 (如 %value%)
                desc = re.sub(r'%[a-zA-Z0-9_]+%', '', desc)
                
                full_desc[a_name] = desc.strip()
        
        # 将该英雄所有技能拼接成一段话
        hero_text_data[hero_name] = full_desc

    return hero_text_data

if __name__ == '__main__':
    # 执行获取英雄数据
    hero_features_save_path = './data/hero_features.xlsx'
    if not os.path.exists(hero_features_save_path):
        hero_features_df = fetch_hero_data()
        hero_features_df.to_excel(hero_features_save_path)
        print(f"成功获取 {len(hero_features_df)} 个英雄的特征，特征维度: {hero_features_df.shape[1]-1}")
        print(hero_features_df.head())

    # 执行获取英雄文本信息
    hero_ability_descriptions_path = "./data/hero_ability_descriptions.json"
    if not os.path.exists(hero_ability_descriptions_path):
        hero_texts = get_hero_full_descriptions()
        with open(hero_ability_descriptions_path, "w", encoding="utf-8") as f:
            json.dump(hero_texts, f, ensure_ascii=False, indent=2)