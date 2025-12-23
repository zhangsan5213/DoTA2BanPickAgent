import requests
import time
import json
import os

# 配置
OUTPUT_FILE = './data/high_mmr_cm_matches.json'
TARGET_COUNT = 100000
MIN_RANK = 50
MIN_DURATION = 20 * 60

# 断点续传逻辑
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'r') as f:
        all_data = json.load(f)
    last_match_id = all_data[-1]['match_id'] if all_data else 9999999999
else:
    all_data = []
    last_match_id = 9999999999

while len(all_data) < TARGET_COUNT:
    # 核心修正：使用 JOIN 关联段位表
    sql = f"""
    SELECT 
        m.match_id, 
        m.picks_bans, 
        m.radiant_win, 
        m.start_time,
        m.duration,
        pm.avg_rank_tier
    FROM matches m
    JOIN public_matches pm ON m.match_id = pm.match_id
    WHERE m.match_id < {last_match_id}
      AND m.game_mode = 2
      AND m.picks_bans IS NOT NULL
      AND m.duration > {MIN_DURATION}
      AND pm.avg_rank_tier >= {MIN_RANK}
    ORDER BY m.match_id DESC
    LIMIT 2000
    """
    
    try:
        url = f"https://api.opendota.com/api/explorer?sql={sql}"
        # 建议：如果你有 API Key，记得带上 ?api_key=YOUR_KEY
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json().get('rows', [])
            if not data:
                print("未匹配到更多数据。")
                break
            
            all_data.extend(data)
            last_match_id = data[-1]['match_id']
            
            # 增量保存
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_data, f)
            
            print(f"进度: {len(all_data)} 场 | 当前 MatchID: {last_match_id}")
        else:
            print(f"API请求失败: {response.status_code}")
            time.sleep(10)
            
    except Exception as e:
        print(f"发生异常: {e}")
        time.sleep(5)

    time.sleep(2) # 遵守速率限制