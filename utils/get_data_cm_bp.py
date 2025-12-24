import requests
import time
import json
import os

def fetch_high_mmr_matches(
    output_file='./data/high_mmr_cm_matches.json',
    target_count=100000,
    min_rank=50,
    min_duration=20 * 60,
    api_key=None
):
    """
    抓取高分段 CM 模式比赛数据，支持从最新比赛开始向下抓取，直到碰到本地已有的最大 MatchID 停止。
    """
    
    # --- 1. 初始化与读取本地断点 ---
    all_data = []
    max_id_in_file = 0
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            if all_data:
                # 假设数据是按 match_id 从大到小排列的，取第一个即可，否则用 max()
                max_id_in_file = max(item['match_id'] for item in all_data)
                print(f"[*] 检测到本地数据: {len(all_data)} 条，最大 MatchID: {max_id_in_file}")
        except Exception as e:
            print(f"[!] 读取文件失败，将重新开始: {e}")
            all_data = []

    # --- 2. 爬取逻辑 ---
    # 从一个极大的 ID 开始向下搜索（代表最新比赛）
    current_search_id = 9999999999 
    new_matches_total = []
    
    print(f"[*] 开始同步最新数据...")

    while True:
        # 如果已经达到目标总量，停止
        if len(all_data) + len(new_matches_total) >= target_count:
            print("[+] 已达到目标抓取数量。")
            break

        # SQL 逻辑：找比 current_search_id 小，但比本地已有的 max_id_in_file 大的比赛
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
        WHERE m.match_id < {current_search_id}
          AND m.match_id > {max_id_in_file}
          AND m.game_mode = 2
          AND m.picks_bans IS NOT NULL
          AND m.duration > {min_duration}
          AND pm.avg_rank_tier >= {min_rank}
        ORDER BY m.match_id DESC
        LIMIT 2000
        """
        
        try:
            url = "https://api.opendota.com/api/explorer"
            params = {'sql': sql}
            if api_key:
                params['api_key'] = api_key
                
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                rows = response.json().get('rows', [])
                
                if not rows:
                    print("[*] 增量更新完成：未发现更多新比赛。")
                    break
                
                # 记录抓取到的数据
                new_matches_total.extend(rows)
                
                # 更新下一次搜索的上限（当前批次的最小值）
                current_search_id = rows[-1]['match_id']
                
                print(f"[*] 本轮抓取 {len(rows)} 场 | 范围: {rows[0]['match_id']} -> {rows[-1]['match_id']}")
                
                # --- 3. 增量保存 ---
                # 将新抓到的（更新的）数据放在旧数据前面
                combined_data = new_matches_total + all_data
                # 截断超出部分
                if len(combined_data) > target_count:
                    combined_data = combined_data[:target_count]
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(combined_data, f)
                
                # 如果返回的数量少于 LIMIT，说明中间的断层已经填满了
                if len(rows) < 2000:
                    print("[*] 数据已对接完成。")
                    break

            elif response.status_code == 429:
                print("[!] 触发速率限制，等待 30 秒...")
                time.sleep(30)
            else:
                print(f"[!] API 请求失败: {response.status_code}")
                time.sleep(10)
                
        except Exception as e:
            print(f"[!] 发生异常: {e}")
            time.sleep(5)

        # 频率控制（有 Key 可以缩短，没 Key 建议 2-3 秒）
        time.sleep(1.5 if api_key else 2.0)

    print(f"[√] 任务结束，本地文件总计: {len(all_data) + len(new_matches_total)} 条比赛记录。")

# --- 调用示例 ---
if __name__ == "__main__":
    # 如果你有 OpenDota API Key，请填入
    MY_API_KEY = None 
    
    fetch_high_mmr_matches(
        output_file='./data/high_mmr_cm_matches.json',
        target_count=100000,
        min_rank=50,
        min_duration=18 * 60,
        api_key=MY_API_KEY
    )