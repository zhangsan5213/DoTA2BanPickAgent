import requests
import time
import json
import os

def fetch_player_hero_stats(account_ids, api_key=None):
    """
    批量获取指定玩家的英雄使用统计
    """
    if not account_ids:
        return {}

    # 过滤掉匿名玩家 (account_id 为 0 或 None)
    valid_ids = [str(aid) for aid in account_ids if aid and aid > 0]
    if not valid_ids:
        return {}

    # SQL: 统计每个玩家每个英雄的 场次 和 胜场
    # 逻辑：(player_slot < 128 和 radiant_win 相同则为胜)
    sql = f"""
    SELECT 
        pm.account_id,
        pm.hero_id,
        count(*) as games,
        sum(case when (pm.player_slot < 128) = m.radiant_win then 1 else 0 end) as wins
    FROM player_matches pm
    JOIN matches m ON pm.match_id = m.match_id
    WHERE pm.account_id IN ({','.join(valid_ids)})
    GROUP BY pm.account_id, pm.hero_id
    """
    
    try:
        url = "https://api.opendota.com/api/explorer"
        params = {'sql': sql}
        if api_key:
            params['api_key'] = api_key
        
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            rows = resp.json().get('rows', [])
            # 整理成字典 {account_id: {hero_id: {games, wins}}}
            stats_map = {}
            for r in rows:
                aid = r['account_id']
                if aid not in stats_map:
                    stats_map[aid] = {}
                stats_map[aid][r['hero_id']] = {
                    'games': r['games'],
                    'wins': r['wins']
                }
            return stats_map
    except Exception as e:
        print(f"[!] 玩家统计抓取异常: {e}")
    return {}

def fetch_high_mmr_matches(
    output_file='./data/high_mmr_with_stats.json',
    target_count=1000, # 演示建议先设小一点
    min_rank=50,
    min_duration=15 * 60,
    api_key=None
):
    all_data = []
    max_id_in_file = 0
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            if all_data:
                max_id_in_file = max(item['match_id'] for item in all_data)
                print(f"[*] 本地已有: {len(all_data)} 条")
        except: pass

    current_search_id = 9999999999 
    
    while len(all_data) < target_count:
        # 1. 抓取比赛基础信息
        match_sql = f"""
        WITH target_matches AS (
            SELECT m.match_id, m.picks_bans, m.radiant_win, m.start_time, m.duration, pm.avg_rank_tier
            FROM matches m
            JOIN public_matches pm ON m.match_id = pm.match_id
            WHERE m.match_id < {current_search_id} AND m.match_id > {max_id_in_file}
              AND m.game_mode = 2 AND m.picks_bans IS NOT NULL AND m.duration > {min_duration}
              AND pm.avg_rank_tier >= {min_rank}
            ORDER BY m.match_id DESC LIMIT 50
        )
        SELECT tm.*, 
            (SELECT json_agg(json_build_object('account_id', pmt.account_id, 'hero_id', pmt.hero_id, 'player_slot', pmt.player_slot))
             FROM player_matches pmt WHERE pmt.match_id = tm.match_id) as players
        FROM target_matches tm ORDER BY tm.match_id DESC
        """
        
        try:
            url = "https://api.opendota.com/api/explorer"
            res = requests.get(url, params={'sql': match_sql, 'api_key': api_key} if api_key else {'sql': match_sql}, timeout=40)
            
            if res.status_code != 200:
                print(f"[!] 错误: {res.status_code}"); time.sleep(10); continue
            
            matches = res.json().get('rows', [])
            if not matches:
                print("[*] 无更多数据。"); break

            # 2. 提取当前批次所有玩家 ID
            current_batch_account_ids = set()
            for m in matches:
                for p in m['players']:
                    if p['account_id']:
                        current_batch_account_ids.add(p['account_id'])

            # 3. 抓取这些玩家的英雄统计
            print(f"[*] 正在查询 {len(current_batch_account_ids)} 名玩家的英雄场次统计...")
            player_stats_map = fetch_player_hero_stats(list(current_batch_account_ids), api_key)

            # 4. 合并数据
            for m in matches:
                for p in m['players']:
                    aid = p['account_id']
                    # 将该玩家所有英雄的统计注入到 player 对象中
                    # 如果该玩家匿名或无数据，返回空字典
                    p['hero_history'] = player_stats_map.get(aid, {})
                
                all_data.append(m)
            
            current_search_id = matches[-1]['match_id']
            print(f"[+] 已抓取 {len(all_data)} 场比赛")

            # 存盘
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f)

            if len(matches) < 50: break
            
        except Exception as e:
            print(f"[!] 循环异常: {e}"); time.sleep(5)

        time.sleep(2.0 if not api_key else 0.5)

if __name__ == "__main__":
    # 替换为你自己的 API KEY 速度会快很多
    fetch_high_mmr_matches(
        output_file='./data/high_mmr_with_stats-rank_40-duration_15.json',
        target_count=10000,
        min_rank=40,
        min_duration=15*60,
        api_key=None,
    )