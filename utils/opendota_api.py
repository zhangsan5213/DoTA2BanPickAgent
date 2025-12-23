import os
import json
import time
import requests

from tqdm import tqdm

def get_teams() -> dict:
    '''returns all teams'''
    base_url = 'https://api.opendota.com/api/teams'
    response = requests.get(base_url)
    return response.json()

def get_team_players(team_id: int) -> dict:
    '''returns all players of a team'''
    base_url = f'https://api.opendota.com/api/teams/{team_id}/players'
    response = requests.get(base_url)
    return response.json()

def get_team_matches(team_id: int) -> dict:
    '''returns all matches of a team'''
    base_url = f'https://api.opendota.com/api/teams/{team_id}/matches'
    response = requests.get(base_url)
    return response.json()

def get_pro_players() -> dict:
    '''returns the top 100 players'''
    base_url = 'https://api.opendota.com/api/proPlayers'
    response = requests.get(base_url)
    return response.json()

def get_public_matches(min_rank: int = 70) -> dict:
    '''returns the recent public matches'''
    response = requests.get('https://api.opendota.com/api/publicMatches', json={'min_rank': min_rank})
    return response.json()

def get_pro_matches():
    '''returns the recent 100 pro matches'''
    response = requests.get('https://api.opendota.com/api/proMatches')
    return response.json()

def get_player_matches(account_id: int) -> dict:
    '''returns all matches for a player'''
    base_url = f'https://api.opendota.com/api/players/{account_id}/matches'
    response = requests.get(base_url)
    return response.json()

def get_match_details(match_id: int) -> dict:
    '''returns the details of a match'''
    base_url = f'https://api.opendota.com/api/matches/{match_id}'
    response = requests.get(base_url)
    return response.json()

def get_heroes():
    '''returns the heroes'''
    response = requests.get('https://api.opendota.com/api/heroes')
    return response.json()

def get_hero_matches(hero_id: int):
    '''returns the hero ids'''
    response = requests.get(f'https://api.opendota.com/api/heroes/{hero_id}/matches')
    return response.json()

def get_hero_winrates():
    hero_winrates = dict()
    heroes = get_heroes()
    for hero in tqdm(heroes, ncols=90):
        hero_matches = get_hero_matches(hero['id'])
        try:
            winrate = sum(
                int(
                    match['radiant_win'] and match['radiant'] or
                    not match['radiant_win'] and not match['radiant']
                )
                for match in hero_matches
            ) / len(hero_matches)
        except:
            winrate = None
        hero_winrates[hero['id']] = {
            'name': hero['localized_name'],
            'winrate': winrate,
        }
        time.sleep(1.1)
    return hero_winrates

hero_winrate_save_path = './data/hero_winrates.json'
if not os.path.exists(hero_winrate_save_path):
    hero_winrates = get_hero_winrates()
    with open(hero_winrate_save_path, 'w', encoding='utf-8') as f:
        json.dump(hero_winrates, f, ensure_ascii=False, indent=2)