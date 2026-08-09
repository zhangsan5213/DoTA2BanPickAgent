"""Hero data utilities for Dash app."""

import os
import pandas as pd
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

_HERO_DATA_CACHE = None
_HERO_ID_TO_NAME = None
_HERO_NAME_TO_ID = None


def load_hero_data():
    """Load hero data from Excel file."""
    global _HERO_DATA_CACHE
    if _HERO_DATA_CACHE is None:
        df = pd.read_excel("data/hero_features.xlsx")
        _HERO_DATA_CACHE = df
    return _HERO_DATA_CACHE


def get_hero_id_to_name():
    """Get mapping from hero ID to name."""
    global _HERO_ID_TO_NAME
    if _HERO_ID_TO_NAME is None:
        df = load_hero_data()
        _HERO_ID_TO_NAME = {
            int(row["id"]): clean_hero_name(row["name"])
            for _, row in df.iterrows()
        }
    return _HERO_ID_TO_NAME


def get_hero_name_to_id():
    """Get mapping from hero name to ID."""
    global _HERO_NAME_TO_ID
    if _HERO_NAME_TO_ID is None:
        id_to_name = get_hero_id_to_name()
        _HERO_NAME_TO_ID = {v: k for k, v in id_to_name.items()}
    return _HERO_NAME_TO_ID


def clean_hero_name(name):
    """Convert npc_dota_hero_antimage to antimage."""
    if name.startswith("npc_dota_hero_"):
        return name[len("npc_dota_hero_"):]
    return name


def get_hero_name(hero_id):
    """Get hero name from ID."""
    id_to_name = get_hero_id_to_name()
    return id_to_name.get(hero_id, f"Hero_{hero_id}")


def get_avatar_path(hero_id):
    """Get path to hero avatar image."""
    avatar_path = f"/assets/hero-avatars/{hero_id}.png"
    return avatar_path


def get_valid_hero_ids():
    """Get list of valid hero IDs."""
    df = load_hero_data()
    return [int(row["id"]) for _, row in df.iterrows()]


def get_hero_features(hero_id):
    """Get features for a specific hero."""
    df = load_hero_data()
    row = df[df["id"] == hero_id]
    if len(row) == 0:
        return None
    return row.iloc[0].to_dict()
