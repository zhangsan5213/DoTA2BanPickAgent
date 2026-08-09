#!/usr/bin/env python3
"""Download DOTA2 hero avatars from OpenDota CDN."""

import os
import requests
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import io

HERO_AVATAR_DIR = "dash_app/assets/hero-avatars"
CDN_BASE_URL = "https://cdn.cloudflare.steamstatic.com/apps/dota2/images/dota_react/heroes/"


def clean_hero_name(name):
    """Convert npc_dota_hero_antimage to antimage."""
    if name.startswith("npc_dota_hero_"):
        return name[len("npc_dota_hero_"):]
    return name


def download_hero_avatar(hero_name, output_path):
    """Download hero avatar from CDN."""
    url = f"{CDN_BASE_URL}{hero_name}.png"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Verify it's a valid image
        img = Image.open(io.BytesIO(response.content))
        img.save(output_path, "PNG")
        return True
    except Exception as e:
        print(f"  [!] Failed to download {hero_name}: {e}")
        return False


def create_fallback_avatar(hero_name, hero_id, output_path):
    """Create a fallback avatar with hero initials."""
    size = (128, 128)
    img = Image.new("RGB", size, color=(60, 60, 80))
    draw = ImageDraw.Draw(img)

    # Get initials
    initials = hero_name[:2].upper() if hero_name else str(hero_id)[:2]

    # Try to use a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except:
        font = ImageFont.load_default()

    # Center text
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    draw.text((x, y), initials, fill=(200, 200, 200), font=font)
    img.save(output_path, "PNG")


def main():
    """Download all hero avatars."""
    # Create directory
    os.makedirs(HERO_AVATAR_DIR, exist_ok=True)

    # Load hero data
    print("[+] Loading hero data...")
    df = pd.read_excel("data/hero_features.xlsx")

    success_count = 0
    fallback_count = 0

    for _, row in df.iterrows():
        hero_id = int(row["id"])
        hero_name = clean_hero_name(row["name"])
        output_path = os.path.join(HERO_AVATAR_DIR, f"{hero_id}.png")

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"[ ] Hero {hero_id} ({hero_name}) already exists, skipping")
            continue

        print(f"[+] Downloading hero {hero_id}: {hero_name}...")

        if download_hero_avatar(hero_name, output_path):
            success_count += 1
        else:
            print(f"  [ ] Creating fallback avatar for {hero_name}")
            create_fallback_avatar(hero_name, hero_id, output_path)
            fallback_count += 1

    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"  Success: {success_count}")
    print(f"  Fallback: {fallback_count}")
    print(f"  Total: {len(df)}")
    print(f"  Saved to: {HERO_AVATAR_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
