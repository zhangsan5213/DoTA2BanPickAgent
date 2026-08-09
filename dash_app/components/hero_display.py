"""Hero avatar display component."""

from dash import html
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from dash_app.utils.hero_data import get_avatar_path, get_hero_name

SIZE_CLASSES = {
    "tiny": "w-6 h-6",
    "small": "w-10 h-10",
    "medium": "w-16 h-16",
    "large": "w-24 h-24",
}


def create_hero_avatar(hero_id, size="small", is_banned=False, show_tooltip=True, additional_classes=""):
    """Create a hero avatar component.

    Args:
        hero_id: Hero ID (int)
        size: One of 'tiny', 'small', 'medium', 'large'
        is_banned: Whether to show as banned (grayed out)
        show_tooltip: Whether to show tooltip with hero name
        additional_classes: Additional CSS classes
    """
    if hero_id is None or hero_id == 0:
        return html.Div(
            className=f"bg-gray-700 rounded border-2 border-gray-600 flex items-center justify-center {SIZE_CLASSES.get(size, 'w-10 h-10')} {additional_classes}",
            children=[html.Span(className="text-gray-500 text-xs", children="?")]
        )

    avatar_path = get_avatar_path(hero_id)
    hero_name = get_hero_name(hero_id)

    ban_style = "grayscale opacity-50" if is_banned else ""
    size_class = SIZE_CLASSES.get(size, "w-10 h-10")

    img = html.Img(
        src=avatar_path,
        alt=hero_name,
        className=f"object-cover rounded {size_class} {ban_style}",
        **{"loading": "lazy"}
    )

    if show_tooltip:
        return html.Div(
            className=f"relative group inline-block {additional_classes}",
            children=[
                img,
                html.Div(
                    className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50",
                    children=hero_name
                )
            ]
        )

    return html.Div(className=additional_classes, children=[img])
