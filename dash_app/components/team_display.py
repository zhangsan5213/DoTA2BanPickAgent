"""Team display component - shows Radiant and Dire picks/bans."""

from dash import html, dcc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from dash_app.components.hero_display import create_hero_avatar


def create_team_display():
    """Create the main team display component."""
    return html.Div(
        className="flex gap-8 justify-center items-start",
        children=[
            # Radiant team
            html.Div(
                className="flex-1 max-w-xs",
                children=[
                    html.H3(
                        "🔴 Radiant",
                        className="text-xl font-bold text-center mb-4 text-red-400"
                    ),
                    # Picks section
                    html.Div(
                        className="mb-6",
                        children=[
                            html.H4("Picks", className="text-sm font-semibold mb-2 text-gray-300"),
                            html.Div(
                                id="radiant-picks",
                                className="grid grid-cols-5 gap-2",
                                children=[
                                    create_hero_avatar(None, size="medium", show_tooltip=True)
                                    for _ in range(5)
                                ]
                            )
                        ]
                    ),
                    # Bans section
                    html.Div(
                        children=[
                            html.H4("Bans", className="text-sm font-semibold mb-2 text-gray-400"),
                            html.Div(
                                id="radiant-bans",
                                className="grid grid-cols-5 gap-2",
                                children=[
                                    create_hero_avatar(None, size="small", is_banned=True, show_tooltip=True)
                                    for _ in range(5)
                                ]
                            )
                        ]
                    )
                ]
            ),
            # VS divider
            html.Div(
                className="flex flex-col items-center justify-center py-8",
                children=[
                    html.Div(
                        className="text-4xl font-bold text-gray-500",
                        children="VS"
                    ),
                    html.Div(
                        id="win-probability",
                        className="mt-4 text-sm text-gray-400",
                        children="Win Prob: -"
                    )
                ]
            ),
            # Dire team
            html.Div(
                className="flex-1 max-w-xs",
                children=[
                    html.H3(
                        "🔵 Dire",
                        className="text-xl font-bold text-center mb-4 text-blue-400"
                    ),
                    # Picks section
                    html.Div(
                        className="mb-6",
                        children=[
                            html.H4("Picks", className="text-sm font-semibold mb-2 text-gray-300"),
                            html.Div(
                                id="dire-picks",
                                className="grid grid-cols-5 gap-2",
                                children=[
                                    create_hero_avatar(None, size="medium", show_tooltip=True)
                                    for _ in range(5)
                                ]
                            )
                        ]
                    ),
                    # Bans section
                    html.Div(
                        children=[
                            html.H4("Bans", className="text-sm font-semibold mb-2 text-gray-400"),
                            html.Div(
                                id="dire-bans",
                                className="grid grid-cols-5 gap-2",
                                children=[
                                    create_hero_avatar(None, size="small", is_banned=True, show_tooltip=True)
                                    for _ in range(5)
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )


def update_team_display(state):
    """Update team display from state.

    Returns:
        (radiant_picks, dire_picks, radiant_bans, dire_bans)
    """
    radiant_picks = [
        create_hero_avatar(h, size="medium", show_tooltip=True)
        for h in state.get("radiant_heroes", [])
    ] + [
        create_hero_avatar(None, size="medium", show_tooltip=True)
        for _ in range(5 - len(state.get("radiant_heroes", [])))
    ]

    dire_picks = [
        create_hero_avatar(h, size="medium", show_tooltip=True)
        for h in state.get("dire_heroes", [])
    ] + [
        create_hero_avatar(None, size="medium", show_tooltip=True)
        for _ in range(5 - len(state.get("dire_heroes", [])))
    ]

    radiant_bans = [
        create_hero_avatar(h, size="small", is_banned=True, show_tooltip=True)
        for h in state.get("radiant_bans", [])
    ] + [
        create_hero_avatar(None, size="small", is_banned=True, show_tooltip=True)
        for _ in range(5 - len(state.get("radiant_bans", [])))
    ]

    dire_bans = [
        create_hero_avatar(h, size="small", is_banned=True, show_tooltip=True)
        for h in state.get("dire_bans", [])
    ] + [
        create_hero_avatar(None, size="small", is_banned=True, show_tooltip=True)
        for _ in range(5 - len(state.get("dire_bans", [])))
    ]

    return radiant_picks, dire_picks, radiant_bans, dire_bans
