"""Action history component - shows timeline of BP actions."""

from dash import html, dcc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from dash_app.components.hero_display import create_hero_avatar
from dash_app.components.bp_stage import CM_SEQUENCE


def create_action_history():
    """Create action history component."""
    return html.Div(
        className="bg-gray-800 rounded-lg p-4",
        children=[
            html.H4(
                "Action History",
                className="text-sm font-semibold mb-3 text-gray-300"
            ),
            html.Div(
                id="action-history",
                className="space-y-1 max-h-64 overflow-y-auto",
                children=[
                    html.Div(
                        className="text-gray-500 text-sm",
                        children="No actions yet"
                    )
                ]
            )
        ]
    )


def create_action_entry(step_idx, hero_id, is_clickable=True):
    """Create a single action history entry."""
    team, action_type = CM_SEQUENCE[step_idx]
    team_name = "Radiant" if team == 0 else "Dire"
    team_color = "text-red-400" if team == 0 else "text-blue-400"
    team_badge = "🔴" if team == 0 else "🔵"
    action_icon = "🚫" if action_type == "ban" else "✓"

    entry_classes = "flex items-center gap-2 p-2 rounded hover:bg-gray-700 transition-colors"
    if is_clickable:
        entry_classes += " cursor-pointer"

    return html.Div(
        id=f"history-step-{step_idx}",
        className=entry_classes,
        **{"data-step": step_idx} if is_clickable else {},
        children=[
            html.Span(className="text-xs text-gray-500 w-6", children=f"{step_idx+1}"),
            html.Span(className="text-sm", children=team_badge),
            html.Span(className=f"text-xs font-medium {team_color}", children=f"{team_name}"),
            html.Span(className="text-sm", children=action_icon),
            html.Span(className="text-xs uppercase", children=action_type),
            create_hero_avatar(hero_id, size="tiny", show_tooltip=True)
        ]
    )


def update_action_history(state):
    """Update action history from state.

    Returns:
        list of action entry components
    """
    history = state.get("history", [])

    if not history:
        return [
            html.Div(
                className="text-gray-500 text-sm",
                children="No actions yet"
            )
        ]

    entries = []
    for step_idx, action in enumerate(history):
        hero_id = action.get("hero_id")
        entries.append(create_action_entry(step_idx, hero_id, is_clickable=True))

    return entries
