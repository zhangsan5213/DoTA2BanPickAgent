"""BP stage timeline component - shows the 20-step CM sequence."""

from dash import html
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from dash_app.components.hero_display import create_hero_avatar

# Phase definitions
PHASES = [
    ("Ban 1", 0, 4),
    ("Pick 1", 4, 8),
    ("Ban 2", 8, 12),
    ("Pick 2", 12, 16),
    ("Ban 3", 16, 18),
    ("Pick 3", 18, 20),
]

# CM sequence: (team, action_type) where team 0=Radiant, 1=Dire
CM_SEQUENCE = [
    (0, "ban"), (1, "ban"), (0, "ban"), (1, "ban"),
    (0, "pick"), (1, "pick"), (1, "pick"), (0, "pick"),
    (1, "ban"), (0, "ban"), (1, "ban"), (0, "ban"),
    (1, "pick"), (0, "pick"), (0, "pick"), (1, "pick"),
    (0, "ban"), (1, "ban"),
    (0, "pick"), (1, "pick"),
]


def get_phase_name(step_idx):
    """Get phase name for step index."""
    for phase_name, start, end in PHASES:
        if start <= step_idx < end:
            return phase_name
    return "Complete"


def create_bp_stage():
    """Create the BP stage timeline component."""
    return html.Div(
        className="bg-gray-800 rounded-lg p-4 mb-6",
        children=[
            html.Div(
                id="phase-label",
                className="text-center text-lg font-semibold mb-4 text-yellow-400",
                children="Ban 1"
            ),
            html.Div(
                id="bp-timeline",
                className="grid grid-cols-10 gap-1",
                children=[
                    create_step_cell(i) for i in range(20)
                ]
            )
        ]
    )


def create_step_cell(step_idx):
    """Create a single step cell."""
    team, action_type = CM_SEQUENCE[step_idx]
    team_color = "bg-red-900/50" if team == 0 else "bg-blue-900/50"
    team_border = "border-red-500" if team == 0 else "border-blue-500"
    action_icon = "🚫" if action_type == "ban" else "✓"

    return html.Div(
        id=f"step-{step_idx}",
        className=f"flex flex-col items-center p-1 rounded border-2 border-gray-600 {team_color} transition-all",
        children=[
            html.Div(className="text-xs text-gray-400", children=f"{step_idx+1}"),
            html.Div(className="text-xs", children=action_icon),
            html.Div(
                id=f"step-hero-{step_idx}",
                className="mt-1",
                children=create_hero_avatar(None, size="tiny", show_tooltip=False)
            )
        ]
    )


def update_bp_stage(state):
    """Update BP stage from state.

    Returns:
        (phase_label, timeline_children)
    """
    step_idx = state.get("step_idx", 0)
    history = state.get("history", [])
    phase_label = get_phase_name(step_idx)

    # Create timeline children
    timeline_children = []
    for i in range(20):
        team, action_type = CM_SEQUENCE[i]
        team_color = "bg-red-900/50" if team == 0 else "bg-blue-900/50"
        action_icon = "🚫" if action_type == "ban" else "✓"

        # Highlight current step
        border_class = "border-yellow-400 ring-2 ring-yellow-400" if i == step_idx else "border-gray-600"
        if i < step_idx:
            border_class = "border-green-500"

        # Get hero if step completed
        hero_id = None
        if i < len(history):
            hero_id = history[i].get("hero_id")

        cell = html.Div(
            className=f"flex flex-col items-center p-1 rounded border-2 {border_class} {team_color} transition-all",
            children=[
                html.Div(className="text-xs text-gray-400", children=f"{i+1}"),
                html.Div(className="text-xs", children=action_icon),
                html.Div(
                    className="mt-1",
                    children=create_hero_avatar(hero_id, size="tiny", show_tooltip=True)
                )
            ]
        )
        timeline_children.append(cell)

    return phase_label, timeline_children
