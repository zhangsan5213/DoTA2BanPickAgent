"""App layout definition for DOTA2 BP visualization."""

from dash import html, dcc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dash_app.components.hero_display import create_hero_avatar
from dash_app.components.team_display import create_team_display
from dash_app.components.bp_stage import create_bp_stage
from dash_app.components.player_preferences import create_player_preferences
from dash_app.components.action_history import create_action_history
from dash_app.components.mcts_tree import create_mcts_tree
from dash_app.components.mcts_details import create_mcts_details


def create_top_bar():
    """Create the top control bar."""
    return html.Div(
        className="bg-gray-800 border-b border-gray-700 p-4",
        children=[
            html.Div(
                className="flex flex-wrap items-center gap-4 justify-between",
                children=[
                    # Left side: Checkpoint and mode
                    html.Div(
                        className="flex items-center gap-4",
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Checkpoint:", className="text-sm text-gray-400 mr-2"),
                                    dcc.Dropdown(
                                        id="checkpoint-selector",
                                        options=[],
                                        placeholder="Select checkpoint...",
                                        className="w-64"
                                    )
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Mode:", className="text-sm text-gray-400 mr-2"),
                                    dcc.RadioItems(
                                        id="mode-toggle",
                                        options=[
                                            {"label": "Replay", "value": "replay"},
                                            {"label": "Real-time", "value": "realtime"}
                                        ],
                                        value="replay",
                                        className="flex gap-4"
                                    )
                                ]
                            )
                        ]
                    ),
                    # Center: Playback controls
                    html.Div(
                        className="flex items-center gap-2",
                        children=[
                            html.Button(
                                "⏮",
                                id="btn-first",
                                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white"
                            ),
                            html.Button(
                                "◀",
                                id="btn-prev",
                                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white"
                            ),
                            html.Button(
                                "▶",
                                id="btn-play",
                                className="px-4 py-1 bg-green-600 hover:bg-green-500 rounded text-white font-bold"
                            ),
                            html.Button(
                                "▶",
                                id="btn-next",
                                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white"
                            ),
                            html.Button(
                                "⏭",
                                id="btn-last",
                                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-white"
                            ),
                        ]
                    ),
                    # Right side: Speed and Start BP
                    html.Div(
                        className="flex items-center gap-4",
                        children=[
                            html.Div(
                                className="flex items-center gap-2",
                                children=[
                                    html.Label("Speed:", className="text-sm text-gray-400"),
                                    dcc.Slider(
                                        id="speed-slider",
                                        min=0.5,
                                        max=3,
                                        step=0.5,
                                        value=1,
                                        marks={0.5: "0.5x", 1: "1x", 2: "2x", 3: "3x"},
                                        className="w-32"
                                    )
                                ]
                            ),
                            html.Button(
                                "🔄 New BP",
                                id="btn-new-bp",
                                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded text-white font-bold"
                            ),
                        ]
                    )
                ]
            ),
            # Timeline slider
            html.Div(
                className="mt-4",
                children=[
                    html.Div(
                        className="flex items-center gap-4",
                        children=[
                            html.Span("Step:", className="text-sm text-gray-400 w-12"),
                            dcc.Slider(
                                id="timeline-slider",
                                min=0,
                                max=20,
                                step=1,
                                value=0,
                                marks={i: str(i) for i in range(0, 21, 5)},
                                className="flex-1"
                            ),
                            html.Span(id="step-label", className="text-sm text-gray-400 w-20 text-right", children="0 / 20")
                        ]
                    )
                ]
            )
        ]
    )


def create_main_layout():
    """Create the main 3-column layout."""
    return html.Div(
        className="flex gap-4 p-4 h-[calc(100vh-160px)]",
        children=[
            # Left panel
            html.Div(
                className="w-1/4 flex flex-col gap-4 overflow-y-auto",
                children=[
                    create_player_preferences("radiant"),
                    create_player_preferences("dire"),
                    create_action_history()
                ]
            ),
            # Center panel
            html.Div(
                className="w-2/4 flex flex-col gap-4 overflow-y-auto",
                children=[
                    create_bp_stage(),
                    create_team_display()
                ]
            ),
            # Right panel
            html.Div(
                className="w-1/4 flex flex-col gap-4 overflow-y-auto",
                children=[
                    create_mcts_tree(),
                    create_mcts_details()
                ]
            )
        ]
    )


def create_layout():
    """Create the complete app layout."""
    return html.Div(
        className="min-h-screen bg-gray-900 text-white",
        children=[
            # Header
            html.Div(
                className="bg-gray-800 border-b border-gray-700 p-4",
                children=[
                    html.H1(
                        "DOTA2 Ban/Pick Agent Visualizer",
                        className="text-2xl font-bold text-center"
                    )
                ]
            ),
            # Top control bar
            create_top_bar(),
            # Main content
            create_main_layout(),
            # Hidden stores
            dcc.Store(id="trajectory-store", data=[]),
            dcc.Store(id="current-step-store", data=0),
            dcc.Store(id="is-playing-store", data=False),
            dcc.Interval(id="play-interval", interval=1000, disabled=True),
        ]
    )
