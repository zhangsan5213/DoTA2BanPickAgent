"""Player preferences heatmap component."""

from dash import html, dcc
import plotly.graph_objects as go
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from dash_app.utils.hero_data import get_hero_name


def create_player_preferences(team="radiant"):
    """Create player preferences component.

    Args:
        team: 'radiant' or 'dire'
    """
    team_color = "red" if team == "radiant" else "blue"
    team_title = "Radiant" if team == "radiant" else "Dire"

    return html.Div(
        className="bg-gray-800 rounded-lg p-4 mb-4",
        children=[
            html.H4(
                f"{team_title} Player Preferences",
                className=f"text-sm font-semibold mb-3 text-{team_color}-400"
            ),
            dcc.Graph(
                id=f"{team}-preferences",
                config={"displayModeBar": False},
                style={"height": "200px"},
                figure=create_empty_preferences_heatmap()
            )
        ]
    )


def create_empty_preferences_heatmap():
    """Create an empty preferences heatmap."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        annotations=[
            dict(
                text="No data loaded",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="gray")
            )
        ]
    )
    return fig


def update_player_preferences(player_matrix, team="radiant", picked_heroes=None, banned_heroes=None):
    """Update player preferences heatmap.

    Args:
        player_matrix: [5, NUM_HEROES] array of win rates
        team: 'radiant' or 'dire'
        picked_heroes: list of picked hero IDs
        banned_heroes: list of banned hero IDs
    """
    if player_matrix is None:
        return create_empty_preferences_heatmap()

    picked_heroes = picked_heroes or []
    banned_heroes = banned_heroes or []

    # Convert to numpy array
    matrix = np.array(player_matrix)

    # For each player, find their top heroes
    top_n = 10
    top_heroes_per_player = []
    top_values_per_player = []

    for player_idx in range(5):
        player_values = matrix[player_idx]
        # Get top N hero indices (0-based)
        top_indices = np.argsort(player_values)[-top_n:][::-1]
        top_heroes_per_player.append([i + 1 for i in top_indices])  # convert to 1-based
        top_values_per_player.append(player_values[top_indices])

    # Create heatmap data
    heatmap_data = np.zeros((5, top_n))
    hero_labels = []

    for player_idx in range(5):
        heatmap_data[player_idx] = top_values_per_player[player_idx]
        hero_labels.append([get_hero_name(h) for h in top_heroes_per_player[player_idx]])

    # Create annotations
    annotations = []
    for i in range(5):
        for j in range(top_n):
            hero_id = top_heroes_per_player[i][j]
            is_picked = hero_id in picked_heroes
            is_banned = hero_id in banned_heroes

            border_color = None
            if is_picked:
                border_color = "green"
            elif is_banned:
                border_color = "red"

            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{heatmap_data[i, j]:.2f}",
                    showarrow=False,
                    font=dict(color="white", size=9),
                    xanchor="center",
                    yanchor="middle"
                )
            )

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"Top {j+1}" for j in range(top_n)],
        y=[f"Player {i+1}" for i in range(5)],
        colorscale="RdYlGn",
        zmin=0.45,
        zmax=0.7,
        showscale=False,
        hoverongaps=False,
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=10, t=10, b=20),
        annotations=annotations,
        xaxis=dict(showticklabels=True, tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(showticklabels=True, tickfont=dict(size=10)),
    )

    return fig
