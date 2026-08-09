"""MCTS node details panel component."""

from dash import html, dash_table
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from dash_app.utils.hero_data import get_hero_name


def create_mcts_details():
    """Create MCTS node details panel."""
    return html.Div(
        className="bg-gray-800 rounded-lg p-4",
        children=[
            html.H4(
                "MCTS Node Details",
                className="text-sm font-semibold mb-3 text-purple-400"
            ),
            html.Div(
                id="mcts-details",
                children=[
                    html.Div(
                        className="text-gray-500 text-sm",
                        children="Select a node in the tree to see details"
                    )
                ]
            )
        ]
    )


def update_mcts_details(node_data):
    """Update MCTS details panel with selected node data.

    Args:
        node_data: Cytoscape node data dict

    Returns:
        Details panel component
    """
    if node_data is None:
        return [
            html.Div(
                className="text-gray-500 text-sm",
                children="Select a node in the tree to see details"
            )
        ]

    hero_id = node_data.get("action")
    hero_name = get_hero_name(hero_id) if hero_id else "Root"

    details = []

    # Hero info
    if hero_id:
        details.append(
            html.Div(
                className="mb-3",
                children=[
                    html.Span(className="text-gray-400 text-sm", children="Hero: "),
                    html.Span(className="text-white font-medium", children=hero_name)
                ]
            )
        )

    # Prior probability
    prior = node_data.get("prior", 0)
    details.append(
        html.Div(
            className="mb-2",
            children=[
                html.Span(className="text-gray-400 text-sm", children="Prior: "),
                html.Span(className="text-white", children=f"{prior:.4f}")
            ]
        )
    )

    # Visit count
    visit_count = node_data.get("visit_count", 0)
    details.append(
        html.Div(
            className="mb-2",
            children=[
                html.Span(className="text-gray-400 text-sm", children="Visits: "),
                html.Span(className="text-white", children=f"{visit_count}")
            ]
        )
    )

    # Value
    value = node_data.get("value", 0)
    details.append(
        html.Div(
            className="mb-2",
            children=[
                html.Span(className="text-gray-400 text-sm", children="Value (Q): "),
                html.Span(
                    className=f"font-medium {'text-green-400' if value > 0 else 'text-red-400'}",
                    children=f"{value:.4f}"
                )
            ]
        )
    )

    # Value sum
    value_sum = node_data.get("value_sum")
    if value_sum is not None:
        details.append(
            html.Div(
                className="mb-2",
                children=[
                    html.Span(className="text-gray-400 text-sm", children="Value Sum: "),
                    html.Span(className="text-white", children=f"{value_sum:.4f}")
                ]
            )
        )

    # Eval value
    eval_value = node_data.get("eval_value")
    if eval_value is not None:
        details.append(
            html.Div(
                className="mb-2",
                children=[
                    html.Span(className="text-gray-400 text-sm", children="Eval Value: "),
                    html.Span(className="text-white", children=f"{eval_value:.4f}")
                ]
            )
        )

    # Terminal flag
    is_terminal = node_data.get("is_terminal")
    if is_terminal is not None:
        terminal_text = "Yes" if is_terminal else "No"
        terminal_color = "text-yellow-400" if is_terminal else "text-gray-300"
        details.append(
            html.Div(
                className="mb-3",
                children=[
                    html.Span(className="text-gray-400 text-sm", children="Terminal: "),
                    html.Span(className=terminal_color, children=terminal_text)
                ]
            )
        )

    return details
