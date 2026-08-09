"""MCTS tree visualization component using Cytoscape."""

from dash import html
import dash_cytoscape as cyto
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from dash_app.utils.hero_data import get_hero_name


def create_mcts_tree():
    """Create MCTS tree component."""
    return html.Div(
        className="bg-gray-800 rounded-lg p-4 mb-4",
        children=[
            html.H4(
                "MCTS Search Tree",
                className="text-sm font-semibold mb-3 text-purple-400"
            ),
            cyto.Cytoscape(
                id="mcts-tree",
                layout={
                    "name": "breadthfirst",
                    "roots": ["[id = 'root']"],
                    "directed": True,
                },
                style={"width": "100%", "height": "300px"},
                stylesheet=[
                    {
                        "selector": "node",
                        "style": {
                            "label": "data(label)",
                            "width": "mapData(visit_count, 0, 100, 20, 60)",
                            "height": "mapData(visit_count, 0, 100, 20, 60)",
                            "background-color": "mapData(value, -1, 1, #ef4444, #22c55e)",
                            "color": "#fff",
                            "font-size": "10px",
                            "text-valign": "center",
                            "text-halign": "center",
                            "border-width": "2px",
                            "border-color": "#4b5563",
                        }
                    },
                    {
                        "selector": "edge",
                        "style": {
                            "width": "mapData(prior, 0, 1, 1, 8)",
                            "line-color": "#6b7280",
                            "target-arrow-color": "#6b7280",
                            "target-arrow-shape": "triangle",
                            "curve-style": "bezier",
                        }
                    },
                    {
                        "selector": ":selected",
                        "style": {
                            "border-width": "3px",
                            "border-color": "#f59e0b",
                        }
                    }
                ],
                elements=create_empty_mcts_tree()
            )
        ]
    )


def create_empty_mcts_tree():
    """Create an empty MCTS tree placeholder."""
    return [
        {
            "data": {
                "id": "root",
                "label": "Root",
                "visit_count": 1,
                "value": 0
            }
        }
    ]


def convert_mcts_tree_to_cytoscape(root_node, hero_id_to_name=None):
    """Convert MCTSNode tree to Cytoscape elements.

    Args:
        root_node: MCTSNode root
        hero_id_to_name: Optional dict mapping hero ID to name

    Returns:
        list of Cytoscape elements (nodes + edges)
    """
    if hero_id_to_name is None:
        hero_id_to_name = {}

    elements = []
    node_counter = 0

    def traverse(node, parent_id=None):
        nonlocal node_counter

        # Create node ID
        if parent_id is None:
            node_id = "root"
            label = "Root"
        else:
            node_id = f"node-{node_counter}"
            node_counter += 1
            hero_name = hero_id_to_name.get(node.action, f"Hero {node.action}")
            label = hero_name[:8]

        # Add node
        elements.append({
            "data": {
                "id": node_id,
                "label": label,
                "action": node.action,
                "prior": node.prior,
                "visit_count": node.visit_count,
                "value": node.value() if node.visit_count > 0 else 0,
                "value_sum": node.value_sum,
                "eval_value": node.eval_value,
                "is_terminal": node.is_terminal,
            }
        })

        # Add edge from parent
        if parent_id is not None:
            elements.append({
                "data": {
                    "source": parent_id,
                    "target": node_id,
                    "prior": node.prior
                }
            })

        # Traverse children
        for child in node.children.values():
            traverse(child, node_id)

    traverse(root_node)
    return elements
