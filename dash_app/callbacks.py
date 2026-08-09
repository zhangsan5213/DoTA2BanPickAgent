"""Dash callbacks for DOTA2 BP visualization."""

from dash import Input, Output, State, callback, no_update, ctx
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dash_app.components.team_display import update_team_display
from dash_app.components.bp_stage import update_bp_stage
from dash_app.components.player_preferences import update_player_preferences
from dash_app.components.action_history import update_action_history
from dash_app.components.mcts_details import update_mcts_details
from dash_app.utils.bp_simulator import BPSimulator

# Global simulator instance
simulator = BPSimulator()


def register_callbacks(app):
    """Register all Dash callbacks."""

    @app.callback(
        Output("checkpoint-selector", "options"),
        Input("checkpoint-selector", "id")
    )
    def load_checkpoints(_):
        """Load available checkpoints from ckpts directory."""
        ckpt_dir = Path("ckpts")
        if not ckpt_dir.exists():
            return []

        checkpoints = []
        # Look for bp_agent checkpoints
        for run_dir in ckpt_dir.glob("bp_agent-*"):
            if run_dir.is_dir():
                for ckpt_file in run_dir.glob("bp_agent_epoch*.pth"):
                    checkpoints.append({
                        "label": f"{run_dir.name}/{ckpt_file.name}",
                        "value": str(ckpt_file)
                    })
                # Also look for final checkpoint
                final_ckpt = run_dir / "bp_agent_final.pth"
                if final_ckpt.exists():
                    checkpoints.append({
                        "label": f"{run_dir.name}/bp_agent_final.pth",
                        "value": str(final_ckpt)
                    })

        return sorted(checkpoints, key=lambda x: x["label"], reverse=True)

    @app.callback(
        [Output("trajectory-store", "data"),
         Output("current-step-store", "data"),
         Output("timeline-slider", "value"),
         Output("timeline-slider", "max")],
        Input("btn-new-bp", "n_clicks"),
        prevent_initial_call=False
    )
    def start_new_bp(n_clicks):
        """Start a new BP simulation."""
        if n_clicks is None or n_clicks == 0:
            # Initialize on first load
            simulator.initialize()
            trajectory = simulator.run_full_simulation()
            return trajectory, 0, 0, len(trajectory) - 1

        simulator.initialize()
        trajectory = simulator.run_full_simulation()
        return trajectory, 0, 0, len(trajectory) - 1

    @app.callback(
        [Output("phase-label", "children"),
         Output("bp-timeline", "children"),
         Output("radiant-picks", "children"),
         Output("dire-picks", "children"),
         Output("radiant-bans", "children"),
         Output("dire-bans", "children"),
         Output("radiant-preferences", "figure"),
         Output("dire-preferences", "figure"),
         Output("action-history", "children"),
         Output("step-label", "children")],
        [Input("timeline-slider", "value"),
         Input("trajectory-store", "data")],
        prevent_initial_call=False
    )
    def update_visualizations(step_idx, trajectory):
        """Update all visualizations based on current step."""
        if not trajectory or step_idx >= len(trajectory):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        state = trajectory[step_idx]

        # Update BP stage
        phase_label, timeline_children = update_bp_stage(state)

        # Update team display
        radiant_picks, dire_picks, radiant_bans, dire_bans = update_team_display(state)

        # Update player preferences
        radiant_prefs = update_player_preferences(
            state.get("radiant_players"),
            "radiant",
            state.get("radiant_heroes", []),
            state.get("radiant_bans", [])
        )
        dire_prefs = update_player_preferences(
            state.get("dire_players"),
            "dire",
            state.get("dire_heroes", []),
            state.get("dire_bans", [])
        )

        # Update action history
        action_history = update_action_history(state)

        # Step label
        step_label = f"{step_idx} / {len(trajectory) - 1}"

        return (phase_label, timeline_children, radiant_picks, dire_picks,
                radiant_bans, dire_bans, radiant_prefs, dire_prefs,
                action_history, step_label)

    @app.callback(
        Output("timeline-slider", "value"),
        [Input("btn-first", "n_clicks"),
         Input("btn-prev", "n_clicks"),
         Input("btn-next", "n_clicks"),
         Input("btn-last", "n_clicks")],
        [State("timeline-slider", "value"),
         State("timeline-slider", "max"),
         State("trajectory-store", "data")],
        prevent_initial_call=True
    )
    def handle_playback_controls(btn_first, btn_prev, btn_next, btn_last, current, max_val, trajectory):
        """Handle playback control buttons."""
        if not trajectory:
            return no_update

        triggered = ctx.triggered_id
        if triggered == "btn-first":
            return 0
        elif triggered == "btn-prev":
            return max(0, current - 1)
        elif triggered == "btn-next":
            return min(max_val, current + 1)
        elif triggered == "btn-last":
            return max_val
        return no_update

    @app.callback(
        [Output("btn-play", "children"),
         Output("is-playing-store", "data"),
         Output("play-interval", "disabled"),
         Output("play-interval", "interval")],
        [Input("btn-play", "n_clicks"),
         Input("speed-slider", "value")],
        [State("is-playing-store", "data")],
        prevent_initial_call=True
    )
    def toggle_playback(play_clicks, speed, is_playing):
        """Toggle playback and update speed."""
        triggered = ctx.triggered_id

        if triggered == "speed-slider":
            # Just update interval, don't change play state
            interval = int(1000 / speed)
            return no_update, no_update, no_update, interval

        if is_playing:
            return "▶", False, True, no_update
        else:
            return "⏸", True, False, int(1000 / (speed or 1))

    @app.callback(
        Output("timeline-slider", "value", allow_duplicate=True),
        Input("play-interval", "n_intervals"),
        [State("timeline-slider", "value"),
         State("timeline-slider", "max"),
         State("is-playing-store", "data")],
        prevent_initial_call=True
    )
    def auto_advance_playback(n_intervals, current, max_val, is_playing):
        """Auto-advance timeline when playing."""
        if not is_playing:
            return no_update

        if current >= max_val:
            return 0  # Loop back to start
        return current + 1

    @app.callback(
        Output("mcts-details", "children"),
        Input("mcts-tree", "tapNodeData"),
        prevent_initial_call=True
    )
    def on_mcts_node_select(node_data):
        """Handle MCTS tree node selection."""
        return update_mcts_details(node_data)
