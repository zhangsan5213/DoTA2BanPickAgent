"""Main Dash app initialization."""

import os
import sys
from dash import Dash

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def create_app():
    """Create and configure Dash app."""
    app = Dash(
        __name__,
        title="DOTA2 BP Agent Visualizer",
        suppress_callback_exceptions=True,
        assets_folder=os.path.join(os.path.dirname(__file__), "assets")
    )

    # Set layout
    from dash_app.layout import create_layout
    app.layout = create_layout()

    # Register callbacks
    from dash_app.callbacks import register_callbacks
    register_callbacks(app)

    return app


# For running directly
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)
