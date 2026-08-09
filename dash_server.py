#!/usr/bin/env python3
"""Entry point for DOTA2 BP Visualizer Dash server."""

import os
import sys

# Ensure we use the right environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dash_app.app import create_app


def main():
    """Run the Dash server."""
    print("=" * 60)
    print("DOTA2 Ban/Pick Agent Visualizer")
    print("=" * 60)
    print()
    print("Starting server...")
    print("Open http://localhost:8050 in your browser")
    print()

    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
