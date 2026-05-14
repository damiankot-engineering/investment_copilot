"""FastAPI surface for Investment Copilot.

A pure addition over services + orchestrator. Co-exists with the CLI and
Streamlit GUI; never bypasses the ServiceContainer.
"""

from investment_copilot.api.main import app, create_app

__all__ = ["app", "create_app"]
