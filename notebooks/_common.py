"""Shared imports + setup used at the top of every notebook.

Notebooks import this in their first cell so the boilerplate (env
loading, rich display, cost banner) is consistent.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel


def load_env() -> None:
    """Load .env from the repo root (or the notebooks/ dir) if present."""
    here = Path(__file__).resolve()
    for candidate in [here.parent, here.parent.parent]:
        env = candidate / ".env"
        if env.exists():
            load_dotenv(env)
            return


def cost_banner(notebook: str, estimate: str, model: str) -> None:
    """Render a cost-estimate banner at the top of a notebook."""
    rprint(Panel.fit(
        f"[bold]{notebook}[/bold]\n"
        f"Default model: [cyan]{model}[/cyan]\n"
        f"Estimated cost per full run: [yellow]{estimate}[/yellow]",
        title="Notebook info",
    ))
