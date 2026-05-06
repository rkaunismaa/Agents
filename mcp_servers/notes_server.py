"""Canonical notes server (FastMCP).

The low-level `Server`-based version of this same surface lives inline
in NB 09 as a teaching artifact. This file is what NB 10 spawns as a
subprocess and what NB 12's spine connects to via stdio.

Persistence: a single JSON file at `data/notes.json` (relative to repo
root). The server validates the file on read and refuses to start if
the schema doesn't match — it never silently overwrites learner data.

Run standalone:
    python mcp_servers/notes_server.py

Inspect interactively:
    mcp dev mcp_servers/notes_server.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTES_PATH = REPO_ROOT / "data" / "notes.json"

mcp = FastMCP("notes-server")


def _load_notes() -> dict[str, dict]:
    """Return {key: {content, created_at}} or {} if file is missing.

    On schema mismatch, prints to stderr and exits non-zero so the
    caller (notebook subprocess or `mcp dev`) sees the error.
    """
    if not NOTES_PATH.exists():
        return {}
    try:
        raw = json.loads(NOTES_PATH.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"notes-server: {NOTES_PATH} is not valid JSON ({exc}); "
            f"refusing to start. Inspect or delete the file manually.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not isinstance(raw, dict):
        print(
            f"notes-server: {NOTES_PATH} top level must be an object "
            f"({{key: {{content, created_at}}}}); refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "content" not in entry or "created_at" not in entry:
            print(
                f"notes-server: malformed entry for key '{key}' in "
                f"{NOTES_PATH}; expected {{content, created_at}}. "
                f"Refusing to start.",
                file=sys.stderr,
            )
            sys.exit(2)
    return raw


def _save_notes(notes: dict[str, dict]) -> None:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(json.dumps(notes, indent=2, sort_keys=True))


@mcp.tool()
def add_note(key: str, content: str) -> str:
    """Add or overwrite a note with the given key."""
    notes = _load_notes()
    notes[key] = {
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_notes(notes)
    return f"stored '{key}'"


@mcp.tool()
def get_note(key: str) -> str:
    """Return the content of the note with the given key, or an error message."""
    notes = _load_notes()
    if key not in notes:
        return f"key not found: {key}"
    return notes[key]["content"]


@mcp.tool()
def list_notes() -> str:
    """Return all note keys, one per line, sorted alphabetically."""
    notes = _load_notes()
    if not notes:
        return "(no notes)"
    return "\n".join(sorted(notes.keys()))


@mcp.tool()
def delete_note(key: str) -> str:
    """Delete the note with the given key. Returns whether it existed."""
    notes = _load_notes()
    if key not in notes:
        return f"key not found: {key}"
    del notes[key]
    _save_notes(notes)
    return f"deleted '{key}'"


@mcp.resource("notes://{key}")
def note_resource(key: str) -> str:
    """Read a single note as a resource."""
    notes = _load_notes()
    if key not in notes:
        raise ValueError(f"unknown note: {key}")
    return notes[key]["content"]


@mcp.prompt()
def summarize_notes() -> str:
    """Prompt template that asks the agent to digest all current notes."""
    notes = _load_notes()
    if not notes:
        return "There are no notes to summarize."
    body = "\n\n".join(
        f"### {key}\n{entry['content']}" for key, entry in sorted(notes.items())
    )
    return (
        "Below are the user's saved notes. Produce a concise digest "
        "grouping related items and surfacing recurring themes. Use "
        "neutral, direct language.\n\n"
        f"{body}"
    )


if __name__ == "__main__":
    mcp.run()
