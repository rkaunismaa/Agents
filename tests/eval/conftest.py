"""Fixtures for the agent eval suite."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_TASKS_PATH = REPO_ROOT / "data" / "eval_tasks.jsonl"


def _load_tasks() -> list[dict]:
    if not EVAL_TASKS_PATH.exists():
        return []
    with EVAL_TASKS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pytest_collection_modifyitems(config, items):
    """Skip eval tests if ANTHROPIC_API_KEY is missing."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    """Inject the eval task list into any test that takes a `task` parameter."""
    if "task" in metafunc.fixturenames:
        tasks = _load_tasks()
        metafunc.parametrize("task", tasks, ids=[t["id"] for t in tasks])


@pytest.fixture(scope="session")
def eval_tasks() -> list[dict]:
    return _load_tasks()


@pytest.fixture(scope="session")
def client():
    from agentlab.llm import get_client
    return get_client()
