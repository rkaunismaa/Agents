"""Pytest fixtures shared across the test suite."""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load .env once at the start of the test session if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@pytest.fixture
def has_api_key() -> bool:
    """True if ANTHROPIC_API_KEY is set; tests that need a real call check this."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
