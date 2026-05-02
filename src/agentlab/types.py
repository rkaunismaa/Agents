"""Shared Pydantic types used across notebooks.

These types model the structured output of the recurring research-assistant
spine project introduced in NB 03 and reused in later modules.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class Citation(BaseModel):
    """A single source citation."""

    url: HttpUrl
    title: str = Field(..., min_length=1)
    quote: str | None = None

    model_config = {"frozen": True}


class Answer(BaseModel):
    """A researched answer with at least one citation."""

    summary: str = Field(..., min_length=1)
    citations: list[Citation] = Field(..., min_length=1)
