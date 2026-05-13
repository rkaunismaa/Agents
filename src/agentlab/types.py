"""Shared Pydantic types used across notebooks.

These types model the structured output of the recurring research-assistant
spine project introduced in NB 03 and reused in later modules.

Why two models instead of one flat dict?
  - Citation is reusable: any other response type can embed it.
  - Answer is the spine's *output contract*: every notebook that builds on
    the spine agrees that a valid response has a summary and at least one
    citation. Pydantic enforces this at parse time, not at the call site.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class Citation(BaseModel):
    """A single source citation."""

    # HttpUrl triggers Pydantic URL validation. This catches the common
    # model hallucination of returning a bare string like "python.org"
    # instead of a valid "https://python.org" URL.
    url: HttpUrl

    # min_length=1 prevents the model from satisfying the schema with an
    # empty string while technically passing validation.
    title: str = Field(..., min_length=1)
    quote: str | None = None

    # frozen=True makes Citation instances immutable and hashable.
    # Immutability is the right default for value objects: a citation
    # identifies a source; it shouldn't be mutated after creation.
    model_config = {"frozen": True}


class Answer(BaseModel):
    """A researched answer with at least one citation."""

    # min_length=1 guards against an empty summary passing the schema
    # check. Without it, {"summary": "", "citations": [...]} is valid JSON
    # that satisfies the schema but is useless as a response.
    summary: str = Field(..., min_length=1)

    # min_length=1 on the list enforces "at least one citation". An answer
    # with zero citations cannot be attributed and should not be accepted.
    citations: list[Citation] = Field(..., min_length=1)
