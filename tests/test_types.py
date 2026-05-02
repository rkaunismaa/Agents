"""Tests for agentlab.types — shared Pydantic types."""
import pytest
from pydantic import ValidationError

from agentlab.types import Answer, Citation


def test_citation_minimum_fields():
    c = Citation(url="https://example.com", title="Example")
    # Pydantic v2 HttpUrl normalises bare hosts by adding a trailing slash,
    # so we compare the string form against the normalised URL.
    assert str(c.url) == "https://example.com/"
    assert c.title == "Example"
    assert c.quote is None


def test_citation_with_quote():
    c = Citation(url="https://example.com", title="Example", quote="lorem ipsum")
    assert c.quote == "lorem ipsum"


def test_citation_rejects_non_url():
    with pytest.raises(ValidationError):
        Citation(url="not-a-url", title="x")


def test_answer_with_citations():
    a = Answer(
        summary="Bears eat fish.",
        citations=[Citation(url="https://example.com/bears", title="Bears 101")],
    )
    assert a.summary == "Bears eat fish."
    assert len(a.citations) == 1


def test_answer_requires_at_least_one_citation():
    with pytest.raises(ValidationError):
        Answer(summary="Bears eat fish.", citations=[])


def test_answer_serializes_to_json():
    a = Answer(
        summary="Bears eat fish.",
        citations=[Citation(url="https://example.com/", title="x")],
    )
    payload = a.model_dump_json()
    assert "Bears eat fish." in payload
    assert "example.com" in payload
