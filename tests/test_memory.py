"""TDD tests for agentlab.memory."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ── ConversationBuffer ─────────────────────────────────────────────


def test_conversation_buffer_appends_and_returns_messages():
    from agentlab.memory import ConversationBuffer

    buf = ConversationBuffer(max_tokens=1000)
    buf.append({"role": "user", "content": "hello"})
    buf.append({"role": "assistant", "content": "hi"})

    assert buf.messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_conversation_buffer_truncates_oldest_first_to_fit_token_budget():
    from agentlab.memory import ConversationBuffer

    buf = ConversationBuffer(max_tokens=20)  # ~5 tokens per "word " * 4 = ~20
    for i in range(10):
        buf.append({"role": "user", "content": f"message-{i} " * 4})

    truncated = buf.truncate()
    # Truncation drops oldest messages until token estimate fits.
    assert len(truncated) < 10
    # Newest message must be retained.
    assert truncated[-1]["content"].startswith("message-9")


def test_conversation_buffer_summarize_calls_client_with_buffer():
    from agentlab.memory import ConversationBuffer

    captured: dict = {}

    class _StubResponse:
        content = [type("B", (), {"text": "User asked about X. Assistant answered Y."})()]

    class _StubClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                return _StubResponse()

    buf = ConversationBuffer(max_tokens=1000)
    buf.append({"role": "user", "content": "What is X?"})
    buf.append({"role": "assistant", "content": "Y."})

    summary = buf.summarize(_StubClient())

    assert "X" in summary and "Y" in summary
    # The buffer's own messages should appear inside the prompt sent to the client.
    sent = json.dumps(captured)
    assert "What is X?" in sent


# ── KeyValueMemory ─────────────────────────────────────────────────


def test_key_value_memory_set_and_get():
    from agentlab.memory import KeyValueMemory

    kv = KeyValueMemory()
    kv.set("user_name", "Rob")
    kv.set("favorite_color", "purple")

    assert kv.get("user_name") == "Rob"
    assert kv.get("favorite_color") == "purple"
    assert kv.get("missing") is None
    assert kv.get("missing", default="fallback") == "fallback"


def test_key_value_memory_save_and_load_roundtrip(tmp_path: Path):
    from agentlab.memory import KeyValueMemory

    kv = KeyValueMemory()
    kv.set("a", 1)
    kv.set("b", {"nested": [1, 2, 3]})

    target = tmp_path / "kv.json"
    kv.save(target)

    fresh = KeyValueMemory()
    fresh.load(target)

    assert fresh.get("a") == 1
    assert fresh.get("b") == {"nested": [1, 2, 3]}


# ── SemanticMemory ─────────────────────────────────────────────────


class _StubEmbedder:
    """Deterministic 4-d embedder for tests. Avoids loading sentence-transformers."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), float(len(t) % 11), float(t.count("a")), float(t.count("e"))] for t in texts]


def test_semantic_memory_add_and_query_returns_match_with_metadata():
    from agentlab.memory import SemanticMemory

    sm = SemanticMemory(collection_name="test_basic", embedder=_StubEmbedder())
    sm.add("Anthropic released Claude Opus 4.7 in 2026.", metadata={"source": "doc-1"})
    sm.add("MCP is the Model Context Protocol.", metadata={"source": "doc-2"})

    matches = sm.query("Anthropic released Claude Opus 4.7 in 2026.", top_k=1)
    assert len(matches) == 1
    assert matches[0].metadata["source"] == "doc-1"
    assert "Anthropic" in matches[0].text
    assert isinstance(matches[0].score, float)


def test_semantic_memory_query_respects_top_k():
    from agentlab.memory import SemanticMemory

    sm = SemanticMemory(collection_name="test_top_k", embedder=_StubEmbedder())
    varied = ["a", "ab", "abc", "abcd", "abcde"]
    for i, text in enumerate(varied):
        sm.add(text, metadata={"i": i})

    matches = sm.query("abc", top_k=3)
    assert len(matches) == 3


def test_semantic_memory_add_without_metadata():
    from agentlab.memory import SemanticMemory

    sm = SemanticMemory(collection_name="test_no_meta", embedder=_StubEmbedder())
    doc_id = sm.add("no metadata provided")

    assert doc_id == "doc-0"
    matches = sm.query("no metadata provided", top_k=1)
    assert len(matches) == 1
    assert matches[0].text == "no metadata provided"
    assert matches[0].metadata == {}
