"""Memory primitives for agentic workflows.

Three flavors, one file:

- ConversationBuffer — short-term, token-bounded, optional summarization.
- KeyValueMemory     — long-term, dict + JSON persistence.
- SemanticMemory     — embedding-based recall via Chroma + sentence-transformers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Good enough for buffer sizing."""
    return max(1, len(text) // 4)


def _message_tokens(message: dict) -> int:
    content = message.get("content", "")
    if isinstance(content, str):
        return _approx_tokens(content)
    # Anthropic content blocks (list of dicts).
    return sum(_approx_tokens(json.dumps(block)) for block in content)


class ConversationBuffer:
    """Token-bounded message buffer.

    Stores messages in order. `truncate()` returns the newest suffix that fits
    within `max_tokens`; `summarize()` (used in the next step) rolls older
    messages into a single system summary.
    """

    def __init__(self, max_tokens: int = 4096) -> None:
        self.max_tokens = max_tokens
        self._messages: list[dict] = []

    def append(self, message: dict) -> None:
        self._messages.append(message)

    def messages(self) -> list[dict]:
        return list(self._messages)

    def truncate(self) -> list[dict]:
        kept: list[dict] = []
        budget = self.max_tokens
        for m in reversed(self._messages):
            cost = _message_tokens(m)
            if cost > budget and kept:
                break
            kept.append(m)
            budget -= cost
        kept.reverse()
        return kept

    def summarize(self, client) -> str:
        """Roll the buffer into a single short summary string via Claude."""
        if not self._messages:
            return ""
        rendered = "\n".join(
            f"{m['role']}: {m['content']}" if isinstance(m['content'], str) else f"{m['role']}: <blocks>"
            for m in self._messages
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system="Summarize the conversation in 2-3 sentences. Preserve key facts and decisions.",
            messages=[{"role": "user", "content": rendered}],
        )
        return response.content[0].text


class KeyValueMemory:
    """Long-term key-value store with JSON persistence."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def save(self, path: Path) -> None:
        """Persist to JSON. All values must be JSON-serializable; raises TypeError otherwise."""
        Path(path).write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        self._store = json.loads(Path(path).read_text(encoding="utf-8"))


class Match(BaseModel):
    """A semantic-search hit. `score` is cosine similarity in [-1, 1]; higher = more similar."""

    text: str
    score: float
    metadata: dict


def _default_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load sentence-transformers so import doesn't pay for it."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, convert_to_numpy=False, show_progress_bar=False)

    return embed


class SemanticMemory:
    """Embedding-backed recall via Chroma.

    Pass `embedder=callable` (taking list[str], returning list[list[float]]) to
    override the default sentence-transformers embedder — useful in tests.
    """

    def __init__(
        self,
        collection_name: str = "default",
        embedder=None,
        embedder_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Construct a fresh in-process semantic store.

        Note: Uses chromadb.EphemeralClient (process-local). On construction, any
        existing collection with the same name is dropped. If you swap to
        PersistentClient later, remove the delete_collection call to avoid data loss.
        """
        import chromadb

        self._client = chromadb.EphemeralClient()
        self._embedder = embedder or _default_embedder(embedder_model)
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._next_id = 0

    def add(self, text: str, metadata: dict | None = None) -> str:
        doc_id = f"doc-{self._next_id}"
        self._next_id += 1
        embedding = self._embedder([text])[0]
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[list(embedding)],
            metadatas=[metadata] if metadata else None,
        )
        return doc_id

    def query(self, text: str, top_k: int = 3) -> list[Match]:
        embedding = self._embedder([text])[0]
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
        )
        out: list[Match] = []
        metadatas = result["metadatas"][0] if result["metadatas"] is not None else [None] * len(result["documents"][0])
        for doc, dist, meta in zip(
            result["documents"][0],
            result["distances"][0],
            metadatas,
        ):
            out.append(Match(text=doc, score=1.0 - float(dist), metadata=meta or {}))
        return out
