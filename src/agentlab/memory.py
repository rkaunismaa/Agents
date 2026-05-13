"""Memory primitives for agentic workflows.

Three flavors, one file:

- ConversationBuffer — short-term, token-bounded, optional summarization.
- KeyValueMemory     — long-term, dict + JSON persistence.
- SemanticMemory     — embedding-based recall via Chroma + sentence-transformers.

All three are in one file because NB 06 compares them side by side. They are
intentionally independent — no shared base class — so each can be understood
in isolation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _approx_tokens(text: str) -> int:
    """Estimate the token count for a plain-text string.

    Returns at least 1 so callers never divide-by-zero or allocate zero budget.
    """
    # ~4 chars per token is the standard rule of thumb for English text with
    # Claude/GPT tokenizers. Exact tokenization would require the Anthropic
    # tokenizer library, which is an unnecessary dependency for buffer sizing.
    return max(1, len(text) // 4)


def _message_tokens(message: dict) -> int:
    """Estimate the token count for a single Anthropic message dict.

    Handles both plain-string content and structured content-block lists
    (the two shapes Anthropic messages can take).
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return _approx_tokens(content)
    # Anthropic content blocks (list of dicts): tool_use, tool_result, text.
    # Serialize each to JSON for a conservative size estimate.
    return sum(_approx_tokens(json.dumps(block)) for block in content)


class ConversationBuffer:
    """Token-bounded message buffer.

    Stores messages in order. `truncate()` returns the newest suffix that fits
    within `max_tokens`; `summarize()` (used in the next step) rolls older
    messages into a single system summary.
    """

    def __init__(self, max_tokens: int = 4096) -> None:
        """Create a buffer that will hold at most max_tokens worth of messages."""
        self.max_tokens = max_tokens
        self._messages: list[dict] = []

    def append(self, message: dict) -> None:
        """Add a message to the end of the buffer; does not enforce the token limit."""
        self._messages.append(message)

    def messages(self) -> list[dict]:
        """Return a copy of all stored messages in chronological order, without truncation."""
        return list(self._messages)

    def truncate(self) -> list[dict]:
        """Return the most recent messages that fit within max_tokens.

        Older messages are dropped when the budget is exceeded. The result is
        always a contiguous tail of the full message list, preserving turn order.
        """
        # Walk from newest to oldest, keeping messages until the budget runs
        # out. This is a recency bias: for agent conversations the most recent
        # exchanges are more relevant than older context.
        # Edge case: if a single message exceeds the budget and `kept` is empty,
        # we still include it (the `and kept` guard) so we never return nothing.
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
        # Haiku is used here deliberately: summarization doesn't need a capable
        # model, and Haiku is substantially cheaper and faster than Sonnet.
        # The summary replaces a potentially long conversation history, so
        # quality matters less than cost.
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
        """Create an empty in-memory store."""
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        """Store value under key, overwriting any existing value for that key."""
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve the value stored under key, or default if the key is absent."""
        return self._store.get(key, default)

    def keys(self) -> list[str]:
        """Return a snapshot of all currently stored key names."""
        return list(self._store.keys())

    def save(self, path: Path) -> None:
        """Persist to JSON. All values must be JSON-serializable; raises TypeError otherwise."""
        Path(path).write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        """Replace the store's contents with key-value pairs read from a JSON file."""
        self._store = json.loads(Path(path).read_text(encoding="utf-8"))


class Match(BaseModel):
    """A semantic-search hit.

    `score` is cosine *similarity* in [-1, 1]; higher = more similar.
    Note: Chroma returns cosine *distance*; SemanticMemory.query converts
    it to similarity via `1.0 - distance`.
    """

    text: str
    score: float
    metadata: dict


def _default_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Build and return a sentence-transformers embedding function.

    Returns a callable that takes a list of strings and returns a list of
    float vectors. The model is loaded once and closed over; subsequent calls
    reuse the loaded weights.
    """
    # sentence_transformers is a heavy import (~500 ms, large model files).
    # Lazy-loading it here means notebooks that never use SemanticMemory
    # don't pay the import cost at module load time.
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

        Uses chromadb.EphemeralClient (in-process, no server, no disk I/O).
        This is the right choice for notebook demos; swap to PersistentClient
        for a store that survives kernel restarts — but remove the
        delete_collection call below to avoid wiping data on re-instantiation.
        """
        import chromadb

        self._client = chromadb.EphemeralClient()
        self._embedder = embedder or _default_embedder(embedder_model)

        # Drop any existing collection with this name before creating a fresh
        # one. Without this, re-running a notebook cell that constructs
        # SemanticMemory would raise "collection already exists" from Chroma.
        # The try/except swallows the "does not exist" case on first run.
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=collection_name,
            # cosine space: queries return cosine distance, which we convert
            # to similarity in query(). Without this, Chroma defaults to L2
            # distance, which doesn't work well for high-dimensional embeddings.
            metadata={"hnsw:space": "cosine"},
        )
        self._next_id = 0

    def add(self, text: str, metadata: dict | None = None) -> str:
        """Embed text and add it to the vector store; return the assigned document ID."""
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
        """Return the top_k stored documents most similar to text, ranked by cosine similarity."""
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
            # Chroma returns cosine distance (0 = identical, 2 = opposite).
            # Convert to similarity (1 = identical, -1 = opposite) so callers
            # get an intuitive "higher score = better match" interface.
            out.append(Match(text=doc, score=1.0 - float(dist), metadata=meta or {}))
        return out
