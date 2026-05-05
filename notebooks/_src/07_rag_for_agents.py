# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .agents
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NB 07 — RAG for agents
#
# **Goal:** make retrieval a tool the agent *decides* to use, not
# unconditional context injection. We chunk + embed this curriculum's
# own source (notebooks/_src + specs) into Chroma, then run two
# variants:
#
# 1. **Naive RAG** — always inject top-3 chunks. Wastes tokens when the
#    question doesn't need retrieval.
# 2. **Tool-shaped retrieval** — `search_notes(query)` is a tool the
#    agent calls when it judges it useful. The agent answers
#    weather-style questions without hitting the corpus.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="07 — RAG for agents",
    estimate="$0.03 (+ ~80 MB one-time embedding model download)",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Build the corpus
#
# Glob `notebooks/_src/*.py` and `docs/superpowers/specs/*.md`. Chunk
# by paragraphs (target ~500 tokens, ~2000 chars). Embed and index.

# %%
from pathlib import Path

from agentlab.memory import SemanticMemory

CHUNK_CHARS = 2000

def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 > max_chars and buf:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


sources = sorted(
    list(Path("notebooks/_src").glob("*.py"))
    + list(Path("docs/superpowers/specs").glob("*.md"))
)
print(f"Source files: {len(sources)}")

corpus = SemanticMemory(collection_name="agents_curriculum")
chunk_count = 0
for src in sources:
    text = src.read_text(encoding="utf-8")
    for i, chunk in enumerate(chunk_text(text)):
        corpus.add(chunk, metadata={"source": str(src), "chunk_index": i})
        chunk_count += 1

print(f"Indexed {chunk_count} chunks across {len(sources)} files.")

# %% [markdown]
# ## Step 2 — Naive RAG (always inject top-3)
#
# Run two questions: one the corpus can answer ("how does NB 02
# register tools?"), one it cannot ("what's today's weather in
# Toronto?"). Watch the second waste tokens on irrelevant context.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client

client = get_client()


def naive_rag(question: str) -> str:
    matches = corpus.query(question, top_k=3)
    context = "\n\n".join(f"[{i}] {m.text[:600]}" for i, m in enumerate(matches))
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        system="Answer using the provided context. If the context is irrelevant, say so.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return "".join(getattr(b, "text", "") for b in response.content)


for q in [
    "How does NB 02 register tools with the registry?",
    "What's the weather in Toronto today?",
]:
    print(f"\nQ: {q}")
    print(f"A: {naive_rag(q)[:400]}")

# %% [markdown]
# Notice the second answer wasted tokens on irrelevant chunks. We want
# the agent to *decide* whether to retrieve.

# %% [markdown]
# ## Step 3 — Retrieval as a tool
#
# Expose `search_notes(query)` via the tool registry. The agent calls
# it when it judges retrieval would help.

# %%
from agentlab.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool(description="Search the curriculum's own source code and specs for relevant context. Use when the question is about how this codebase works.")
def search_notes(query: str) -> str:
    matches = corpus.query(query, top_k=5)
    blocks = []
    for m in matches:
        src = m.metadata.get("source")
        idx = m.metadata.get("chunk_index")
        # Synthesize an HttpUrl-shaped identifier so callers can cite it via Citation.url.
        url = f"https://corpus.local/{src}#chunk={idx}"
        blocks.append(f"[source url: {url}]\n[path: {src}]\n{m.text}")
    return "\n\n".join(blocks)


from agentlab.llm import run_agent_loop

for q in [
    "How does NB 02 register tools with the registry?",
    "What's the weather in Toronto today?",
]:
    print(f"\n=== Q: {q} ===")
    result = run_agent_loop(
        client=client,
        system=(
            "You are a helpful assistant. If the question is about this "
            "codebase, call search_notes. If it's a general or current-events "
            "question, answer from your own knowledge or say you can't."
        ),
        user_message=q,
        tools=registry.schemas(),
        tool_handlers=registry.handlers(),
        max_turns=8,
    )
    print(f"Turns: {result.turns}")
    print(f"Answer: {result.final_text[:400]}")

# %% [markdown]
# Look at the turn counts. The codebase question used `search_notes`;
# the weather question didn't — the agent (correctly) noted it can't
# answer that with this tool.

# %% [markdown]
# ## Step 4 — Citations
#
# When the agent retrieves, we want it to cite which chunk it used.
# Reuse NB 03's `Answer` model so the contract is consistent.

# %%
from agentlab.types import Answer

submit_answer_tool = {
    "name": "submit_answer",
    "description": (
        "Submit final answer with citations to the chunks you used. "
        "Each citation's url MUST be the 'source url' shown above each chunk in "
        "the search results. At least one citation is required."
    ),
    "input_schema": Answer.model_json_schema(),
}


def cited_research(question: str, max_turns: int = 6) -> Answer:
    messages = [{"role": "user", "content": question}]
    tools = registry.schemas() + [submit_answer_tool]
    handlers = registry.handlers()
    for turn in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=4096,
            system=(
                "Answer questions about this codebase.\n"
                "1. Call search_notes to find relevant chunks.\n"
                "2. Each chunk in the search results begins with a `[source url: ...]` "
                "line — use exactly that URL as the citation url. Use the file path "
                "as the citation title, and a short verbatim snippet as the quote.\n"
                "3. Always finish by calling submit_answer with at least one citation. "
                "Do not invent URLs — only cite urls that appeared in the search results."
            ),
            tools=tools, messages=messages,
        )
        block_descs = [
            f"{getattr(b, 'type', '?')}"
            + (f":{b.name}" if getattr(b, "name", None) else "")
            for b in response.content
        ]
        print(
            f"  [debug turn {turn}] stop={response.stop_reason} "
            f"out_tokens={response.usage.output_tokens} blocks={block_descs}"
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                print(f"  [debug] submit_answer keys={list(block.input.keys())}")
                return Answer.model_validate(block.input)
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name in handlers:
                try:
                    out = handlers[block.name](**block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(out)})
                except Exception as e:
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True})
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("Out of turns.")


answer = cited_research("How does NB 03 use Pydantic to enforce structured output?")
print(answer.summary)
print()
for c in answer.citations:
    print(f"  • {c.title} — {c.url}")
    if c.quote:
        print(f"      \"{c.quote[:140]}\"")

# %% [markdown]
# ## Reflect
#
# - **Naive RAG injects context unconditionally — the agent has no
#   choice.** Tokens wasted on irrelevant matches; latency spent on
#   embeddings the agent didn't need.
# - **Tool-shaped retrieval restores agency.** The agent uses search
#   when the question warrants it, skips it otherwise.
# - **Citations are cheap with Pydantic.** Reusing `Answer` from NB 03
#   meant we got typed citations for free.
# - **Self-referential corpus is fun for learning** but real corpora
#   need re-indexing on document change. Production: a content hash
#   tracker; this NB just re-indexes on each run.
# - **Next:** NB 08 makes "is the agent working?" measurable with evals
#   + traces.
