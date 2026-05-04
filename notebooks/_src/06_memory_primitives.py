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
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 06 — Memory primitives (✦ spine touchpoint)
#
# **Goal:** introduce three memory shapes via `agentlab.memory`:
# `ConversationBuffer`, `KeyValueMemory`, `SemanticMemory`. Then layer
# `KeyValueMemory` into the research-assistant spine so it can answer
# follow-up questions.
#
# Heads up: NB 07 will depend on `SemanticMemory`. NB 08 will trace
# everything.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="06 — Memory primitives (✦ spine + memory)",
    estimate="$0.03",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Short-term: `ConversationBuffer`
#
# Token-bounded message buffer with optional summarization. The simplest
# memory: just keep the last N tokens of conversation.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.memory import ConversationBuffer

client = get_client()

buf = ConversationBuffer(max_tokens=80)  # tiny on purpose: forces truncation
for i in range(8):
    buf.append({"role": "user", "content": f"This is message {i}, which has some words in it for token weight."})
    buf.append({"role": "assistant", "content": f"Acknowledged message {i}."})

print(f"All messages: {len(buf.messages())}")
truncated = buf.truncate()
dropped = len(buf.messages()) - len(truncated)
print(f"After truncate: {len(truncated)} ({dropped} oldest dropped to fit budget)")
print("Oldest kept:", truncated[0]["content"][:60])
print("Newest kept:", truncated[-1]["content"][:60])

# %% [markdown]
# ### Summarization keeps context at a price
#
# `truncate()` *drops* old messages. `summarize()` *compresses* them
# into a single summary string. Use it when the conversation has
# load-bearing context older than your budget.

# %%
summary = buf.summarize(client)
print("Summary:", summary)

# %% [markdown]
# ## Step 2 — Long-term: `KeyValueMemory`
#
# Persistent dict. Use cases: user preferences, learned facts,
# pre-computed results. `save / load` round-trip via JSON.

# %%
from pathlib import Path

from agentlab.memory import KeyValueMemory

kv = KeyValueMemory()
kv.set("user_name", "Rob")
kv.set("project_focus", "AI agents curriculum")
kv.set("last_question", "How do I build NB 06?")

scratch = Path(".kv-demo.json")
kv.save(scratch)
print(scratch.read_text())

fresh = KeyValueMemory()
fresh.load(scratch)
assert fresh.get("user_name") == "Rob"
print(f"\nRe-loaded: user_name = {fresh.get('user_name')}")

scratch.unlink()

# %% [markdown]
# ## Step 3 — Semantic recall: `SemanticMemory`
#
# Embedding-backed retrieval. The first run downloads the
# sentence-transformers model (~80 MB). Subsequent runs are fast.

# %%
from agentlab.memory import SemanticMemory

sm = SemanticMemory(collection_name="nb06_demo")
sm.add("Anthropic's Claude API uses an HTTP-based message protocol.", metadata={"topic": "api"})
sm.add("MCP (Model Context Protocol) standardizes how tools are exposed to LLMs.", metadata={"topic": "mcp"})
sm.add("ReAct interleaves reasoning and acting in a loop.", metadata={"topic": "react"})
sm.add("Pydantic validates structured outputs from LLMs.", metadata={"topic": "structured"})

for query in ["How does the Anthropic API work?", "What is the protocol for tool exposure?"]:
    matches = sm.query(query, top_k=2)
    print(f"\nQuery: {query}")
    for m in matches:
        print(f"  [{m.score:.3f}] {m.text}  (topic={m.metadata.get('topic')})")

# %% [markdown]
# ## Step 4 — Spine: research assistant with memory
#
# We redefine `research()` from NB 03 inline (per the curriculum's
# "no library extraction until NB 15" rule), but this time the
# assistant has a `KeyValueMemory` it can read on each call. We expose
# two extra tools: `remember(key, value)` and `recall(key)`.

# %%
from agentlab.tools import ToolRegistry
from agentlab.types import Answer, Citation

registry = ToolRegistry()
session_memory = KeyValueMemory()


@registry.tool(description="Save a fact for later retrieval. Use when the user mentions a preference or fact.")
def remember(key: str, value: str) -> str:
    session_memory.set(key, value)
    return f"Saved {key}={value}"


@registry.tool(description="Recall a previously-saved fact by key. Returns the stored value or 'not found'.")
def recall(key: str) -> str:
    val = session_memory.get(key)
    return str(val) if val is not None else "not found"


submit_answer_tool = {
    "name": "submit_answer",
    "description": "Submit your final answer with citations.",
    "input_schema": Answer.model_json_schema(),
}

SPINE_SYSTEM = """You are a research assistant with persistent memory.

Workflow (do these in order, every turn):
1. If the user mentions a preference, project, or fact about themselves, call remember(key, value) to save it.
2. If a follow-up question references something earlier, call recall(key) to look it up.
3. ALWAYS call web_search at least once before answering — even for recommendations or
   follow-ups, you need fresh sources to cite. Do not skip this step.
4. Finish by calling submit_answer exactly once with:
   - summary: 2-4 sentences answering the question (weave in any recalled facts).
   - citations: list of {url, title, optional quote} drawn from your web_search results.
     At least one entry; ideally 2-3. URLs must come from web_search — do not invent any.
5. Do not produce free-form text instead of submit_answer — your final output must be a submit_answer call."""


def research_with_memory(question: str, max_turns: int = 6) -> Answer:
    # Surface what's in memory so the agent doesn't have to guess key names.
    keys = session_memory.keys()
    system = SPINE_SYSTEM + (
        f"\n\nCurrent memory keys (call recall(key) to read each value): {keys}"
        if keys
        else "\n\nMemory is currently empty."
    )
    messages = [{"role": "user", "content": question}]
    tools = registry.schemas() + [
        {"type": "web_search_20250305", "name": "web_search"},
        submit_answer_tool,
    ]
    handlers = registry.handlers()
    for turn in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=2048, system=system,
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
                return Answer.model_validate(block.input)
        # Run any local tool calls; web_search is managed.
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
        if response.stop_reason == "end_turn":
            raise RuntimeError("end_turn without submit_answer.")
    raise RuntimeError("Out of turns.")


# Turn 1: introduce a preference.
answer1 = research_with_memory(
    "I'm working on a Python async-first project. Remember that. "
    "What's the most popular async HTTP library?"
)
print("Q1 →", answer1.summary[:200])
print(f"\n[memory now contains: {session_memory.keys()}]")

# Turn 2: follow-up that needs the remembered context.
answer2 = research_with_memory(
    "Given my project's focus, which library do you recommend I use? "
    "(Look up what you remembered.)"
)
print("\nQ2 →", answer2.summary[:300])

# %% [markdown]
# ## Reflect
#
# - **Three memory shapes, three jobs.** Conversation buffers handle
#   token budgets; key-value stores hold facts; semantic memory finds
#   things by meaning.
# - **Memory becomes a tool the agent uses.** `remember/recall` are
#   plain `ToolRegistry` tools — the agent decides when to use them.
# - **Why `KeyValueMemory` for the spine, not `ConversationBuffer`?**
#   Because the spine is multi-turn but stateless across calls — we
#   want explicit, named facts (preferences, project context), not
#   raw transcripts.
# - **Foreshadow:** NB 07 puts `SemanticMemory` to work — but exposed as
#   a *retrieval tool*, not unconditional context injection.
