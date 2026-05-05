"""A copy of the spine `research()` function for the eval suite to exercise.

Per the curriculum design, the spine stays inline in notebooks until
NB 15 (capstone). The eval suite needs *something* to import, so it
keeps its own copy here. NB 15 will unify all of these.
"""
from __future__ import annotations

from agentlab.llm import DEFAULT_MODEL
from agentlab.types import Answer


SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final answer with citations.",
    "input_schema": Answer.model_json_schema(),
}

SYSTEM_PROMPT = """You are a careful technical research assistant.

Workflow:
1. If you need facts you aren't sure of, use web_search.
2. When ready, call submit_answer with summary + citations.
3. Do not produce free-form text after submit_answer.
"""


def research(client, question: str, max_turns: int = 8) -> tuple[Answer, list[str]]:
    """Run the spine. Returns (Answer, list of tool names called)."""
    tool_calls: list[str] = []
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[
                {"type": "web_search_20250305", "name": "web_search"},
                SUBMIT_ANSWER_TOOL,
            ],
            messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(block.name)
                if block.name == "submit_answer":
                    return Answer.model_validate(block.input), tool_calls
            if block.type == "server_tool_use":
                tool_calls.append(getattr(block, "name", "server_tool"))
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            raise RuntimeError("end_turn without submit_answer.")
    raise RuntimeError(f"Out of turns ({max_turns}).")
