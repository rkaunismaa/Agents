"""Three eval styles on the spine research assistant.

Run with: `pytest -m eval tests/eval/ -v`
"""
from __future__ import annotations

import pytest

from agentlab.types import Answer
from tests.eval.research_under_test import research


# ── Eval Style 1 — deterministic asserts ───────────────────────────


@pytest.mark.eval
def test_research_deterministic(client, task):
    """Asserts: returned object validates as Answer, expected tools were called,
    at least one citation present."""
    answer, tool_calls = research(client, task["question"])

    assert isinstance(answer, Answer)
    assert len(answer.citations) >= 1, "expected at least one citation"

    for expected in task["expected_tool_calls"]:
        assert expected in tool_calls, (
            f"expected tool '{expected}' to be called; saw {tool_calls}"
        )


# ── Eval Style 2 — LLM-as-judge ────────────────────────────────────


JUDGE_SYSTEM = """You are an evaluator. Score the answer 1-5 against the rubric:
1 = wildly wrong / off-topic
2 = partially relevant but missing key facts
3 = correct but minimal
4 = correct, well-cited, complete
5 = exemplary

Output ONLY the integer score on its own line."""


@pytest.mark.eval
def test_research_llm_judge(client, task):
    answer, _ = research(client, task["question"])

    judge_prompt = (
        f"Question: {task['question']}\n\n"
        f"Rubric: {task['judge_rubric']}\n\n"
        f"Answer summary: {answer.summary}\n\n"
        f"Citations: {[c.url for c in answer.citations]}"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content).strip()
    score = int(text.split()[0])

    # Treat scores ≤ 2 as failures; 3+ as passes (with note).
    assert score >= 3, f"judge gave {score}/5 for task {task['id']}"


# ── Eval Style 3 — reference-based (substring/exact match) ─────────


@pytest.mark.eval
def test_research_reference(client, task):
    if task["reference_answer"] is None:
        pytest.skip("no reference answer for this task")

    answer, _ = research(client, task["question"])
    ref = task["reference_answer"].lower()
    summary = answer.summary.lower()
    cited = " ".join(c.title.lower() + " " + (c.quote or "").lower() for c in answer.citations)

    assert ref in summary or ref in cited, (
        f"reference '{task['reference_answer']}' not found in summary or citations"
    )
