"""Tests for agentlab.spine: Subagent dataclass + Orchestrator.

We do not call any LLM here. The Orchestrator under test is dispatched
via an injected `dispatch` callable so workers are deterministic
fakes.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentlab.spine import Orchestrator, Subagent, WorkerResult


# ── Fixtures ───────────────────────────────────────────────────────


def _make_workers() -> list[Subagent]:
    return [
        Subagent(role="researcher", system_prompt="be a researcher"),
        Subagent(role="summarizer", system_prompt="be a summarizer"),
        Subagent(role="ranker", system_prompt="be a ranker"),
    ]


# ── Sync run: dispatch is called once per worker, results collected ─


def test_orchestrator_run_dispatches_each_worker_once():
    """In sync mode, every worker runs once and produces a WorkerResult."""
    seen: list[tuple[str, str]] = []

    def fake_dispatch(worker: Subagent, task: str) -> str:
        seen.append((worker.role, task))
        return f"{worker.role}: {task[:20]}"

    orch = Orchestrator(workers=_make_workers(), dispatch=fake_dispatch)
    results = orch.run_sync("compare httpx, aiohttp, requests")

    assert len(results) == 3
    assert {r.role for r in results} == {"researcher", "summarizer", "ranker"}
    assert all(isinstance(r, WorkerResult) for r in results)
    assert all(r.error is None for r in results)
    assert {role for role, _ in seen} == {"researcher", "summarizer", "ranker"}


def test_orchestrator_run_records_dispatch_errors():
    """If dispatch raises, the WorkerResult carries the error string and result is None."""

    def flaky_dispatch(worker: Subagent, task: str) -> str:
        if worker.role == "summarizer":
            raise RuntimeError("boom")
        return f"{worker.role}: ok"

    orch = Orchestrator(workers=_make_workers(), dispatch=flaky_dispatch)
    results = orch.run_sync("test")

    bad = next(r for r in results if r.role == "summarizer")
    good = [r for r in results if r.role != "summarizer"]
    assert bad.result is None
    assert "boom" in bad.error
    assert all(g.error is None and g.result for g in good)


# ── Async run: workers fan out via asyncio.gather ──────────────────


@pytest.mark.asyncio
async def test_orchestrator_run_async_runs_workers_concurrently():
    """Three workers each sleep 0.1s; total wall-clock should be <0.2s."""
    import time

    async def slow_dispatch(worker: Subagent, task: str) -> str:
        await asyncio.sleep(0.1)
        return f"{worker.role}-done"

    orch = Orchestrator(workers=_make_workers(), dispatch_async=slow_dispatch)
    start = time.perf_counter()
    results = await orch.run_async("test")
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    assert elapsed < 0.5, f"async run took {elapsed:.3f}s, expected <0.5s (3× the per-worker sleep)"


@pytest.mark.asyncio
async def test_orchestrator_run_async_timeout_cancels_slow_worker():
    """One worker exceeds its timeout; result has error, others succeed."""

    async def mixed_dispatch(worker: Subagent, task: str) -> str:
        if worker.role == "researcher":
            await asyncio.sleep(0.5)  # will exceed timeout
        else:
            await asyncio.sleep(0.05)
        return f"{worker.role}-done"

    orch = Orchestrator(workers=_make_workers(), dispatch_async=mixed_dispatch)
    results = await orch.run_async("test", worker_timeout=0.1)

    by_role = {r.role: r for r in results}
    assert by_role["researcher"].result is None
    assert "timeout" in by_role["researcher"].error.lower()
    assert by_role["summarizer"].result == "summarizer-done"
    assert by_role["ranker"].result == "ranker-done"


# ── Checkpointing ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_writes_checkpoint_per_worker(tmp_path):
    """Each successful worker appends a JSONL line under checkpoint_dir/run_id.jsonl."""
    async def quick(worker: Subagent, task: str) -> str:
        return f"{worker.role}-result"

    orch = Orchestrator(
        workers=_make_workers(), dispatch_async=quick, checkpoint_dir=tmp_path
    )
    await orch.run_async("test", run_id="abc123")

    checkpoint = tmp_path / "abc123.jsonl"
    assert checkpoint.exists()
    lines = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert {entry["role"] for entry in lines} == {"researcher", "summarizer", "ranker"}
    assert all(entry["run_id"] == "abc123" for entry in lines)


@pytest.mark.asyncio
async def test_orchestrator_resume_skips_completed_workers(tmp_path):
    """Resume reads cached results from JSONL; only re-runs missing workers."""
    # Pre-populate the checkpoint with one completed worker.
    cp_path = tmp_path / "abc123.jsonl"
    cp_path.write_text(json.dumps({
        "run_id": "abc123",
        "role": "researcher",
        "result": "cached-researcher",
        "error": None,
    }) + "\n")

    seen_roles: list[str] = []

    async def dispatch(worker: Subagent, task: str) -> str:
        seen_roles.append(worker.role)
        return f"{worker.role}-fresh"

    orch = Orchestrator(
        workers=_make_workers(), dispatch_async=dispatch, checkpoint_dir=tmp_path
    )
    results = await orch.run_async("test", run_id="abc123")

    by_role = {r.role: r for r in results}
    assert by_role["researcher"].result == "cached-researcher"
    assert set(seen_roles) == {"summarizer", "ranker"}  # researcher skipped


# ── Subagent dataclass ─────────────────────────────────────────────


def test_subagent_is_a_dataclass_with_role_and_system_prompt():
    s = Subagent(role="x", system_prompt="y")
    assert s.role == "x" and s.system_prompt == "y"
