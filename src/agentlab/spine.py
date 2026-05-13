"""Multi-agent orchestrator + worker primitives.

Extracted from NB 11 Stage A and extended for NB 12: parallel async
fan-out via asyncio.gather, per-worker timeout cancellation, and
JSONL checkpointing for resumability.

Construction is dispatch-injected so the same Orchestrator can be
exercised with a fake (in tests) or with a real Claude-backed worker
loop (in NB 12).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

DispatchSync = Callable[["Subagent", str], str]
DispatchAsync = Callable[["Subagent", str], Awaitable[str]]


# frozen=True makes Subagent immutable and hashable. A worker definition is a
# value object: the role name and system prompt identify *what* a worker is,
# not mutable state. Freezing makes accidental mutation impossible and allows
# Subagent instances to be used as dict keys or in sets.
@dataclass(frozen=True)
class Subagent:
    """A worker definition: a role name + the system prompt that role uses."""
    role: str
    system_prompt: str


@dataclass
class WorkerResult:
    role: str
    result: Optional[str]
    error: Optional[str]


class Orchestrator:
    """Dispatch a fixed set of subagents over a single question.

    Two execution modes:
    - `run_sync(question)` — sequential, requires `dispatch` (sync callable)
    - `run_async(question, run_id=..., worker_timeout=...)` — parallel,
      requires `dispatch_async` (awaitable callable)

    Optional checkpointing: pass `checkpoint_dir` and a `run_id`. Each
    successful worker's result is appended to `checkpoint_dir/<run_id>.jsonl`;
    failed workers are not cached so they are retried on the next call with
    the same `run_id`.
    """

    def __init__(
        self,
        workers: list[Subagent],
        dispatch: DispatchSync | None = None,
        dispatch_async: DispatchAsync | None = None,
        checkpoint_dir: Path | None = None,
    ):
        if dispatch is None and dispatch_async is None:
            raise ValueError("Provide at least one of dispatch / dispatch_async")
        # Dispatch is injected rather than hardcoded to a Claude API call.
        # This means the Orchestrator knows nothing about Claude: any callable
        # (worker, task) -> str works. Tests inject a synchronous echo function
        # without mocking the SDK; notebooks inject a real Claude call.
        self.workers = list(workers)
        self.dispatch = dispatch
        self.dispatch_async = dispatch_async
        self.checkpoint_dir = checkpoint_dir

    # ── sync ───────────────────────────────────────────────────────

    def run_sync(self, question: str) -> list[WorkerResult]:
        # run_sync exists for NB 11 (sequential orchestrator). NB 12 uses
        # run_async for parallel fan-out. Both modes coexist so the notebooks
        # can show the progression from sequential to parallel.
        if self.dispatch is None:
            raise RuntimeError("run_sync requires `dispatch` (sync callable)")
        results: list[WorkerResult] = []
        for worker in self.workers:
            try:
                text = self.dispatch(worker, question)
                results.append(WorkerResult(role=worker.role, result=text, error=None))
            except Exception as exc:
                results.append(WorkerResult(role=worker.role, result=None, error=repr(exc)))
        return results

    # ── async + checkpointing ──────────────────────────────────────

    async def run_async(
        self,
        question: str,
        run_id: str | None = None,
        worker_timeout: float = 60.0,
    ) -> list[WorkerResult]:
        if self.dispatch_async is None:
            raise RuntimeError("run_async requires `dispatch_async` (awaitable callable)")
        # Auto-generate a short run_id if none provided. hex[:8] = 32-bit
        # random ID; collision probability is negligible for the number of runs
        # in a single session, and it's short enough to read in a filename.
        run_id = run_id or uuid.uuid4().hex[:8]
        cached = self._load_checkpoint(run_id)

        async def _one(worker: Subagent) -> WorkerResult:
            # If this worker already succeeded in a previous run with the same
            # run_id, return its cached result immediately without making an
            # API call. This gives free retry semantics: re-run the same
            # run_id and only the failed workers are re-dispatched.
            if worker.role in cached:
                return cached[worker.role]
            try:
                text = await asyncio.wait_for(
                    self.dispatch_async(worker, question), timeout=worker_timeout
                )
                wr = WorkerResult(role=worker.role, result=text, error=None)
                # Only write successful results to the checkpoint. A failed
                # result is not cached so the worker is retried next time.
                self._append_checkpoint(run_id, wr)
            except asyncio.TimeoutError:
                wr = WorkerResult(
                    role=worker.role,
                    result=None,
                    error=f"timeout after {worker_timeout}s",
                )
            except asyncio.CancelledError:
                # Re-raise CancelledError so asyncio.gather can propagate the
                # cancellation to the caller. Swallowing it would leave the
                # gather hanging if the outer coroutine is cancelled.
                raise
            except Exception as exc:
                wr = WorkerResult(role=worker.role, result=None, error=repr(exc))
            return wr

        return await asyncio.gather(*(_one(w) for w in self.workers))

    def resume(self, run_id: str) -> list[WorkerResult]:
        """Return cached results from the checkpoint without re-running.

        For partial resumes (some workers missing), prefer `run_async`
        with the same `run_id` — it skips cached workers and runs the
        rest.
        """
        path = self._checkpoint_path(run_id)
        if path is None or not path.exists():
            raise FileNotFoundError(f"No checkpoint file for run_id={run_id}")
        cached = self._load_checkpoint(run_id)
        if not cached:
            raise ValueError(f"Checkpoint for run_id={run_id} exists but is empty")
        return [cached[w.role] for w in self.workers if w.role in cached]

    # ── checkpoint internals ───────────────────────────────────────

    def _checkpoint_path(self, run_id: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return self.checkpoint_dir / f"{run_id}.jsonl"

    def _load_checkpoint(self, run_id: str) -> dict[str, WorkerResult]:
        path = self._checkpoint_path(run_id)
        if path is None or not path.exists():
            return {}
        out: dict[str, WorkerResult] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[entry["role"]] = WorkerResult(
                role=entry["role"], result=entry.get("result"), error=entry.get("error")
            )
        return out

    def _append_checkpoint(self, run_id: str, wr: WorkerResult) -> None:
        path = self._checkpoint_path(run_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # JSONL (one JSON object per line, append-only) is chosen over a
        # single JSON file because appending a line is atomic on most
        # filesystems. A crash mid-write leaves the file valid up to the last
        # complete line; there is no read-modify-write step that could corrupt
        # the file if the process is killed between the read and the write.
        entry = {
            "run_id": run_id,
            "role": wr.role,
            "result": wr.result,
            "error": wr.error,
        }
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
