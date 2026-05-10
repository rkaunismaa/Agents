#!/usr/bin/env python
"""Research assistant CLI — wraps agentlab.spine for terminal use."""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import typer
from anthropic import RateLimitError as _RateLimitError
from rich.console import Console

from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.spine import Orchestrator, Subagent, WorkerResult

app = typer.Typer(add_completion=False, help="Parallel research assistant powered by agentlab.spine.")
console = Console()

WORKERS = [
    Subagent(role="researcher", system_prompt=(
        "You are a focused technical researcher. Use web_search to gather "
        "current information about the topic in the question. Return factual "
        "findings with at least one source URL per claim."
    )),
    Subagent(role="summarizer", system_prompt=(
        "You are a precise summarizer. In one tight paragraph (~80 words), "
        "synthesize an answer to the question from your training knowledge. "
        "Be direct, no hedging, no preamble."
    )),
    Subagent(role="ranker", system_prompt=(
        "You are a comparative analyst. If the question involves comparing or "
        "ranking items, rank them explicitly with one-sentence reasoning per "
        "item and a final recommendation. If not comparative, give your most "
        "direct answer."
    )),
]


async def _claude_dispatch(worker: Subagent, task: str, model: str, client) -> str:
    """Run a worker as a single Claude call with jitter backoff on rate limits."""
    tools = [{"type": "web_search_20250305", "name": "web_search"}] if worker.role == "researcher" else []
    messages = [{"role": "user", "content": task}]
    for delay in (10, 20, 40, 90):
        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model=model,
                max_tokens=1024,
                system=worker.system_prompt,
                tools=tools,
                messages=messages,
            )
            return "".join(getattr(b, "text", "") for b in response.content).strip()
        except _RateLimitError:
            actual = delay + random.uniform(0, delay * 0.5)
            console.print(f"  [yellow][rate limit] {worker.role} waiting {actual:.0f}s...[/yellow]")
            await asyncio.sleep(actual)
    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=1024,
        system=worker.system_prompt,
        tools=tools,
        messages=messages,
    )
    return "".join(getattr(b, "text", "") for b in response.content).strip()


def _setup_tracing():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("agentlab.spine"), provider


@app.command()
def main(
    question: str = typer.Argument(..., help="Research question to answer"),
    run_id: str = typer.Option(
        None, "--run-id", help="Resume from checkpoint (same run_id skips completed workers)"
    ),
    trace: bool = typer.Option(False, "--trace", help="Enable OTel console tracing"),
    model: str = typer.Option(
        os.getenv("AGENTLAB_MODEL", DEFAULT_MODEL), "--model", help="Claude model to use"
    ),
    checkpoint_dir: Path = typer.Option(
        Path(os.getenv("AGENTLAB_CHECKPOINT_DIR", "data/checkpoints")),
        "--checkpoint-dir",
        help="Directory for JSONL checkpoint files",
    ),
) -> None:
    """Run the parallel research assistant spine from the command line."""
    tracer, provider = _setup_tracing() if trace else (None, None)
    client = get_client()

    async def _dispatch(worker: Subagent, task: str) -> str:
        if tracer:
            with tracer.start_as_current_span(f"subagent.{worker.role}") as span:
                span.set_attribute("worker.role", worker.role)
                with tracer.start_as_current_span("llm.complete"):
                    return await _claude_dispatch(worker, task, model, client)
        return await _claude_dispatch(worker, task, model, client)

    orch = Orchestrator(
        workers=WORKERS,
        dispatch_async=_dispatch,
        checkpoint_dir=checkpoint_dir,
    )

    console.print(f"\n[bold blue]Research Assistant[/bold blue] — [dim]{model}[/dim]")
    console.print(f"[dim]Question:[/dim] {question}\n")

    async def _run() -> list[WorkerResult]:
        if tracer:
            with tracer.start_as_current_span("agent.run") as span:
                span.set_attribute("question.length", len(question))
                return await orch.run_async(question, run_id=run_id, worker_timeout=240)
        return await orch.run_async(question, run_id=run_id, worker_timeout=240)

    results = asyncio.run(_run())

    if trace and provider:
        provider.force_flush()

    for r in results:
        if r.error:
            console.print(f"[red][{r.role}] ERROR: {r.error}[/red]")
        else:
            console.print(f"[bold green][{r.role}][/bold green]")
            console.print(r.result or "")
            console.print()


if __name__ == "__main__":
    app()
