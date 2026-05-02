# Agents — a notebook-driven curriculum for building AI agents on Claude

A 15-notebook progressive curriculum that takes you from "I've built a toy
agent" to "I can ship multi-agent, MCP-integrated systems on Claude — and
port them to local models when I need to."

The curriculum is **scratch-first**: you build the bare agent loop by hand
before any framework abstraction shows up, so you understand what
frameworks are hiding from you. Frameworks (Claude Agent SDK, MCP) appear
once they earn their keep.

## Modules

| Module | Notebooks | Status |
|---|---|---|
| **A — Foundations** | 01 bare loop · 02 tool use · 03 structured outputs | ✅ this plan |
| **B — Workflows** | 04 ReAct · 05 planning · 06 memory · 07 RAG · 08 evals + observability | ⏳ next plan |
| **C — Multi-agent + MCP** | 09 build MCP server · 10 consume MCP · 11 orchestrator + subagents · 12 parallel + durable | ⏳ |
| **D — Autonomous + local** | 13 computer use · 14 local models (Ollama / LM Studio) · 15 capstone | ⏳ |

The "research assistant" spine project recurs at NB 03, NB 08, NB 12, and
NB 15 to integrate the patterns you've learned.

Full design: [`docs/superpowers/specs/2026-05-02-ai-agents-curriculum-design.md`](docs/superpowers/specs/2026-05-02-ai-agents-curriculum-design.md)

## Quickstart

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv) ≥ 0.11.

```bash
git clone git@github.com:rkaunismaa/Agents.git
cd Agents

# Tell uv to use the included .agents venv (or set in .envrc + direnv allow):
export UV_PROJECT_ENVIRONMENT=.agents

uv sync --extra dev

cp .env.example .env
# edit .env to add your ANTHROPIC_API_KEY

# Register the kernel so VS Code's notebook UI can find it:
.agents/bin/python -m ipykernel install --user --name agents --display-name "Python (.agents)"

# Then open notebooks/01_bare_agent_loop.ipynb in VS Code.
```

## Repo layout

```
notebooks/                 # the .ipynb files you execute
notebooks/_src/            # jupytext-format .py source for the notebooks
src/agentlab/              # shared utilities imported by every notebook
tests/                     # pytest tests for agentlab
mcp_servers/               # MCP servers (added in Module C)
data/                      # tiny seed corpora for RAG/eval notebooks
docs/superpowers/specs/    # design specs
docs/superpowers/plans/    # implementation plans
```

## Editing notebooks

Notebook content is canonical in `notebooks/_src/*.py` (jupytext "percent"
format). The `.ipynb` files in `notebooks/` are generated. After editing
a `.py` source, regenerate its `.ipynb`:

```bash
.agents/bin/jupytext --sync notebooks/_src/01_bare_agent_loop.py
```

This pairing is configured in `pyproject.toml` (`[tool.jupytext]`).

## Tests

```bash
.agents/bin/python -m pytest -v
```

Tests run against `agentlab` only; notebook content is verified by
executing the notebooks (see `.github/workflows/test.yml`).

## Costs

Each notebook prints a cost estimate at the top. Most notebooks cost a
few cents to run; the multi-agent and capstone notebooks default to
Claude Haiku where pedagogically acceptable to keep costs low.

## License

MIT — see [LICENSE](LICENSE).
