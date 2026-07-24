# Multi-Agent Research & Task Automation Assistant

Two implementations of the same problem — "break a request into subtasks,
research each one with tools, synthesize a final answer" — built to compare
orchestration styles head to head:

- **`src/graph.py`** — [LangGraph](https://github.com/langchain-ai/langgraph): an
  explicit state machine. A `planner` node breaks the request into subtasks,
  fans them out in parallel with LangGraph's `Send` API, and an `aggregator`
  node reduces the results.
- **`src/crew.py`** — [CrewAI](https://github.com/crewAIInc/crewAI): three
  role-based agents (Researcher → Writer → Reviewer) working sequentially,
  each with its own goal/backstory and shared access to the same tools.

Both paths use the same tool set (`src/tools.py`): a sandboxed calculator, a
DuckDuckGo web search, and a local file lookup — and both are exposed through
one CLI (`src/cli.py`).

## Why two implementations of the same thing

They're genuinely different orchestration philosophies. LangGraph gives you
an explicit graph you control node-by-node — good when you need parallel
fan-out, custom retry/branch logic, or fine-grained state. CrewAI gives you
higher-level role/goal/backstory abstractions and sequential hand-off — good
for getting a multi-agent workflow running quickly, at the cost of less
control over execution flow. Building the same task both ways was the
fastest way to actually feel that trade-off instead of just reading about it.

## Architecture

```
                        ┌────────────┐
   request ───────────► │  planner   │
                        └─────┬──────┘
                              │ Send(subtask) per subtask
                     ┌────────┼────────┐
                     ▼        ▼        ▼
               ┌─────────┐┌─────────┐┌─────────┐
               │run_task │ │run_task │ │run_task │   (parallel, LangGraph fan-out)
               │ (ReAct) │ │ (ReAct) │ │ (ReAct) │
               └────┬────┘└────┬────┘└────┬────┘
                    └──────────┼──────────┘
                               ▼
                        ┌────────────┐
                        │ aggregator │───► final_answer
                        └────────────┘
```

Each `run_task` node runs a small hand-rolled ReAct loop (plan → act → observe,
capped at 3 steps) rather than relying solely on a provider's native
function-calling, so the same loop works identically across OpenAI,
Anthropic, and local models — the trade-off is giving up some of the
structured-output guarantees a provider's native tool-calling gives you.

## Pluggable LLM backend

`src/llm.py` selects a backend from `LLM_PROVIDER`:

| Value | Backend | Needs |
|---|---|---|
| `mock` (default) | `src/mock_llm.py`, a deterministic offline stand-in | nothing |
| `openai` | `langchain-openai` | `OPENAI_API_KEY` |
| `anthropic` | `langchain-anthropic` | `ANTHROPIC_API_KEY` |
| `ollama` | `langchain-ollama`, local model | a running Ollama server |

The mock backend is **not** a language model — it pattern-matches on the
structured prompts the graph sends (`### TASK: PLAN`, `### TASK: REACT_STEP`,
`### TASK: SYNTHESIZE`) and returns a valid-shaped response. It exists so the
graph wiring, the fan-out/reduce, and the ReAct loop can all be verified with
`pytest` — zero cost, zero network, zero API key — before ever spending money
on a real model call. Once a key is available, flipping `LLM_PROVIDER` is the
only change needed; none of the graph code changes.

**CrewAI is different**: it drives LLM calls itself through `litellm`
(`CREWAI_MODEL=openai/gpt-4.1` etc.), so it does not go through `llm.py` and
does not support the mock backend. `tests/test_crew.py` only checks that the
agents/tasks/tools wire up correctly — running `crew.kickoff()` for real
needs an actual key.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in a provider, or leave LLM_PROVIDER=mock
```

## Run it

```bash
# LangGraph path, offline (no key needed)
python -m src.cli langgraph "What is 12 * (4 + 3) and search for what LangGraph is used for?"

# LangGraph path, with a real model
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... python -m src.cli langgraph "..."

# CrewAI path (always needs a real key)
CREWAI_MODEL=openai/gpt-4.1 OPENAI_API_KEY=sk-... python -m src.cli crewai "..."
```

## Test

```bash
pytest -q
```

11 tests cover: the calculator's AST-based safe eval (and that it rejects
code-injection attempts), the file-lookup tool, graceful degradation when
web search has no network, each branch of the ReAct loop, the full
LangGraph run end-to-end, and that the CrewAI crew wires its three agents
and task-context chain correctly.

## Docker

```bash
docker build -t research-assistant .
docker run research-assistant langgraph "What is 12 * (4 + 3)?"
```

Defaults to `LLM_PROVIDER=mock` so the container runs with zero
configuration; pass `--env-file .env` to use a real provider.

## Known limitations / honest caveats

- The ReAct loop is capped at 3 steps and uses simple string parsing
  (`ACTION: tool | input` / `FINAL: answer`) rather than a provider's
  structured function-calling output — deliberately, for portability across
  providers, but it means malformed model output can fall through to a
  literal-text answer instead of retrying.
- `web_search` depends on outbound network access; some sandboxed
  environments (this one included, during development) block it, in which
  case the tool returns an `error:` string rather than crashing the agent.
- The mock LLM's routing is a small set of keyword heuristics, not real
  language understanding — it's a stand-in for exercising control flow, not
  a claim that offline testing = testing against a real model's behavior.
