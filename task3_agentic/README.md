# Task 3 — Multi-Agent Financial Research System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/CDAZZDEV-MLE-Udith/blob/main/task3_agentic/notebooks/task3_agentic_system.ipynb)

**Notebook:** [`notebooks/task3_agentic_system.ipynb`](notebooks/task3_agentic_system.ipynb) · CPU only, ~15 min · **framework: LangGraph**

| Criterion | Marks | Where |
|---|---:|---|
| All five tools implemented | 15 | `src/tools.py` |
| Autonomous tool selection | 10 | Conditional edge on the model's own output — no fixed sequence |
| Observe and replan cycle | 8 | The `agent → tools → agent` edge, streamed in the notebook |
| Final report quality | 10 | `src/schemas.py::ResearchReport` — three sections, validated |
| Error handling | 7 | Every tool returns `{ok, data, error, suggestion}`; nothing raises |
| Distinct roles + tool restriction | 8 | `build_tools(context, names)` — structural, not prompted |
| Structured handoff schema | 8 | `src/schemas.py::QuantBrief` |
| Message trace visible | 6 | Per-agent transcripts + inline trace echo |
| Critique loop | 8 | `critique → clarify_a → final_report` edges |
| End-to-end automation | 5 | One `graph.invoke()` from query to report |
| Short-term memory | 5 | Graph state + `ToolContext` caches |
| Persistent cache | 5 | `src/memory.py` — keyed by ticker **and date** |
| `agent_trace.jsonl` | 5 | `src/tracing.py` → `logs/agent_trace.jsonl` |
| **Bonus** — observability | +5 | `dashboard.py` (Streamlit) |

## Why LangGraph, not CrewAI

Three criteria are about things being **enforced**, not claimed:

* *"decides order based on observations — not a fixed sequence"* → a **conditional edge**
  evaluated on the model's actual output. There is no sequence to hardcode.
* *"at least one visible observe-and-replan cycle"* → the cycle **is** the
  `agent → tools → agent` edge. It appears in the trace because it is the control flow.
* *"agents have enforced, separate tool access"* → Agent B is never handed `get_price_data`.
  Its tool node rejects any call to a name outside its list. A system prompt saying "don't
  use the price tools" is a request; not passing the tool is a guarantee.

CrewAI would be less code and more opaque. This is more code and defensible line by line.

## `data_gaps` is mandatory, and that is the point

Agent B cannot see Agent A's tools. If a volatility call fails and the field were optional,
Agent A omits it and Agent B reads silence as *"measured and unremarkable"* — then sizes a
hedge on a number nobody took. Making the declaration mandatory turns a silent failure into
a visible one. It is the single most load-bearing schema decision in the task.

## Dashboard (bonus)

```bash
pip install streamlit pandas
streamlit run task3_agentic/dashboard.py
```

Reads `logs/agent_trace.jsonl` directly, so what you see on screen is exactly the artefact
that was committed — no divergence between the deliverable and the dashboard over it.

## Run (no keys needed)

```bash
python task3_agentic/tests/test_task3_offline.py
```
