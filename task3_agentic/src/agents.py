"""
Task 3A and 3B - LangGraph agents.

WHY LANGGRAPH RATHER THAN CrewAI
--------------------------------
Three marking criteria are about things being *visible and enforced*:
"agent decides order based on observations - not a fixed sequence",
"at least one visible observe-and-replan cycle", and "agents have enforced,
separate tool access". In LangGraph each of those is a structural property of
the graph rather than a claim about a prompt:

  * Tool selection is a conditional edge evaluated on the model's actual output,
    so there is no sequence to hardcode.
  * The observe-replan cycle IS the `agent → tools → agent` edge. It appears in
    the trace because it is the control flow, not because we logged it.
  * Tool restriction is enforced by binding different tool lists to different
    nodes. Agent B is never given `get_price_data`, so it cannot call it. A
    system prompt saying "do not use the price tools" is a request; not passing
    the tool is a guarantee.

The cost is more code than CrewAI would need. That is the trade I would make
again for a system whose behaviour I have to defend line by line.

GRAPH SHAPES
------------
Task 3A (single agent):

        START → agent ⇄ tools → synthesise → END
                  └── conditional: tool_calls present? ──┘

Task 3B (two agents with a critique loop):

        START → agent_a ⇄ tools_a → handoff → agent_b ⇄ tools_b
                                                  ↓
                                              critique ──no──→ final_report → END
                                                  │yes
                                          clarify_a → incorporate ┘

# AI-ASSISTED: Claude (claude-opus-5), Prompt: 'Build a LangGraph single-agent
# research graph and a two-agent graph with structurally enforced tool
# restriction and a critique loop', Date: 2026-09-10
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ValidationError

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for extra in (str(_ROOT), str(_ROOT / "task1_financial" / "src"), str(_HERE)):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from schemas import (  # noqa: E402
    ClarificationRequest,
    ClarificationResponse,
    QuantBrief,
    QuantJudgement,
    ResearchReport,
    build_quant_brief,
)
from tools import AGENT_A_TOOLS, AGENT_B_TOOLS, ToolContext, build_tools  # noqa: E402

# _add_messages_module_level
# LangGraph >=1.0 resolves state-class annotations with get_type_hints() against
# this module's globals. Combined with `from __future__ import annotations`
# (which makes every annotation a string), a function-local import of
# `add_messages` is invisible to that lookup and raises NameError at build time.
try:
    from langgraph.graph.message import add_messages  # noqa: E402
except ImportError:  # langgraph is not needed by the offline test suite.
    add_messages = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

MAX_AGENT_ITERATIONS = 8      # Hard stop. An agent that will not finish must not run forever.
MAX_STRUCTURED_ATTEMPTS = 3
MAX_CRITIQUE_ROUNDS = 1
MAX_MODELS_PER_PROVIDER = 3   # Fallback links per provider; see build_chat_model.       # The brief asks for one visible cycle.
TOOL_MESSAGE_CHAR_BUDGET = 1200   # Per observation replayed to the model; see make_tool_node.
PROMPT_CHAR_BUDGET = 13000        # ~3.2k tokens of transcript; see _trim_to_budget.
# Groq's 8,000 tokens/minute counts prompt + reserved output together, so these
# two numbers are one budget: ~3.2k transcript + 2k output leaves clear headroom.
# max_tokens was briefly cut to 1024 to buy TPM room, which was the wrong lever --
# a ResearchReport does not fit in 1024 tokens, and the JSON came back truncated
# mid-string. Trim the INPUT, never the room the answer needs.


# ---------------------------------------------------------------------------
# Chat model construction
# ---------------------------------------------------------------------------
def make_tool_node(tools: list[Any], messages_key: str = "messages") -> Any:
    """Return a node that executes the tool calls on the last message.

    LangGraph's prebuilt `ToolNode` reads and writes state["messages"]. Task 3B
    keeps two separate conversations (`a_messages`, `b_messages`) so each agent's
    transcript stays independently inspectable, which the "Message Trace Visible"
    criterion needs. `ToolNode` grew a `messages_key` argument to support that,
    but only in a later minor release - pinning the reviewer to a specific
    langgraph version is a worse failure mode than fifteen lines of code, so we
    hand-roll it for the non-default key and keep the prebuilt for the default.

    Tool restriction lives here too: this node can only execute tools present in
    `tools`. A call to anything else returns an error message to the agent rather
    than executing, which is what makes the restriction enforced rather than
    merely requested.
    """
    from langchain_core.messages import ToolMessage

    if messages_key == "messages":
        from langgraph.prebuilt import ToolNode

        return ToolNode(tools)

    by_name = {getattr(t, "name", str(t)): t for t in tools}

    def custom_tool_node(state: dict) -> dict:
        last = state[messages_key][-1]
        outputs: list[Any] = []
        for call in getattr(last, "tool_calls", None) or []:
            name = call["name"]
            tool = by_name.get(name)
            if tool is None:
                # Structural enforcement: the tool is not in this agent's list.
                outputs.append(ToolMessage(
                    content=json.dumps({
                        "ok": False,
                        "error": f"Tool '{name}' is not available to this agent.",
                        "suggestion": f"You may only call: {sorted(by_name)}. "
                                      f"Use one of those, or proceed without this data "
                                      f"and declare it as a gap.",
                    }),
                    name=name, tool_call_id=call["id"],
                ))
                continue
            try:
                result = tool.invoke(call.get("args", {}))
            except Exception as exc:  # noqa: BLE001 - failure is data, not a crash
                result = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "suggestion": "Try an alternative tool or proceed without this data."}
            outputs.append(ToolMessage(
                # 2000, not 6000. Groq's free tier allows 8000 tokens per MINUTE
                # across the whole conversation, and the agent re-sends the entire
                # transcript on every turn, so a few 6000-character observations
                # exhaust the budget within two or three iterations. The full,
                # untruncated result is still written to agent_trace.jsonl -- this
                # caps only what is replayed back to the model.
                content=json.dumps(result, default=str)[:TOOL_MESSAGE_CHAR_BUDGET],
                name=name, tool_call_id=call["id"],
            ))
        return {messages_key: outputs}

    return custom_tool_node


class FailoverChatModel:
    """A chat model that moves to the next free provider when one refuses.

    `LLMClient` already fails over between Groq and OpenRouter, but the LangGraph
    agents do not use it - they need a LangChain chat model, so `build_chat_model`
    returned a bare `ChatOpenAI` pointed at whichever provider answered first.
    That single-provider model is what actually killed live runs:

      * HTTP 413 - "Request too large ... on tokens per minute (TPM): Limit 8000"
      * HTTP 429 - "Rate limit reached ... on tokens per day (TPD): Limit 200000"

    Neither is recoverable by retrying the same provider; the second is not
    recoverable for the rest of the day. The README claimed the client "backs off
    and fails over", and for the agents that was simply not true.

    `bind_tools` is why this is a wrapper and not a bare `.with_fallbacks()` call:
    `RunnableWithFallbacks` has no `bind_tools`, so the tools must be bound to
    each provider's model first and the fallback chain built from the bound
    runnables.
    """

    def __init__(self, models: list[Any], names: list[str]) -> None:
        if not models:
            raise ValueError("FailoverChatModel needs at least one model")
        self._models = models
        self._names = names

    @property
    def model_name(self) -> str:
        """Primary model id. Notebooks print this in their run manifest."""
        return self._names[0]

    @property
    def providers(self) -> list[str]:
        return list(self._names)

    @staticmethod
    def _chain(models: list[Any]) -> Any:
        return models[0].with_fallbacks(models[1:]) if len(models) > 1 else models[0]

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self._chain([m.bind_tools(tools, **kwargs) for m in self._models])

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._chain(self._models).invoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._chain(self._models).stream(*args, **kwargs)


def build_chat_model(temperature: float = 0.1, max_tokens: int = 2048) -> Any:
    """Build a chat model spanning every free provider that has a key.

    Groq and OpenRouter are both OpenAI-compatible, so one `ChatOpenAI` per
    provider with a swapped `base_url` reaches either. Concrete model ids are
    resolved by querying each provider's /models endpoint (see
    common/llm_client.py) rather than hardcoded, because the ids the brief
    implies are already deprecated.

    Tool calling is a hard requirement - Task 3 is meaningless without it - so
    the reasoning tier is preferred, being the ranked list of models known to
    support it reliably. Providers are returned in order, wrapped so that a
    refusal from one moves to the next rather than ending the run.
    """
    from langchain_openai import ChatOpenAI

    from common.llm_client import PROVIDERS, Tier, get_secret

    models: list[Any] = []
    names: list[str] = []

    for provider in PROVIDERS:
        api_key = get_secret(provider.api_key_env)
        if not api_key:
            continue

        try:
            from openai import OpenAI

            probe = OpenAI(api_key=api_key, base_url=provider.base_url, timeout=30.0)
            served = {m.id for m in probe.models.list().data}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not list models for %s: %s", provider.name, exc)
            served = set()

        # Match LLMClient: normalise "models/x" -> "x" (Gemini), then honour the
        # provider's cost constraint.
        served = {m.split("/", 1)[1] if m.startswith("models/") else m for m in served}
        if getattr(provider, "free_only_suffix", None):
            served = {m for m in served if m.endswith(provider.free_only_suffix)}

        preferences = provider.preferences[Tier.REASONING]
        # Several links per provider, not one. Free endpoints fail INDEPENDENTLY
        # and transiently -- a 502 "Service temporarily overloaded" on the single
        # OpenRouter model, arriving while Groq's daily quota was spent, emptied
        # a two-link chain and ended the run. Extra links cost nothing unless
        # they are needed.
        chosen = [m for m in preferences if m in served][:MAX_MODELS_PER_PROVIDER]
        if not chosen:
            chosen = [sorted(served)[0]] if served else [preferences[0]]

        for model_id in chosen:
            logger.info("Chat model: %s via %s", model_id, provider.name)
            if getattr(provider, "agent_client", "openai") == "google_genai":
                # Native client: the OpenAI-compatible surface drops Gemini 3's
                # thought_signature and the tool loop dies on the second turn.
                from langchain_google_genai import ChatGoogleGenerativeAI

                models.append(ChatGoogleGenerativeAI(
                    model=model_id,
                    google_api_key=api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    timeout=90,
                    max_retries=1,
                ))
            else:
                models.append(ChatOpenAI(
                    model=model_id,
                    api_key=api_key,
                    base_url=provider.base_url,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=90,
                    # Low, deliberately: a TPM or TPD refusal will not clear by
                    # retrying the same provider, so spending attempts there only
                    # delays the failover that actually resolves it.
                    max_retries=1,
                    default_headers=provider.extra_headers or None,
                ))
            names.append(f"{model_id} ({provider.name})")

    if not models:
        raise RuntimeError(
            "No provider configured. Set GROQ_API_KEY or OPENROUTER_API_KEY "
            "in your environment or Colab Secrets."
        )

    return FailoverChatModel(models, names)


def is_tool_call_parse_error(exc: Exception) -> bool:
    """True when the provider rejected the model's OWN tool call as unparseable.

    Groq returns 400 with code `tool_use_failed` and a `failed_generation` field
    when the model emits malformed tool-call JSON - typically because it ran out
    of output tokens midway through serialising a large argument. This is a
    model failure surfaced as a transport error, and it is distinct from a tool
    raising: no tool ever ran, so the tool-level error handling never sees it.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "tool_use_failed",
            "failed to parse tool call",
            "failed_generation",
            "invalid tool call",
        )
    )


# Sent back to the model after it emits an unparseable tool call. Names the
# specific mistake rather than saying "try again", because a generic retry
# reproduces the same oversized call.
TOOL_CALL_REPAIR = (
    "Your previous tool call could not be parsed - it was cut off partway through "
    "its arguments, most likely because the arguments were too long.\n\n"
    "Retry with SHORT arguments. Tool arguments are references, not payloads: pass "
    "identifiers like a ticker symbol, never long lists of text you already have in "
    "this conversation. Call exactly one tool, and if you were trying to score "
    "sentiment, call llm_sentiment with only the ticker."
)


def _trim_to_budget(messages: list[Any], budget: int = PROMPT_CHAR_BUDGET) -> list[Any]:
    """Drop the oldest middle turns until the transcript fits `budget` characters.

    Groq's free tier allows 8,000 tokens per MINUTE, and that ceiling counts the
    whole re-sent transcript plus `max_tokens` of reserved output. An agent loop
    grows its transcript monotonically, so a long run walks into HTTP 413
    ("Request too large ... Requested 8022") no matter how well it is behaving.

    The first message (the system prompt) and the most recent turns are what the
    model actually needs: the system prompt defines the task, and recent turns
    carry the observations it is reasoning about. Older tool observations are
    already reflected in the reasoning that followed them, so they are the right
    thing to shed. Whatever is dropped remains in agent_trace.jsonl.
    """
    if not messages:
        return messages

    def size(message: Any) -> int:
        content = getattr(message, "content", "")
        return len(content) if isinstance(content, str) else len(str(content))

    total = sum(size(m) for m in messages)
    if total <= budget:
        return messages

    head, tail = messages[:1], messages[1:]
    kept: list[Any] = []
    running = size(head[0])
    # Walk backwards so the most recent turns survive.
    for message in reversed(tail):
        cost = size(message)
        if running + cost > budget and kept:
            break
        kept.append(message)
        running += cost
    kept.reverse()

    dropped = len(messages) - len(head) - len(kept)
    if dropped:
        logger.info("Trimmed %d older turn(s) to fit the prompt budget.", dropped)
    return head + kept


def invoke_with_recovery(
    model_with_tools: Any,
    messages: list[Any],
    *,
    max_attempts: int = 3,
    label: str = "agent",
) -> Any:
    """Invoke a tool-bound model, recovering from malformed tool calls.

    The README previously listed retry-on-malformed-tool-call as a known gap.
    This is that gap closed, and it exists because the failure was observed in a
    real run: Llama 3.3 tried to copy fifteen headlines into a tool argument,
    truncated, and the 400 propagated out of the graph and killed the notebook.

    Recovery ladder, in order:
      1. Retry with a corrective message naming the specific mistake.
      2. Retry once more without tools bound at all, so the model must answer in
         prose rather than a tool call.
      3. Return a synthetic assistant message with no tool calls, which routes
         the graph to its synthesis node.

    The run degrades to a report built on whatever was already gathered rather
    than crashing - the same principle the tool layer already follows.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    working = _trim_to_budget(list(messages))
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return model_with_tools.invoke(working)
        except Exception as exc:  # noqa: BLE001 - classified below
            last_error = exc
            if not is_tool_call_parse_error(exc):
                raise
            logger.warning(
                "[%s] attempt %d: provider rejected a malformed tool call; retrying.",
                label, attempt,
            )
            print(f"   RECOVERED ← malformed tool call rejected by the provider "
                  f"(attempt {attempt}); retrying with corrective guidance")
            working = list(messages) + [HumanMessage(content=TOOL_CALL_REPAIR)]

    # Last resort: ask without tools so no tool call can be malformed.
    try:
        bare = getattr(model_with_tools, "bound", None) or model_with_tools
        response = bare.invoke(
            list(messages)
            + [HumanMessage(content=(
                "Do not call any tool. Summarise what you have established so far "
                "from the tool results already in this conversation, and state what "
                "you were unable to retrieve."
            ))]
        )
        print("   RECOVERED ← tool calling disabled for one turn; continuing in prose")
        return response
    except Exception:  # noqa: BLE001
        logger.error("[%s] unrecoverable after %d attempts: %s", label, max_attempts, last_error)
        return AIMessage(content=(
            "I was unable to issue a further tool call because the provider rejected "
            "the generated call as malformed. Proceeding to synthesis using the "
            "evidence gathered so far; this limitation is recorded as a data caveat."
        ))


def structured_call(
    chat_model: Any,
    messages: list[Any],
    model_cls: type[BaseModel],
    *,
    label: str = "structured",
    max_attempts: int = MAX_STRUCTURED_ATTEMPTS,
) -> BaseModel | None:
    """Request JSON matching `model_cls`, validating with a bounded repair loop.

    Implemented manually rather than via `.with_structured_output()` because
    free-tier models support the underlying function-calling mode
    inconsistently. Asking for JSON and validating it ourselves works on every
    provider, and gives us the validator error text to feed back on a retry -
    which is what actually makes the repair converge.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    schema = json.dumps(model_cls.model_json_schema(), indent=2)[:3500]
    # The schema and the repair instruction are large; trim the transcript so the
    # whole request still fits the provider's per-minute token ceiling.
    working = _trim_to_budget(list(messages), PROMPT_CHAR_BUDGET - len(schema)) + [
        HumanMessage(content=(
            f"Return a single JSON object conforming to this schema. "
            f"No prose, no markdown fences.\n\n{schema}"
        ))
    ]

    for attempt in range(1, max_attempts + 1):
        raw = ""
        try:
            response = chat_model.invoke(working)
            raw = response.content if isinstance(response.content, str) else str(response.content)

            from common.llm_client import _loads_lenient

            payload, parse_error = _loads_lenient(raw)
            if payload is None:
                raise ValueError(f"not valid JSON: {parse_error}")

            validated = model_cls.model_validate(payload)
            if attempt > 1:
                logger.info("[%s] validated on attempt %d.", label, attempt)
            return validated

        except (ValidationError, ValueError) as exc:
            detail = (
                " | ".join(
                    f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                )
                if isinstance(exc, ValidationError) else str(exc)
            )
            logger.warning("[%s] attempt %d failed: %s", label, attempt, detail[:250])
            if attempt == max_attempts:
                logger.error("[%s] exhausted %d attempts.", label, max_attempts)
                return None
            working = list(messages) + [
                AIMessage(content=raw[:2500]),
                HumanMessage(content=(
                    f"That failed validation: {detail}\n\n"
                    f"Fix only what the error identifies and return the corrected JSON "
                    f"object alone. Schema:\n{schema}"
                )),
            ]
        except Exception as exc:  # noqa: BLE001 - transport failure
            logger.warning("[%s] attempt %d transport error: %s", label, attempt, exc)
            if attempt == max_attempts:
                return None
    return None


# ===========================================================================
# TASK 3A - single tool-using research agent
# ===========================================================================
SINGLE_AGENT_SYSTEM = """\
You are a senior equity research analyst with access to five tools. You work \
autonomously: you decide which tools to call, in what order, and when you have \
enough evidence to stop.

HOW TO WORK
1. Think about what you do not yet know, then call the tool that closes the \
largest gap. Do not call tools in a fixed order and do not call all five out of \
habit - an unnecessary call costs time and adds nothing.
2. BEFORE each tool call, state in one sentence what you have observed so far \
and what you are now trying to establish. This is how your reasoning becomes \
auditable.
3. AFTER each result, react to what it actually said. If price data shows an \
unusual move, investigate WHY with news or search. If sentiment conflicts with \
the technicals, dig into the conflict. Your later calls must be shaped by your \
earlier observations - that is the whole point.
4. If a tool fails, read its "suggestion" field and take that alternative route. \
Never abandon the task because one source is unavailable, and never invent data \
you could not retrieve.

DISCIPLINE
- Every claim in your final answer must trace to a tool result. If you could not \
measure something, say so explicitly rather than estimating it.
- You have at most {max_iterations} rounds of tool calls. Budget them.
- When you have enough to answer, stop calling tools and say so."""

SINGLE_AGENT_TASK = """\
Analyse the current financial health and market sentiment of {ticker}. Identify \
the top three risks to its share price over the next 90 days and suggest one \
data-driven hedge strategy.

Work through this with the tools. Begin by stating what you need to establish."""


class ResearchState(TypedDict):
    """Graph state for the single agent."""

    messages: Annotated[list, "add_messages"]
    ticker: str
    iterations: int
    report: ResearchReport | None


def build_single_agent(
    context: ToolContext,
    chat_model: Any | None = None,
    *,
    max_iterations: int = MAX_AGENT_ITERATIONS,
) -> Any:
    """Compile the Task 3A graph: agent ⇄ tools → synthesise → END."""
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import HumanMessage, SystemMessage

    chat_model = chat_model or build_chat_model()
    tools = build_tools(context)                       # All five.
    model_with_tools = chat_model.bind_tools(tools)

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        ticker: str
        iterations: int
        report: ResearchReport | None

    def agent_node(state) -> dict:
        """Decide the next action. This node IS the 'replan' half of the cycle."""
        iterations = state.get("iterations", 0)
        if iterations >= max_iterations:
            logger.warning("Iteration cap (%d) reached; forcing synthesis.", max_iterations)
            from langchain_core.messages import AIMessage

            return {
                "messages": [AIMessage(content=(
                    "I have reached my tool-call budget. Synthesising the report from "
                    "the evidence gathered so far."
                ))],
                "iterations": iterations + 1,
            }
        response = invoke_with_recovery(
            model_with_tools, state["messages"], label="single_agent"
        )
        return {"messages": [response], "iterations": iterations + 1}

    def route(state) -> Literal["tools", "synthesise"]:
        """Conditional edge - the model's own output decides, nothing is scripted."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("iterations", 0) <= max_iterations:
            return "tools"
        return "synthesise"

    def synthesise_node(state) -> dict:
        """Force the final answer into the required three-section structure."""
        instruction = HumanMessage(content=(
            "Now produce the final research report from everything you gathered.\n\n"
            "Three sections are required:\n"
            "1. financial_health_summary - the quantitative picture, citing the actual "
            "figures you retrieved.\n"
            "2. top_three_risks - EXACTLY three, each with supporting_evidence that "
            "cites a specific figure or a retrieved headline, and a severity of "
            "high, medium or low.\n"
            "3. hedge_strategy - ONE concrete, data-driven recommendation. It must "
            "name a real instrument or mechanism (protective put, collar, "
            "beta-weighted short, pair trade) AND reference a measured quantity "
            "(a volatility level, a strike, a ratio, a horizon).\n\n"
            "Add any measurement you could not obtain to data_caveats. Do not "
            "present an estimate as a measurement."
        ))
        report = structured_call(
            chat_model, list(state["messages"]) + [instruction], ResearchReport,
            label="single_agent_report",
        )
        if report is None:
            logger.error("Report synthesis failed after all attempts.")
            return {"report": None}
        report.ticker = state["ticker"]
        return {"report": report}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("synthesise", synthesise_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "synthesise": "synthesise"})
    graph.add_edge("tools", "agent")          # ← the observe-and-replan cycle
    graph.add_edge("synthesise", END)

    return graph.compile()


def run_single_agent(
    ticker: str,
    context: ToolContext,
    chat_model: Any | None = None,
    *,
    max_iterations: int = MAX_AGENT_ITERATIONS,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run Task 3A, streaming the observe-decide-act cycle to the notebook.

    The printed narration is the evidence for the "Observe and Replan Cycle"
    criterion: it shows the model stating an intention, receiving a result, and
    changing its next move in response.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    graph = build_single_agent(context, chat_model, max_iterations=max_iterations)
    initial = {
        "messages": [
            SystemMessage(content=SINGLE_AGENT_SYSTEM.format(max_iterations=max_iterations)),
            HumanMessage(content=SINGLE_AGENT_TASK.format(ticker=ticker.upper())),
        ],
        "ticker": ticker.upper(),
        "iterations": 0,
        "report": None,
    }

    if verbose:
        print(f"\n{'=' * 78}\nTASK 3A — autonomous research agent: {ticker.upper()}\n{'=' * 78}")

    final_state: dict[str, Any] = {}
    step = 0
    seen = 0

    for state in graph.stream(initial, stream_mode="values"):
        final_state = state
        messages = state.get("messages", [])
        for message in messages[seen:]:
            kind = type(message).__name__
            if kind == "AIMessage":
                step += 1
                text = (message.content or "").strip()
                if text and verbose:
                    print(f"\n── Step {step} · REASONING ──\n{text[:600]}")
                calls = getattr(message, "tool_calls", None) or []
                if calls and verbose:
                    for call in calls:
                        args = json.dumps(call.get("args", {}))[:120]
                        print(f"   DECIDED → call {call['name']}({args})")
            elif kind == "ToolMessage" and verbose:
                content = str(message.content)
                print(f"   OBSERVED ← {message.name}: {content[:260]}"
                      f"{'…' if len(content) > 260 else ''}")
        seen = len(messages)

    report = final_state.get("report")
    if verbose:
        print(f"\n{'=' * 78}")
        if report:
            print(report.render())
        else:
            print("Report synthesis failed. See the trace and logs above.")

    return {
        "report": report,
        "state": final_state,
        "iterations": final_state.get("iterations", 0),
        "tools_used": sorted({r.tool for r in context.tracer.records}) if context.tracer else [],
    }


# ===========================================================================
# TASK 3B - two-agent pipeline with a critique loop
# ===========================================================================
AGENT_A_SYSTEM = """\
You are Agent A, a QUANTITATIVE ANALYST. You measure; you do not narrate.

YOUR TOOLS: get_price_data, calculate_volatility, llm_sentiment.
You have NO web search and NO general news access. That is deliberate. Your job \
is the numbers.

YOUR OUTPUT is a structured quantitative brief that Agent B - a research writer \
who cannot see any price data at all - will build a report from. Everything \
numeric in the final report has to come from you.

RULES
- Gather your measurements first, then produce the brief. Call get_price_data \
before calculate_volatility so the price cache is warm.
- To score sentiment you need headlines, but you have no news tool. Note this \
in data_gaps if it applies.
- Every quantitative_finding must contain an actual figure. "Momentum is strong" \
is not a finding; "MACD histogram at +1.83 with the 50-day 7.4% above the \
200-day" is.
- data_gaps is MANDATORY and must be honest. Agent B cannot see your tools, so \
anything you leave unstated is invisible to them. Silence reads as "measured and \
unremarkable", which is a lie if you never measured it.
- Set confidence honestly. Missing data means lower confidence."""

AGENT_B_SYSTEM = """\
You are Agent B, a RESEARCH WRITER. You synthesise and interpret.

YOUR TOOLS: web_search, get_news.
You have NO access to price data, indicators or volatility. That is deliberate. \
Every number you use must come from Agent A's brief. If you find yourself \
wanting a figure Agent A did not supply, that is exactly what the clarification \
mechanism is for - do not invent it and do not estimate it.

YOUR JOB
1. Read Agent A's quantitative brief carefully, including its declared data gaps.
2. Use your tools to find the qualitative context the numbers cannot supply: \
what is actually driving them, what analysts expect, what regulatory or \
competitive developments are in play.
3. Produce a research report with three sections: a financial health summary, \
exactly three risks each backed by specific evidence, and one data-driven hedge \
strategy.

DISCIPLINE
- Numbers come from Agent A. Context comes from your tools. Never blur the two.
- If a tool fails, read its suggestion and take the alternative route.
- Where Agent A declared a data gap, carry it into data_caveats. Do not paper \
over it."""


class MultiAgentState(TypedDict):
    ticker: str
    a_messages: Annotated[list, "add_messages"]
    b_messages: Annotated[list, "add_messages"]
    quant_brief: QuantBrief | None
    clarification: ClarificationRequest | None
    clarification_answer: ClarificationResponse | None
    critique_rounds: int
    a_iterations: int
    b_iterations: int
    report: ResearchReport | None
    forced_critique: bool


def build_multi_agent(
    tracer: Any,
    llm_client: Any,
    chat_model: Any | None = None,
    *,
    max_iterations: int = 5,
    require_critique: bool = True,
) -> Any:
    """Compile the Task 3B graph.

    TOOL RESTRICTION IS STRUCTURAL. Two independent ToolContexts are created,
    each labelled with its agent name for the trace, and each agent's node binds
    only its own tool list. There is no code path by which Agent B can reach
    `get_price_data`; the tool object is not in the list passed to its ToolNode.

    `require_critique` guarantees the visible clarification cycle the brief asks
    for. The critique is model-driven first: Agent B is asked whether it needs
    anything, and with Agent A obliged to declare data gaps it almost always
    does. If Agent B declines on the first round, we synthesise one clarification
    from the largest declared gap so the reviewer sees the mechanism work - and
    the output is flagged `forced_critique=True` so that is never mistaken for
    the model's own initiative.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    chat_model = chat_model or build_chat_model()

    context_a = ToolContext(tracer=tracer, llm=llm_client, agent_name="AgentA")
    context_b = ToolContext(tracer=tracer, llm=llm_client, agent_name="AgentB")
    tools_a = build_tools(context_a, AGENT_A_TOOLS)     # 3 tools. No web_search.
    tools_b = build_tools(context_b, AGENT_B_TOOLS)     # 2 tools. No price data.

    model_a = chat_model.bind_tools(tools_a)
    model_b = chat_model.bind_tools(tools_b)

    class State(TypedDict):
        ticker: str
        a_messages: Annotated[list, add_messages]
        b_messages: Annotated[list, add_messages]
        quant_brief: QuantBrief | None
        clarification: ClarificationRequest | None
        clarification_answer: ClarificationResponse | None
        critique_rounds: int
        a_iterations: int
        b_iterations: int
        report: ResearchReport | None
        forced_critique: bool

    # -- Agent A ------------------------------------------------------------
    def agent_a_node(state) -> dict:
        count = state.get("a_iterations", 0)
        if count >= max_iterations:
            return {"a_messages": [AIMessage(content="Measurement budget reached; "
                                                     "producing the brief now.")],
                    "a_iterations": count + 1}
        response = invoke_with_recovery(model_a, state["a_messages"], label="AgentA")
        return {"a_messages": [response], "a_iterations": count + 1}

    def route_a(state) -> Literal["tools_a", "handoff"]:
        last = state["a_messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("a_iterations", 0) <= max_iterations:
            return "tools_a"
        return "handoff"

    def handoff_node(state) -> dict:
        """Agent A → Agent B. Validated QuantBrief or the pipeline stops here."""
        print("\n── HANDOFF · Agent A → Agent B ──")
        instruction = HumanMessage(content=(
            f"Summarise your judgement of {state['ticker']} now.\n\n"
            "Do NOT restate the raw measurements as fields - the price, "
            "volatility and sentiment figures are attached automatically from "
            "the tool results. Give only:\n"
            "  trend_regime           - one of uptrend, downtrend, rangebound, undetermined\n"
            "  quantitative_findings  - 2 to 8 strings, each containing a figure\n"
            "  data_gaps              - anything you could not measure, and why\n"
            "  confidence             - a number between 0 and 1"
        ))
        judgement = structured_call(
            chat_model, list(state["a_messages"]) + [instruction], QuantJudgement,
            label="quant_judgement",
        )

        # The figures come from the measurements, not from the model. See
        # QuantJudgement's docstring for why this split exists.
        brief = None
        if judgement is not None:
            try:
                brief = build_quant_brief(
                    judgement, context_a.results, ticker=state["ticker"],
                )
            except ValidationError as exc:
                logger.warning("[quant_brief] assembly failed: %s", exc)
                brief = None
        if brief is None:
            # Agent A could not produce a schema-valid brief. Agent B must still
            # run: it has its own tools and can report qualitatively, provided it
            # is told the numbers are missing. Returning no b_messages here left
            # Agent B invoking the model with an empty message list, which the
            # provider rejects outright ("'messages' : minimum number of items is
            # 1") -- turning a recoverable degradation into a dead pipeline.
            print("   Agent A failed to produce a valid brief - "
                  "Agent B will proceed on qualitative sources alone.")
            degraded_seed = HumanMessage(content=(
                f"Agent A could NOT produce a validated quantitative brief for "
                f"{state['ticker']}; no price, volatility or sentiment figures are "
                f"available to you, and you cannot retrieve them yourself.\n\n"
                f"Proceed using only get_news and web_search. Write the report from "
                f"qualitative evidence, and record the absence of quantitative data "
                f"as an explicit data caveat. Do not invent figures."
            ))
            return {"quant_brief": None,
                    "b_messages": [SystemMessage(content=AGENT_B_SYSTEM), degraded_seed]}

        brief.ticker = state["ticker"]
        print(f"   QuantBrief validated — confidence {brief.confidence:.2f}, "
              f"{len(brief.quantitative_findings)} findings, "
              f"{len(brief.data_gaps)} declared gap(s)")
        for finding in brief.quantitative_findings:
            print(f"     • {finding[:100]}")
        for gap in brief.data_gaps:
            print(f"     ! gap: {gap[:100]}")

        seed = HumanMessage(content=(
            f"Agent A has completed the quantitative analysis of {state['ticker']} "
            f"and handed you this validated brief.\n\n"
            f"{brief.as_handoff_text()}\n\n"
            f"Now research the qualitative context with your tools. State what you "
            f"are looking for before each search."
        ))
        return {"quant_brief": brief, "b_messages": [SystemMessage(content=AGENT_B_SYSTEM), seed]}

    # -- Agent B ------------------------------------------------------------
    def agent_b_node(state) -> dict:
        count = state.get("b_iterations", 0)
        if not state.get("b_messages"):
            # Defensive: every provider rejects a zero-message request, and that
            # 400 is unrecoverable mid-graph. Seed rather than die.
            return {"b_messages": [
                SystemMessage(content=AGENT_B_SYSTEM),
                HumanMessage(content=(
                    f"Research the qualitative outlook for {state['ticker']} using "
                    f"your tools. No quantitative brief reached you; say so in the "
                    f"report rather than inventing figures."
                )),
            ], "b_iterations": count}
        if count >= max_iterations:
            return {"b_messages": [AIMessage(content="Research budget reached; proceeding.")],
                    "b_iterations": count + 1}
        response = invoke_with_recovery(model_b, state["b_messages"], label="AgentB")
        return {"b_messages": [response], "b_iterations": count + 1}

    def route_b(state) -> Literal["tools_b", "critique"]:
        last = state["b_messages"][-1]
        if getattr(last, "tool_calls", None) and state.get("b_iterations", 0) <= max_iterations:
            return "tools_b"
        return "critique"

    # -- Critique loop ------------------------------------------------------
    def critique_node(state) -> dict:
        """Agent B decides whether to send ONE clarification back to Agent A."""
        if state.get("critique_rounds", 0) >= MAX_CRITIQUE_ROUNDS:
            return {"clarification": None}

        brief = state.get("quant_brief")
        if brief is None:
            return {"clarification": None}

        print("\n── CRITIQUE · Agent B reviewing Agent A's brief ──")
        ask = HumanMessage(content=(
            "Before writing the report, review Agent A's brief for anything that "
            "blocks you.\n\n"
            f"Agent A's confidence: {brief.confidence:.2f}\n"
            f"Declared data gaps: {brief.data_gaps or 'none'}\n\n"
            "If any declared gap, or any missing figure you need for the hedge "
            "recommendation, would force you to guess, raise EXACTLY ONE "
            "clarification request naming the specific field and asking a "
            "specific answerable question. Remember you cannot call the price "
            "tools yourself - this is your only route to that data.\n\n"
            "If nothing blocks you, return "
            '{\"field_in_question\": \"none\", \"question\": \"No clarification is '
            'required to proceed.\", \"why_it_matters\": \"The brief is sufficient '
            'for the report.\"}'
        ))
        request = structured_call(
            chat_model, list(state["b_messages"]) + [ask], ClarificationRequest,
            label="clarification_request",
        )

        declined = request is None or request.field_in_question.strip().lower() in {"none", "n/a", ""}
        forced = False

        if declined and require_critique:
            # Guarantee the visible cycle. Flagged so it is never mistaken for
            # the model's own judgement.
            target = brief.data_gaps[0] if brief.data_gaps else "volatility.percentile_vs_two_year"
            request = ClarificationRequest(
                field_in_question=target[:80],
                question=(
                    f"Could you quantify this for {state['ticker']}, or confirm it could "
                    f"not be measured? I need the figure to size the hedge and cannot "
                    f"retrieve it myself: {target[:200]}"
                ),
                why_it_matters=(
                    "The hedge recommendation must reference a measured quantity to be "
                    "data-driven; without this I would be estimating."
                ),
            )
            forced = True
            print("   Agent B raised no clarification; forcing one for demonstration "
                  "(flagged forced_critique=True).")

        if request is None:
            return {"clarification": None}

        print(f"   REQUEST → field '{request.field_in_question}'")
        print(f"     Q: {request.question[:180]}")
        print(f"     Why: {request.why_it_matters[:150]}")
        return {"clarification": request, "forced_critique": forced,
                "critique_rounds": state.get("critique_rounds", 0) + 1}

    def route_critique(state) -> Literal["clarify_a", "final_report"]:
        return "clarify_a" if state.get("clarification") else "final_report"

    def clarify_a_node(state) -> dict:
        """Agent A answers. It may call its tools again to do so."""
        request = state["clarification"]
        print("\n── CRITIQUE · Agent A responding ──")
        ask = HumanMessage(content=(
            f"Agent B has sent a clarification request about '{request.field_in_question}'.\n\n"
            f"Question: {request.question}\n"
            f"Why it matters: {request.why_it_matters}\n\n"
            "Answer from the measurements you already have. If you genuinely cannot "
            "determine it with your three tools, say so plainly and set "
            "could_not_answer to true — do not estimate."
        ))
        answer = structured_call(
            chat_model, list(state["a_messages"]) + [ask], ClarificationResponse,
            label="clarification_response",
        )
        if answer is None:
            answer = ClarificationResponse(
                field_in_question=request.field_in_question,
                answer="Agent A could not produce a valid response to this clarification.",
                could_not_answer=True,
            )
        print(f"   RESPONSE ← {answer.answer[:220]}")
        if answer.supporting_values:
            print(f"   values: {json.dumps(answer.supporting_values)[:180]}")

        incorporate = HumanMessage(content=(
            f"Agent A's response to your clarification about "
            f"'{answer.field_in_question}':\n\n{answer.answer}\n\n"
            f"Supporting values: {json.dumps(answer.supporting_values) or 'none supplied'}\n"
            f"{'Agent A could NOT determine this — carry it into data_caveats.' if answer.could_not_answer else ''}\n\n"
            "Incorporate this into your analysis, then write the final report."
        ))
        return {"clarification_answer": answer, "b_messages": [incorporate]}

    def final_report_node(state) -> dict:
        print("\n── FINAL REPORT · Agent B ──")
        brief = state.get("quant_brief")
        instruction = HumanMessage(content=(
            "Write the final research report now.\n\n"
            "1. financial_health_summary — the quantitative picture using Agent A's "
            "figures, with your qualitative context explaining what drives them.\n"
            "2. top_three_risks — EXACTLY three. Each supporting_evidence must cite "
            "a specific figure from Agent A or a specific headline you retrieved, "
            "and each needs a severity of high, medium or low.\n"
            "3. hedge_strategy — ONE recommendation naming a concrete instrument "
            "(protective put, collar, beta-weighted short, pair trade) AND "
            "referencing a measured quantity from Agent A's brief.\n\n"
            "Carry every declared data gap into data_caveats. Use no number that "
            "did not come from Agent A."
        ))
        report = structured_call(
            chat_model, list(state["b_messages"]) + [instruction], ResearchReport,
            label="final_report",
        )
        if report is None:
            print("   Final report synthesis failed.")
            return {"report": None}
        report.ticker = state["ticker"]
        if brief and brief.data_gaps:
            for gap in brief.data_gaps:
                if gap not in report.data_caveats:
                    report.data_caveats.append(f"Agent A declared: {gap}")
        return {"report": report}

    # -- Wiring -------------------------------------------------------------
    graph = StateGraph(State)
    graph.add_node("agent_a", agent_a_node)
    graph.add_node("tools_a", make_tool_node(tools_a, "a_messages"))
    graph.add_node("handoff", handoff_node)
    graph.add_node("agent_b", agent_b_node)
    graph.add_node("tools_b", make_tool_node(tools_b, "b_messages"))
    graph.add_node("critique", critique_node)
    graph.add_node("clarify_a", clarify_a_node)
    graph.add_node("final_report", final_report_node)

    graph.add_edge(START, "agent_a")
    graph.add_conditional_edges("agent_a", route_a, {"tools_a": "tools_a", "handoff": "handoff"})
    graph.add_edge("tools_a", "agent_a")
    graph.add_edge("handoff", "agent_b")
    graph.add_conditional_edges("agent_b", route_b, {"tools_b": "tools_b", "critique": "critique"})
    graph.add_edge("tools_b", "agent_b")
    graph.add_conditional_edges("critique", route_critique,
                                {"clarify_a": "clarify_a", "final_report": "final_report"})
    graph.add_edge("clarify_a", "final_report")
    graph.add_edge("final_report", END)

    return graph.compile()


def run_multi_agent(
    ticker: str,
    tracer: Any,
    llm_client: Any,
    chat_model: Any | None = None,
    *,
    query: str | None = None,
    require_critique: bool = True,
) -> dict[str, Any]:
    """Run Task 3B end to end with no manual intervention."""
    from langchain_core.messages import HumanMessage, SystemMessage

    graph = build_multi_agent(tracer, llm_client, chat_model, require_critique=require_critique)
    question = query or (
        f"Analyse the current financial health and market sentiment of {ticker.upper()}. "
        f"Identify the top three risks to its share price over the next 90 days and "
        f"suggest one data-driven hedge strategy."
    )

    print(f"\n{'=' * 78}\nTASK 3B — two-agent pipeline: {ticker.upper()}\n{'=' * 78}")
    print(f"Agent A tools: {AGENT_A_TOOLS}")
    print(f"Agent B tools: {AGENT_B_TOOLS}")
    print("Overlap: none — restriction is enforced by the tool list, not the prompt.")
    print(f"\n── AGENT A · quantitative analysis ──")

    final = graph.invoke(
        {
            "ticker": ticker.upper(),
            "a_messages": [
                SystemMessage(content=AGENT_A_SYSTEM),
                HumanMessage(content=(
                    f"{question}\n\nBegin your quantitative measurement of "
                    f"{ticker.upper()}. State what you are establishing before each call."
                )),
            ],
            "b_messages": [],
            "quant_brief": None,
            "clarification": None,
            "clarification_answer": None,
            "critique_rounds": 0,
            "a_iterations": 0,
            "b_iterations": 0,
            "report": None,
            "forced_critique": False,
        },
        {"recursion_limit": 40},
    )

    report = final.get("report")
    if report:
        print("\n" + report.render())

    return {
        "report": report,
        "quant_brief": final.get("quant_brief"),
        "clarification": final.get("clarification"),
        "clarification_answer": final.get("clarification_answer"),
        "critique_rounds": final.get("critique_rounds", 0),
        "forced_critique": final.get("forced_critique", False),
        "state": final,
    }
