# REFLECTION

*Covers all three tasks. 594 words.*

## Architectural decisions

**Correctness before cleverness (Task 1).** The 25-mark indicator criterion looks trivial
and isn't. RSI needs Wilder's smoother, not `rolling(14).mean()` — that is Cutler's RSI, a
different indicator, and my tests measure them diverging 6.7 points on average, 22 at worst.
MACD needs `adjust=False` or the crossovers land in the wrong places. Bollinger needs the
population σ; pandas defaults to `ddof=1`, inflating a 2σ band by 2.6%. I verified each
against an independent pure-Python reference, because two derivations sharing an assumption
fail together silently.

**Enforcing rubric criteria in code, not prose (Task 1B).** The brief asks that the model
reason over indicator *combinations* rather than echo values. Rather than hope for it, the
`TradingSignal` validator rejects a justification unless it spans two named indicators and
contains relational language. Rejections are logged, the model gets one repair attempt, and
a deterministic fallback takes over if it still can't — flagged as such, because a degraded
honest answer beats a confident fabricated one.

**Structure over instruction (Task 3).** I chose LangGraph over CrewAI because three
criteria are about things being *enforced*, not claimed. Agent B cannot call the price tools
because the tool objects are never placed in its node's list — a system prompt saying "don't"
is a request; not passing the tool is a guarantee. Similarly the observe-replan cycle isn't
logged, it *is* the `agent → tools → agent` edge. More code than CrewAI; far easier to defend.

**Making silence impossible (Task 3B).** The `data_gaps` field on the handoff schema is
mandatory, not optional. Agent B cannot see Agent A's tools, so if a volatility call fails
and the field is optional, Agent A simply omits it and Agent B reads silence as "measured
and unremarkable" — then sizes a hedge on a number nobody took.

## What I'd improve with more time

Task 2's test set is synthetic and shares its teacher's blind spots. The format improvement
would transfer; the *accuracy* improvement is measured against a distribution the model has
effectively seen the shape of. I'd want 50 real clauses labelled by an analyst as a true
holdout, and I'd expect field accuracy to fall.

At ~96 examples a few points is inside the noise, and I report one run — a 3-seed sweep with
variance bars would be honest. The critique loop is capped at one round because I had no
principled convergence criterion. Both are real gaps, not design choices.

## Limitations encountered

The most instructive bug was Task 2A's duplicate guard. I used TF-IDF cosine similarity —
the obvious choice — and a test caught it: on a mode-collapsed corpus it scored 0.41 and
flagged 0.3% of pairs. IDF down-weights terms common to many documents, so when every
document is near-identical the shared text carries no weight and similarity is computed on
the tokens that *differ*. Plain term frequency scores 0.97 and flags 100%. The guard would
have accepted a zero-marks dataset while reporting a clean profile. I found it only because
I wrote a test asserting the analyser *fails* on bad input.

The second surfaced in a live run, not a test. Task 3 died on a provider 400,
`tool_use_failed`: the model was copying fifteen headlines verbatim into `llm_sentiment`'s
arguments and ran out of tokens mid-string. The instinct is to raise the token budget; the
fault was my signature. Tool arguments should be references, not payloads — the headlines
were already in context. Passing a ticker instead cut arguments from ~1,500 tokens to ~20,
removed the failure class, and closed a hole I had not considered: a model copying text can
silently paraphrase it.
