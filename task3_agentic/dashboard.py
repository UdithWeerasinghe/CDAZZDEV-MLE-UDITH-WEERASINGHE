"""
Task 3C bonus - Streamlit dashboard over agent_trace.jsonl.

    pip install streamlit pandas
    streamlit run task3_agentic/dashboard.py

Reads the trace produced by `tracing.AgentTracer` and renders a run visually:
per-run summary metrics, a call-sequence timeline coloured by success, latency
by tool, and the raw records with their inputs and outputs.

The bonus offers a choice between integrating LangSmith and building this.
LangSmith would be less work, but it needs an account, ships trace data to a
third party, and shows *its* view of the run rather than the trace the brief
actually specifies. This reads the required artefact directly, so what you see
on screen is exactly the file that was committed - no divergence between the
deliverable and the dashboard over it.

# AI-ASSISTED: Claude (claude-sonnet-5), Prompt: 'Build a Streamlit dashboard
# reading agent_trace.jsonl showing a call timeline, per-tool latency and
# failure highlighting', Date: 2026-09-10
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_TRACE = Path(__file__).parent / "logs" / "agent_trace.jsonl"

st.set_page_config(page_title="Agent Trace", page_icon="🔎", layout="wide")


@st.cache_data(ttl=5)
def load_trace(path: str) -> pd.DataFrame:
    """Read JSONL, tolerating malformed lines rather than failing the page."""
    file = Path(path)
    if not file.exists():
        return pd.DataFrame()

    rows, malformed = [], 0
    for line in file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            malformed += 1

    if malformed:
        st.warning(f"Skipped {malformed} malformed trace line(s).")
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


st.title("Agent trace")
st.caption("CDAZZDEV Senior MLE assessment · Task 3C observability")

with st.sidebar:
    st.header("Source")
    trace_path = st.text_input("Trace file", value=str(DEFAULT_TRACE))
    if st.button("Reload"):
        st.cache_data.clear()

frame = load_trace(trace_path)

if frame.empty:
    st.info(
        "No trace records found. Run the Task 3 notebook first — it writes to "
        "`task3_agentic/logs/agent_trace.jsonl`."
    )
    st.stop()

# --- Run selection ---------------------------------------------------------
runs = (
    frame.groupby("run_id")
    .agg(calls=("seq", "size"), started=("timestamp", "min"),
         failures=("ok", lambda s: int((~s).sum())))
    .sort_values("started", ascending=False)
)
labels = {
    run: f"{run}  ·  {row.calls} calls  ·  "
         f"{row.started.strftime('%Y-%m-%d %H:%M') if pd.notna(row.started) else 'unknown'}"
         f"{'  ·  ' + str(row.failures) + ' failed' if row.failures else ''}"
    for run, row in runs.iterrows()
}

with st.sidebar:
    st.header("Run")
    choice = st.selectbox("Select", options=["All runs"] + list(runs.index),
                          format_func=lambda r: labels.get(r, r))
    agents = sorted(frame["agent"].dropna().unique())
    selected_agents = st.multiselect("Agents", agents, default=agents)
    show_failures_only = st.checkbox("Failures only", value=False)

view = frame if choice == "All runs" else frame[frame["run_id"] == choice]
if selected_agents:
    view = view[view["agent"].isin(selected_agents)]
if show_failures_only:
    view = view[~view["ok"]]

if view.empty:
    st.warning("No records match the current filters.")
    st.stop()

# --- Headline metrics ------------------------------------------------------
columns = st.columns(5)
columns[0].metric("Tool calls", len(view))
columns[1].metric("Failures", int((~view["ok"]).sum()),
                  delta=None if not (~view["ok"]).any() else "needs attention",
                  delta_color="inverse")
columns[2].metric("Total time", f"{view['duration_ms'].sum() / 1000:.1f}s")
columns[3].metric("Slowest call", f"{view['duration_ms'].max():.0f}ms")
columns[4].metric("Distinct tools", view["tool"].nunique())

st.divider()

# --- Charts ----------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Latency by tool")
    latency = (
        view.groupby("tool")["duration_ms"]
        .agg(["count", "mean", "max"])
        .rename(columns={"count": "calls", "mean": "mean_ms", "max": "max_ms"})
        .round(1)
        .sort_values("mean_ms", ascending=False)
    )
    st.bar_chart(latency["mean_ms"], height=260)
    st.dataframe(latency, use_container_width=True)

with right:
    st.subheader("Call sequence")
    timeline = view.sort_values("seq").copy()
    timeline["label"] = (
        timeline["seq"].astype(str) + ". " + timeline["agent"].fillna("agent")
        + ":" + timeline["tool"]
    )
    timeline["status"] = timeline["ok"].map({True: "ok", False: "failed"})
    st.dataframe(
        timeline[["label", "duration_ms", "status"]].set_index("label"),
        use_container_width=True, height=300,
        column_config={
            "duration_ms": st.column_config.ProgressColumn(
                "duration (ms)", format="%.0f ms",
                min_value=0, max_value=float(max(timeline["duration_ms"].max(), 1)),
            ),
        },
    )

# --- Per-agent breakdown ---------------------------------------------------
if view["agent"].nunique() > 1:
    st.subheader("Tool access by agent")
    st.caption(
        "Task 3B requires enforced, separate tool access. This is the observed "
        "record of which agent actually called what."
    )
    matrix = pd.crosstab(view["agent"], view["tool"])
    st.dataframe(matrix, use_container_width=True)

st.divider()

# --- Records ---------------------------------------------------------------
st.subheader("Records")
for row in view.sort_values("seq").itertuples():
    status = "✅" if row.ok else "❌"
    truncated = " (truncated)" if getattr(row, "output_truncated", False) else ""
    with st.expander(
        f"{status}  #{row.seq}  {row.agent}:{row.tool}  ·  {row.duration_ms:.0f}ms{truncated}"
    ):
        left_col, right_col = st.columns([1, 2])
        with left_col:
            st.caption("Inputs")
            st.json(row.inputs)
            if not row.ok and getattr(row, "error", None):
                st.error(row.error)
        with right_col:
            st.caption(
                f"Output (first 200 chars of {getattr(row, 'output_full_length', '?')})"
            )
            st.code(row.output, language="json")

with st.sidebar:
    st.divider()
    st.caption(f"{len(frame)} records · {frame['run_id'].nunique()} runs")
    st.download_button(
        "Download filtered trace (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="agent_trace_filtered.csv",
        mime="text/csv",
    )
