import json
import time
from datetime import datetime

import streamlit as st

from agent import run_agent_stream, MAX_STEPS as DEFAULT_MAX_STEPS, MODEL as DEFAULT_MODEL
from multi_agent import run_multi_agent_stream


# ----------------------------------------------------------------------
# Page config + light styling
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

TOOL_ICONS = {
    "web_search": "🔍",
    "calculator": "🧮",
    "final_answer": "✅",
    "STOP": "⛔",
}

st.markdown(
    """
    <style>
    .step-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        background: rgba(128,128,128,0.04);
    }
    .step-badge {
        display: inline-block;
        padding: 0.1rem 0.6rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(99,102,241,0.15);
        color: rgb(99,102,241);
        margin-right: 0.5rem;
    }
    .agent-badge-researcher {
        background: rgba(59,130,246,0.15);
        color: rgb(59,130,246);
    }
    .agent-badge-writer {
        background: rgba(16,185,129,0.15);
        color: rgb(16,185,129);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of past run dicts
if "running" not in st.session_state:
    st.session_state.running = False


# ----------------------------------------------------------------------
# Sidebar: configuration
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Agent Configuration")

    mode = st.radio(
        "Mode",
        options=["Single Agent (ReAct)", "Multi-Agent (Researcher → Writer)"],
        help="Single agent does everything itself. Multi-agent hands raw findings "
             "from a Researcher agent to a separate Writer agent for the final report.",
    )

    model = st.selectbox(
        "Model",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
        index=0,
        help="gpt-oss-120b: best quality. gpt-oss-20b: faster/cheaper. "
             "qwen3.6-27b: multimodal, currently a preview model on Groq.",
    )

    max_steps = st.slider(
        "Max steps (guardrail)",
        min_value=2, max_value=15, value=DEFAULT_MAX_STEPS,
        help="Hard cap on ReAct loop iterations, to prevent infinite loops.",
    )

    show_raw_json = st.toggle("Show raw JSON per step", value=False)
    auto_scroll_delay = st.slider(
        "Step reveal delay (sec)", 0.0, 1.5, 0.15, 0.05,
        help="Small artificial delay between steps so the reasoning is easier to follow live.",
    )

    st.markdown("---")
    st.markdown("### 📜 Run History")
    if st.session_state.history:
        for i, run in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - i
            label = f"#{idx} · {run['task'][:32]}{'...' if len(run['task']) > 32 else ''}"
            if st.button(label, key=f"hist_{idx}", use_container_width=True):
                st.session_state.selected_history = idx - 1
    else:
        st.caption("No runs yet.")

    if st.session_state.history and st.button("🗑️ Clear history", use_container_width=True):
        st.session_state.history = []
        st.session_state.pop("selected_history", None)
        st.rerun()


# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("🔎 Autonomous Research Agent")
st.caption(
    "An LLM agent that reasons step by step (ReAct: Thought → Action → Observation), "
    "calling web search and a calculator until it can give you a grounded final answer."
)

# Example task chips
st.markdown("**Try an example, or write your own task below:**")
examples = [
    "Compare the GDP growth of India vs Vietnam in 2024 and summarize which grew faster.",
    "What is the current population of Japan, and how has it changed over the last 5 years?",
    "Find the current price of Bitcoin and calculate what a $1,000 investment a year ago would be worth today.",
    "Who won the most recent Formula 1 World Championship, and by how many points?",
]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex[:40] + "...", key=f"ex_{i}", use_container_width=True, help=ex):
        st.session_state.task_input = ex


# ----------------------------------------------------------------------
# Task input form
# ----------------------------------------------------------------------

with st.form("task_form", clear_on_submit=False):
    task = st.text_area(
        "Research task",
        key="task_input",
        placeholder="e.g. Compare the GDP growth of India vs Vietnam in 2024 and summarize...",
        height=100,
    )
    submitted = st.form_submit_button(
        "🚀 Run Agent", use_container_width=True, disabled=st.session_state.running
    )


# ----------------------------------------------------------------------
# Helpers to render a step / run
# ----------------------------------------------------------------------

def render_step(container, event, show_json=False):
    """Render a single step event as a card."""
    step_num = event.get("step")
    action = event.get("action", "")
    thought = event.get("thought", "")
    action_input = event.get("action_input", {})
    observation = event.get("observation", "")
    agent_name = event.get("agent")  # "researcher" / "writer" in multi-agent mode

    icon = TOOL_ICONS.get(action, "🔧")
    badge_class = "step-badge"
    agent_label = ""
    if agent_name:
        badge_class += f" agent-badge-{agent_name}"
        agent_label = f" · {agent_name.title()}"

    with container.container():
        st.markdown(
            f'<div class="step-card">'
            f'<span class="{badge_class}">Step {step_num}{agent_label}</span>'
            f'<b>{icon} {action}</b>'
            f"</div>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Details — step {step_num}", expanded=(action == "final_answer")):
            if thought:
                st.markdown("**🧠 Thought**")
                st.write(thought)
            if action_input:
                st.markdown("**📥 Action Input**")
                st.code(json.dumps(action_input, indent=2), language="json")
            st.markdown("**👁️ Observation**")
            obs_display = observation if len(observation) < 1500 else observation[:1500] + "\n...[truncated]"
            st.text(obs_display)
            if show_json:
                st.markdown("**Raw event**")
                st.json(event)


def render_finished_run(run: dict):
    """Render a completed run (from history or just-finished) in full."""
    st.subheader("📋 Final Answer")
    if run["mode"] == "multi":
        st.info("Multi-agent mode: shown below is the Writer agent's polished report.")
    st.markdown(run["answer"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Steps used", run.get("steps_used", len(run.get("trace", []))))
    tool_calls = sum(1 for s in run.get("trace", []) if s.get("action") not in ("final_answer", "STOP"))
    m2.metric("Tool calls", tool_calls)
    m3.metric("Stop reason", run.get("stopped_reason", "—"))

    if run["mode"] == "multi" and run.get("raw_findings"):
        with st.expander("🔬 Researcher agent's raw findings (pre-handoff)"):
            st.write(run["raw_findings"])

    st.subheader("🪜 Full Trace")
    for i, step in enumerate(run.get("trace", [])):
        render_step(st, step, show_json=show_raw_json)

    st.subheader("⬇️ Export")
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download final answer (.md)",
        data=run["answer"],
        file_name=f"answer_{run['id']}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    c2.download_button(
        "Download full trace (.json)",
        data=json.dumps(run.get("trace", []), indent=2),
        file_name=f"trace_{run['id']}.json",
        mime="application/json",
        use_container_width=True,
    )


# ----------------------------------------------------------------------
# Run the agent (streamed live into the UI)
# ----------------------------------------------------------------------

if submitted and task.strip():
    st.session_state.running = True
    st.session_state.pop("selected_history", None)

    status = st.status("Agent is thinking...", expanded=True)
    step_area = st.container()
    progress_bar = st.progress(0.0)

    trace_events = []
    final_answer_text = None
    stopped_reason = None
    steps_used = 0
    raw_findings = None
    error_msg = None
    mode_key = "multi" if mode.startswith("Multi") else "single"

    stream = (
        run_multi_agent_stream(task, max_steps=max_steps, model=model)
        if mode_key == "multi"
        else run_agent_stream(task, max_steps=max_steps, model=model)
    )

    try:
        for event in stream:
            if event["type"] == "error":
                error_msg = event["message"]
                status.update(label="❌ Agent hit an error", state="error", expanded=True)
                st.error(error_msg)
                break

            elif event["type"] == "step":
                trace_events.append(event)
                steps_used = event["step"]
                status.update(label=f"Step {steps_used}: running `{event['action']}`...")
                render_step(step_area, event, show_json=show_raw_json)
                progress_bar.progress(min(steps_used / max_steps, 1.0))
                if auto_scroll_delay:
                    time.sleep(auto_scroll_delay)

            elif event["type"] == "handoff":
                raw_findings = event["raw_findings"]
                status.update(label="🤝 Handing off to Writer agent...")
                with step_area:
                    st.markdown("---")
                    st.markdown("**🤝 Handoff: Researcher → Writer**")
                    with st.expander("Raw findings passed to Writer", expanded=False):
                        st.write(raw_findings)
                    st.markdown("---")

            elif event["type"] == "done":
                final_answer_text = event["answer"]
                stopped_reason = event["stopped_reason"]
                steps_used = event["steps_used"]

            elif event["type"] == "final":
                final_answer_text = event["final_report"]
                raw_findings = event["raw_findings"]
                trace_events = event["researcher_trace"]
                stopped_reason = "final_answer_called"
                steps_used = len(trace_events)

        if error_msg is None:
            status.update(label="✅ Done!", state="complete", expanded=False)
            progress_bar.progress(1.0)

    except Exception as e:
        status.update(label="❌ Unexpected error", state="error")
        st.error(f"Something went wrong: {e}")
        error_msg = str(e)

    st.session_state.running = False

    if error_msg is None and final_answer_text is not None:
        run_record = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "task": task,
            "mode": mode_key,
            "answer": final_answer_text,
            "raw_findings": raw_findings,
            "trace": trace_events,
            "steps_used": steps_used,
            "stopped_reason": stopped_reason,
            "model": model,
        }
        st.session_state.history.append(run_record)

        st.markdown("---")
        render_finished_run(run_record)

elif submitted and not task.strip():
    st.warning("Please enter a research task first.")


# ----------------------------------------------------------------------
# Viewing a past run from history (when nothing is currently running)
# ----------------------------------------------------------------------

if not submitted and "selected_history" in st.session_state and st.session_state.history:
    idx = st.session_state.selected_history
    if 0 <= idx < len(st.session_state.history):
        st.markdown("---")
        st.markdown(f"### 🕘 Viewing past run #{idx + 1}")
        render_finished_run(st.session_state.history[idx])