import json
from groq import Groq
from dotenv import load_dotenv
from agent import run_agent, run_agent_stream, MODEL  # reuse the ReAct loop from agent.py

load_dotenv()

RESEARCHER_TASK_TEMPLATE = """You are the RESEARCHER agent in a two-agent pipeline.
Your job is ONLY to gather and verify facts using your tools — do NOT write a
polished report. Call final_answer with a structured dump of raw findings:
key numbers, facts, and sources, in bullet form.

Task to research: {task}
"""

WRITER_SYSTEM_PROMPT = """You are the WRITER agent in a two-agent pipeline.
You receive raw research findings (facts, numbers, sources) from a Researcher
agent. Your job is to turn them into a clear, well-structured final report for
the end user: an intro, a comparison, and a concise conclusion. You have no
tools — you only write. Do not invent facts beyond what was provided.
"""


def run_writer(raw_findings: str, original_task: str) -> str:
    client = Groq()
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original task: {original_task}\n\n"
                    f"Raw findings from Researcher agent:\n{raw_findings}\n\n"
                    "Write the final report."
                ),
            },
        ],
    )
    return response.choices[0].message.content or ""


def run_multi_agent(task: str) -> dict:
    # Step 1: Researcher agent gathers facts via the ReAct loop
    researcher_result = run_agent(RESEARCHER_TASK_TEMPLATE.format(task=task))
    raw_findings = researcher_result["answer"]

    print("\n\n===== HANDOFF: Researcher -> Writer =====")
    print(raw_findings)

    # Step 2: Writer agent turns findings into a polished report
    final_report = run_writer(raw_findings, task)

    return {
        "raw_findings": raw_findings,
        "final_report": final_report,
        "researcher_trace": researcher_result["trace"],
    }


def run_multi_agent_stream(task: str, max_steps: int = None, model: str = None):
    """
    Streaming version for UIs: yields researcher step events first (same shape
    as run_agent_stream), then a "handoff" event, then a "final" event with the
    Writer agent's polished report.
    """
    raw_findings = None
    researcher_trace = None

    for event in run_agent_stream(RESEARCHER_TASK_TEMPLATE.format(task=task), max_steps=max_steps, model=model):
        if event["type"] == "step":
            yield {**event, "agent": "researcher"}
        elif event["type"] == "error":
            yield event
            return
        elif event["type"] == "done":
            raw_findings = event["answer"]
            researcher_trace = event["trace"]

    yield {"type": "handoff", "raw_findings": raw_findings}

    try:
        final_report = run_writer(raw_findings, task)
    except Exception as e:
        yield {"type": "error", "message": f"Writer agent failed: {e}"}
        return

    yield {
        "type": "final",
        "raw_findings": raw_findings,
        "final_report": final_report,
        "researcher_trace": researcher_trace,
    }


if __name__ == "__main__":
    task = (
        "Compare the GDP growth of India vs Vietnam in 2024 and summarize "
        "which economy grew faster and by how many percentage points."
    )
    result = run_multi_agent(task)

    print("\n\n===== FINAL REPORT (Writer agent) =====")
    print(result["final_report"])

    with open("multi_agent_trace.json", "w") as f:
        json.dump(result["researcher_trace"], f, indent=2)