# Autonomous Research Agent (Tool-Using Agent)

A minimal but complete implementation of the ReAct (Reason → Act → Observe) agent
pattern using native tool/function calling.

## Files
- `agent.py` — core single-agent ReAct loop (search + calculator + final_answer tools)
- `multi_agent.py` — stretch goal: Researcher agent hands off findings to a Writer agent
- `streamlit_app.py` — interactive web UI: live step-by-step streaming, mode/model
  switcher, run history, and export buttons
- `requirements.txt` — dependencies

## Setup
1. Get a **free** Groq API key at https://console.groq.com/keys (no credit card needed).
2. Copy `.env.example` to `.env` and paste your key in:
   ```
   GROQ_API_KEY=gsk_your_real_key_here
   ```
3. Install dependencies and run:
```bash
pip install -r requirements.txt
python agent.py
```

## Web UI (Streamlit)

For an interactive experience instead of the terminal, run:
```bash
streamlit run streamlit_app.py
```
This opens a browser tab where you can:
- Type (or click an example) research task and hit **Run Agent**
- Watch each Thought → Action → Observation step stream in live, in real time,
  with an icon per tool (🔍 search, 🧮 calculator, ✅ final answer)
- Switch between **Single Agent** and **Multi-Agent (Researcher → Writer)** mode
  from the sidebar
- Switch models (`gpt-oss-120b` / `gpt-oss-20b` / `qwen3.6-27b`) and the max-step
  guardrail without touching code
- Expand any step to see the raw tool input/output, or toggle raw JSON per step
- See run metrics (steps used, tool calls, stop reason) once finished
- Browse **run history** in the sidebar and revisit any past run
- **Download** the final answer as Markdown or the full trace as JSON

## How it works
1. **Tools** are defined with a name, description, and JSON input schema
   (`web_search`, `calculator`, `final_answer`).
2. **Agent loop**: the model is called with the task + tool list. It either
   emits a `tool_use` block (an action) or plain text.
3. **Execute + observe**: our code runs the chosen tool, wraps the result as a
   `tool_result`, and appends it to the conversation so the model can decide
   its next step.
4. **Stop condition**: the loop ends when the model calls `final_answer`, or
   after `MAX_STEPS` (default 8) — whichever comes first. This guardrail
   prevents infinite loops if the model keeps calling tools forever.
5. **Trace**: every step logs `thought → action → observation` to the console
   and to `trace.json` for debugging.

## Swapping components
- **Search provider**: `tool_web_search` uses free DuckDuckGo search
  (`duckduckgo-search`). Swap in Tavily by replacing that function's body.
- **LLM**: uses Groq's OpenAI-compatible chat completions API with native tool
  calling (free tier, very fast inference). To switch models, change `MODEL`
  in `agent.py` — any Groq model that supports tool calling works (see
  https://console.groq.com/docs/models). To swap providers entirely (e.g. back
  to Anthropic, or to NVIDIA NIM / another OpenAI-compatible endpoint), replace
  the `client.chat.completions.create` call and adapt the tool-call parsing —
  the ReAct loop structure itself stays the same.
- **Orchestration framework**: this is hand-rolled to show the mechanics
  explicitly. The same loop maps directly onto LangGraph (as a graph node with
  a conditional edge back to itself) or CrewAI (as an Agent with a Task) if
  you want to move to a framework.

## Stretch: multi-agent
`multi_agent.py` splits the work into two agents:
- **Researcher**: runs the full ReAct loop, outputs raw findings only.
- **Writer**: no tools, just turns those findings into a polished report.

This is the simplest form of "orchestration" — a fixed sequential handoff.
A next step would be a supervisor/router agent that decides dynamically which
sub-agent to call next.