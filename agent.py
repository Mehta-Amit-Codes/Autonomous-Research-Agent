import os
import json
import ast
import operator
from dataclasses import dataclass, field
from typing import Any

from groq import Groq
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()  # reads .env file in the project root and loads it into os.environ


# ----------------------------------------------------------------------
# 1. TOOL DEFINITIONS  (OpenAI-style function schema, used by Groq)
# ----------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information, facts, statistics, or news. "
                "Use this when you need up-to-date or factual information you are not "
                "certain about. Returns a list of result snippets with titles and URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query. Be specific (e.g. 'India GDP growth rate 2024').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression. Supports +, -, *, /, **, %, parentheses. "
                "Use this for any math instead of doing it in your head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A math expression, e.g. '(7.2 - 6.1) / 6.1 * 100'.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": (
                "Call this ONLY when you have gathered enough information and are ready "
                "to give the complete final answer to the user's task. This ends the loop."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The complete, well-written final answer / report.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


# ----------------------------------------------------------------------
# 2. TOOL IMPLEMENTATIONS (execute + observe)
# ----------------------------------------------------------------------

def tool_web_search(query: str, max_results: int = 5) -> str:
    """Free web search via DuckDuckGo. Returns formatted snippets."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', '')}\n"
                f"    {r.get('body', '')}\n"
                f"    URL: {r.get('href', '')}"
            )
        return "\n".join(formatted)
    except Exception as e:
        return f"SEARCH_ERROR: {e}"


# Safe arithmetic evaluator (no eval() — avoids arbitrary code execution)
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def tool_calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"CALC_ERROR: {e}"


TOOL_IMPL = {
    "web_search": lambda inp: tool_web_search(inp["query"]),
    "calculator": lambda inp: tool_calculator(inp["expression"]),
}


# ----------------------------------------------------------------------
# 3. AGENT LOOP (ReAct: Thought -> Action -> Observation, repeat)
# ----------------------------------------------------------------------

MAX_STEPS = 8  # guardrail: hard cap to prevent infinite loops
# Groq deprecated llama-3.3-70b-versatile (Aug 2026). Current recommended
# general-purpose/tool-calling model per Groq's migration guidance:
# https://console.groq.com/docs/deprecations
MODEL = "openai/gpt-oss-120b"
# Faster/cheaper fallback if you hit rate limits: "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are an autonomous research agent.

You solve tasks by reasoning step by step and using the tools available to you.
For each step:
1. Think about what you need to find out or compute next.
2. Call exactly one tool to make progress (web_search, calculator, or final_answer).
3. You will be shown the tool's result (observation) before your next step.

Rules:
- Always ground factual claims in web_search results; don't guess statistics.
- Use calculator for any arithmetic (e.g. computing differences, percentages).
- When you have enough information, call final_answer with a complete, well-organized
  written answer (not just a number) that cites what you found.
- Be efficient: don't repeat identical searches.
"""


@dataclass
class Trace:
    """Records every thought -> action -> observation for debugging."""
    steps: list = field(default_factory=list)

    def log(self, step_num: int, thought: str, action: str, action_input: Any, observation: str):
        entry = {
            "step": step_num,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "observation": observation,
        }
        self.steps.append(entry)
        print(f"\n--- Step {step_num} ---")
        if thought:
            print(f"Thought: {thought}")
        print(f"Action: {action}({json.dumps(action_input)})")
        obs_preview = observation if len(observation) < 400 else observation[:400] + "...[truncated]"
        print(f"Observation: {obs_preview}")

    def save(self, path: str = "trace.json"):
        with open(path, "w") as f:
            json.dump(self.steps, f, indent=2)


def run_agent_stream(task: str, max_steps: int = None, model: str = None):
    """
    Generator version of the ReAct loop — yields one event dict per step so a UI
    (e.g. Streamlit) can render progress live instead of waiting for the full run.

    Yields dicts of shape:
      {"type": "step", "step": int, "thought": str, "action": str,
       "action_input": dict, "observation": str}
      {"type": "done", "answer": str, "steps_used": int, "stopped_reason": str, "trace": [...]}
      {"type": "error", "message": str}
    """
    max_steps = max_steps or MAX_STEPS
    model = model or MODEL

    try:
        client = Groq()
    except Exception as e:
        yield {"type": "error", "message": f"Could not initialize Groq client: {e}"}
        return

    trace = Trace()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=1500,
                tools=TOOLS,
                tool_choice="auto",
                messages=messages,
            )
        except Exception as e:
            yield {"type": "error", "message": f"API call failed at step {step}: {e}"}
            return

        msg = response.choices[0].message
        thought_text = msg.content or ""
        tool_calls = msg.tool_calls or []
        messages.append(msg.model_dump(exclude_none=True))

        if not tool_calls:
            trace.log(step, thought_text, "final_answer", {}, thought_text)
            yield {
                "type": "step", "step": step, "thought": thought_text,
                "action": "final_answer", "action_input": {}, "observation": thought_text,
            }
            yield {
                "type": "done", "answer": thought_text, "steps_used": step,
                "trace": trace.steps, "stopped_reason": "model_returned_text_without_tool_call",
            }
            return

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            if tool_name == "final_answer":
                answer = tool_input.get("answer", "")
                trace.log(step, thought_text, tool_name, tool_input, answer)
                yield {
                    "type": "step", "step": step, "thought": thought_text,
                    "action": tool_name, "action_input": tool_input, "observation": answer,
                }
                yield {
                    "type": "done", "answer": answer, "steps_used": step,
                    "trace": trace.steps, "stopped_reason": "final_answer_called",
                }
                return

            try:
                impl = TOOL_IMPL.get(tool_name)
                if impl is None:
                    observation = f"ERROR: unknown tool '{tool_name}'"
                else:
                    observation = impl(tool_input)
            except Exception as e:
                observation = f"ERROR: tool '{tool_name}' raised an exception: {e}"

            trace.log(step, thought_text, tool_name, tool_input, observation)
            yield {
                "type": "step", "step": step, "thought": thought_text,
                "action": tool_name, "action_input": tool_input, "observation": observation,
            }

            messages.append(
                {"role": "tool", "tool_call_id": tool_call.id, "content": observation}
            )

    trace.log(max_steps, "", "STOP", {}, "Max step limit reached without final_answer.")
    yield {
        "type": "done",
        "answer": "Agent stopped: reached max step limit before producing a final answer.",
        "steps_used": max_steps, "trace": trace.steps, "stopped_reason": "max_steps_reached",
    }


def run_agent(task: str, verbose: bool = True) -> dict:
    """
    Non-streaming wrapper around run_agent_stream — runs the loop to completion
    and returns the final result dict. Kept for CLI / programmatic use.
    """
    for event in run_agent_stream(task):
        if event["type"] == "error":
            raise RuntimeError(event["message"])
        if event["type"] == "done":
            return event


# ----------------------------------------------------------------------
# 4. DEMO
# ----------------------------------------------------------------------

if __name__ == "__main__":
    task = (
        "Compare the GDP growth of India vs Vietnam in 2024 and summarize "
        "which economy grew faster and by how many percentage points."
    )
    result = run_agent(task)

    print("\n\n===== FINAL ANSWER =====")
    print(result["answer"])
    print(f"\n(steps used: {result['steps_used']}, stop reason: {result['stopped_reason']})")

    # Save full trace for debugging / grading
    with open("trace.json", "w") as f:
        json.dump(result["trace"], f, indent=2)
    print("\nFull trace saved to trace.json")