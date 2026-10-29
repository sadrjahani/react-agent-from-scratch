"""Parses raw model text into an action, a final answer or an error.

The keywords must match the format taught in prompt.py.
"""

import re
from typing import Literal, TypedDict

__all__ = ["ActionResult", "ErrorResult", "FinalResult", "Parsed", "parse"]


class ActionResult(TypedDict):
    type: Literal["action"]
    tool: str
    input: str


class FinalResult(TypedDict):
    type: Literal["final"]
    answer: str


class ErrorResult(TypedDict):
    type: Literal["error"]
    message: str


Parsed = ActionResult | FinalResult | ErrorResult

# Reasoning models (for example qwen3) may print their thoughts first. A stray
# "Action:" inside them must not count; an unclosed block is dropped to the end.
_THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:[ \t]*([^\n]+)\n+[ \t]*Action Input:[ \t]*([^\n]+)")
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _strip_quotes(value: str) -> str:
    """Remove one pair of matching quotes, which models often add around input."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def parse(text: str) -> Parsed:
    """Turn model output into an action, a final answer or an error.

    Never raises: bad output becomes an ``error`` result whose message the
    agent loop sends back to the model as an Observation.
    """
    text = _THINK_RE.sub("", text)
    action = _ACTION_RE.search(text)
    final = _FINAL_RE.search(text)

    # An Action wins over a Final Answer after it: the model may invent the
    # answer before the tool has run.
    if action and (not final or action.start() < final.start()):
        return {
            "type": "action",
            "tool": action.group(1).strip(),
            "input": _strip_quotes(action.group(2)),
        }

    if final and final.group(1).strip():
        return {"type": "final", "answer": final.group(1).strip()}

    if "Action:" in text:
        return {
            "type": "error",
            "message": "Found 'Action:' but no 'Action Input:' on the next line. "
            "Use the exact format.",
        }
    return {
        "type": "error",
        "message": "Could not find 'Action:' with 'Action Input:', or 'Final Answer:'. "
        "Reply in the exact format.",
    }