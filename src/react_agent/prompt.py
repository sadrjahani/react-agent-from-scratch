"""Builds the ReAct system prompt from the list of available tools.

The keywords in the template (``Thought:``, ``Action:``, ``Action Input:``,
``Observation:``, ``Final Answer:``) are a contract with ``parser.py``: if one
changes here, the parser must change too.
"""

from collections.abc import Sequence

from react_agent.tools import Tool

__all__ = ["PROMPT_TEMPLATE", "build_system_prompt"]

# Plain English on purpose: small models follow an English format more reliably.
# Placeholders use str.format, so never write a literal "{" or "}" in this text
# (write "{{" and "}}" if you ever need one).
PROMPT_TEMPLATE = """You are a helpful assistant that solves tasks step by step using tools.

You have access to these tools:
{tool_descriptions}

Use exactly this format:

Thought: reason about what to do next
Action: the tool name, one of [{tool_names}]
Action Input: the input for the tool
Observation: the result of the tool (written by the system, NOT by you)
... (Thought/Action/Action Input/Observation can repeat)
Thought: I now know the final answer
Final Answer: the answer to the original question

Rules:
- Write only ONE Action per reply, then stop and wait for the Observation.
- Never write the Observation yourself.
- If you can answer without a tool, go straight to Thought and Final Answer.
- If an Observation starts with "Error", read it, fix your Action or Action Input, and try again.
"""


def build_system_prompt(tools: Sequence[Tool]) -> str:
    """Fill the template with the names and descriptions of ``tools``.

    Args:
        tools: Tools the model may call. The model only knows a tool through
            its ``name`` and ``description``.

    Returns:
        The system prompt text to pass to the model.

    Raises:
        ValueError: If ``tools`` is empty (a ReAct agent without tools makes
            no sense, and ``Action: ... one of []`` would confuse the model).
    """
    if not tools:
        raise ValueError("at least one tool is required")
    tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    tool_names = ", ".join(t.name for t in tools)
    return PROMPT_TEMPLATE.format(
        tool_descriptions=tool_descriptions,
        tool_names=tool_names,
    )