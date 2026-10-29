
"""ReAct agent loop: call the model, parse, run a tool, feed the Observation back."""

from react_agent.llm import chat
from react_agent.parser import parse
from react_agent.prompt import build_system_prompt


def _build_tool_map(tools):
    """Validate tool definitions and build a unique name-to-tool mapping."""
    tool_map = {}

    for tool in tools:
        if not all(hasattr(tool, attr) for attr in ("name", "description", "func")):
            raise TypeError(
                "Each tool must have name, description, and func attributes"
            )

        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ValueError("Tool name must be a non-empty string")

        if tool.name in tool_map:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")

        if not isinstance(tool.description, str) or not tool.description.strip():
            raise ValueError(f"Tool {tool.name!r} must have a non-empty description")

        if not callable(tool.func):
            raise TypeError(f"Tool {tool.name!r} func must be callable")

        tool_map[tool.name] = tool

    return tool_map


def run_agent(question, tools, llm=chat, max_iters=6):
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")

    tools = list(tools)
    tool_map = _build_tool_map(tools)
    system = build_system_prompt(tools)
    messages = [{"role": "user", "content": f"Question: {question}"}]
    trace = []

    for step in range(1, max_iters + 1):
        text = llm(messages, system=system, stop=["Observation:"])
        parsed = parse(text)

        trace.append({"step": step, "model": text})
        messages.append({"role": "assistant", "content": text})

        if parsed["type"] == "final":
            return {
                "answer": parsed["answer"],
                "trace": trace,
                "finished": True,
            }

        # Do not execute an action on the last model call:
        # there would be no next call to process its Observation.
        if step == max_iters:
            return {
                "answer": None,
                "trace": trace,
                "finished": False,
            }

        if parsed["type"] == "error":
            observation = f"Error: {parsed['message']}"
        elif parsed["tool"] not in tool_map:
            observation = (
                f"Error: unknown tool '{parsed['tool']}'. "
                f"Available: {', '.join(tool_map)}"
            )
        else:
            try:
                observation = str(
                    tool_map[parsed["tool"]].func(parsed["input"])
                )
            except Exception as e:
                observation = f"Error: tool failed: {e}"

        trace[-1]["observation"] = observation
        messages.append(
            {"role": "user", "content": f"Observation: {observation}"}
        )

    return {"answer": None, "trace": trace, "finished": False}
