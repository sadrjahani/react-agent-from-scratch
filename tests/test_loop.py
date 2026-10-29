
"""Tests for the ReAct agent loop. No real LLM or network is used."""

import pytest

from react_agent.loop import run_agent
from react_agent.tools import Tool


def make_tool(func=None):
    """Create a simple tool for isolated loop tests."""
    if func is None:
        func = lambda text: f"result: {text}"
    return Tool("calculator", "Evaluate a test input", func)


def make_llm(*responses):
    """Return a fake LLM that yields predefined responses and records calls."""
    calls = []
    remaining = iter(responses)

    def fake_llm(messages, system="", stop=None):
        calls.append({
            "messages": [message.copy() for message in messages],
            "system": system,
            "stop": stop,
        })
        return next(remaining)

    fake_llm.calls = calls
    return fake_llm


def test_direct_final_answer():
    llm = make_llm("Thought: I know the answer\nFinal Answer: 42")

    result = run_agent("What is the answer?", [make_tool()], llm=llm)

    assert result["answer"] == "42"
    assert result["finished"] is True
    assert len(result["trace"]) == 1
    assert len(llm.calls) == 1


def test_tool_observation_is_sent_back_to_llm():
    llm = make_llm(
        "Thought: calculate\nAction: calculator\nAction Input: 2+3",
        "Thought: now I know\nFinal Answer: 5",
    )

    result = run_agent("What is 2+3?", [make_tool()], llm=llm)

    assert result["answer"] == "5"
    assert result["finished"] is True
    assert len(llm.calls) == 2

    second_messages = llm.calls[1]["messages"]
    assert second_messages[-1] == {
        "role": "user",
        "content": "Observation: result: 2+3",
    }

    assert result["trace"][0]["observation"] == "result: 2+3"


def test_unknown_tool_returns_observation_and_can_recover():
    llm = make_llm(
        "Action: unknown_tool\nAction Input: hello",
        "Final Answer: recovered",
    )

    result = run_agent("Try a tool", [make_tool()], llm=llm)

    assert result["answer"] == "recovered"
    assert result["finished"] is True
    assert "unknown tool" in result["trace"][0]["observation"]
    assert len(llm.calls) == 2


def test_parser_error_is_returned_as_observation():
    llm = make_llm(
        "Action: calculator",
        "Final Answer: recovered",
    )

    result = run_agent("Try again", [make_tool()], llm=llm)

    assert result["answer"] == "recovered"
    assert result["finished"] is True
    assert "Action Input" in result["trace"][0]["observation"]


def test_tool_exception_becomes_observation():
    def broken_tool(_):
        raise ZeroDivisionError("division by zero")

    llm = make_llm(
        "Action: calculator\nAction Input: 1/0",
        "Final Answer: handled the error",
    )

    result = run_agent(
        "Divide by zero",
        [make_tool(broken_tool)],
        llm=llm,
    )

    assert result["answer"] == "handled the error"
    assert "tool failed" in result["trace"][0]["observation"]
    assert "division by zero" in result["trace"][0]["observation"]


def test_last_iteration_does_not_execute_tool():
    executed = []

    def tracked_tool(text):
        executed.append(text)
        return "result"

    llm = make_llm(
        "Action: calculator\nAction Input: 2+3",
        "Action: calculator\nAction Input: 4+5",
    )

    result = run_agent(
        "Calculate",
        [make_tool(tracked_tool)],
        llm=llm,
        max_iters=2,
    )

    assert result["answer"] is None
    assert result["finished"] is False
    assert len(llm.calls) == 2
    assert executed == ["2+3"]
    assert "observation" not in result["trace"][-1]


@pytest.mark.parametrize("max_iters", [0, -1])
def test_non_positive_max_iters_is_rejected(max_iters):
    llm = make_llm("Final Answer: never called")

    with pytest.raises(ValueError, match="max_iters"):
        run_agent(
            "Test",
            [make_tool()],
            llm=llm,
            max_iters=max_iters,
        )

    assert llm.calls == []

def test_duplicate_tool_names_are_rejected():
    tools = [
        Tool("calculator", "First calculator", lambda text: "first"),
        Tool("calculator", "Second calculator", lambda text: "second"),
    ]
    llm = make_llm("Final Answer: never called")

    with pytest.raises(ValueError, match="Duplicate tool name"):
        run_agent("Test", tools, llm=llm)

    assert llm.calls == []


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_tool_names_are_rejected(name):
    tool = Tool(name, "A test tool", lambda text: text)
    llm = make_llm("Final Answer: never called")

    with pytest.raises(ValueError, match="Tool name"):
        run_agent("Test", [tool], llm=llm)

    assert llm.calls == []


def test_empty_tool_description_is_rejected():
    tool = Tool("calculator", "  ", lambda text: text)
    llm = make_llm("Final Answer: never called")

    with pytest.raises(ValueError, match="description"):
        run_agent("Test", [tool], llm=llm)

    assert llm.calls == []


def test_non_callable_tool_func_is_rejected():
    tool = Tool("calculator", "A test tool", None)
    llm = make_llm("Final Answer: never called")

    with pytest.raises(TypeError, match="callable"):
        run_agent("Test", [tool], llm=llm)

    assert llm.calls == []


def test_tool_missing_required_attributes_is_rejected():
    class InvalidTool:
        name = "calculator"

    llm = make_llm("Final Answer: never called")

    with pytest.raises(TypeError, match="name, description, and func"):
        run_agent("Test", [InvalidTool()], llm=llm)

    assert llm.calls == []
