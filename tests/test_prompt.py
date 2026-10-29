"""Tests for react_agent.prompt: tool listing and the format contract with parser.py."""

import pytest

from react_agent.prompt import build_system_prompt
from react_agent.tools import TOOLS, Tool


def _tool(name: str = "calculator", description: str = "Evaluate a math expression") -> Tool:
    return Tool(name, description, lambda text: text)


def test_prompt_lists_every_tool_name_and_description() -> None:
    prompt = build_system_prompt(TOOLS)
    for tool in TOOLS:
        assert f"- {tool.name}: {tool.description}" in prompt


def test_prompt_lists_allowed_action_names() -> None:
    prompt = build_system_prompt(TOOLS)
    assert "one of [calculator, search, file_lookup]" in prompt


@pytest.mark.parametrize(
    "keyword",
    ["Thought:", "Action:", "Action Input:", "Observation:", "Final Answer:"],
)
def test_prompt_contains_format_keywords_used_by_parser(keyword: str) -> None:
    """If this fails, prompt.py and parser.py no longer agree on the format."""
    assert keyword in build_system_prompt([_tool()])


def test_prompt_has_no_unfilled_placeholders() -> None:
    prompt = build_system_prompt([_tool()])
    assert "{" not in prompt
    assert "}" not in prompt


def test_prompt_with_single_custom_tool() -> None:
    prompt = build_system_prompt([_tool("weather", "Get the weather")])
    assert "- weather: Get the weather" in prompt
    assert "one of [weather]" in prompt


def test_empty_tool_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_system_prompt([])