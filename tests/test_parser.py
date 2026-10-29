"""Tests for react_agent.parser."""

import pytest

from react_agent.parser import parse


def test_parses_action() -> None:
    text = "Thought: need math\nAction: calculator\nAction Input: 2+3*4"
    assert parse(text) == {"type": "action", "tool": "calculator", "input": "2+3*4"}


def test_action_tolerates_extra_spaces_and_windows_line_endings() -> None:
    text = "Action:   calculator  \r\nAction Input:   2+3  \r\n"
    assert parse(text) == {"type": "action", "tool": "calculator", "input": "2+3"}


@pytest.mark.parametrize("raw", ['"2+3"', "'2+3'"])
def test_strips_matching_quotes_from_input(raw: str) -> None:
    result = parse(f"Action: calculator\nAction Input: {raw}")
    assert result["input"] == "2+3"


def test_parses_final_answer() -> None:
    text = "Thought: I now know the final answer\nFinal Answer: 14"
    assert parse(text) == {"type": "final", "answer": "14"}


def test_multiline_final_answer_is_kept() -> None:
    result = parse("Final Answer: line one\nline two")
    assert result == {"type": "final", "answer": "line one\nline two"}


def test_action_before_final_answer_wins() -> None:
    """The model may invent an answer before the tool has run."""
    text = "Action: calculator\nAction Input: 1+1\nFinal Answer: 2"
    assert parse(text)["type"] == "action"


def test_action_without_input_is_an_error_that_names_the_problem() -> None:
    result = parse("Action: calculator")
    assert result["type"] == "error"
    assert "Action Input" in result["message"]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I think the answer is 14.",
        "Final Answer:",
        "Final Answer:   \n",
    ],
)
def test_unusable_output_is_an_error(text: str) -> None:
    assert parse(text)["type"] == "error"


def test_think_block_is_ignored() -> None:
    text = "<think>Action: calculator\nAction Input: 1+1</think>\nFinal Answer: 4"
    assert parse(text) == {"type": "final", "answer": "4"}


def test_unclosed_think_block_hides_everything_after_it() -> None:
    text = "<think>Action: calculator\nAction Input: 1+1"
    assert parse(text)["type"] == "error"