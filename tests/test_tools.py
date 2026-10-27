"""Tests for react_agent.tools: calculator safety, fake search, sandboxed files.

No LLM or network is involved; every test is deterministic.
"""

from pathlib import Path

import pytest

from react_agent import tools
from react_agent.tools import TOOLS, calculator, file_lookup, search


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the file sandbox at a temporary folder for one test.

    monkeypatch restores the real DATA_DIR afterwards, so tests never touch
    the project's real data/ folder.
    """
    monkeypatch.setattr(tools, "DATA_DIR", tmp_path)
    return tmp_path


# --- calculator ----------------------------------------------------------------
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2+3*4", "14"),  # operator precedence
        ("(1+2)*3", "9"),  # parentheses
        ("7/2", "3.5"),  # true division
        ("-(2+3)*2", "-10"),  # unary minus
        ("2**10", "1024"),  # power within the limit
    ],
)
def test_calculator_valid_expressions(expression: str, expected: str) -> None:
    assert calculator(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').getcwd()",  # code injection: function calls are blocked
        "x+1",  # names are not allowed
        "'a'+'b'",  # only numbers are allowed
        "True+1",  # bool must not count as a number
        "2 % 3",  # operator not in the whitelist
        "9**9**9",  # would freeze the program without the exponent limit
        "2**101",  # just above the exponent limit
        "1+" * 150 + "1",  # longer than the expression length limit
    ],
)
def test_calculator_rejects_unsafe_input(expression: str) -> None:
    with pytest.raises(ValueError):
        calculator(expression)


def test_calculator_division_by_zero_is_raised() -> None:
    """The tool lets this propagate; loop.py turns it into an Observation."""
    with pytest.raises(ZeroDivisionError):
        calculator("1/0")


def test_calculator_invalid_syntax_is_raised() -> None:
    with pytest.raises(SyntaxError):
        calculator("2+")


# --- search --------------------------------------------------------------------
def test_search_finds_fact_inside_a_longer_query() -> None:
    assert "330" in search("What is the height of Eiffel Tower?")


def test_search_returns_message_when_nothing_matches() -> None:
    assert search("something unknown") == "No results found."


# --- file_lookup ---------------------------------------------------------------
def test_file_lookup_reads_file(data_dir: Path) -> None:
    (data_dir / "a.txt").write_text("hello", encoding="utf-8")
    assert file_lookup("a.txt") == "hello"


def test_file_lookup_truncates_long_files(data_dir: Path) -> None:
    (data_dir / "big.txt").write_text("a" * (tools.MAX_FILE_CHARS + 50), encoding="utf-8")
    assert len(file_lookup("big.txt")) == tools.MAX_FILE_CHARS


@pytest.mark.parametrize("filename", ["../secret.txt", "../../etc/passwd"])
def test_file_lookup_blocks_path_traversal(data_dir: Path, filename: str) -> None:
    """'..' must not let the model escape the sandbox folder."""
    assert "outside" in file_lookup(filename)


def test_file_lookup_blocks_absolute_paths(data_dir: Path) -> None:
    """An absolute path replaces the base path, so it must be rejected too."""
    outside = data_dir.parent / "outside.txt"
    assert "outside" in file_lookup(str(outside))


def test_file_lookup_reports_missing_file(data_dir: Path) -> None:
    assert "not found" in file_lookup("nope.txt")


def test_file_lookup_reports_non_utf8_file(data_dir: Path) -> None:
    # 0xFF 0xFE 0x00 0x80 is not valid UTF-8, so decoding must fail.
    (data_dir / "bin.dat").write_bytes(b"\xff\xfe\x00\x80")
    assert "not a UTF-8" in file_lookup("bin.dat")


# --- registry (contract with prompt.py and loop.py) -----------------------------
def test_registry_has_expected_unique_tool_names() -> None:
    names = [t.name for t in TOOLS]
    assert sorted(names) == ["calculator", "file_lookup", "search"]
    assert len(names) == len(set(names))  # loop.py builds a dict by name


def test_every_tool_has_a_description() -> None:
    """The description is all the model knows about a tool."""
    assert all(t.description.strip() for t in TOOLS)