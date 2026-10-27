
import pytest

from react_agent import tools
from react_agent.tools import calculator, file_lookup, search


def test_calculator_precedence():
    assert calculator("2+3*4") == "14"


def test_calculator_float():
    assert calculator("7/2") == "3.5"


def test_calculator_rejects_code():
    with pytest.raises(ValueError):
        calculator("__import__('os').getcwd()")


def test_calculator_rejects_huge_power():
    with pytest.raises(ValueError):
        calculator("9**9**9")


def test_search_found_and_missing():
    assert "330" in search("What is the height of Eiffel Tower?")
    assert search("something unknown") == "No results found."


def test_file_lookup_reads_file(tmp_path, monkeypatch):
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    monkeypatch.setattr(tools, "DATA_DIR", tmp_path)
    assert file_lookup("a.txt") == "hello"


def test_file_lookup_blocks_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DATA_DIR", tmp_path)
    assert "outside" in file_lookup("../pyproject.toml")


def test_file_lookup_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "DATA_DIR", tmp_path)
    assert "not found" in file_lookup("nope.txt")