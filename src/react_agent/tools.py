"""Tools the agent can call: safe calculator, fake search, sandboxed file reader."""

import ast
import operator

# from typing import Callable
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression):
    tree = ast.parse(expression.strip(), mode="eval")
    return str(_eval_node(tree.body))


_FAKE_DB = {
    "capital of france": "Paris is the capital of France.",
    "population of france": "France has a population of about 68 million people.",
    "height of eiffel tower": "The Eiffel Tower is about 330 meters tall.",
    "creator of python": "Python was created by Guido van Rossum.",
}


def search(query):
    q = query.lower()
    for key, answer in _FAKE_DB.items():
        if key in q:
            return answer
    return "No results found."


def file_lookup(filename):
    base = DATA_DIR.resolve()
    path = (base / filename.strip()).resolve()
    if not path.is_relative_to(base):
        return "Error: access outside the data folder is not allowed."
    if not path.is_file():
        return f"Error: file '{filename}' not found in the data folder."
    return path.read_text(encoding="utf-8")[:2000]


TOOLS = [
    Tool(
        "calculator",
        "Evaluate a math expression using + - * / ** and parentheses. "
        "Input: an expression like 2+3*4",
        calculator,
    ),
    Tool(
        "search",
        "Look up a short fact. Input: a search query like 'height of eiffel tower'",
        search,
    ),
    Tool(
        "file_lookup",
        "Read a text file from the data folder. Input: a file name like notes.txt",
        file_lookup,
    ),
]