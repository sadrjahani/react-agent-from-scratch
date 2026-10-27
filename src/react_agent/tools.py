"""Tools the agent can call.

Every tool is a plain function ``str -> str`` wrapped in a :class:`Tool`.
The model sends the input as text (``Action Input:``) and receives the result
as text (``Observation:``), so strings in and strings out keep the tools simple
and easy to test without any LLM.

Contract with the rest of the project:
    - ``prompt.py`` reads ``Tool.name`` and ``Tool.description``.
    - ``loop.py`` reads ``Tool.name`` and calls ``Tool.func``.
    - Expected problems the model can fix (missing file, outside the sandbox)
      are *returned* as ``"Error: ..."`` text. Unexpected problems are raised;
      the loop turns them into an Observation.
"""

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TOOLS", "Tool", "calculator", "file_lookup", "search"]

# --- Limits (named so they are easy to find and change) ---------------------
MAX_EXPRESSION_LENGTH = 200  # blocks huge or deeply nested input (RecursionError)
MAX_EXPONENT = 100  # blocks expressions like 9**9**9 that freeze the program
MAX_FILE_CHARS = 2000  # keeps one Observation from filling the model's context

# Sandbox folder, relative to the directory the program is started from.
DATA_DIR = Path("data")

Number = int | float


@dataclass(frozen=True)
class Tool:
    """A tool the agent can call.

    Attributes:
        name: Exact name the model must write after ``Action:``.
        description: What the tool does and an example input. This text is
            all the model knows about the tool, so write it carefully.
        func: Function that takes the ``Action Input`` text and returns the
            result text.
    """

    name: str
    description: str
    func: Callable[[str], str]


# --- calculator -------------------------------------------------------------
# Why not eval()? eval("__import__('os')...") runs arbitrary code. Instead we
# parse the text into a syntax tree and evaluate only whitelisted node types.
_BINARY_OPS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node: ast.AST) -> Number:
    """Evaluate one syntax-tree node, allowing only numbers and + - * / **.

    Raises:
        ValueError: If the node type is not whitelisted or the exponent is
            too large.
    """
    # type(...) in (int, float) instead of isinstance: bool is a subclass of
    # int, and we do not want True/False (or strings) to count as numbers.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)  # recursion handles 2+3*4 correctly
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression such as ``2+3*4``.

    Args:
        expression: Text with numbers, ``+ - * / **`` and parentheses.

    Returns:
        The result as text, e.g. ``"14"``.

    Raises:
        ValueError: Unsupported or too long expression, or exponent too large.
        SyntaxError: The text is not a valid expression.
        ZeroDivisionError: Division by zero.
    """
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")  # parses only; never executes
    return str(_eval_node(tree.body))


# --- search (fake, offline) ---------------------------------------------------
# A tiny fixed knowledge base so the project needs no API key and tests are
# deterministic. Replace with a real search later without changing the tool
# interface.
_FACTS = {
    "capital of france": "Paris is the capital of France.",
    "population of france": "France has a population of about 68 million people.",
    "height of eiffel tower": "The Eiffel Tower is about 330 meters tall.",
    "creator of python": "Python was created by Guido van Rossum.",
}


def search(query: str) -> str:
    """Look up a short fact in the built-in fake knowledge base.

    A fact matches when its key appears anywhere in the query (ignoring case),
    so "What is the height of Eiffel Tower?" finds "height of eiffel tower".

    Returns:
        The fact, or ``"No results found."``.
    """
    q = query.lower()
    for key, answer in _FACTS.items():
        if key in q:
            return answer
    return "No results found."


# --- file_lookup (sandboxed) ---------------------------------------------------
def file_lookup(filename: str) -> str:
    """Read a UTF-8 text file from the sandbox folder ``DATA_DIR``.

    Security: the path is resolved first (removing ``..`` and symlinks) and
    must stay inside ``DATA_DIR``; otherwise the model could read any file on
    the machine (path traversal).

    Returns:
        The first ``MAX_FILE_CHARS`` characters of the file, or an
        ``"Error: ..."`` message the model can act on.
    """
    base = DATA_DIR.resolve()
    path = (base / filename.strip()).resolve()
    if not path.is_relative_to(base):
        return "Error: access outside the data folder is not allowed."
    if not path.is_file():
        return f"Error: file '{filename}' not found in the data folder."
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: file '{filename}' is not a UTF-8 text file."
    return text[:MAX_FILE_CHARS]


# --- registry ---------------------------------------------------------------------
# To add a tool: write a ``str -> str`` function and add one Tool(...) here.
# prompt.py and loop.py need no changes.
TOOLS: list[Tool] = [
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