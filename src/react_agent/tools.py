"""Tools the agent can call: a safe calculator, a fake search and a sandboxed file reader.

Every tool is a plain function ``str -> str`` wrapped in a :class:`Tool`.
Problems the model can fix (missing file, outside the sandbox) are returned as
``"Error: ..."`` text. Unexpected ones are raised; the loop turns them into an
Observation.
"""

import ast
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["TOOLS", "Tool", "calculator", "file_lookup", "search"]

# Limits that stop huge input from freezing the program or flooding the context.
MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100
MAX_FILE_CHARS = 2000

# Relative to the directory the program is started from.
DATA_DIR = Path("data")

Number = int | float


@dataclass(frozen=True)
class Tool:
    """A tool the agent can call.

    ``description`` is all the model knows about the tool, so include an
    example input.
    """

    name: str
    description: str
    func: Callable[[str], str]


# No eval(): it would run any code the model writes. Only whitelisted node
# types are evaluated.
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
    """Evaluate a syntax-tree node; only numbers and whitelisted operators pass."""
    # type() instead of isinstance: bool is a subclass of int, and True must not
    # count as a number.
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("exponent too large")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression such as ``2+3*4``.

    Raises:
        ValueError: Unsupported, too long, or too large an exponent.
        SyntaxError: The text is not a valid expression.
        ZeroDivisionError: Division by zero.
    """
    expression = expression.strip()
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError("expression too long")
    tree = ast.parse(expression, mode="eval")  # parses only; never executes
    return str(_eval_node(tree.body))


# Fixed facts keep the project offline and the tests deterministic.
_FACTS = {
    "capital of france": "Paris is the capital of France.",
    "population of france": "France has a population of about 68 million people.",
    "height of eiffel tower": "The Eiffel Tower is about 330 meters tall.",
    "creator of python": "Python was created by Guido van Rossum.",
}


def search(query: str) -> str:
    """Return the fact whose key appears in the query (any case), or "No results found."."""
    q = query.lower()
    for key, answer in _FACTS.items():
        if key in q:
            return answer
    return "No results found."


def file_lookup(filename: str) -> str:
    """Read a UTF-8 text file inside ``DATA_DIR``.

    The path is resolved first, so ``..`` and absolute paths cannot escape the
    folder. Returns the first ``MAX_FILE_CHARS`` characters or an ``Error: ...``
    message.
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


# To add a tool, write a ``str -> str`` function and register it here.
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