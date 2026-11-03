
"""Command-line interface: run the agent on a question from the terminal."""

import argparse

from react_agent.llm import chat
from react_agent.loop import run_agent
from react_agent.tools import TOOLS

__all__ = ["main"]


def main(
    argv: list[str] | None = None,
    *,
    llm=chat,
) -> int:
    """Run the agent on one question and print every step.

    Returns:
        0 if the agent reached a final answer, 1 if the iteration limit
        was reached first.
    """
    parser = argparse.ArgumentParser(
        prog="react_agent",
        description="Run the ReAct agent on a question.",
    )
    parser.add_argument("question", help="the question for the agent")
    parser.add_argument(
        "--max-iters",
        type=int,
        default=6,
        help="maximum number of model calls (default: 6)",
    )
    args = parser.parse_args(argv)

    result = run_agent(
        args.question,
        TOOLS,
        max_iters=args.max_iters,
        llm=llm,
    )

    for step in result["trace"]:
        print(f"--- step {step['step']} ---")
        print(step["model"])
        if "observation" in step:
            print(f"Observation: {step['observation']}")

    if result["finished"]:
        print(f"\nFinal Answer: {result['answer']}")
        return 0

    print(
        "\nNo final answer: the iteration limit was reached. "
        "Try again with a larger --max-iters."
    )
    return 1