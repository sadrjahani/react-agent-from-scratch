"""Tests for the command-line interface. No real LLM or network is used."""

from react_agent import cli


def test_cli_prints_answer_and_exits_zero(monkeypatch, capsys) -> None:
    calls = {}

    def fake_run_agent(question, tools, max_iters=6):
        calls.update(question=question, tools=tools, max_iters=max_iters)
        return {
            "answer": "42",
            "finished": True,
            "trace": [{"step": 1, "model": "Thought: done\nFinal Answer: 42"}],
        }

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)

    exit_code = cli.main(["What is 2+2?"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "42" in out
    assert calls["question"] == "What is 2+2?"
    assert calls["max_iters"] == 6


def test_cli_max_iters_flag_is_passed_through(monkeypatch) -> None:
    calls = {}

    def fake_run_agent(question, tools, max_iters=6):
        calls["max_iters"] = max_iters
        return {"answer": "ok", "finished": True, "trace": []}

    monkeypatch.setattr(cli, "run_agent", fake_run_agent)

    assert cli.main(["A question", "--max-iters", "3"]) == 0
    assert calls["max_iters"] == 3


def test_cli_exits_one_when_iteration_limit_is_reached(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "run_agent",
        lambda question, tools, max_iters=6: {
            "answer": None,
            "finished": False,
            "trace": [],
        },
    )

    assert cli.main(["A question"]) == 1
    assert "iteration limit" in capsys.readouterr().out


def test_cli_prints_every_step_and_observation(monkeypatch, capsys) -> None:
    result = {
        "answer": "660",
        "finished": True,
        "trace": [
            {
                "step": 1,
                "model": "Thought: calculate\nAction: calculator\nAction Input: 330*2",
                "observation": "660",
            },
            {"step": 2, "model": "Final Answer: 660"},
        ],
    }
    monkeypatch.setattr(cli, "run_agent", lambda q, t, max_iters=6: result)

    exit_code = cli.main(["A question"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "--- step 1 ---" in out
    assert "--- step 2 ---" in out
    assert "Observation: 660" in out
    assert "Final Answer: 660" in out