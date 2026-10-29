from react_agent.prompt import build_system_prompt
from react_agent.tools import Tool


def test_prompt_contains_tools():
    tools = [Tool("calculator", "Evaluate a math expression", lambda x: x)]
    prompt = build_system_prompt(tools)
    assert "calculator" in prompt
    assert "Evaluate a math expression" in prompt
    assert "Final Answer:" in prompt